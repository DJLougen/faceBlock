/**
 * Scheduling primitives for the offscreen document. Pure — no DOM, no
 * chrome APIs — so the policy is unit-testable in Bun.
 *
 * Two separate lanes, deliberately:
 *
 *   - createJobQueue: the SERIAL model lane. At most one job runs at a time
 *     because ORT sessions must never overlap. Frame jobs (sampled video)
 *     are prioritized over ordinary inference and supersede a still-pending
 *     earlier frame — a stale frame is worthless. A job that outlives its
 *     timeout rejects its caller, but the lane stays locked until the
 *     underlying work actually settles: a model call that cannot be
 *     cancelled must never overlap the next one.
 *
 *   - createPool: a bounded concurrency lane for network/download work.
 *     Downloads must never hold the model lane — a stalled fetch would
 *     otherwise starve inference behind it.
 */

export type JobKind = "inference" | "frame";

/** Opaque platform timer handle. */
type TimerHandle = ReturnType<typeof setTimeout>;

export interface JobQueueOptions {
  /** Queued jobs older than this are dropped before they run. */
  maxWaitMs: number;
  /** Maximum pending jobs; beyond this enqueue rejects immediately. */
  maxPending: number;
  /** Hard ceiling per job; the caller is rejected, the lane stays locked. */
  jobTimeoutMs: number;
}

interface Job {
  run: () => Promise<unknown>;
  resolve: (v: unknown) => void;
  reject: (e: unknown) => void;
  enqueuedAt: number;
  kind: JobKind;
  /** Runs once when the job can never run again; see JobQueue.enqueue. */
  cleanup?: () => void;
}

export interface JobQueue {
  /**
   * Enqueue work. `cleanup` runs exactly once when the job can never run
   * again: immediately when rejected before starting (queue full, expired,
   * superseded) or after the underlying work settles — including after a
   * caller-facing timeout, so resources the job still holds are not
   * released out from under it.
   */
  enqueue<T>(run: () => Promise<T>, kind?: JobKind, cleanup?: () => void): Promise<T>;
  /** Pending jobs, for diagnostics/tests. */
  pending(): number;
}

export function createJobQueue(opts: JobQueueOptions): JobQueue {
  if (!(opts.maxWaitMs > 0) || !(opts.maxPending > 0) || !(opts.jobTimeoutMs > 0)) {
    throw new Error("faceBlock: job queue requires positive maxWaitMs/maxPending/jobTimeoutMs");
  }
  const queue: Job[] = [];
  let pumping = false;

  /** Invoke a job's cleanup exactly once, guarded so a throwing cleanup cannot break the pump. */
  function cleanupJob(job: Job): void {
    const fn = job.cleanup;
    job.cleanup = undefined;
    if (!fn) return;
    try {
      fn();
    } catch {
      /* cleanup must never break scheduling */
    }
  }

  function expireStale(now: number): void {
    for (let i = queue.length - 1; i >= 0; i--) {
      const job = queue[i]!;
      if (now - job.enqueuedAt > opts.maxWaitMs) {
        queue.splice(i, 1);
        job.reject(new Error("faceBlock: request expired waiting in the inference queue"));
        cleanupJob(job);
      }
    }
  }

  function enqueue<T>(
    run: () => Promise<T>,
    kind: JobKind = "inference",
    cleanup?: () => void,
  ): Promise<T> {
    const now = Date.now();
    expireStale(now);
    if (kind === "frame") {
      // A fresher frame supersedes a still-pending one — queueing stale
      // frames behind each other just delays the only one that matters.
      for (let i = queue.length - 1; i >= 0; i--) {
        const job = queue[i]!;
        if (job.kind === "frame") {
          queue.splice(i, 1);
          job.reject(new Error("faceBlock: superseded by a fresher video frame"));
          cleanupJob(job);
        }
      }
    }
    if (queue.length >= opts.maxPending) {
      if (cleanup) {
        try {
          cleanup();
        } catch {
          /* cleanup must never break scheduling */
        }
      }
      return Promise.reject(
        new Error(
          `faceBlock: inference queue full (${opts.maxPending} pending) — try again shortly`,
        ),
      );
    }
    return new Promise<T>((resolve, reject) => {
      queue.push({
        run: run as () => Promise<unknown>,
        resolve: resolve as (v: unknown) => void,
        reject,
        enqueuedAt: now,
        kind,
        cleanup,
      });
      void pump();
    });
  }

  async function pump(): Promise<void> {
    if (pumping) return;
    pumping = true;
    try {
      while (queue.length > 0) {
        // Frames jump ahead of ordinary inference: a sampled video frame is
        // only useful while it is still roughly current.
        let idx = queue.findIndex((j) => j.kind === "frame");
        if (idx === -1) idx = 0;
        const job = queue.splice(idx, 1)[0]!;
        if (Date.now() - job.enqueuedAt > opts.maxWaitMs) {
          job.reject(new Error("faceBlock: request expired waiting in the inference queue"));
          cleanupJob(job);
          continue;
        }
        const work = Promise.resolve().then(job.run);
        // The caller is rejected at jobTimeoutMs, but the lane stays locked
        // until run() actually settles — an uncancellable model call must
        // never overlap the next job.
        const settled = work.then(
          () => undefined,
          () => undefined,
        );
        let timer: TimerHandle | undefined;
        try {
          job.resolve(
            await Promise.race([
              work,
              new Promise<never>((_, reject) => {
                timer = setTimeout(
                  () =>
                    reject(
                      new Error(
                        `faceBlock: inference job timed out after ${Math.round(opts.jobTimeoutMs / 1000)}s`,
                      ),
                    ),
                  opts.jobTimeoutMs,
                );
              }),
            ]),
          );
        } catch (e) {
          job.reject(e);
        } finally {
          clearTimeout(timer);
        }
        await settled;
        cleanupJob(job);
      }
    } finally {
      pumping = false;
    }
  }

  return { enqueue, pending: () => queue.length };
}

/**
 * Bounded concurrency pool for network/download work. Tasks beyond the
 * limit wait in FIFO order; `maxPending` caps the backlog so a flood of
 * candidates cannot accumulate unbounded waiters.
 */
export interface Pool {
  run<T>(task: () => Promise<T>): Promise<T>;
  pending(): number;
}

export function createPool(concurrency: number, maxPending: number): Pool {
  if (!(concurrency > 0) || !(maxPending > 0)) {
    throw new Error("faceBlock: pool requires positive concurrency and maxPending");
  }
  let active = 0;
  const waiters: Array<{
    task: () => Promise<unknown>;
    resolve: (v: unknown) => void;
    reject: (e: unknown) => void;
  }> = [];

  function kick(): void {
    while (active < concurrency && waiters.length > 0) {
      const w = waiters.shift()!;
      active += 1;
      Promise.resolve()
        .then(w.task)
        .then(w.resolve, w.reject)
        .finally(() => {
          active -= 1;
          kick();
        });
    }
  }

  return {
    run<T>(task: () => Promise<T>): Promise<T> {
      if (waiters.length >= maxPending) {
        return Promise.reject(
          new Error(`faceBlock: download queue full (${maxPending} pending)`),
        );
      }
      return new Promise<T>((resolve, reject) => {
        waiters.push({
          task: task as () => Promise<unknown>,
          resolve: resolve as (v: unknown) => void,
          reject,
        });
        kick();
      });
    },
    pending: () => waiters.length,
  };
}
