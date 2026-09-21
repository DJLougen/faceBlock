import { describe, expect, test } from "bun:test";
import { createJobQueue, createPool } from "../../extension/jobqueue.ts";

function deferred<T>(): { promise: Promise<T>; resolve: (v: T) => void; reject: (e: unknown) => void } {
  const { promise, resolve, reject } = Promise.withResolvers<T>();
  return { promise, resolve, reject };
}

const OPTS = { maxWaitMs: 5_000, maxPending: 8, jobTimeoutMs: 200 };

describe("createJobQueue", () => {
  test("rejects non-positive options", () => {
    expect(() => createJobQueue({ ...OPTS, maxPending: 0 })).toThrow();
    expect(() => createJobQueue({ ...OPTS, jobTimeoutMs: 0 })).toThrow();
    expect(() => createJobQueue({ ...OPTS, maxWaitMs: -1 })).toThrow();
  });

  test("serializes jobs: at most one runs at a time", async () => {
    const q = createJobQueue(OPTS);
    const started: string[] = [];
    const firstStarted = deferred<void>();
    const gate = deferred<void>();
    const j1 = q.enqueue(async () => {
      started.push("one");
      firstStarted.resolve();
      await gate.promise;
      started.push("one-done");
    });
    const j2 = q.enqueue(async () => {
      started.push("two");
    });
    await firstStarted.promise;
    // The lane is locked by j1, so j2 cannot have started — no timing guess.
    expect(started).toEqual(["one"]);
    gate.resolve();
    await Promise.all([j1, j2]);
    expect(started).toEqual(["one", "one-done", "two"]);
  });

  test("a synchronous throw rejects that job and the queue continues", async () => {
    const q = createJobQueue(OPTS);
    await expect(
      q.enqueue(() => {
        throw new Error("sync boom");
      }),
    ).rejects.toThrow("sync boom");
    await expect(q.enqueue(async () => 7)).resolves.toBe(7);
  });

  test("frame jobs jump ahead of pending inference jobs", async () => {
    const q = createJobQueue(OPTS);
    const gate = deferred<void>();
    const order: string[] = [];
    const first = q.enqueue(async () => {
      order.push("a");
      await gate.promise;
    });
    const inf = q.enqueue(async () => {
      order.push("b");
    });
    const frame = q.enqueue(async () => {
      order.push("f");
    }, "frame");
    gate.resolve();
    await Promise.all([first, inf, frame]);
    expect(order).toEqual(["a", "f", "b"]);
  });

  test("a fresher frame supersedes a still-pending frame", async () => {
    const q = createJobQueue(OPTS);
    const gate = deferred<void>();
    const first = q.enqueue(() => gate.promise);
    const stale = q.enqueue(async () => "stale", "frame");
    const fresh = q.enqueue(async () => "fresh", "frame");
    gate.resolve();
    await first;
    await expect(stale).rejects.toThrow("superseded");
    await expect(fresh).resolves.toBe("fresh");
  });

  // Real-timer exception: jobTimeoutMs is the queue's own wall-clock deadline,
  // so this test exercises it with a small genuine timeout.
  test("a timed-out job rejects its caller but the lane stays locked until it settles", async () => {
    const q = createJobQueue({ ...OPTS, jobTimeoutMs: 30 });
    const gate = deferred<string>();
    const order: string[] = [];
    const slow = q.enqueue(async () => {
      await gate.promise;
      order.push("slow-settled");
      return "slow";
    });
    const next = q.enqueue(async () => {
      order.push("next-ran");
      return "next";
    });
    await expect(slow).rejects.toThrow("timed out");
    // The rejection is the signal: the next job must not have run while the
    // lane was still locked by the unsettled slow job.
    expect(order).toEqual([]);
    gate.resolve("done");
    await expect(next).resolves.toBe("next");
    expect(order).toEqual(["slow-settled", "next-ran"]);
  });

  test("enqueue rejects when the pending backlog is full", async () => {
    const q = createJobQueue({ ...OPTS, maxPending: 1 });
    const gate = deferred<void>();
    const running = q.enqueue(() => gate.promise);
    const pending = q.enqueue(async () => 1);
    await expect(q.enqueue(async () => 2)).rejects.toThrow("queue full");
    gate.resolve();
    await running;
    await pending;
  });
});

describe("createPool", () => {
  test("rejects non-positive options", () => {
    expect(() => createPool(0, 4)).toThrow();
    expect(() => createPool(2, 0)).toThrow();
  });

  test("bounds concurrency and drains waiters", async () => {
    const pool = createPool(2, 8);
    let active = 0;
    let maxActive = 0;
    const release = deferred<void>();
    const twoStarted = deferred<void>();
    const task = async () => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      if (active === 2) twoStarted.resolve();
      await release.promise;
      active -= 1;
    };
    const runs = Array.from({ length: 6 }, () => pool.run(task));
    await twoStarted.promise;
    expect(maxActive).toBe(2);
    expect(pool.pending()).toBe(4);
    release.resolve();
    await Promise.all(runs);
    expect(maxActive).toBe(2);
  });

  test("a synchronous throw rejects that task and frees the slot", async () => {
    const pool = createPool(1, 4);
    await expect(
      pool.run(() => {
        throw new Error("sync boom");
      }),
    ).rejects.toThrow("sync boom");
    await expect(pool.run(async () => 3)).resolves.toBe(3);
  });

  test("rejects when the waiter backlog is full", async () => {
    const pool = createPool(1, 1);
    const gate = deferred<void>();
    const running = pool.run(() => gate.promise);
    const waiting = pool.run(async () => 1);
    await expect(pool.run(async () => 2)).rejects.toThrow("download queue full");
    gate.resolve();
    await running;
    await waiting;
  });
});

describe("candidate-lifetime composition (mirrors offscreen wiring)", () => {
  test("resident candidates stay bounded and a frame interleaves between units", async () => {
    // Mirrors extension/offscreen.ts: candidate units hold a resource from
    // "download" through the serialized inference job, released by the job's
    // cleanup; the pool slot is held until cleanup actually ran.
    const candidates = createPool(4, 64);
    const model = createJobQueue({ maxWaitMs: 5_000, maxPending: 64, jobTimeoutMs: 5_000 });
    let resident = 0;
    let maxResident = 0;
    const order: string[] = [];
    const gate = deferred<void>();
    const firstStarted = deferred<void>();

    const unit = (i: number) =>
      candidates.run(async () => {
        resident += 1;
        maxResident = Math.max(maxResident, resident);
        const released = deferred<void>();
        try {
          await model.enqueue(
            async () => {
              order.push(`c${i}`);
              if (i === 0) firstStarted.resolve();
              await gate.promise;
            },
            "inference",
            () => {
              resident -= 1;
              released.resolve();
            },
          );
        } finally {
          await released.promise;
        }
      });

    const units = Array.from({ length: 10 }, (_, i) => unit(i));
    // Wait until the first candidate's model job is actually running before
    // enqueueing the frame — otherwise the frame could land first and the
    // interleave assertion would be vacuous.
    await firstStarted.promise;
    const frame = model.enqueue(async () => {
      order.push("frame");
    }, "frame");
    gate.resolve();
    await Promise.all([...units, frame]);
    expect(maxResident).toBeLessThanOrEqual(4);
    // The frame ran strictly between candidate units: after c0 and before
    // the last candidate finished.
    const fi = order.indexOf("frame");
    expect(fi).toBeGreaterThan(0);
    expect(fi).toBeLessThan(10);
  });

  test("a caller-facing timeout keeps the pool slot held until the job settles", async () => {
    // The lifecycle guarantee: enqueue rejects the caller at jobTimeoutMs,
    // but the underlying model call may still be running — the candidate
    // pool slot (and its decoded bitmap) must stay held until cleanup runs.
    const candidates = createPool(1, 8);
    const model = createJobQueue({ maxWaitMs: 5_000, maxPending: 8, jobTimeoutMs: 30 });
    const gate = deferred<void>();
    const callerTimedOut = deferred<void>();
    let resident = 0;
    let maxResident = 0;

    const slow = candidates.run(async () => {
      resident += 1;
      maxResident = Math.max(maxResident, resident);
      const released = deferred<void>();
      try {
        await model.enqueue(() => gate.promise, "inference", () => {
          resident -= 1;
          released.resolve();
        });
      } catch {
        // Caller observed the timeout; the underlying job is still gated.
        callerTimedOut.resolve();
      } finally {
        await released.promise;
      }
    });

    // The caller has rejected, but the model job is still running — the
    // second unit must wait because the slot is still held.
    await callerTimedOut.promise;
    let secondFinished = false;
    const second = candidates
      .run(async () => {
        resident += 1;
        maxResident = Math.max(maxResident, resident);
        const released = deferred<void>();
        try {
          await model.enqueue(async () => 1, "inference", () => {
            resident -= 1;
            released.resolve();
          });
        } finally {
          await released.promise;
        }
      })
      .then(() => {
        secondFinished = true;
      });
    // Give the pool a macrotask to wrongly admit the second unit.
    await Bun.sleep(20);
    expect(secondFinished).toBe(false); // slot still held by the timed-out job
    gate.resolve();
    await Promise.all([slow, second]);
    expect(secondFinished).toBe(true);
    expect(maxResident).toBe(1); // never two resident bitmaps on a 1-slot pool
  });

  test("cleanup runs on pre-run rejection so resources are not leaked", async () => {
    const model = createJobQueue({ maxWaitMs: 5_000, maxPending: 1, jobTimeoutMs: 5_000 });
    const gate = deferred<void>();
    let cleaned = 0;
    const running = model.enqueue(() => gate.promise);
    const pending = model.enqueue(async () => 1, "inference", () => {
      cleaned += 1;
    });
    // Queue full: cleanup fires immediately on the rejected enqueue.
    await expect(
      model.enqueue(async () => 2, "inference", () => {
        cleaned += 1;
      }),
    ).rejects.toThrow("queue full");
    expect(cleaned).toBe(1);
    gate.resolve();
    await running;
    await pending;
    // cleanup runs a microtask after the job's promise resolves.
    await Bun.sleep(0);
    expect(cleaned).toBe(2);
  });
});
