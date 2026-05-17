---
title: "Benchmarking KPipe against the parallel-Kafka libraries you would actually pick"
date: 2026-05-16
description: "How KPipe stacks up against Confluent Parallel Consumer, Reactor Kafka, and a hand-rolled
  KafkaConsumer + virtual-thread executor — first published numbers from the new 4-runtime harness."
tags:
  - kafka
  - java
  - benchmarks
  - virtual-threads
  - jmh
---

KPipe is a Kafka consumer library built on Java 25 virtual threads. The pitch is that you get the
performance of a hand-rolled `KafkaConsumer` + `Thread.ofVirtual()` loop with the operational stack
already wired up: lowest-pending-offset commits, retry, DLQ producer, backpressure with hysteresis,
circuit breaker, OTel metrics + tracing, batch sinks, `Result<T>` typed pipeline outcomes, graceful
shutdown.

The question this post answers is what that pitch costs. KPipe is now benchmarked against three
alternatives: **Confluent Parallel Consumer**, **Reactor Kafka**, and a hand-rolled **raw
`KafkaConsumer` + virtual-thread executor** baseline. Three workload regimes. Real Kafka 4.2.0 broker
(Testcontainers, not in-process). JMH-published scores, GC profiler, raw JSON committed alongside
this post in [`benchmarks/results/`][bench-results].

The short answer: **KPipe captures 85–95% of raw Loom throughput** and **degrades gracefully under
blocking work**, with about a 9% drop from 0 → 1000 µs of per-record I/O. Confluent drops 35% over
the same sweep. Reactor Kafka drops 96%. The rest of this post is the numbers, the methodology, and
the saga of getting Reactor onto the bench at all.

## Headline

Records / second, higher is better. `workMicros` is per-record simulated work via `LockSupport.parkNanos`
— 0 µs is pure framework overhead, 100 µs is local enrichment, 1000 µs is a blocking I/O round trip.
All four runtimes run against the same Testcontainers-managed Kafka 4.2.0 broker, same 25,000-record
seed, same eight partitions, two JMH forks × five measurement iterations.

| Runtime                         |      `workMicros=0` |     `workMicros=100` |    `workMicros=1000` |
|---------------------------------|--------------------:|---------------------:|---------------------:|
| **Raw `KafkaConsumer` + VT**    |    542,859 ± 34,077 |     500,231 ± 40,168 |     482,639 ± 50,406 |
| **KPipe**                       |**473,491 ± 79,218** | **461,273 ± 55,179** |**430,526 ± 117,613** |
| **Reactor Kafka 1.3.25**        |    256,648 ± 25,508 |       77,542 ± 1,054 |        **8,979 ± 34**|
| **Confluent Parallel Consumer** |    100,106 ± 19,160 |      108,194 ± 6,652 |       64,715 ± 3,091 |

![Parallel-consumer throughput, allocation, and GC profile](parallel-gc-baseline.svg)

## The Reactor cliff

The headline finding from this run isn't where KPipe lands. It's where Reactor Kafka lands.

At `workMicros=0` Reactor is fine — 257k ops/sec, between Confluent and KPipe. At `workMicros=100`
it falls *below* Confluent. At `workMicros=1000` it does **8,979 ops/sec** — that's **48× slower
than KPipe** and **7× slower than Confluent** at the same workload. The error band of ±34 on 8,979
says it's consistently terrible, not noise. Something pathological happens to `Flux.parallel(100)`
when records park on `LockSupport.parkNanos`.

I went into this run expecting Reactor to land between Confluent and the Loom runtimes — a third
option for teams who want reactive composition without a thread pool ceiling. The data says the
opposite: if your per-record work blocks at all, Reactor Kafka is the worst of the four. The "don't
pick Reactor Kafka for blocking work" claim is now evidence-backed, not a vibe.

## Three things the throughput says cleanly

**1. Raw `KafkaConsumer + VT` is the fastest runtime; KPipe trails by single digits to low teens.**
At `workMicros=0`, Raw 543k vs KPipe 473k — about 13%. At `workMicros=100`, ~8%. At `workMicros=1000`,
~11%. The earlier smoke-run claim of "KPipe ≈ raw within error" was too strong; with this run's tighter
error bands they're close but not statistically identical. There's a small measurable framework cost.

Given everything KPipe layers on top of the raw loop — offset manager with lowest-pending-offset
commits, retry, DLQ producer, backpressure with hysteresis, circuit breaker, OTel metric + tracing
hooks, batch sinks, `Result<T>` typed pipeline outcomes, lifecycle management with graceful shutdown —
a 5–15% framework cost is the right shape. The framework is not free; it's also not making Loom slower
in any practical sense.

**2. Both Loom-based runtimes are 4.3× – 7.5× ahead of Confluent Parallel Consumer.** The gap widens
at `workMicros=1000`. Confluent's 100-worker thread pool can only have 100 records simultaneously
parked on the simulated I/O wait; under blocking work the pool serialises. KPipe and raw don't
serialise — virtual threads scale beyond the partition count, each blocked record costs kilobytes
instead of megabytes.

**3. KPipe degrades gracefully across the workload sweep.** 473k → 461k → 431k. The drop from
`workMicros=0` to `workMicros=1000` is about 9%. Confluent drops 35% over the same sweep. Reactor
drops 96%. This is the Loom thesis paying off in the only way that matters: blocking work doesn't
crater the runtime.

## What this does not say

A bench number with no caveat is a marketing pitch. The honest framing:

- **The Loom runtimes win because of Loom, not because of KPipe.** "KPipe is 5× Confluent" is true but
  unfairly framed — the right comparison is "Loom-based parallel consumption is 5× a platform-thread
  pool." KPipe's contribution is bringing that win with a small, measurable framework cost.
- **KPipe allocates more per record.** ~924 B/op at `workMicros=0` vs raw's 55 B/op vs Confluent's
  34 B/op. On this run that didn't depress throughput — the JVM's young gen absorbed it. On a tight
  heap, or a workload sensitive to GC tail latency, watch this number.
- **One payload shape, one broker config.** Small JSON, single-broker Testcontainers, replication
  factor 1. Production with a network broker and replication 3 / acks=all has different absolute
  numbers. Headline ordering between runtimes usually survives the move; the gaps shift.
- **No latency-percentile data yet.** `ParallelProcessingLatencyBenchmark` was not exercised in this
  run. Average throughput is half the story; p99 is the other half, and a runtime can rank one way on
  throughput and the opposite way on tail. Coming in the next snapshot.

## Allocation and GC

| Runtime                     | `workMicros=0` B/op | `workMicros=100` B/op | `workMicros=1000` B/op |
|-----------------------------|--------------------:|----------------------:|-----------------------:|
| Confluent Parallel Consumer |                  34 |                    34 |                     35 |
| Raw `KafkaConsumer` + VT    |                  55 |                   431 |                    455 |
| Reactor Kafka               |                 175 |                    83 |                    193 |
| KPipe                       |                 924 |                 1,513 |                  1,436 |

Confluent allocates least (~34 B/op, flat across the sweep — its long-lived worker pool doesn't
allocate much per record). Raw is second-leanest at `workMicros=0` then jumps when blocked-record
Runnable lambdas pile up. Reactor allocates moderately (83–193 B/op) — surprising given its throughput
collapse, but its problem is scheduling, not bytes. KPipe allocates most (924–1,513 B/op) because of
`Result<T>` wrappers, the per-record pipeline builder hand-off, and a fresh virtual thread per record.

The GC story is counter-intuitive but consistent with throughput. Confluent's smaller per-record
allocations still produce more total GC events than KPipe or Raw because its slower iterations stay
in the young gen longer. Reactor's GC numbers at `workMicros=1000` look low (33 events / 98 ms) only
because the benchmark is barely running — at ~9 records/sec there's almost no allocation pressure
to clear. Don't read that as well-behaved; the throughput number tells the real story.

## How the harness got here

The first attempt at the new bench was wrong, and the wrongness was instructive.

**Attempt 1 — in-process Kafka via `KafkaClusterTestKit`.** Same approach as the prior 10k baseline.
Worked at 10k records, didn't scale. At 25k records with four invocation contexts loaded, the in-process
broker collapsed under load on a shared-core laptop — KRaft controller events, group coordinator,
producer seed, and consumer under test all fighting for the same CPUs. Smoke tests timed out at ~50
records/sec across every framework. **The bench was measuring broker contention, not framework
throughput.**

**Attempt 2 — Testcontainers Kafka.** Real Kafka 4.2.0 broker in a Docker container, on its own JVM
and own cores. Consumer is the bottleneck again. The same smoke test that got 50 records/sec on the
in-process harness now gets 500,000+ records/sec. That's not a 10x improvement — that's "measuring
the right thing now."

The lesson: **for parallel-consumer benchmarks, the broker has to be external.** Testcontainers is the
cheap path; a sidecar on dedicated hardware is the production-faithful path. In-process Kafka is fine
for "does my code compile and run end-to-end" tests. It is not fine for performance comparison.

## The Reactor Kafka saga

Getting Reactor onto the bench at all took two version bumps and a fat-jar exorcism.

Reactor Kafka 1.3.23 (the latest stable on Maven Central when I first set up the bench) **crashes on
first record** against `kafka-clients:4.x`:

```
java.lang.NoSuchMethodError: 'void org.apache.kafka.clients.consumer.ConsumerRecord.<init>(...)'
    at reactor.kafka.receiver.ReceiverRecord.<init>(ReceiverRecord.java:48)
```

The `ConsumerRecord` ctor `ReceiverRecord` calls was removed in kafka-clients 4.0. Issue
[#420](https://github.com/reactor/reactor-kafka/issues/420) on the upstream tracks this — opened March
2025, fix landed November 2025 as part of 1.3.25. The fix avoids the deprecated ctor in favour of one
that exists in both kafka-clients 3.x and 4.x.

Two gotchas worth recording:

- **Maven Central's search API was stale** when I checked. It returned 1.3.23 as the latest version
  even though 1.3.25 was already deployed. Always cross-check the direct directory listing
  (`https://repo1.maven.org/maven2/<groupId>/<artifactId>/`).
- **The fat JMH jar caches transitive bytecode.** After bumping `reactor-kafka` from 1.3.23 to 1.3.25
  in `libs.versions.toml`, the first smoke test still produced the old `NoSuchMethodError`. The fat
  jar had the 1.3.23 `ReceiverRecord.class` baked in. Force-cleaning fixed it:

  ```bash
  rm -rf benchmarks/build && ./gradlew :benchmarks:jmhJar --rerun-tasks
  ```

  Verified the right bytecode landed in the fresh jar with `javap -c -p`.

On 1.3.25 Reactor runs cleanly across the full sweep — which is how I got the cliff numbers above.
The compatibility fix unlocked the measurement; the measurement is what revealed the blocking-work
problem.

## Reproduce locally

The bench code is on `main`:

```bash
just bench               # full 4-runtime publishing run, ~30–60 min on a quiet machine
just bench mode=smoke    # ~3–5 min KPipe-only sanity iteration
just bench mode=latency  # p50 / p95 / p99 / p999 companion
```

Output lands in `benchmarks/results/<date>.json` and `benchmarks/results/<date>.log`. The companion
human-readable summary template is in [`benchmarks/results/TEMPLATE.md`][template]. Methodology + the
runtime config matrix is in [`benchmarks/METHODOLOGY.md`][methodology]. **Docker must be running** —
Testcontainers will pull `apache/kafka:4.2.0` on first invocation.

## What's next

- Latency-percentile companion run — `ParallelProcessingLatencyBenchmark` reports `p50 / p95 / p99 /
  p999` per runtime. The Reactor cliff in average throughput says nothing about whether the few
  records that *do* get through have decent or terrible tail latency. Worth knowing.
- Multi-partition sweep (8 / 32 / 128) to expose how each runtime scales with parallelism opportunity.
- Payload-size sweep (100 B / 1 KB / 10 KB) — Reactor's `flatMap` strategy can behave very differently
  under larger payloads.
- Dig into the Reactor pathology. Is it `Flux.parallel(100)` over-scheduling and starving its own
  workers? A poll-loop interaction with blocking downstream? A real answer would be a follow-up post
  in itself.

The committed JMH JSON in `benchmarks/results/` is the source of truth; this post is the readable
surface over it.

[GitHub repo][gh] · [Benchmarks README][bench-readme] · [Raw JMH JSON][bench-results]

[gh]: https://github.com/eschizoid/kpipe

[bench-readme]: https://github.com/eschizoid/kpipe/tree/main/benchmarks

[bench-results]: https://github.com/eschizoid/kpipe/tree/main/benchmarks/results

[template]: https://github.com/eschizoid/kpipe/blob/main/benchmarks/results/TEMPLATE.md

[methodology]: https://github.com/eschizoid/kpipe/blob/main/benchmarks/METHODOLOGY.md
