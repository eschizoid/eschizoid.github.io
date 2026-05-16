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

If you write a library that "consumes Kafka in parallel," eventually someone asks "compared to what?"

For KPipe that question had a partial answer. Up to 1.12 the benchmark compared KPipe to the Confluent
Parallel Consumer only, at 10,000 records per invocation, with no per-record workload. One competitor,
one workload, one number.

This post is about the upgrade and the first numbers from it. KPipe now has a four-runtime competitive
bench: **KPipe**, **Confluent Parallel Consumer**, **Reactor Kafka**, and a hand-rolled **raw
`KafkaConsumer` + virtual-thread executor** baseline. Three workload regimes. Real Kafka broker
(Testcontainers, not in-process). JMH-published scores, GC profiler, raw JSON committed alongside this
post in [`benchmarks/results/`][bench-results].

The headline first. The methodology and the saga of getting here second.

## Headline

Records / second, higher is better. `workMicros` is per-record simulated work via `LockSupport.parkNanos`
— 0 µs is pure framework overhead, 100 µs is local enrichment, 1000 µs is a blocking I/O round trip.
All four runtimes run against the same Testcontainers-managed Kafka 4.2.0 broker, same 25,000-record
seed, same eight partitions, two JMH forks × five measurement iterations.

| Runtime | `workMicros=0` | `workMicros=100` | `workMicros=1000` |
| --- | ---: | ---: | ---: |
| **KPipe** | **553,704 ± 54,924** | **500,903 ± 95,758** | **495,803 ± 41,069** |
| **Raw `KafkaConsumer` + VT** | 570,890 ± 38,369 | 526,568 ± 44,444 | 498,016 ± 51,323 |
| **Reactor Kafka** | ~300,000 (preliminary)¹ | — | — |
| **Confluent Parallel Consumer** | 107,877 ± 8,892 | 112,985 ± 2,653 | 66,454 ± 2,963 |

¹ *Reactor Kafka 1.3.23 crashed on first fetch — `kafka-clients` 4.x incompatibility, [tracked
upstream][reactor-issue]. Bumped to 1.3.25 (the fix shipped 2025-11-06) and the smoke iteration at
`workMicros=0` came back at ~300k ops/sec. The full three-cell sweep is running as I write this post;
this section will be updated in place when it lands.*

![Parallel-consumer throughput, allocation, and GC profile](parallel-gc-baseline.svg)

**Reading the graph — the Reactor bar is not apples-to-apples.** The KPipe, Raw, and Confluent bars
come from a full publishing run on the same JMH JAR: two forks, five measurement iterations each,
GC profiler attached, error bands quoted. The Reactor bar is a **single iteration** from a separate
smoke run after I bumped the Reactor Kafka dependency from 1.3.23 (which crashed) to 1.3.25 (which
runs). The lighter fill on the Reactor bar is meant to flag that. The full Reactor sweep — three
`workMicros` cells, two forks, error band — is in flight and will replace the bar in the next
snapshot. Until then: the position is directional, not yet load-bearing.

[reactor-issue]: https://github.com/reactor/reactor-kafka/issues/420

## Three things this measures cleanly

**1. KPipe ≈ raw `KafkaConsumer + VT` within error bars.** Across the entire `workMicros` sweep
KPipe's score sits on top of the hand-rolled baseline. The framework wraps the consumer loop without
slowing it down. The interesting overhead question — "what does the framework cost on the hot path?" —
the data says **statistically zero**.

That doesn't mean KPipe is uniquely fast. It means KPipe doesn't make Loom slower. Which, given everything
KPipe layers on top of the raw loop — offset manager with lowest-pending-offset commits, retry, DLQ
producer, backpressure with hysteresis, circuit breaker, OTel metric + tracing hooks, batch sinks,
`Result<T>` typed pipeline outcomes, lifecycle management with graceful shutdown — is the right answer.

**2. Both Loom-based runtimes are 4.4× – 7.5× ahead of Confluent Parallel Consumer.** The gap widens at
`workMicros=1000`. Confluent's 100-worker thread pool can only have 100 records simultaneously parked on
the simulated I/O wait; under blocking work the pool serialises. KPipe and raw don't serialise — virtual
threads scale beyond the partition count, each blocked record costs kilobytes instead of megabytes.

**3. Reactor Kafka 1.3.25 lands between Confluent and the Loom runtimes.** Preliminary 300k ops/sec is
~3× Confluent but ~half KPipe at `workMicros=0`. The full sweep will tell whether Reactor falls off
under blocking work the way Confluent does. Hypothesis: yes, because `Flux.parallel(100)` is also a
fixed worker pool. We'll see.

## What this does not say

A bench number with no caveat is a marketing pitch. The honest framing:

- **The Loom runtimes win because of Loom, not because of KPipe.** "KPipe is 5× Confluent" is true but
  unfairly framed — the right comparison is "Loom-based parallel consumption is 5× a platform-thread
  pool." KPipe's contribution is bringing that win without overhead.
- **KPipe allocates more per record.** ~923 B/op at `workMicros=0` vs raw's 55 B/op vs Confluent's 34
  B/op. On this run that didn't depress throughput — the JVM's young gen absorbed it. On a tight heap,
  or a workload sensitive to GC tail latency, watch this number.
- **One payload shape, one broker config.** Small JSON, single-broker Testcontainers, replication
  factor 1. Production with a network broker and replication 3 / acks=all has different absolute
  numbers. Headline ordering between runtimes usually survives the move; the gaps shift.
- **No latency-percentile data yet.** `ParallelProcessingLatencyBenchmark` was not exercised in this
  run. Average throughput is half the story; p99 is the other half, and a runtime can rank one way on
  throughput and the opposite way on tail. Coming in the next snapshot.

## Allocation and GC

The cost-profile panels in the graph above carry the headline. The story is counter-intuitive but
consistent with the throughput numbers: **Confluent allocates least per record (~34 B/op) but has the
most GC events overall** because its slower iterations run longer wall-clock and the young gen cycles
more times. KPipe and the raw baseline allocate faster per second but each iteration ends sooner so
cumulative GC time is lower. **Throughput dominates allocation rate in this workload.**

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
in-process harness now gets 553,000 records/sec. That's not a 10x improvement — that's "measuring the
right thing now."

The lesson: **for parallel-consumer benchmarks, the broker has to be external.** Testcontainers is the
cheap path; a sidecar on dedicated hardware is the production-faithful path. In-process Kafka is fine
for "does my code compile and run end-to-end" tests. It is not fine for performance comparison.

## The Reactor Kafka saga

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

- Full four-runtime sweep with the Reactor row populated (running as I publish this).
- Latency-percentile companion run.
- Multi-partition sweep (8 / 32 / 128) to expose how each runtime scales with parallelism opportunity.
- Payload-size sweep (100 B / 1 KB / 10 KB) — Reactor's `flatMap` strategy can behave very differently
  under larger payloads.

The next post in this series will swap in the full four-runtime graph and probably one or two of those
parameter sweeps. The committed JMH JSON in `benchmarks/results/` is the source of truth; this post is
the readable surface over it.

[GitHub repo][gh] · [Benchmarks README][bench-readme] · [Raw JMH JSON][bench-results]

[gh]: https://github.com/eschizoid/kpipe

[bench-readme]: https://github.com/eschizoid/kpipe/tree/main/benchmarks

[bench-results]: https://github.com/eschizoid/kpipe/tree/main/benchmarks/results

[template]: https://github.com/eschizoid/kpipe/blob/main/benchmarks/results/TEMPLATE.md

[methodology]: https://github.com/eschizoid/kpipe/blob/main/benchmarks/METHODOLOGY.md
