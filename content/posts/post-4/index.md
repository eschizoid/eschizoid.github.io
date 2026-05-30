---
title: "Kafka consumers get messy fast. KPipe is a simpler way to structure them."
date: 2026-04-12
description: "Modern Java Kafka consumer library: virtual threads, composable pipelines, predictable consumers."
tags:
  - kafka
  - java
  - virtual-threads
  - event-driven
  - stream-processing
  - apache-kafka
---

Kafka consumers start simple.

Then they turn into tightly coupled, hard-to-test, side-effect-heavy code.

I built KPipe to fix that.

[GitHub repo][gh]

## External write-ups

- [KPipe: A Modern, High-Performance Kafka Consumer in Java — Powered by Java 25 Features][substack-writeup]

---

## What it looks like

The 80% path is a fluent facade. Five lines from `main`:

```java
try (final var handle =
    KPipe.json("events", kafkaProps)
        .pipe(Operators.addField("processedAt", System.currentTimeMillis()))
        .pipe(Operators.removeFields("password", "ssn"))
        .toConsole()
        .start()) {
  handle.awaitShutdown();
}
```

That is a real consumer. It deserializes JSON, applies two transforms, logs the result, runs on virtual
threads, commits offsets safely under parallel processing, and closes gracefully on JVM shutdown.

Avro and Protobuf follow the same shape, with the format supplied explicitly:

```java
final var format = AvroFormat.of(schemaJson);
try (final var handle =
    KPipe.avro(format, "users", kafkaProps)
        .skipBytes(5) // Confluent wire envelope
        .pipe(
            record -> {
              record.put("name", record.get("name").toString().toLowerCase());
              return record;
            })
        .toConsole()
        .start()) {
  handle.awaitShutdown();
}
```

---

## Why I built it

For most Kafka work in a JVM service I had three options:

1. Hand-roll a `KafkaConsumer` loop. Cheap to start, expensive to maintain. Every team writes a slightly
   different version of retries, offset commits, backpressure, and graceful shutdown.
2. Spring Kafka. Powerful, but only if the rest of the app is already Spring.
3. Kafka Streams. Full topology engine. Overkill for "consume, transform, write somewhere."

KPipe is option four. A code-first library that owns the operational concerns (offset safety, retries,
DLQ, backpressure, metrics, tracing, circuit breaker, graceful shutdown) so the business logic stays as
`UnaryOperator<T>` calls.

No annotations. No Spring container. No DSL.

---

## KPipe vs Spring Kafka

This is the comparison most people actually need.

| Concern               | Spring Kafka                                            | KPipe                                                    |
|-----------------------|---------------------------------------------------------|----------------------------------------------------------|
| **Container**         | Spring required                                         | None: `kafka-clients` + Java 25                          |
| **Programming model** | `@KafkaListener` on a bean method                       | `Stream<T>` fluent pipeline; `UnaryOperator<T>` per step |
| **Concurrency**       | `concurrency=N` per listener (one thread per partition) | Virtual thread per record, by default                    |
| **Error semantics**   | `ErrorHandler`, exceptions bubble out                   | Sealed `Result<T>` (Passed/Filtered/Failed)              |
| **Retries**           | `@RetryableTopic` or `DefaultErrorHandler`              | `.withRetry(maxRetries, backoff)`                        |
| **DLQ**               | `DeadLetterPublishingRecoverer`                         | `.withDeadLetterTopic("events-dlq")`                     |
| **Backpressure**      | Manual pause/resume from the listener                   | `.withBackpressure()` with hysteresis (in-flight or lag) |
| **Tracing**           | Spring Cloud Sleuth / Micrometer Tracing                | `kpipe-tracing-otel`, W3C propagation via Kafka headers  |
| **Schema Registry**   | Confluent client via Spring auto-config                 | `kpipe-schema-registry-confluent`, plain HTTP client     |
| **Testing**           | `EmbeddedKafka` + Spring context                        | Testcontainers; `kpipe-test` (no Kafka) on the roadmap   |
| **Health checks**     | Spring Boot Actuator                                    | `HttpHealthServer.fromEnv(...)` in `kpipe-consumer`      |
| **Java floor**        | 17                                                      | 25                                                       |

**Pick Spring Kafka when** the rest of the app is Spring, the team already has Spring muscle memory, and
Actuator / Sleuth integration is load-bearing.

**Pick KPipe when** the service is a small focused consumer, the team values explicit code over magic
configuration, Java 25 is on the table, or the project does not pull Spring for any other reason. Smaller
dependency tree, smaller surface area to learn, smaller production footprint.

---

## KPipe vs Kafka Streams

Different problem.

Kafka Streams is a topology engine. State stores, joins, windowed aggregations, exactly-once via
transactions. If you need any of those, use it. KPipe will not get there and will not try.

KPipe is for the case where the right code shape is "consume a topic, transform each record, write
somewhere else." Most consumer services look like that.

---

## Getting started in 5 minutes

### 1. Add the dependencies

```kotlin
// Gradle Kotlin DSL
implementation("io.github.eschizoid:kpipe-api:1.13.0")
implementation("io.github.eschizoid:kpipe-format-json:1.13.0")
```

`kpipe-api` transitively pulls `kpipe-consumer` + `kpipe-producer` + `kpipe-core`. The format module is
the only other thing you need for JSON. For Avro or Protobuf, add `kpipe-format-avro` or
`kpipe-format-protobuf` instead.

There is also a `kpipe-bom` if you prefer to pin one version across modules.

### 2. Write a consumer

```java
import java.util.Properties;
import org.kpipe.KPipe;
import org.kpipe.registry.Operators;

public final class App {
  public static void main(final String[] args) {
    final var props = new Properties();
    props.setProperty("bootstrap.servers", System.getenv("KAFKA_BOOTSTRAP"));
    props.setProperty("group.id", "events-consumer");
    props.setProperty("auto.offset.reset", "earliest");

    try (final var handle =
        KPipe.json("events", props)
            .pipe(Operators.addField("processedAt", System.currentTimeMillis()))
            .pipe(Operators.removeFields("password", "ssn"))
            .toConsole()
            .start()) {
      handle.awaitShutdown();
    }
  }
}
```

That is the whole app. Try-with-resources gives you a 5-second graceful shutdown for free.

### 3. Run Kafka

```yaml
# docker-compose.yml
services:
  kafka:
    image: apache/kafka:4.2.0
    ports: ["9092:9092"]
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
```

`docker compose up -d kafka`, set `KAFKA_BOOTSTRAP=localhost:9092`, and run the app.

### 4. Add the operational layer

```java
KPipe.json("events", props)
    .pipe(Operators.removeFields("password"))
    .withRetry(3, Duration.ofSeconds(1))
    .withDeadLetterTopic("events-dlq")
    .withBackpressure()
    .withCircuitBreaker(0.5, 100, Duration.ofSeconds(30))
    .toConsole()
    .start();
```

That is the full operational story for a typical consumer: three retries, DLQ on permanent failure,
pause when 10k records are in flight, and a circuit breaker that trips at 50% failure rate over a
100-sample window and probes after 30s.

### 5. Add OpenTelemetry (optional)

```kotlin
implementation("io.github.eschizoid:kpipe-metrics-otel:1.13.0")
implementation("io.github.eschizoid:kpipe-tracing-otel:1.13.0")
```

```java
KPipe.json("events", props)
    .withMetrics(new OtelConsumerMetrics(otel, "events-consumer"))
    .withTracer(new OtelTracer(otel, "events-consumer"))
    .pipe(Operators.addField("ts", System.currentTimeMillis()))
    .toConsole()
    .start();
```

The base library has no OTel dependency. Adding the two modules opts you in without changing the rest
of the code.

---

## What you do not get

- **No topology DSL.** No joins, no windowed aggregations, no state stores. Use Kafka Streams.
- **No annotation-driven configuration.** Pipelines are values, not bean declarations.
- **No exactly-once transactions.** At-least-once with idempotent processing is the contract.
- **No Spring auto-wiring** (a starter is on the roadmap, gated on demand).

If any of those is a hard requirement, KPipe is the wrong tool.

---

## The design decisions worth knowing

A few choices that shape everything else.

**Byte boundary at the consumer entry point.** `KPipeConsumer` always reads `byte[]` values from Kafka.
SerDe is part of the pipeline, not part of the consumer config. This makes Confluent wire envelopes,
mixed-format topics, and bring-your-own-format trivial.

**Single SerDe cycle.** A pipeline deserializes once, applies every `UnaryOperator<T>` against the same
object, and serializes once. Composing five transforms costs one deserialize and one serialize, not
five round-trips.

**Lowest-pending-offset commits.** Parallel processing produces out-of-order completions. KPipe commits
the highest contiguous offset, never a "gap." At-least-once safety survives the parallelism.

**Result&lt;T&gt; for filter-vs-fail.** `MessagePipeline.process()` returns a sealed `Result<T>`
(`Passed | Filtered | Failed`). The compiler enforces exhaustive handling. The 1.12 era convention was
"null means filter, throw means fail" — readable but trust-based. 1.13 moved that guarantee from
convention to type.

**ConsumerHealthController.** One bitmask coordinates manual pause, backpressure, and circuit-breaker
pause sources. The three can hold the consumer paused simultaneously without fighting each other on
resume.

**No-deprecation policy.** When a public API needs to change, we delete it and migrate every caller in
the same PR. The codebase carries no `@Deprecated` and no compatibility shims. The cost is loud break
notes in the release; the win is no `since="..."` rot.

---

## Modules

12 published artifacts on Maven Central. Pull what you need:

| Module                            | What it gives you                                                                         |
|-----------------------------------|-------------------------------------------------------------------------------------------|
| `kpipe-api`                       | Fluent facade: `KPipe`, `Stream<T>`, `Sink<T>`, `Handle`                                  |
| `kpipe-bom`                       | Version-pinning BOM                                                                       |
| `kpipe-core`                      | Pipeline machinery: `MessageProcessorRegistry`, `MessageFormat`, `Operators`, `Result<T>` |
| `kpipe-consumer`                  | `KPipeConsumer`, backpressure, circuit breaker, offset manager, health server             |
| `kpipe-producer`                  | Kafka producer wrapper, DLQ producer, `Tracer` SPI                                        |
| `kpipe-metrics`                   | Metrics interfaces + log-based reporters (no OTel on classpath)                           |
| `kpipe-metrics-otel`              | OpenTelemetry-backed metrics (opt-in)                                                     |
| `kpipe-tracing-otel`              | W3C trace context propagation (opt-in)                                                    |
| `kpipe-schema-registry-confluent` | Confluent Schema Registry client (opt-in)                                                 |
| `kpipe-format-json`               | `JsonFormat`, `JsonConsoleSink`                                                           |
| `kpipe-format-avro`               | `AvroFormat`, `AvroSchemaCatalog`, `AvroConsoleSink`                                      |
| `kpipe-format-protobuf`           | `ProtobufFormat`, `ProtobufDescriptorCatalog`, `ProtobufConsoleSink`                      |

---

## Closing

Kafka itself isn't the problem.

The way we structure consumers around it usually is.

If you have ever maintained a Kafka service whose `Consumer.poll` loop has slowly grown into a 400-line
method with hand-rolled retries and a TODO about offset commits, this library is for you.

[GitHub repo][gh] · [Maven Central][mvn]

[gh]: https://github.com/eschizoid/kpipe
[mvn]: https://central.sonatype.com/artifact/io.github.eschizoid/kpipe-api
[substack-writeup]: https://topicigor.substack.com/p/kpipe-a-modern-high-performance-kafka
