---
title: "Kafka consumers get messy fast. KPipe is a simpler way to structure them."
date: 2026-04-12
description: "A lightweight Kafka processing library for modern Java built around virtual threads, composable pipelines, and predictable consumer behavior."
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

GitHub: https://github.com/eschizoid/kpipe

---

## Why I built KPipe

KPipe is a lightweight Kafka processing library for modern Java.

The goal is to keep the programming model code-first and composable, while still handling the operational
concerns that real Kafka consumers need:

- retries
- offset tracking and safe commits
- metrics and observability
- optional backpressure

---

## What KPipe is

At a high level, KPipe focuses on:

- modern Java concurrency with virtual threads
- composable functional pipelines
- safe at-least-once processing
- optional backpressure
- high throughput without heavy framework overhead

It is not trying to replace full streaming frameworks.

It is designed for Kafka consumer services where you want direct control over code, predictable behavior, and minimal
abstraction overhead.

---

## The programming model

A KPipe consumer is built by composing processors into a pipeline and attaching it to a Kafka consumer.

A minimal example:

```java
final var registry = new MessageProcessorRegistry("demo");

final var sanitizeKey = RegistryKey.json("sanitize");
registry.register(sanitizeKey, JsonMessageProcessor.removeFieldsOperator("password"));

final var stampKey = RegistryKey.json("stamp");
registry.register(stampKey, JsonMessageProcessor.addTimestampOperator("processedAt"));

final var pipeline = registry
  .pipeline(MessageFormat.JSON)
  .add(sanitizeKey, stampKey)
  .toSink(MessageSinkRegistry.JSON_LOGGING)
  .build();

final var consumer = KPipeConsumer.<byte[], byte[]>builder()
  .withProperties(kafkaProps)
  .withTopic("users")
  .withPipeline(pipeline)
  .withRetry(3, Duration.ofSeconds(1))
  .build();

final var runner = KPipeRunner.builder(consumer).build();
runner.start();
```

This sets up a consumer with processing pipeline, retries, offset tracking, metrics, and safe commit handling.

---

## Where KPipe fits

KPipe sits between raw KafkaConsumer code and full frameworks like Kafka Streams.

It’s meant for cases where Kafka Streams is too heavy, but writing everything manually becomes hard to maintain.

Typical use cases:

- Kafka consumer microservices
- enrichment pipelines
- transformation services
- I/O-bound processing (REST calls, DB lookups)
- teams adopting virtual threads

---

## Why not Kafka Streams?

Kafka Streams is powerful, but it introduces a full processing model.

KPipe is intentionally simpler:

- no topology DSL
- no framework lifecycle
- just code-first pipelines

---

## Single SerDe cycle

A core design choice in KPipe is the Single SerDe Cycle.

Instead of repeatedly doing:

byte[] -> object -> byte[] -> object -> byte[]

KPipe:

- deserializes once
- applies transformations on the same representation
- serializes once at the end

For JSON, this is typically a `Map<String, Object>`. For Avro, a `GenericRecord`.

---

## Virtual threads as the concurrency model

KPipe is built for modern Java.

Each message can be processed using a virtual thread, following a thread-per-record model.

It also uses ScopedValue to cache heavier objects per virtual thread (parsers, buffers, encoders), instead of relying
on ThreadLocal.

---

## Delivery guarantees and offset management

KPipe is designed around at-least-once processing.

It uses a lowest-pending-offset strategy:
- messages can finish out of order
- commits only happen when it is safe

It also supports pluggable offset management, including external storage.

---

## Parallel vs sequential processing

Parallel (default):
- virtual-thread-based
- best for stateless processing

Sequential:

```java
.withSequentialProcessing(true)
```

- one message per partition
- preserves ordering

---

## Backpressure

Backpressure is available when configured:

```java
.withBackpressure(highWatermark, lowWatermark)
```

Parallel mode monitors in-flight work. Sequential mode monitors consumer lag.

---

## Error handling and retries

```java
.withRetry(maxRetries, backoff)
.withDeadLetterTopic("events-dlq")
```

Custom handlers are also supported.

---

## Metrics and observability

Includes built-in metrics, log reporting, and OpenTelemetry support.

---

## Modular architecture

- kpipe-metrics
- kpipe-producer
- kpipe-consumer

Dependency chain:
kpipe-metrics <- kpipe-producer <- kpipe-consumer

---

## Performance notes

Includes JMH benchmarks for JSON, Avro, and parallel processing.

---

## Installation

Current release: v1.8.2

```kotlin
implementation("io.github.eschizoid:kpipe:1.8.2")
```

---

## Closing

Kafka itself isn’t the problem.

How we structure consumers around it is.

KPipe focuses on composable pipelines, modern Java concurrency, and predictable Kafka consumer behavior.

GitHub: https://github.com/eschizoid/kpipe
