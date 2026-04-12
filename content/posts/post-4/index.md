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

[GitHub repo](https://github.com/eschizoid/kpipe)

## External write-ups

- [KPipe: A modern high-performance Kafka library](https://topicigor.substack.com/p/kpipe-a-modern-high-performance-kafka)

---

## Why I built KPipe

KPipe is a lightweight Kafka processing library for modern Java.

The goal is simple: keep Kafka consumers composable and predictable, while still handling the operational concerns that
real systems need:

- retries
- offset tracking and safe commits
- metrics and observability
- optional backpressure

---

## Design goals

KPipe is built around a few constraints:

- pipelines should be composable  
- processing should be predictable  
- concurrency should be simple  
- operational concerns should not leak into business logic  

---

## What KPipe is

At a high level, KPipe provides:

- virtual-thread-based concurrency
- composable processing pipelines
- at-least-once delivery semantics
- optional backpressure
- minimal framework overhead

It is not a full streaming framework.

It is designed for Kafka consumer services where you want direct control over code.

---

## Where KPipe fits

KPipe sits between:

- raw KafkaConsumer code  
- full frameworks like Kafka Streams  

It’s useful when Kafka Streams is too heavy, but manual consumers become hard to maintain.

---

## When not to use KPipe

KPipe is not meant for every Kafka use case.

You probably don’t need it if:

- you are already using Kafka Streams with complex topologies  
- you need full stateful stream processing  
- your problem fits well into an existing framework  

---

## The programming model

Instead of structuring a consumer as a single handler, KPipe builds it as a pipeline:

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

---

## Why not Kafka Streams?

Kafka Streams is powerful, but introduces a full processing model.

KPipe is intentionally simpler:

- no topology DSL
- no framework lifecycle
- just code-first pipelines

---

## Single SerDe cycle

Instead of repeatedly doing:

byte[] -> object -> byte[] -> object -> byte[]

KPipe:

- deserializes once
- transforms in place
- serializes once

For JSON, this is typically a `Map<String, Object>`. For Avro, a `GenericRecord`.

---

## Virtual threads

Each message is processed using a virtual thread (thread-per-record model).

KPipe also uses ScopedValue for per-thread resource reuse.

---

## Delivery guarantees

KPipe provides at-least-once processing.

Offsets are committed only when safe, using a lowest-pending-offset strategy.

---

## Processing modes

Parallel (default): higher throughput
Sequential: preserves ordering per partition

```java
.withSequentialProcessing(true)
```

---

## Backpressure

Enabled when configured:

```java
.withBackpressure(highWatermark, lowWatermark)
```

Allows slowing down consumption when downstream systems are the bottleneck.

---

## Error handling

```java
.withRetry(maxRetries, backoff)
.withDeadLetterTopic("events-dlq")
```

---

## Metrics

Built-in metrics and OpenTelemetry support included.

---

## Modules

- kpipe-metrics
- kpipe-producer
- kpipe-consumer

---

## Installation

```kotlin
implementation("io.github.eschizoid:kpipe:1.8.2")
```

---

## Closing

Kafka itself isn’t the problem.

The way we structure consumers around it usually is.

[GitHub repo](https://github.com/eschizoid/kpipe)
