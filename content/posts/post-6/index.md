---
title: "From a Monocle port to one fluent type: how telescope happened"
date: 2026-05-30
description: "Telescope started as a converter registry, drifted into an academic port of Scala
  Monocle's optic lattice, and turned — through five rewrites — into a single-type Java DSL with
  compile-time codegen, a generated Path navigator, and a benchmark suite that says it pays.
  The story of getting category theory to disappear."
tags:
  - java
  - dsl
  - optics
  - codegen
  - records
  - jmh
---

Telescope is a Java 25 deep-copy DSL for records and POJOs. The pitch: build a path through an
immutable graph, then read it, write it, update it, traverse it, convert it, or thread an
effect through it — without writing copy constructors by hand and without ever typing `Iso`,
`Lens`, `Prism`, `Affine`, or `Traversal`. The end-state example is one method-ref chain on the
runtime DSL or one fluent chain on the generated navigator, both bottoming out at the same
value:

```java
// Reflective (runtime resolution, ~262 ns/op for a 3-level path)
Telescope.of(Company.class)
  .each(Company::departments).each(Department::teams)
  .each(Team::users).field(User::email)
  .update(company, String::toLowerCase);

// Compile-time, reflection-free (~45 ns/op — same Telescope, generator-built)
CompanyPath.start()
  .departments().each().teams().each()
  .users().each().email()
  .update(company, String::toLowerCase);
```

That is not how it started. The first commits were a small converter registry — function-based,
multi-hop composition. Within a couple of iterations it had drifted into a full port of Scala's
Monocle library: eight category-theory interfaces
(`Iso`, `Lens`, `Prism`, `Affine`, `Traversal`, `Getter`, `Setter`, `Fold`) exposed as the
public API, a composition lattice that picked the most-specific return type at every
`.then(...)`, and lens laws verified in a test suite. It worked. It was correct. It was also
unusable.

This post is the arc from that academic port to the DSL above, plus the codegen story that
landed on top of it. It is also, honestly, a record of how often I got the shape wrong and had
to start over.

[GitHub repo][telescope]

## The optics itch and the slide into Monocle

The original problem was real and concrete: deep updates to immutable graphs in Java are
ergonomically awful. To change one field five levels down in a `Company → Department → Team →
User → Address`, you write 25 lines of nested `new Company(company.name(), company.departments()
.stream().map(d -> new Department(...)))`, every constructor enumerated, every untouched field
threaded through by hand. Stream + Optional + records make this *cleaner* than the pre-records
version. They do not make it short.

The first version was a registry of named converters that composed in chains. It worked for the
synthetic test cases. It collapsed at the third level of nesting because composition needed
real rules — what happens when you compose a partial-focus optic with a many-focus one — and I
was reinventing those rules badly. So I went to the source.

Optics solve exactly this problem. A `Lens<S, A>` is two functions, a `get: S → A` and a `set:
S → A → S`, plus laws that make those two play nicely (`set(s, get(s)) == s`, etc.). A `Prism`
is the same trick for sealed-type cases. A `Traversal` generalises to many-focus. The
composition rules between them form a lattice — compose a `Lens` with a `Prism` and you get an
`Affine`, compose an `Affine` with a `Traversal` and you get a `Traversal`. Those rules are
decades old, already proven, already implemented in Haskell `lens`, Scala Monocle, Arrow
Optics, Higher-Kinded-J.

Within two refactors of the converter registry, I had a working version of the lattice. The
test suite hit lens laws, prism partial round-trips, iso reversibility, and the diamond
resolution rule for `Iso.then(Lens)` returning `Lens`. The composition table:

| Outer ↘ Inner | Lens   | Prism  | Iso    | Affine | Traversal |
| ------------- | ------ | ------ | ------ | ------ | --------- |
| **Lens**      | Lens   | Affine | Lens   | Affine | Traversal |
| **Prism**     | Affine | Prism  | Prism  | Affine | Traversal |
| **Iso**       | Lens   | Prism  | Iso    | Affine | Traversal |
| **Affine**    | Affine | Affine | Affine | Affine | Traversal |
| **Traversal** | Trav.  | Trav.  | Trav.  | Trav.  | Traversal |

That table is intellectually satisfying. The lattice rules drop out of category theory; the
laws drop out of the type signatures. As a piece of academic Java, it was fine.

As an API, it was a disaster.

## Why the academic shape hurt in Java

Monocle lives in Scala, and Scala has three things that make optics elegant:

- **Higher-kinded types.** A `Traversal[S, A]` in Monocle is defined as a function over any
  applicative `F[_]`. That's how `modifyF` (effectful update) drops out of the same definition
  as plain `modify`. In Java, `F[_]` does not exist.
- **Implicit type classes.** Scala's `cats` library makes "thread any applicative through a
  traversal" a one-line implicit lookup. Java has no implicits; every applicative would have to
  be passed explicitly.
- **Macros.** Monocle's `@Lenses` annotation generates per-field lens constants at compile
  time from a single annotation. Without macros, the choice is between writing the constants
  by hand or bolting on an annotation processor and a multi-module build.

I had none of those. So the Java port surfaced every piece of the inheritance: users had to
import `Lens` and `Prism` and `Iso` and `Affine` and know which composition produced which.
They had to remember that `Iso.then(Lens) = Lens` but `Lens.then(Prism) = Affine`. They had to
think in category-theoretic vocabulary to navigate a record.

**That is a reasonable experience for a Haskell library. It is an unshippable experience for a
Java one.** No Java team is going to learn five interfaces and a composition lattice just to
update a nested field.

So I tried to throw the lattice away.

## The detour that taught me what not to do

The next iteration replaced the eight interfaces with a single `Path<S, A>` class. `Path` had a
`Function<S, Stream<A>>` reader and a custom `Updater<S, A>` updater inside it, hand-rolled.
The composition rules were re-implemented from scratch as `Path` methods. The API surface
collapsed to one type. The lattice was gone.

This felt great for about two days.

Then I started adding edge cases. Iso reversibility — gone. Prism partial round-trip — gone.
The diamond resolution that made `Iso.then(Lens)` return `Lens` instead of widening to `Affine`
— gone. The lens laws were no longer verified by composition; they were now my problem to
enforce by hand inside `Path`'s update logic. The `Path` internals grew until they were a
worse version of the lattice I had just deleted.

This is the lesson: **the lattice was earning its keep.** The fact that users didn't want to
see it didn't mean the implementation didn't need it. What needed to go away was the *exposure*
of the lattice, not the lattice itself.

I reverted the deletion. The lattice came back into `org.telescope.internal.optics`,
package-private, where the compiler enforces that no user-facing code ever names it. The
public surface became one class:

```java
public final class Telescope<S, A>
```

`Telescope` wraps a `Traversal<S, A>` from the internal lattice. Every navigation method
(`field`, `each`, `as`, `filter`) builds the appropriate optic and composes it via the
lattice's rules. Every read/write operation (`read`, `update`, `toList`, `set`, ...) delegates.
Users never type `Lens`, `Prism`, `Iso`, `Affine`, `Traversal`, `Getter`, `Setter`, or `Fold`.

Two layers. Proven concepts internally, convenient DSL externally. That is the shape that
stuck.

## What the DSL looks like

The 30-second pitch is one path against a domain:

```java
record Address(String city, String zip) {}
record User(String name, int age, String email, Address address) {}
record Team(String name, List<User> users) {}
record Department(String name, List<Team> teams) {}
record Company(String name, List<Department> departments) {}

final Telescope<Company, String> emails = Telescope.of(Company.class)
  .each(Company::departments)
  .each(Department::teams)
  .each(Team::users)
  .field(User::email);

final Company lowered = emails.update(company, String::toLowerCase);
final List<String> all = emails.toList(company);
final long count = emails.count(company);
```

One path, build once, use many ways. The vanilla equivalent is about 25 lines of nested
`new Company(company.name(), company.departments().stream().map(d -> new Department(...)))` —
every constructor enumerated by hand, every untouched field threaded through.

Everything past that — sealed-type narrowing with `.as(...)`, `Optional` traversal with
`.whenPresent(...)`, indexed traversals, type conversion between records via `from / to / using`
and `map / to / field / build`, the four effectful update methods (`updateAsync`,
`updateOptional`, `updateEither`, `updateValidated`) — was added on top of this substrate
without changing it. The `Kind<F, A>` machinery that makes the four effects work lives in
`internal.optics`. It never appears in user code.

## Then codegen happened, twice

The reflective DSL above resolves field names at runtime through
`SerializedLambda.getImplMethodName()` and `RecordComponent.getAccessor().invoke()`. It works.
It is sub-microsecond. It is also reflection, with all the costs that implies.

The first codegen pass shipped `@Focus` for records and `@BeanFocus` for POJOs. Annotate a
type, get a sibling class with per-field lens constants built from direct method-ref + canonical-
constructor calls. No runtime reflection, no `SerializedLambda` decode, ~45 ns/op for a 3-level
field path. Container components got generated traversal constants alongside the field lenses,
so a `List<User> users` component on `Team` produced `TeamFocus.eachUsers : Telescope<Team,
User>`, and a fully compile-checked deep path looked like this:

```java
CompanyFocus.eachDepartments
  .then(DepartmentFocus.eachTeams)
  .then(TeamFocus.eachUsers)
  .then(UserFocus.email);
```

Correct. Type-checked. Reflection-free. It also did not read at all like the reflective DSL it
was supposed to be the fast path for. The reflective version reads:

```java
Telescope.of(Company.class)
  .each(Company::departments).each(Department::teams)
  .each(Team::users).field(User::email);
```

The information content is identical. The reading flow is not. So the next pass replaced the
public lens constants with a **fluent typed Path navigator**:

```java
CompanyPath.start()
  .departments().each()
  .teams().each()
  .users().each()
  .email()
  .update(company, String::toLowerCase);
```

Per annotated type, `@Focus` now emits one parameterised navigator class (`<X>Path<R>`) plus
one container step per collection component (`<X><Cap>Step<R>`). Each scalar component yields a
terminal `Telescope<R, T>`; each sub-record component yields a `<Sub>Path<R>` so navigation
continues; each container component (`List` / `Set` / `Iterable`, `Map` values, `Optional`)
yields a step whose `.each()` / `.eachValue()` / `.whenPresent()` returns the element's `Path`
when the element is itself annotated, or a terminal `Telescope` otherwise. The method bodies
all build `Telescope.lens(getter, setter)` directly — no reflection at any hop.

Every `Path` and `Step` also forwards the full `Telescope` operation surface — `read`, `find`,
`toList`, `count`, `exists`, `set`, `update`, `updateIndexed`, plus the four effect variants —
so you do not need to terminate with `.get()` to operate at any hop:

```java
CompanyPath.start()
  .teams().each().users().each()
  .updateAsync(company, svc::lookupAsync, pool);   // CompletableFuture<Company>
```

The `@Bridge` annotation generates a bidirectional `Iso` between any two top-level types
(record↔record, record↔POJO, POJO↔POJO). When a type carries both `@Focus` and `@Bridge`, the
navigator gains an `as<Target>()` method that chains the bridge constant in:

```java
@Focus @Bridge(UserDto.class) record UserEntity(String id, String email) {}
@Focus record UserDto(String id, String email) {}

UserEntityPath.start()
  .asUserDto()
  .email()
  .update(entity, String::toLowerCase);
```

The Iso round-trips, so the result is a new `UserEntity`. The navigator is now a single
compile-checked, reflection-free surface for navigation, container traversal, sync ops, all
four effects, and conversion — including across paradigms (record↔POJO via `@Bridge` works
the same way).

## Benchmarks

JMH, 3 warmup + 5 measurement iterations, JDK 25, Apple Silicon. Numbers shift between runs;
the ratios are the part to read.

| Benchmark                  | ns/op | ±error |             vs hand-copy |
| -------------------------- | ----: | -----: | -----------------------: |
| `bridgeForwardRead`        |  14.9 |   ±0.2 |       codegen conversion |
| `handRolledBeanCopyUpdate` |  22.2 |   ±0.6 |     1.0× (bean baseline) |
| `handRolledCopyUpdate`     |  26.4 |   ±1.9 |          record baseline |
| `lensConstantUpdate`       |  45.2 |   ±3.4 |                     1.7× |
| `fromBeanForwardRead`      | 114.0 |   ±1.7 |                     5.1× |
| `mapperForwardRead`        | 135.4 |  ±90.1 |    record→record (noisy) |
| `mapBeanForwardRead`       | 142.5 |   ±3.7 |                     6.4× |
| `reflectionFieldUpdate`    | 261.6 |  ±15.9 |                    11.8× |
| `ofBeanFieldUpdate`        | 488.1 | ±139.7 |                    22.0× |

The headline:

- **`@Bridge` codegen at ~15 ns/op vs runtime `mapBean` at ~142 ns/op — ~9.5× for the same
  POJO↔POJO conversion.** That is the closest apples-to-apples comparison in the suite.
- **Codegen field navigation at ~45 ns vs reflective at ~262 ns — ~5.8× for a 3-level deep
  field path.** That is what `@Focus` → `*Path` buys.
- **Runtime reflective conversions (`fromBean`, `mapper`, `mapBean`) cluster in 114–142 ns.**
  Sub-microsecond, fine for ordinary use, not the path to pick in a tight loop.
- **Native POJO navigation (`ofBean`) at ~488 ns — ~22× a hand-rolled bean copy.** It
  rebuilds the whole POJO at every level and re-reads every getter to carry siblings over. For
  a hot loop, bridge once to a record with `fromBean` or use `@BeanFocus` codegen.

The codegen surface closes the 5–10× gap with hand-written code while keeping the same end-
value the reflective DSL produces. End users pick a surface (reflective for ergonomics + zero
codegen, navigator for compile-time guarantees + the perf number) without paying for the one
they did not pick.

## What still surprises me

A few honest observations after living with this codebase for a while:

- **The lattice was right.** The detour where I tried to delete it cost more LOC than I saved.
  The "trust the proven implementation" lesson keeps recurring. The lattice doing the work
  internally is why the DSL never needed to expand its public surface to handle effectful
  update or codegen or bridge hops — those each landed as a few methods on `Telescope` plus
  helpers in `internal.optics`, never a restructuring.
- **Codegen is a behaviour multiplier, not a perf trick.** The first version of `@Focus` was
  sold internally as "make `.field(...)` faster." The actual win was different: the navigator
  makes paths *type-checked at javac*. A typo in `Company.teams` is a compile error, not a
  runtime exception. The ~5.8× speedup is the consolation prize.
- **Bridge hops are the most underrated feature.** A record's navigator can convert to its DTO
  navigator in one fluent step, reflection-free, with the Iso round-trip handling the reverse
  for free. That is not in any other Java lens library I have seen. It dropped out of having
  `@Bridge` already shipped — the codegen pattern is "if the source has both annotations, one
  extra method on the `Path`." Two days of work for what feels like a step-change in
  ergonomics.
- **The reflective DSL stayed honest.** I expected it to feel obsolete after codegen landed.
  It has not. Anyone who wants to use telescope without wiring an annotation processor still
  has the full DSL, sub-microsecond, with build-time fail-fast validation that catches bad
  method refs at path construction. Codegen is the opt-in that pays back tight loops; the
  reflective path is the default that just works.

Five rewrites in. The thing that started as a converter registry, drifted into an academic
Monocle port, almost got deleted, and came back as two layers is now a Java DSL with a single
public type, a fluent generated navigator, four effect variants, three styles of type
conversion, and a benchmark suite that says it pays its way. None of the rewrites felt obvious
at the time. The one that mattered was the one where I admitted the academic library was the
wrong public surface and pushed it under the floorboards twice over — first behind `Telescope`,
then behind the generated `*Path` navigators.

The category theory is still doing the work. You just do not have to know the vocabulary to
use it.

---

[GitHub repo][telescope]

[telescope]: https://github.com/eschizoid/telescope
