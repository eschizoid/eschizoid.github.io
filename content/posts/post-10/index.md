---
title: "Two agent tools taught me the same thing: the work is a graph"
date: 2026-09-02
description: "Beads models what an AI agent should do next as a dependency graph. My fleet-merge
  skill decides when a pull request is actually done and merges it. They look unrelated until you
  notice they are the same move: represent the work as a graph, define ready and done as strict
  gates, and let an engine walk it. That move has a name, and it is worth doing on purpose."
tags:
  - ai-agents
  - graph
  - workflow
  - beads
  - automation
  - claude-code
draft: false
---

## The setup

Hand real work to a coding agent and two problems show up, one at each end of the same job.

At the start: what should it do next? A long project is not a list, it is a web. Task B needs A
finished first, C and D can go in parallel, E is blocked on both. Hold that in a flat TODO and the
agent picks the wrong thing, or redoes finished work, or loses the plan the moment its context
window rolls over.

At the end: is this actually done? The agent opened six pull requests. CI is green on four. One has
a review comment nobody resolved. One looks merged-ready but the approval is sitting on a commit
three pushes old. Which of the six can you actually merge, right now, without breaking `main`?

I reach for a different tool at each end. [Beads](https://github.com/steveyegge/beads) for the
first, a skill I wrote called `fleet-merge` for the second. I used them for months before I noticed
they are the same idea in two places.

## The first graph: what is ready to start

[Beads](https://steve-yegge.medium.com/the-beads-revolution-how-i-built-the-todo-system-that-ai-agents-actually-want-to-use-228a5f9be2a9)
is Steve Yegge's issue tracker for AI agents. Its one good idea, the one everything else hangs off,
is that work is a directed graph. A task is a node. A dependency is an edge. You do not tell the
agent what to do; you describe the shape of the work and let it ask the graph.

```bash
# two nodes and an explicit edge between them
bd create "Add auth middleware"     # -> bd-1
bd create "Protect /admin routes"   # -> bd-2
bd dep add bd-2 bd-1                 # bd-2 is blocked by bd-1 until it closes

# the only question the agent actually asks:
bd ready
# bd-1  Add auth middleware   (no open blockers)
```

`bd ready` is the whole trick. It returns the nodes whose dependencies are all satisfied, which is
exactly the set of things that can be worked right now. Finish `bd-1`, close it, and `bd-2` falls
into the ready set on its own. The agent never guesses at order, because order is not a plan it has
to remember; it is a property of the graph, recomputed on demand.

The other half is that the graph is git-backed, so it survives the thing agents are worst at:
forgetting. Context windows roll over and the session dies, but the DAG is committed next to the
code. A fresh agent runs `bd ready` and is immediately oriented, no re-briefing. The graph is the
memory.

So beads answers the start-of-work question, and it answers it structurally: **ready work is a node
with no unmet dependencies.**

## The second graph: what is ready to finish

`fleet-merge` is a skill I wrote to close the other end. Point it at a repo with open pull requests
and it watches them until each one is genuinely mergeable, then merges it with whatever method the
repo allows. It is the boring, vigilant reviewer that never gets tired and never waves something
through because it is Friday.

I did not think of it as a graph tool when I wrote it. It is. The pull requests are nodes. Two pull
requests that touch the same file have an edge between them: whichever merges second has to rebase,
so they cannot land in parallel. And every node has the same question hanging over it that beads
asks at the other end, only inverted: not "can this start," but "can this finish."

The answer is a gate, and the gate does all the work. A pull request is done only when all of
these hold at once:

- No real CI failure, and nothing still running. Not "the checkmark is green": a check that is still
  in progress reports no conclusion yet, and if you read that as success you merge with the tests
  half-finished.
- The review sign-off is on the *current* commit. An approval on a commit that has since been pushed
  over is not an approval of the code you are about to merge. It is an approval of code that no
  longer exists.
- No suppressed review comments. This is the one that pays. A reviewer can leave notes that never
  become blocking threads, so the summary still reads "no new comments" and the unresolved count
  still says zero, while a real finding sits one click out of view.
- It is not a release pull request.

None of those four is the naive signal. The green checkmark, the APPROVED badge, the zero-unresolved
count: each looks like done and is not. The value is in refusing the signals that merely resemble
done, the same way `bd ready` refuses to hand you a node whose dependencies only look satisfied.

So fleet-merge answers the end-of-work question, and it answers it structurally too: **finishable
work is a node whose gate passes on its current state.**

## The same move, twice

Line the two up and the shared shape stands out.

|                    | beads                          | fleet-merge                       |
| ------------------ | ------------------------------ | --------------------------------- |
| Node               | a task                         | a pull request                    |
| Edge               | this task needs that one first | these two touch the same file     |
| The gate           | dependencies satisfied         | CI, fresh sign-off, no hidden notes |
| The question       | what can start                 | what can finish                   |
| Who walks it       | the coding agent               | the merge loop                    |

Neither tool is really "a TODO list" or "a merge bot." Both are the same three-part move:

1. Model the work as a graph, so order is a property you compute, not a plan you hold in your head.
2. Define *ready* and *done* as explicit gates, and make the gates refuse the signals that only look
   like readiness.
3. Let an autonomous engine walk the graph, transitioning a node only when its gate actually passes.

That move is worth naming, because once you see it you start building it deliberately. Call it graph
engineering: turning "a pile of work an agent has to keep straight" into "a graph an agent can
query." The agent stops being something you supervise turn by turn and becomes
something that reads the graph, acts, and updates it. You are no longer in the loop; you own the
gates.

## The loop neither one closes

Here is what got me to write this down. Each tool owns one end, and there is an obvious pipe between
the ends that nothing currently connects:

```text
bd ready            ->  agent implements the node  ->  opens a pull request
      ^                                                          |
      |                                                          v
 node closes        <-  fleet-merge converges + merges  <-  the gate passes
```

Beads emits a ready node. The agent builds it and opens a pull request. fleet-merge watches that
pull request until its gate passes, merges it, and the corresponding beads node closes, which drops
the *next* node into the ready set. A graph draining itself to green, with a human watching two sets
of gates instead of driving every step.

I want to be honest about where this stands, because the receipts are the point and I do not have
them yet: those two graphs are not wired together today. I run beads by hand and fleet-merge by
hand. The pipe in that diagram is a claim, not a demo. Building the bridge, and showing a real
dependency graph drain to a stack of merged pull requests without me touching the middle, is the
next post. This one is the idea that makes that post worth writing.

## Closing

I did not set out to build a philosophy of agent workflows. I set out to stop my agents from picking
the wrong task and to stop myself from merging half-reviewed code. Two narrow tools, two opposite
ends of the work.

But the fix at both ends turned out to be identical: stop keeping the work in your head as a list,
put it in a graph, and write down honestly what *ready* and *done* actually mean, so an engine can
tell the real thing from the thing that resembles it. The graph carries the plan and the memory. The
gate carries the judgment you used to spend by hand, one task at a time.

The work is a graph. Build it that way, and the agent stops getting lost. That is the whole post.
