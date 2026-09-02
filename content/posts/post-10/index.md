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

At the start: what should it do next? A long project is a web of dependencies. Task B needs A
finished first, C and D can go in parallel, E is blocked on both. Hold that in a flat TODO and the
agent picks the wrong thing, and the plan dies with the context window that held it.

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

`bd ready` is the one command that matters. It returns the nodes whose dependencies are all satisfied, which is
exactly the set of things that can be worked right now. Finish `bd-1`, close it, and `bd-2` falls
into the ready set on its own. The agent never guesses at order, because the graph recomputes it on
demand; there is no plan to remember.

The other half is that the graph is git-backed, so it survives the thing agents are worst at:
forgetting. Context windows roll over and the session dies, but the DAG is committed next to the
code. A fresh agent runs `bd ready` and is immediately oriented, no re-briefing. The graph is the
memory.

So beads answers the start-of-work question, and it answers it structurally: **ready work is a node
with no unmet dependencies.**

## The second graph: what is ready to finish

`fleet-merge` is a skill I wrote to close the other end. Point it at a repo with open pull requests
and it watches them until each one is genuinely mergeable, then merges it with whatever method the
repo allows. It is the boring, vigilant gatekeeper that never gets tired and never waves something
through because it is Friday.

I did not think of it as a graph tool when I wrote it. It is. The pull requests are nodes. Two pull
requests that touch the same file have an edge between them: whichever merges second has to rebase,
so fleet-merge lands them strictly one at a time. The edge decides merge order the same way a beads
dependency decides start order. And every node has the same question hanging over it that beads asks
at the other end, only inverted: beads asks whether a node can start, fleet-merge asks whether it
can finish.

The answer is a gate, and the gate does all the work. A pull request is done only when all of
these hold at once:

- No real CI failure, and nothing still running. Not "the checkmark is green": a check that is still
  in progress reports no conclusion yet, and if you read that as success you merge with the tests
  half-finished.
- A review sign-off on the *current* commit. Wherever the review came from, if it sits on a commit
  you have since pushed over, it approved code that no longer exists and says nothing about what you
  are about to merge.
- No suppressed review comments. This is the one that catches real bugs. A reviewer can file notes
  that never become blocking threads, so the review summary still reads "no new comments" and the
  unresolved count still says zero, while a real finding sits collapsed one click out of view.
- It is not a release pull request. Cutting a release is a human decision; the loop has no business
  making it as a side effect of being green.

None of those four is the naive signal. The green checkmark, the APPROVED badge, the zero-unresolved
count: each looks like done and is not. The value is in refusing the signals that merely resemble
done, the same way `bd ready` refuses to hand you a node that still has an open blocker.

Where does that sign-off come from? Sometimes a reviewer shows up on their own. When nobody does,
fleet-merge does not wait around; it produces the review itself, and the interesting part is how it
sizes one. It does not pick a number of reviewers and fill it. It reads the diff and assigns one
specialist per failure class: a correctness reviewer when behavior changes, a silent-failure hunter
when the change touches error paths or a check that can pass while proving nothing, a comment
analyzer when prose describes the code, a test analyzer when a test claims something is now guarded.
A one-file fix draws a single reviewer. A migration draws a crowd. The panel is never fixed; the
diff decides who shows up.

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

Strip the domains away and both tools make the same three-part move:

1. Model the work as a graph, so you compute order from the graph instead of holding it in your head.
2. Define *ready* and *done* as explicit gates, and make the gates refuse the signals that only look
   like readiness.
3. Let an autonomous engine walk the graph, transitioning a node only when its gate actually passes.

Once you see the move, you start building it on purpose. I have started calling it graph
engineering: turning "a pile of work an agent has to keep straight" into "a graph an agent can
query." The agent stops being something you supervise turn by turn and becomes something that reads
the graph, acts, and updates it. Your job shrinks to owning the gates.

## The loop neither one closes

Here is what got me to write this down. Each tool owns one end, and there is an obvious pipe between
the ends that nothing currently connects:

![The loop: bd ready feeds the agent, the agent opens a pull request, the gate passes, fleet-merge merges, the node closes, and the next node becomes ready](loop.svg)

Beads emits a ready node. The agent builds it and opens a pull request. fleet-merge watches that
pull request until its gate passes, merges it, and the corresponding beads node closes, which drops
the *next* node into the ready set. A graph draining itself to green, with a human watching two sets
of gates instead of driving every step.

To be clear about where this stands: those two graphs are not wired together today. I run beads by
hand and fleet-merge by hand, and the diagram shows where the pipe would go once I build it. Building
that bridge, and showing a real dependency graph drain to a stack of merged pull requests without me
touching the middle, is the next post. This one is the idea that makes that post worth writing.

## Closing

I set out to stop my agents from picking the wrong task, and to stop myself from merging
half-reviewed code. Two narrow tools, two opposite ends of the work, and the fix at both ends turned
out to be identical. The graph carries the plan and the memory. The gate carries the judgment I used
to spend by hand, one task at a time.

The work is a graph. Build it that way, and the agent stops getting lost. That is the whole post.
