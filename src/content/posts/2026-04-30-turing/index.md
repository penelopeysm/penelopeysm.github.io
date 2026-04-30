---
title: "Au revoir to Turing.jl"
publishDate: "2026-04-30"
tags: ["programming"]
---

Since July 2024 my day job has been to work on [Turing.jl](https://turinglang.org) and associated probabilistic programming libraries.
Due to various reasons this is coming to an end: I'm moving to another project from next week onwards.
(I am currently also the only full-time developer on it, so a corollary is that as of next week, nobody will be working on it.)

This has probably been the most intellectually stimulating thing I've worked on full-time (perhaps even more so than my PhD).
The large-scale, open-source aspect of it is also something that is quite unusual for a project, and I'm glad to have had the opportunity to work on something like that.
For one, it feels really meaningful to do work that leads to immediate, tangible benefits for other people.
But also, I've lurked on Reddit and other forums for many years and I've always looked up to people who are active in programming communities, and it is slightly weird, but also gratifying, to find that I'm now one of these people (in some circles, at least).

Anyway, given that this feels almost like the end of a chapter, I thought I'd write a little bit (or a lot) about my thoughts on the entire thing.
I'll also have a talk on Turing.jl at JuliaCon 2026, so this might turn out to be a sneak peek of the contents (but who knows).

## Things that went well (code)

### DynamicPPL APIs

Over the course of the past two years we've almost completely overhauled everything in [DynamicPPL](https://github.com/TuringLang/DynamicPPL.jl), the package that defines the core modelling primitives.
In particular I'm most proud of how we broke up `VarInfo`, an all-purpose struct that tried to do everything that every algorithm needed, into its constituent components, namely:

- initialisation strategies to specify how parameter values are generated
- accumulators to collect the outputs of model evaluation
- transform strategies to specify whether and how parameters are transformed

These are all thoroughly [explained in the DynamicPPL docs](https://turinglang.org/DynamicPPL.jl/stable/evaluation/), so I won't go into detail here.

The outcome of this decoupling is that instead of bundling everything into a `VarInfo`, you can now pick and choose which bits of it you want.
This is _the_ root of all the performance improvements in Turing in recent months: we simply stopped calculating and storing all the things we didn't need to calculate and store.
It sounds almost trivial when put that way, and indeed maybe the underlying idea is quite simple, but getting there required what I'd consider fairly deep conceptual understanding.

### VarNamedTuple

On top of this API redesign, we also completely rewrote the internal data structure that is used to map variable names to models.
The easiest way to motivate this is to notice that `Dict`s are slow (and potentially type-unstable if the values are heterogeneous).
While `NamedTuple`s can make everything type-stable, it restricts keys to `Symbol`s: this poses a problem for Turing, whose model syntax attempts to be as permissive as possible, meaning that you can have almost any combination of indexing and field access syntax on the left-hand side of a tilde.
That means that `NamedTuple`s are a no-go.

For a long time, DynamicPPL had notions of 'untyped' and 'typed' VarInfo.
'Untyped' was really just a `Dict` under the hood, whereas 'typed' was a `NamedTuple` of `Dict`s (grouped by top-level symbols).
This was 'good enough', but possibly the most outlandish idea we had was that we could actually do better.

The resulting `VarNamedTuple` data structure ended up being one that combined the best of both worlds: it is as performant as a `NamedTuple` in the case where there were only top-level symbols, and is (*almost*) as generalisable as a `Dict`.
In fact, on top of yielding performance improvements which legitimately surprised us, it additionally solved long-standing correctness issues, as [the docs explain](https://turinglang.org/DynamicPPL.jl/stable/vnt/motivation/).

What this means is that across the entirety of Turing and DynamicPPL, essentially every interface is now specified in terms of `VarNamedTuple`.
If you call `rand(model)`, you will get a VNT.
If you do an MLE optimisation on a model it returns a VNT.
If you want to initialise MCMC sampling from somewhere, you can pass in a VNT, and so on.

This makes for a *much* more cohesive experience: in the past, `rand(model)` would return a `NamedTuple`, MLE optimisation would return a `NamedArray`, and MCMC sampling had to be initialised from `AbstractVector{<:Real}`.

### Other things

The above are, by far, the meatiest bits of work we have done on the Turing ecosystem.
They are responsible for versions 0.36 through 0.41 of DynamicPPL, which are large and admittedly quite breaking.
There are still many other smaller things, which I can't list exhaustively, but here are some examples:

- A lot of work has gone into automatic differentiation. Turing is quite naturally a major source of test cases for AD and we have found and reported a ton of issues. We have [a website benchmarking AD backends](https://turinglang.org/ADTests/).

- I alluded to docs earlier, but there are [way more docs](https://turinglang.org/docs/getting-started/index.html) now, especially in the 'User Guide' section. (While there has been a lot of progress on this front, I do actually think there's still a lot of room for improvement: more on that later.)

- We made completely new versions of `AbstractPPL.VarName` (the data structure used to represent variable names) and the Bijectors.jl interface (the old one was not perfectly well-defined, and also caused performance losses). The average user won't notice this, but it makes everything under the hood work more smoothly.

- I've [written FlexiChains](https://pysm.dev/FlexiChains.jl/) as a complete replacement for MCMCChains. (Although I will point out that this was really a side project for me, which is why it's on my personal GitHub account 🙂) If you aren't using it, you probably should. I genuinely don't believe that there is a real use case for MCMCChains any more (but am happy to be corrected).


## Things that went well (not code)

[...]

## Where could we have done better?

- Backwards compatibility and API stability
  - Lack of review doesn't help.

- Docs especially for people new to Bayesian inference

- Identifying and engaging with users

- Cultivating open source contributions

[...]
