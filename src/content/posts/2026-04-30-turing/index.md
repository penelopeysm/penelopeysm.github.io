---
title: "Au revoir to Turing.jl"
publishDate: "2026-04-30"
tags: ["programming"]
---

Since July 2024 my day job has been to work on [Turing.jl](https://turinglang.org) and associated probabilistic programming libraries.
Due to various reasons this is coming to an end: I'm moving to another project from next week onwards.
(I am currently also the only full-time developer on Turing, so a corollary is that as of next week, nobody will be working on it.)

This has probably been the most intellectually stimulating thing I've worked on full-time (perhaps even more so than my PhD).
The large-scale, open-source aspect of it is also something that is quite unusual for a project, and I'm glad to have had the opportunity to work on something like that.
For one, it feels really meaningful to do work that leads to immediate, tangible benefits for other people.
But also, I've lurked on Reddit and other forums for many years and I've always looked up to people who are active in programming communities, and it is slightly weird, but also gratifying, to find that I'm now one of these people (in some circles, at least).

Anyway, given that this feels almost like the end of a chapter, I thought I'd write a little bit (or a lot) about my thoughts on the entire thing.
I'll also have a talk on Turing at JuliaCon 2026, so this might turn out to be a sneak peek of the contents (but who knows).

## Things that went well (code)

### DynamicPPL APIs

Over the course of the past two years we've almost completely overhauled everything in [DynamicPPL](https://github.com/TuringLang/DynamicPPL.jl), the package that defines the core modelling primitives.
In particular I'm most proud of how we broke up `VarInfo`, an all-purpose struct that represented the state of model execution and tried to do everything that every algorithm needed, into its constituent components, namely:

- initialisation strategies to specify how parameter values are generated
- accumulators to collect the outputs of model evaluation
- transform strategies to specify whether and how parameters are transformed

These are all thoroughly explained in [the DynamicPPL docs](https://turinglang.org/DynamicPPL.jl/stable/evaluation/), so I won't go into detail here.

The outcome of this decoupling is that instead of bundling everything into a `VarInfo`, you can now pick and choose which bits of it you want.
This is _the_ root of all the performance improvements in Turing in recent months: we simply stopped calculating and storing all the things we didn't need to calculate and store.
It sounds almost trivial when put that way, and indeed maybe the underlying idea is quite simple, but getting there required what I'd consider fairly deep conceptual understanding.

### VarNamedTuple

On top of this API redesign, we also completely rewrote the internal data structure that is used to map variable names to values.
The easiest way to motivate this is to notice that `Dict`s are slow (and potentially type-unstable if the values are heterogeneous).
While `NamedTuple`s can make everything type-stable, it restricts keys to `Symbol`s: this poses a problem for Turing, whose model syntax attempts to be as permissive as possible, meaning that you can have almost any combination of indexing and field access syntax on the left-hand side of a tilde.
That means that `NamedTuple`s are a no-go.

For a long time, DynamicPPL dealt with this problem by having 'untyped' and 'typed' VarInfo.
'Untyped' was really just a `Dict` under the hood, whereas 'typed' was a `NamedTuple` of `Dict`s (grouped by top-level symbols).
The first time you ran a model you had to use the 'untyped' VarInfo, and then you could switch to the 'typed' one for subsequent runs, which would be much faster.
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

- I've [written FlexiChains](https://pysm.dev/FlexiChains.jl/) as a complete replacement for MCMCChains. (Although I will point out that this was really a side project for me, which is why it's on my personal GitHub account!) If you aren't using it, you probably should. I genuinely don't believe that there is a real use case for MCMCChains any more (but am happy to be corrected).


## Things that went well (not code)

As mentioned above, during my time on Turing I've had the opportunity to work with many people in the Julia community.
This has been incredibly rewarding, both in terms of being exposed to new ideas, but also just on a personal level!
I've met quite a few of these people at various places, whether at my London office or at JuliaCon last year.
I'd like to also give a shoutout to all the people I've spoken to on Slack but haven't met in person.
You are more than welcome to let me know if you are passing by London!

One of the most unique things about Julia is the extent to which different packages are intertwined, particularly via the package extension mechanism.
I suspect that if I had been working on any other language I would have been in more of a walled garden where I'd focus much more on just my own library.
I've also found people in the Julia community to mostly be very nice and open to collaborative discussion.
This might partly be because of the way the language works, but I think the general community vibes also tend to encourage this.

To be fair this hasn't always been rainbows and sunshine, either: there have definitely been disagreements and times where I feel caught between two sides.
But I think I have managed to come out of them not too badly, and have also learnt a lot about dealing with such situations.
I've always tried to be completely honest and unprejudiced and I hope that that comes off.
In fact some of the situations that I regret have come from me _not_ being upfront enough.
If I've been too annoying at any point, I'm sorry!

## Where could we have done better?

To end off, though, I don't think it would really be much of a growth experience if I didn't have some points about where things did _not_ go perfectly.
I feel that my colleagues and I tried our best to do everything we could, and some of these are genuinely difficult problems to solve which I don't have answers to.
In fact these points are the ones which I wanted to focus most on in my JuliaCon talk!

### Backwards compatibility and API stability

As mentioned above, we've made a lot of breaking changes to DynamicPPL and Turing.

Now, one could definitely argue that many of these breaking changes were *necessary*.
We also implemented a release cadence which involved slower, bigger releases, to reduce the amount of CompatHelper churn across the ecosystem.
And we also made sure to put together [incredibly detailed changelogs](https://github.com/TuringLang/DynamicPPL.jl/releases) to help people migrate to newer versions.

That said, I think we definitely still adopted the attitude that it's fine to move fast and break things.
There are a couple of aspects to this that I'm wary of:

- I don't think it translates well to production-level software where one has a large existing userbase to consider. While in my CV I definitely tout the 'Turing.jl has more than 2000 stars...' line, I'm also quite cognisant that it is not currently at a 'widely used in production' level.

- I do think that a cleverer, or wiser, team of software engineers might have been able to reduce the amount of churn, by having a concrete long-term plan for how to do things. I think some of the changes we made (e.g. accumulators, init strategies, `VarNamedTuple`) were quite intentional, but some of them (e.g. transform strategies, `OnlyAccsVarInfo`) were honestly rather coincidental and were developed because it was what I thought of in the shower the previous day.

In the last few months working as the only developer I have felt this even more acutely.
In the past when I had some weird ideas at least I'd have colleagues to bounce ideas off and to refine them, and that would at least save us the pain of having to change it later in another breaking release.
Without that, I tried my best, but I think especially with transform strategies, I introduced some changes in v0.40 which I immediately tore up in v0.41, which felt rather bad.

As with many software engineers nowadays, I try to get Claude to review my PRs.
The problem with Claude is that for the most part it really shares your blind spots.
It sees that you're trying to do X, and will never question whether Y might be better.
That means that often it's only really reliable at detecting 'known unknowns': things that you could fix if you did a very careful manual review.
To deal with 'unknown unknowns' you really need more perspectives in the room.
So my unsolicited advice to anybody out there who has a _team_ of developers is: please do try to keep it as a team, you will probably get better outcomes :).

### Publicity

It turns out that the performance improvements in DynamicPPL have meant that Turing is [competitive with, and sometimes faster, than Stan](https://github.com/JuliaBayes/posteriordb-bench).
(As long as you use reverse-mode Enzyme, it seems!)
Of course there are also many other benefits to Turing, namely complete access to the Julia ecosystem, something which Stan can't even attempt to offer.
However, I don't know if that has translated into any real growth in terms of users.

While I've done community newsletters for a while, I can't help but feel that these were circulated in places that were already 'friendly' to us: our own GitHub, website, and the #turing Slack channel.
I don't think we ever made much of an effort to promote Turing.jl 'externally'.
Basically, I think we did a good job of making probabilistic programming look good to Julia users, but I don't think we did a good job of making Julia look good to probabilistic programming users.

Part of this, unfortunately, comes from my own insecurity about Bayesian inference.
The truth is that I'm a software developer and I don't really know much about inference methods beyond the basics.
For example, I'm capable of [fixing correctness issues with Turing's Gibbs sampler](https://github.com/TuringLang/Turing.jl/issues/2801), but the way I reason about it is by looking at the code and what it does, rather than a more principled approach of sitting down and thinking about what's needed to make the posterior stationary.

This means that I really don't feel like I have the confidence to go to, for example, somewhere like StanCon, give a talk about Turing, and field questions.
There are two ways to deal with this.
One is to say that it's my job, and I should suck it up / git gud, and just do it.
The other would be to rope in someone else in the team who _is_ actually an expert on Bayesian inference.
Unfortunately though the latter is moot because there wasn't really a team.
I'll stop short of saying that I did something _wrong_ by not taking on this responsibility, but I definitely wish that we _had_ had the resources to do this.

### Cultivating open-source contributions

When we started on the project, one of the key aims we had was to make sure that after the project ended (in effect now!), Turing would be in a place where it was possible to be maintained purely by the open-source community.
I'm really not certain that that's happened at all.
While we've made great strides in terms of docs and onboarding (the docs have [a full section on contributing now](https://turinglang.org/docs/contributing/start-contributing/)), I'm not sure that there have really been enough regular contributors to make me feel comfortable about this.

This is quite obviously a problem across the entire Julia ecosystem.
Packages (even important ones) are maintained by small teams, individuals, or in some cases nobody.
To some extent, this just reflects the relative lack of popularity of Julia: there are just fewer people doing the work.
Another problem is that Julia's users lean very academic in nature, and it is quite difficult to get them to contribute upstream, whether it's because they don't have time or because their work is private and they don't want to dump their data (or an MWE) on a GitHub issue.

However, the fact remains that if you were to judge us solely on whether the community could maintain Turing now, it would probably be a fail, and I don't have a good answer for this at all.
I think part of the solution to this might be to bring in people who care about probabilistic programming.
In other words, I think this ties into the previous point.
I think the average Julia person has no real motivation to work on Turing, which is totally fair — I have no real motivation to work on, say, SciML!

### Identifying users

In general, I think there is also a broader question about funding open-source software.
For example, how do you get people like me to spend their day job developing and maintaining libraries?
I've thought about this on and off for a long time but I still don't really have a good answer.

One possible way out here is to notice that it's quite hard to justify 'let's maintain this software' on a funding application / grant, but if one notices that said software is being used _for_ exciting things, the motivation for funding it immediately surfaces.

The problem here is that I think we still don't have a very, very clear idea of what these exciting things are.
We do have some knowledge of our users: at one point we tried to collect a list of publications that were using Turing.jl, and also just the general community engagement that I do has helped with this.

However, I think we could definitely go deeper.
In particular, I'd really be very keen to see _demand for specific new functionality that is driven by downstream usage_.

Most of the things we did on DynamicPPL were not, in and of themselves, motivated by the way people were using Turing.
Of course things like correctness, modularity, and performance are good for downstream users, but they're not something we needed users to tell us about: we could go and fix it simply because we were sensible software engineers.
I can't help but feel that we have almost lived in a bit of a bubble here.

Another problem of not knowing users is that it's very hard to get a sense of real-world use cases.
My job as a software engineer requires me to find minimal examples, so I'm very good at coming up with toy models that demonstrate enough to break something.
However, I have almost no intuition for what a real-world model looks like!
I don't really feel like I have a good sense of what libraries people are using Turing with, what sorts of platforms people are running Turing on, and so on.
This makes it quite difficult to draw up a feature roadmap beyond the obvious 'correctness' / 'performance' stuff.

One of the ideas I've been toying with fairly recently, but had no real time to follow up on, was to have something like a TuringCon: this could be a virtual meetup where we got people to give a short talk on what they were using Turing for, why they chose Turing over other PPLs, and what they still saw as lacking in Turing.

Now, the good news is that I'm probably not vanishing.
If you're interested in using Turing for Something Exciting, please do feel free to get in touch.
I probably can't help you all that much, but we can definitely at least discuss; and if nothing else, I'd just like to know what you're up to!

## Where to now?

In the immediate short-term future I'm going to mostly take a break from Julia, not least because I feel quite burnt out from the past few weeks.

As I said above, though, I'm probably not going to vanish completely!
I'll probably still be around to answer questions and stuff about Turing.
And as I said above, I've really enjoyed working with the Julia community and its people.
You are more than welcome to tag me on GitHub or Slack, and I'll do my best to respond.
There are also some nascent side projects on [the JuliaBayes organisation](https://github.com/JuliaBayes) which I'll still be involved with.

But as far as Julia open-source contributions go, I'd really like to take it in a direction that personally excites me more.
After all, this is going to be my spare time, not my day job :)
In particular, that means compiler-related stuff.
I've [offered to help maintain JuliaFormatter.jl](https://discourse.julialang.org/t/juliaformatter-jl-needs-help/136744) (although that's still in the works).
One day I'd like to get to a point where I understand LLVM and MLIR (more so than I do right now, at least, which is quite superficial).
Thankfully, there are many things in Julia that are related to this.
So... see you on the other side, perhaps!
