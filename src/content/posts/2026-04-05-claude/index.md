---
title: "Coding with Claude"
publishDate: "2026-04-05"
tags: ["programming"]
---

Lots of people have written lots of things about coding agents and stuff, which I won't rehash, so I'll just try to list a few things good and bad.
This is mostly based on my experience writing Julia with Claude Opus 4.6.

I 'only' have a Pro plan, which also means that I've not gone fully down the path of vibe coding.
Nor do I really intend to.
I mostly use it as a place to bounce ideas off and to review my own PRs (since, unfortunately, I don't actually have anybody at work to do this with).

## Good

- It's sometimes pretty good at suggesting things I might not have thought of and mentioning them.
  I think many of us have blind spots where things we know end up being 'too obvious to mention'.
  When writing docs I try really hard to avoid this, and a lifetime of writing Stack Exchange answers has trained me very well in this regard, but when it comes to code I don't think I'm amazing at this.
  I like that to Claude nothing is really obvious: it's often quite useful to take a step back, read its output, and think 'should I document this in a comment?' or 'should I add an explicit test for this?'.

- It's docs on steroids.
  As with all LLM tools one needs to verify and not just trust blindly: this is thankfully easy enough with code because there tends to be an objectively defined right or wrong.
  But it's way, way better than doing a web search for something and hoping that you find the right page.
  Sometimes it even tells you about little corners of the docs that you didn't know.
  (Sometimes it misses these things too, of course: but the point is that even if it does, it doesn't make your code *worse* than if you did it solo.)

- Brilliant at boilerplate, but everyone knew that already.
  I have no qualms getting it to code [most of a React web app](https://github.com/penelopeysm/hut101).

- Most of all, I am really impressed by how adaptable it is to different situations and how quickly it can pick up on my intent.
  (This definitely relies on useful prompting: I haven't explored Claude's settings and memory thingies enough, but I suspect that a lot of common frustrations would be solved by having more carefully constructed instructions.)

## Bad

- It's still way, way, too agreeable.
  While I think I'm a decent programmer, I am conscious that I don't make the right decision all the time, and this is quite evident in the way I sometimes tear up my APIs from minor version to minor version.
  It doesn't really push back on design choices very much, and it's not yet very capable of seeing or anticipating future uses, unless it's something really very bog-standard.

- The corollary of this is that Claude's feedback is only as good as the person driving it.
  It's well-known that bad programmers can create AI slop, of course, but even careful usage of Claude by a bad programmer doesn't magically make them a good programmer.
  The biggest problem with this is that you don't know what you don't know.
  It's easy to go down a route where you think you're doing fine, and Claude makes some improvements and you think you're actually doing quite well, but actually a better programmer would come along and point out flaws.

- For this reason, I actually suspect (but have not yet verified) that Claude is *most harmful* in the prototyping stage of a new project.
  If you have an existing project, it provides a good framework within which Claude can work: it can pick up the general quality of the codebase, design choices, etc. and work from there.
  However, if you set it completely loose on a new project it will probably come up with some ideas that are not great, and then it becomes hard to remove these as time goes on.
  But I don't have enough tokens to really test this out for sure.

Can any of this be mitigated?

I have not yet explored this, but I'm thinking of collecting all the bits of wisdom that I've learnt over the years of working on this project about how to write good-quality code, and shoving them into something that Claude can read from.
That means that even if somebody new comes along who doesn't know the general standard of the codebase, Claude can, in principle, get them up to speed.

## What's the future like?

I don't think that Opus 4.6 will replace all human software engineers.
However, I suspect it can replace human software engineers where the quality of the code doesn't need to be great.
In particular, as someone who used to work in scientific research, I have definitely seen a lot of bad code which could easily be rewritten with Claude to a higher standard.
Technical debt doesn't matter in these situations, because the aim is to get a paper or feature out and forget that the code ever existed.

If we don't want to accept this future, then that leads to the question of how we can incentivise researchers to make their code better.
Despite this literally being the purpose of my current job, I don't yet have a good answer to this.
One could make arguments about how better code leads to increased collaboration and usage.
With that sort of argument, the general sentiment behind it is: let's try and unify the aims of scientific research and software development.
But right now, I can't help but feel that ultimately funding for open-source code has to come with the express purpose of it being high-quality and open-source, rather than being tied to research grants, new papers, or new features.
In other words, stop trying to pander to the academic incentives, and instead say that this funding is about making things maintainable, etc.
I'm not sure that there has been a good solution to this yet...

The alternative of course is to just roll with it, and let the LLMs take over.
Which might just happen.
