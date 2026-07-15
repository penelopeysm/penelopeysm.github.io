---
title: "Liking work, hating AI, and liking work less"
publishDate: "2026-07-12"
tags: ["programming"]
published: false
---

## Liking work

When you discover an efficient new way to do work that lets you do 8 hours' worth of work in 2, by far the most common advice you'll get is to keep it quiet.
That's because you can get paid 8 hours' salary and get to chill for 6 hours.

I don't entirely agree with this advice.
Now, that's hardly because I have any sympathy for big corporations or for employers in general.
However, I do think that this advice comes with an undercurrent of disdain for work, and I don't really *want* to be that sort of person.
You see, I'm still idealistic enough to believe that I should *like* my work.

For many of the past few years I think that that was actually more or less true.
I recognise that this is a very privileged position to be in, but essentially I got to do what I liked to do during the day, and I got paid for it because my employers liked what I did.
Win-win!

When you're in such a situation and you find a way to compress your work, then the natural next step is to do *more* of that work — because you like doing it!
Logically, this means that since AI lets me be more productive, I *should* want to use Claude more and get more work done.
Indeed, I see a number of people online who seem to now be using LLMs to build more and more things.
I think that's great for them, if they're happy.

This hasn't worked out *so* well for me, though, which motivates this post.
I do indeed use AI a lot, but I think I use it in quite different ways to many of the loudest voices online, and in general I have actually found that overusing it not only makes me dumber but also simply *less happy*.

## Where exactly is the fun?

To figure out why AI makes me less happy, it's important to start by figuring out *exactly* what makes me happy.
I think there are a couple parts to it:

### Engineering

Fundamentally, I like programming because I can *make things happen* by changing things.
When you change a bit of code and your computer gives you a different result, that feedback loop is immensely satisfying.

With AI, my actions no longer have a logical correlation with the changes in the behaviour.
Sure, there is still some correlation: by and large, the difference in the output is what I tell Claude to do.
However, it's no longer of the form 'if I do X, then Y will happen': the interaction with the system becomes *linguistic* in nature, with all the uncertainties that that entails, rather than *logical*.

Indeed, I think that this uncertainty was the biggest thing that I disliked about wet-lab chemistry.
I did a year of experimental organic chemistry, during which I found out that we have much less control over reactions than we *think* we do.
Subtle differences can lead to [reactions being difficult to reproduce](https://www.science.org/content/blog-post/phil-baran-blog-syn).
Experiments often have quite different yields depending on the exact chemical structure of the reactants, and this is frequently left unjustified in papers, often because it's just very hard to figure out the real reason.
This was the biggest reason why I chose to do a spectroscopy PhD rather than organic chemistry, and probably why I eventually moved to software engineering.

I do indeed recognise that there are people out there who actually aren't that interested in the actual software engineering, and are more interested in getting the outputs.
In fact I would wager that a non-negligible proportion of my ex-colleagues at the Turing fall into that category.
For these people, this probably isn't actually a benefit, and conversely the fact that LLMs let you get outputs faster is probably a boon.
However, I'm not part of this group!

### Communication

When you've done something nice or figured something new out, you want to show it to other people and get them to be also excited about it.
This can take many forms, whether it's documentation, talks, or just chatting to someone near you.

While one might think that a 'language' model should be good at this, it turns out that they're actually really bad!
LLMs don't (yet) have a good grasp of how to explain things to an audience, and I think it's because the way *they* understand things is fundamentally different to the way *humans* understand things.

*Humans need a story*: something that starts from what they already know, and adds layers on top of that one by one, until they understand all parts of a complex system.
That means that when writing for humans you need to be judicious in choosing what to say and when.
In contrast, LLMs don't need this.
They work very well with fact dumps, and if you ask an LLM to write documentation, chances are you will get a relatively unstructured bunch of facts.
Some of these will be important and some will be irrelevant, but you'll be left to figure out for yourself which is which.
Pair that with the known tendency of LLMs to use corporate jargon ('load-bearing') and you end up with vapid docs that have a very poor signal-to-noise ratio.

If I used AI to write docs, I would lose this ability to connect with an audience and to bring them along.
That's something that I genuinely value: it's satisfying to see a docs page, or more rarely a talk, come together.
Some of the people I look up to the most are people who can communicate well: there are plenty of Haskell people past and present, notably [Alexis King](https://lexi-lambda.github.io) and [Simon Peyton Jones](https://simon.peytonjones.org/).
And there are two textbooks in particular which are *directly* responsible for my PhD, namely, [David Griffiths's *Introduction to Quantum Mechanics*](https://www.cambridge.org/highereducation/books/introduction-to-quantum-mechanics/990799CA07A83FC5312402AF6860311E), and [James Keeler's *Understanding NMR Spectroscopy*](https://www.wiley.com/en-us/shop/general-chemistry/understanding-nmr-spectroscopy-2nd-edition-p-9780470746080): both are excellent examples of technical writing and have greatly influenced the way I write.

## About quality

On top of me enjoying aspects of not using AI, I do also find that AI generated code is worse than what I can do.
[I've written about this before](/posts/2026-04-05-claude) so won't really repeat myself.

However, the effect is actually twofold.
Firstly, the code is less good.
But secondly, it's also way, way easier to *pretend* that it is good.
If it passes the tests and all that, and you don't take the time to really sit down and think about how it works and edge cases and all that, you *will* potentially miss things.

And if you do actually review your AI generated code very carefully, I suspect it will almost certainly take you a similar amount of time it would have taken you to just write it.
As much as some people might not want to admit it, sitting down and spending time writing code forces you to engage deeply and critically with what you're doing.
I think that in the vast majority of circumstances, developing this understanding is *the* primary bottleneck in producing good code.

## Why I hate AI

Given the above, it shouldn't be hard to see why I don't really like using Claude to write code myself.

LLM code is one thing, but what *really* annoys me is seeing people write prose with LLMs.
An increasing proportion of posts on Reddit are just LLM generated.
You might be forgiven for thinking that this is because of the anonymity.
But no — I regularly see GitHub comments or package announcements on Julia Discourse that are clearly AI.
It's almost as if people have forgotten how to just talk.
And considering how bad LLMs are at technical writing, it is pretty disrespectful: the subtext is 'I couldn't be bothered to take the time to explain this to you, so here, you wade through this pile of rubbish'.

## Liking work in the future

Given that the use of AI takes away many of the things I enjoy about my work, I do wonder if I will still be able to claim that I like work in the near future.

As it happens, I'm actually between jobs right now: I'm done with the Turing, and I'll be starting as a software engineer at Jane Street next month.

The truth is that my job at the Turing was never really on a topic that I cared much about.
Sure, I have worked on Julia MCMC stuff for a couple of years.
But I don't have any vested passion in MCMC, Bayesian methods, or machine learning in general!
In fact, machine learning is one of the areas where I've tried *several* times to get myself interested in, but have invariably failed.
It just so happened that this area let me do the fun aspects of software engineering that I described above.
In some ways it was a marriage of convenience.

I'm actually incredibly glad to have found a new job that caters more towards my technical interests of functional programming and compilers.
(Besides, it's a place where I could say I liked these things *and* have someone reply 'yes, me too'!)
That means that, even if the job (and indeed the broader software engineering profession) becomes less interesting because of AI, at least it'll be in an area that I am genuinely passionate about.

With that said, I don't think that it's a particularly bold prediction to say that I expect there to be pressures to be more productive with AI.
I am kind of optimistic that there will be a *healthy* relationship with AI and that good software engineering and communication will still be considered a virtue (this is a large part of the reason why I chose to move there — essentially I think that people will broadly share my values!).
But, barring a complete collapse of the AI industry, I don't think that we'll ever go back to a place where there'll be the same amount of space to do things by hand.

## Taking back control

In light of this, I think it'll become more and more important for me to find spaces where I don't *have* to use AI, and just choose not to do it.

About two weeks ago I got Fable 5 to build something like [this website](https://pysm.dev/bubbletea/) (source code [here](https://github.com/penelopeysm/bubbletea)).
The result you see here actually isn't Fable.
It's me, because immediately after Fable finished it I felt really quite guilty.
Building web apps with maps used to be my specialty!
(I did this at work for about a year.)
So I quite literally `rm -rf`'d all of it and started from scratch with minimal LLM assistance.
(I still used Claude, but only to ask it questions about CSS — the sort that one would go to Stack Overflow for in days of yore.)

It took me approximately a day to do (most of it was the horribly mundane task of getting coordinates for every shop from Google Maps).
Is it *prettier* than what Fable did?
I'm not really sure it is.
But I am a lot *happier* with it.
It feels like it's *my* thing: it reflects a quirk that's uniquely mine (the bubble tea habit), and doing it myself means that there's just a little bit of my soul inside it.
It's like having a home-cooked meal instead of a takeaway.

Likewise, for many of my open source projects I have to split them into 'things I care about' and 'things I don't really care about'.
For example, [FlexiChains.jl](https://pysm.dev/FlexiChains.jl) is kind of my baby.
I think the library is well-designed (I'm biased!), and I still enjoy working on it because I think it's useful to the Julia community.
I might get Claude to do horrible boilerplate stuff (Makie overloads...), but if I'm trying to make a FlexiChains extension for a different package, I'd really rather do it myself.

In contrast, while I care a lot about [JuliaFormatter.jl](https://github.com/JuliaEditorSupport/JuliaFormatter.jl), it's not really *my* codebase.
I don't feel the same sentiment towards it, and it's also turned out to be harder to maintain than I would've liked because there are years of historical baggage and edge-case inconsistencies baked into it.
So, yeah, sure, I'll get Claude to fix bugs and stuff.

And yes, when I submit a PR that was mostly Clauded, you *are* allowed to read it as me not caring as much as I could have.

## Not liking work (so much)

I've spent a lot of time talking about how much I love doing work.
However, over the years I think I've come to accept that this can also be taken to an extreme.
(I never used to think that overworking was possible — which might sound odd, but it unfortunately won't be surprising to anyone who has done a PhD.)
I'm very aware that I habitually overwork, partly because I have an ability to concentrate on doing something for a long time, but largely because if I were to pick amongst *all* the things I might want to do at any given point in time, work is quite high on the list.

Sometimes, perhaps even often, this has made me happy!
As described above, I already have a lot of *intrinsic* motivation for doing work; when that's paired with *extrinsic* motivation such as positive feedback, it's very hard to find something that can compete with it.
Indeed I often feel that what I experience is an addiction.
(Work addiction is a known thing.)

However, sometimes it all falls apart.
Sometimes things aren't progressing, sometimes the people I work with aren't nice, or sometimes I'm not interested in what I'm doing.
Perhaps the craziest thing about me, or the most worrying thing, is that somehow even under such circumstances I *still* overwork.
As much as I wish life was always in the 'happy' regime, the 'sad' regime has unfortunately been quite frequent recently.
And at times like these, pouring too much of myself into something leads to an overwhelmingly negative mental state since I start to feel that it's all for nothing.

Over the past couple of years I've spoken with quite a few people about this, and thought about various solutions, but I don't think any of them have really been lasting.
However, it may perhaps be the case that if work becomes (relatively) less fun, then this might be one way out.
There needs to be a sweet spot where I like work enough to want to do it, but not like it so much that I spend every waking hour doing it.

So maybe... I love AI after all?
