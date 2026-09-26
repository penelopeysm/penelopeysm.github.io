---
title: "New things are hard as an adult"
publishDate: "2026-09-26"
tags: ["programming"]
published: false
---

At the beginning of July, I quit my old job at [The Alan Turing Institute](https://turing.ac.uk).
Since end-August I've been working at [Jane Street](https://www.janestreet.com/functional-programming/).

## Why?

I've always been keen on functional programming, and in recent-ish years have been trying to learn more about compilers.
At the Turing, though, I think I was pretty much alone in having these interests, or very firmly in the minority.
In my experience, people at the Turing were more interested in the actual science, which is great, but the parts of science I could personally get myself interested in (quantum chemistry, i.e., my PhD topic) was very much _not_ part of the Turing's research.

In fact, I don't even think that good software engineering was a massive focus at the Turing.
I think generally we tried to avoid having really bad code, but often the aim was definitely just to be 'good enough'.
I was lucky in that much of my time was [spent working on open-source libraries](/posts/2026-04-30-turing), which taught me a lot about how to write code that stood up to the scrutiny of the outside world.
However, that specific project was pretty unique, and even if my time on it hadn't been abruptly terminated the way it was, I couldn't have stayed on it forever.
All that is to say that I think that there was always a ceiling on how much I could grow as a software engineer.[^1]

[^1]: I don't mean this as a particular criticism of the Turing; there were certainly many things I was disgruntled about, but I think that what I'm describing here is just a misalignment between what it does and what I wanted.

So when I started looking elsewhere for jobs I tried quite hard to find things that I thought I would like.
I didn't apply for generic software engineering jobs: apart from Jane Street, whom I had known for ages because of their OCaml work, the two other roles I interviewed for were both in compiler development.
So to land this job, and on top of that to be put on the same team[^2] as the people who work on [OxCaml](https://oxcaml.org) (Jane Street's fork of OCaml), was in many ways a dream come true.
Of course, no change like this could come without any reservations, but that's for another time!

[^2]: My wording here of 'put on the same team as the people who work on OxCaml' is very carefully constructed.
I don't actually work on the compiler itself, I'm not that clever!
Very broadly speaking, I work on libraries that sit on top of the compiler and are used throughout Jane Street, [some of which are open-source](https://github.com/janestreet).

## Some reflections

So, there are a ton of things I like about my new job:

- My immediate colleagues are very nice people!
- The office is ridiculously well-provisioned, and the food in the office is genuinely good.
  In my previous job I cooked virtually all my meals, including lunches, and I was very happy to be able to control what I ate, so I thought I would be sad to give that up.
  But in all honesty I haven't missed cooking lunch at all.
  (I still cook all my dinners! Despite the temptation, I have not actually eaten out on my own since starting.)
- Going into the office five days a week costs me a lot in Tube fares, but it has been a lot healthier for me in terms of having a work schedule which I can (ostensibly) put down when I get back.

But at the same time, it's been very, very humbling and I have to confess that the struggle with impostor syndrome is extremely real.
I think I used to fancy myself as some kind of pretty good programmer.
Maybe at the Turing I *was*.
I was probably better at writing software than most other people, and after all, I *did* make it through the rather unforgiving interview process at Jane Street.
But as I described earlier, the Turing just generally wasn't a place for serious software engineering, and so maybe the bar I was measuring myself against was not quite as high.

But, well, things are very different now!
I'm working next to people who have spent many years thinking about programming languages, and it's not always possible to understand what they are saying.
In the past I might have been the only person who knew what a [GADT](https://en.wikipedia.org/wiki/Generalized_algebraic_data_type) was, but now that's _practically_ assumed knowledge.
It's become incredibly clear to me that what I might once have thought was a relatively deep understanding of functional programming is in fact really quite superficial.
Now, I'm not delusional; I always knew that at the back of my mind because I'd read [/r/haskell](https://reddit.com/r/haskell) and be acutely aware that I didn't understand what went on there.
But I guess it's easy to fall into the trap when you interact with people who are mostly happy to write Python and call it a day.

(By the way, I'm not trying to accuse my ex-colleagues of being dim.
As I said earlier, the Turing just wasn't really a place for computer science or software engineering.
For everything I could do that others couldn't, there would be something else they could do that I couldn't.
Like, training PyTorch models!)

Even if we ignore the programming language theory stuff, there's still a lot of computer science or software engineering stuff that goes over my head.
Of course, that's probably because I don't have a real computer science background.
There is so much fundamental knowledge that one would ordinarily slog through in an undergraduate degree which I don't have at all.

I don't regret taking this job for a single moment, and I know that a lot of these are 'just' growing pains and that it's precisely the right environment for me to push myself a bit harder than in my previous job (or maybe _as hard as_ in my previous job, given that I _did_ work very hard...).
But it is quite mentally difficult sometimes.
The biggest problem is that _right now_, being so unproductive and feeling like I'm struggling with even the smallest type checking error, is a _sharp_ constrast to my previous work where I was probably the person who knew the codebase the best and could tell exactly what and where to change.
Being in this situation can really make you feel quite helpless and small.
(It certainly didn't help that at the same time I also started using a split keyboard, _and_ also dramatically reduced my AI usage in an effort to force myself to learn properly.)

In fact, my desire to feel like I knew what I was doing was so strong that on many nights I just went back and continued writing some Julia code even though I was really tired, just because that's what I felt like I have control over.

In many ways this was reminiscent of me starting on the flute.
I sounded horrible, couldn't play more than three different notes, and would feel faint after playing for 10 minutes (really!).
That was a pretty stark contrast from the violin, which I had done quite seriously as a young girl and was fairly decent at.

There are also some similarities to my struggle trying to sit down and learn about compilers.
When learning a new topic like that, it's practically impossible to sit down and start magically writing something that's very interesting.
You'll have to struggle through doing toy compilers for toy languages, doing the slightly boring LLVM Kaleidoscope tutorial (which I _still_ haven't finished...), and other things like that.
The truth is probably that in every field you have to do a lot of those boring things to acquire the knowledge that lets you do the fun things.
I couldn't have done my PhD without countless hours of tutorials drawing mechanisms for things that probably didn't really happen in real life.

I don't really know how to solve this, apart from just lots and lots of time.
Of course, the problem is that it's a job, so I will have to start figuring things out sooner rather than later. :)
