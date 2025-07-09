---
title: "Pull request disagreement"
publishDate: "2025-07-08"
tags: ["programming"]
---

I disagree with people a lot on GitHub.
However, not all disagreement is created equal!
I've been meaning to write this post for a while: hopefully it will help make it clearer as to _how much_ I disagree with something.
In particular, I fear that I sometimes give the impression that I disagree more than I really do.

When I disagree with some code ('A') I usually do suggest an alternative ('B') that I prefer.
Then, on a scale of 0 to 5:

-----

**0** I literally don't care! I just wanted to throw the possibility out there.

**1** I consider this essentially a stylistic choice. I would write B if it were my personal codebase, but otherwise I don't care.

**2** I can see benefits for both A and B. I think on balance B is better, but I'm not particularly fussed about it.

**3** I don't think A is necessarily _wrong_, but I do think that B is objectively better (i.e., A is not Pareto optimal).

**4** This is the point where I actually don't like A (in the sense that I wouldn't want to maintain the code associated with it) and I will push very hard for it to be changed.

**5** I _really_ disagree with it, and I will refuse to engage further with the PR until and unless somebody convinces me why I'm wrong.
I expect that this level of disagreement would be reserved for the direction of an entire PR rather than a single line in a PR.
Obviously, this doesn't stop other people from approving and merging the PR. If that happens, they can bear responsibility for it.

-----

I would estimate that most of my disagreements are 2's, with a non-negligible minority of 3's.
On my time on Turing.jl there have maybe been a handful of 4's.
I don't think there has really been a 5 (yet).

Of course, these numbers are always based on my perspective which I recognise to be incomplete.
Thus, the level of disagreement should not be taken as being immutable.

And finally, even though the intent of this post is really to help reduce the amount of perceived negativity, I am cognisant of how negative this post sounds.
I have long found that having my pull requests reviewed can be a bit of a draining experience because I always get comments disagreeing (rather than agreeing) and I have to spend time justifying myself and/or addressing the comments.
It's really very much like submitting a paper to a journal and then finding that you have to do five more experiments because the reviewers asked for it, rather than because you yourself thought that they were good experiments to do.
Even though I do try to be nice, I recognise that others may well feel the same way when I review their pull requests.
Let me know if I'm being too adversarial or if you have suggestions for how to be more positive :)
