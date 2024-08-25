---
title: Annoyances with Julia
publishDate: "2024-08-25"
tags: []
---

Having worked in Julia professionally for a bit of time, I feel like I have enough experience to complain about some of the pain points I've faced.
I don't think Julia is a terrible language at all, but I think [this comment on HackerNews](https://news.ycombinator.com/item?id=22289969) summarises my take on it very well.
Quoting:

> Together with Julia's many strengths I think these design choices and community philosophy lead to a language that is very good for small scale and experimental work but will have major issues scaling to very complex systems development projects and will be ill-suited to mission critical applications.

I think it's a huge improvement over Matlab, but parts of it have, in my opinion, not been developed with principles of good software engineering in mind.
I think this is partly because of the target audience: it's trying to be a language for scientists, who may have used Python, Matlab, or R before.
However, they aren't software engineers, and generally aren't interested in maintainability of large codebases or being able to run reproducible code.
(I mean, that's kind of why my entire day job exists.)

It's not to say that other languages don't have their problems.
However, these problems have generally been solved over time, because people realised that the old way of doing things was suboptimal.
(See e.g. Python's package manager, `pip`, installing packages into the global environment, and often requiring `sudo`.)
But I don't get the sense that the Julia community is generally motivated to fix these problems, or that it even sees them as problems.

By the way, before the specific complaints, I'll say my absolute favourite thing about Julia is its metaprogramming tools.
It's extremely cool that you can write macros which directly modify abstract syntax trees, and this allows for domain-specific languages that are very expressive and powerful.

## Multiple dispatch abuse

Julia's multiple dispatch system is often touted as one of its best features.
Essentially, it allows you to define multiple methods for the same function, and the correct method is chosen based on the types of the arguments.
Unfortunately, it gets abused way too often in code like this, which is a very common pattern in many libraries:

```julia
using Statistics: mean, std

struct Foo
    x::Int
    y::Int
end

function Foo(x::Int)
    return Foo(x, 0)
end

function Foo(x::Real)
    return Foo(round(Int, x), 0)
end

function Foo(x::AbstractVector{<:Real})
    return Foo(round(Int, mean(x)), round(Int, std(x)))
end
```

This is, in my opinion, _one of the worst things about Julia code in the wild_.

1. If you're reading code somewhere else that calls `Foo()`, you have to figure out which method it's calling.
   If there was just one function called `Foo`, this wouldn't be a problem.
   Unfortunately, not only do you have to look at the number of arguments, but also the types.

   Explicitly specifying types is not very common in Julia (because apparently it hurts performance), so it's often not trivial to figure this out by reading the code.
   And types are effectively only resolved at runtime, so not even a language server can help you.
   Pressing go-to-definition will just give you all four definitions and you still have to figure out which is the appropriate one.
   If you're thinking you could have done this with `grep`: yeah.

1. Okay, well, surely you can at least just keep all your definitions close to each other so that people can see all of them at a glance?
   _Nope._
   Any other package that imports your code can define `Foo` for its own types, or any other types it likes (though the latter is not recommended).
   Hey, I guess at least your language server will tell you where those definitions are scattered.
   In this respect it's better than `grep`.

1. The worst thing about this is that it's a wholly unnecessary pattern.
   In cases where you just want to have a default parameter (like the second and third definitions above), many other programming languages just use keyword arguments.
   If you wanted to construct a `Foo` from some other type of input (like the `AbstractVector` case above), what's so difficult with just _using a different name for the function_—like `FooFromVector`—that would make it so much clearer what's going on at the call site?

I don't doubt that multiple dispatch has its uses, but this really should not be one of them.
In my opinion this abuse encourages code that is written once and never touched again.
Instead of using the shiny feature just because they can, people should really think more about writing code that is easy to read, understand, and maintain.


## `include()`

Packages are structured using `include()`, which literally just pastes the contents of the included file into the current file.
This is a very bad idea.
It makes it very difficult to reason about the codebase, because the scope of your code is determined by the file that includes the one you're in, rather than the file you're writing in.

This also leads to annoying things like if you have two submodules `L` and `M` which both depend on `N`.
If `L` and `M` both `include("N.jl")`, then `N` will be included twice.
So, you have to make sure in your top-level module to include `N` one time before `L` and `M`.

It's not as if Julia was made in the 1970s like C.
It could really use a proper module system, like pretty much every other modern language.


## REPL-based workflow

If you want to install a package from the shell, you need to do something like

```sh
julia --project=. -e 'using Pkg; Pkg.add("Package")'
```

If you want to run tests,

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

In fact, even this workflow is strongly discouraged, because every time you launch a new Julia instance it has to recompile your code.
So, Julia _really_ pushes you towards having a REPL open all the time.
Oh, but if you change your code, the REPL won't be able to keep up and you have to use `Revise.jl`.
As impressive as `Revise` is, it's obviously a hack that exists solely to get around this problem, as evidenced by how it doesn't work with struct definitions.

Compare with most other languages, where you can run tests or a script from a shell which will always use the most up-to-date version of the code and where this process won't incur extra overhead.

By the way, persisting a REPL comes with all its usual downsides such as not being able to easily run code in a reproducible manner (who knows if the code you ran in that session 5 minutes ago affects what you're running now).
I guess people are OK with it because that's just what they're used to, but I find it very annoying, despite having set up shell aliases and [vim-slime](https://github.com/jpalardy/vim-slime) to make life easier.
It also makes CI pipelines a lot harder to read.


## Overly opinionated infrastructure

Too much of the infrastructure that keeps the Julia ecosystem running is centred around community tools that are overly opinionated.
This makes it hard to customise and troubleshoot.

Consider, for example, the process of releasing a new version of a package.
The traditional way to do this is something like

```sh
# edit your version number to 1.2.3
git commit -am 'Release version 1.2.3'
git push
git tag v1.2.3 -am 'Release version 1.2.3'
git push --tags
# then you upload your package to a registry
```

In Julia, you do this:

```sh
# edit your version number to 1.2.3
git commit -am 'Release version 1.2.3'
git push
```

Then you _comment_ on the GitHub commit, which [triggers a bot to release your package](https://github.com/JuliaRegistries/Registrator.jl).
That opens a PR to the registry, and once that's merged, [_another bot_ will add the tag to your package](https://github.com/JuliaRegistries/TagBot).
Oh, by the way, if your documentation is built via GitHub Actions and you want to trigger a documentation build for a new tag, you have to make sure you give the tagging bot an extra SSH key.

Let's say you forget to give it the SSH key and the documentation doesn't build.
(This has totally _never_ happened before /s.)
Okay, well, we can just build the documentation for every tag locally and push to the `gh-pages` branch, right?
Well, _good luck_ doing that with Documenter.jl.
It gives you a function that you can call to build specific tags, called... [`deploydocs`](https://documenter.juliadocs.org/stable/lib/public/#Documenter.deploydocs).
You then realise that this function is not meant to be called locally at all; it's only meant to be called during your CI run.

I'm not claiming that CI is an easy thing to get right.
However, it's way easier to get it right when you are in control of what you're doing, instead of trying to put together a number of jigsaw pieces that can interact in unpredictable ways.
