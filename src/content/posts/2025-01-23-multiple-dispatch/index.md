---
title: Multiple dispatch as a substitute for inheritance
publishDate: "2025-01-23"
tags: []
published: false
---

## Python

A common pattern in object-oriented programming is to have a default implementation on a parent class and to reuse that in a child class.
In Python, this is done using the `super()` function.

```python
class Parent:
    def foo(self):
        print("calling Parent.foo")

class Child(Parent):
    def foo(self):
        print("calling Child.foo")
        super().foo()
```

Here, calling `Child().foo()` will print:

```
calling Child.foo
calling Parent.foo
```

This means that you get to reuse any logic in the parent class that is supposed to be shared amongst all children, but you also get to do specialised things in the child class if you need to.

## Julia

Now, how do we do this in Julia?

In Julia, concrete types cannot have subtypes, so the equivalent of the 'parent class' must be an abstract type.

```julia
abstract type Parent end

struct Child <: Parent end

foo(::Parent) = println("calling Parent.foo")

function foo(::Child)
    println("calling Child.foo")
    # we can't call the equivalent of Parent.foo here
end
```

Here, `foo(Child())` will just print:

```
calling Child.foo
```

There are a couple of ways of getting around this.
One is to use the abstract type as a value, instead:

```julia
foo(::Type{Parent}) = println("calling Parent.foo")

function foo(::Child)
    println("calling Child.foo")
    foo(Parent)  # this invokes the method we just defined
end
```

In fact, I think that this is the closest equivalent to the Python code above, in the sense that `super().foo()` isn't calling the parent method on an instance of the class, but rather on the class itself.
I don't think I've seen this pattern used very much in Julia, though.
(If you have an example, please let me know!)

Another approach, and one that is used within the Turing.jl codebase quite substantially, is to use different functions.
Because `parent_foo(::Child)` isn't defined, the method for its abstract supertype will be called instead.

```julia
parent_foo(::Parent) = println("calling Parent.foo")

function foo(c::Child)
    println("calling Child.foo")
    parent_foo(c)
end
```

As a concrete example, even though users might perform sampling using [`sample`](https://github.com/TuringLang/AbstractMCMC.jl/blob/217200e5af0583fed8e476d42186ef610e3f9ddc/src/sample.jl#L54-L62), most of the actual implementation is done in an inner function [`AbstractMCMC.mcmcsample`](https://github.com/TuringLang/AbstractMCMC.jl/blob/217200e5af0583fed8e476d42186ef610e3f9ddc/src/sample.jl#L108-L122).
This allows us to have different sampling setup behaviour depending on the arguments' types, but common shared MCMC logic.

There's also [`DynamicPPL.initialstep`](https://github.com/TuringLang/DynamicPPL.jl/blob/727da635d290c22bc978dd09febe229bb8e7c906/src/sampler.jl#L112-L131), which is similarly related to `AbstractMCMC.step`, but with an inversion: the original method (`step`) is defined on the parent class, and it calls a method (`initialstep`) that is only defined for child classes.
In other words, it looks like this:

```julia
function foo(p::Parent)
    println("calling Parent.foo")
    child_foo(p)
end

child_foo(::Child) = println("calling Child.foo")
```

One might ask why this is any worse than the Python way.
To me, it boils down to how much meaning the code carries.
`parent_foo` is just a random identifier: it could have been swapped out for any other word, and _on its own_ it doesn't tell you anything about what it does, unless you choose the name specifically like we did here.
In Python, when you see `super()`, that immediately tells you that it's trying to call a parent method, so the _intent_ of the code is clear.

_Edit_: I've now been told about a _third_ way, which is to use `invoke`:

```julia
function foo(c::Child)
    println("calling Child.foo")
    invoke(foo, (Parent,), c)
end
```

I'm not fully aware of the limitations of `invoke` ([its docstring](https://docs.julialang.org/en/v1/base/base/#Core.invoke) suggests that there are some weird edge cases), but it _does_ seem like a nice way to do this.
I think that it could be worth trying out a bit more in the Turing codebase.

## Haskell

Notice that in a functional language, say Haskell, you _have_ to use the second technique, where you define functions with different names for the parent and child:

```haskell
class Parent a where
    foo :: a -> IO ()
    foo p = do
        childFoo p
        putStrLn "calling Parent.foo"

    -- Child must implement this
    childFoo :: a -> IO ()

data Child = Child

instance Parent Child where
    childFoo c = do
        putStrLn "calling Child.foo"
```

But there is a practical difference, in that (unlike Julia) Haskell actually enforces the interface: if you try to define 

```haskell
data IllegalChild = IllegalChild

instance Parent IllegalChild where
```

without a corresponding definition of `childFoo`, it will emit a compile-time warning.
(The warning can be promoted to an compiler error, and there's talk about making this an error by default, but it hasn't happened yet, as of the time of writing.)

In contrast, in Julia you can easily write

```julia
abstract type Parent end

function foo(p::Parent)
    println("calling Parent.foo")
    child_foo(p)
end

struct IllegalChild <: Parent end
```

which will then throw a runtime error if `foo(IllegalChild())` is called.
The only real way to enforce this interface is to explicitly add a test for it, which is obviously doable, but it does mean that it's not enforced by the language itself, and is thus reliant on the programmer to 'do the right thing'.
