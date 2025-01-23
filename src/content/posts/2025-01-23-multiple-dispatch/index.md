---
title: Multiple dispatch as a substitute for inheritance
publishDate: "2025-01-23"
tags: []
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

In Julia, types cannot subtype other concrete types, so the equivalent of the 'parent class' must be an abstract type.

```julia
abstract type Parent end

struct Child <: Parent end

foo(::Parent) = println("calling Parent.foo")

function foo(::Child)
    println("calling Child.foo")
    # we can't call the equivalent of Parent.foo here
end
```

and doing `foo(Child())` will just print:

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

Another is to use different functions.
Because `parent_foo(::Child)` isn't defined, the method for its abstract supertype will be called instead.

```julia
parent_foo(::Parent) = println("calling Parent.foo")

function foo(c::Child)
    println("calling Child.foo")
    parent_foo(c)
end
```

This strategy is used quite extensively in Turing.jl code.
For example, even though sampling is done using `sample`, the actual work of sampling is done in an inner function `AbstractMCMC.mcmcsample`.
This allows us to have different sampling setup behaviour depending on the arguments.

One might ask why this is worse than the Python way.
To me, it boils down to how much meaning the code carries.
`parent_foo` is just a random identifier: it could have been swapped out for any other word, and _on its own_ it doesn't tell you anything about what it does, unless you choose the name specifically like we did here.
Unfortunately, this is not always the case: see the `AbstractMCMC.mcmcsample` example above.
Quite often, the parent and child functions are called `foo` and `_foo`, which is also quite unhelpful, because it doesn't tell you what the function does.

In Python, when you see `super()`, that immediately tells you that it's trying to call a parent method, so the _intent_ of the code is clear.

## Haskell

Notice that in a functional language, say Haskell, you _have_ to use the latter method:

```haskell
class Parent a where
    foo :: a -> IO ()

-- the argument seems unnecessary, but imagine that innerFoo was a function
-- which needed inputs to run correctly
parentFoo :: Parent a => a -> IO ()
parentFoo _ = putStrLn "calling Parent.foo"

data Child = Child

instance Parent Child where
    foo c = do
        putStrLn "calling Child.foo"
        parentFoo c
```

But: (1) Haskell at least doesn't pretend that it lets you reuse function names; and (2) unlike Julia, Haskell actually enforces the interface\*, in that if you try to define an `instance Parent IllegalChild` without a corresponding definition of `foo`, it will tell you.

\* Well, sort of: it emits a warning, which can be promoted to an error with `-Werror=missing-methods` if desired.
There's talk about making this an error by default, but it hasn't happened yet, as of the time of writing.
