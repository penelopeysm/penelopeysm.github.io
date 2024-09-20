---
title: MCMC... with a simplified VarInfo; part 5
publishDate: "2024-09-19"
tags: ["inference"]
---

In [Part 4 of this series](/posts/2024-09-17-mcmc4), we saw how our samplers can be generalised to accept a model that satisfies the LogDensityProblems interface.
This allowed us to abstract away the calculation of the log density into the `model` parameter, instead of hardcoding it as part of the sampler.

However, we still haven't achieved full generality, because the sampler still hardcodes information about what the parameters are called, how many of them there are, and how they are distributed.

In this post, we will abstract all of these out of our sampler so that we can have a fully generalised sampler.
The end result will conceptually be reasonably close to what Turing.jl does, except that it is heavily simplified.
At the end, we'll talk about where the remaining major differences lie.

## The existing code

Before we begin, let's set up the code again.

```julia
using AbstractMCMC
using Distributions
using Random
using Statistics: mean, std
using Turing
```

These are the data that we're using.

```julia
years = range(1970, 2017)
latlongs = [
    (51.74055, -2.40512 ), (51.75693, -2.3889  ), (51.78825, -1.53601 ),
    (51.77742, -2.24422 ), (51.77258, -2.67131 ), (51.90814, -1.44842 ),
    (51.93837, -1.12551 ), (51.84574, -1.79373 ), (51.81055, -2.06563 ),
    (52.02035, -1.18665 ), (52.07196, -0.99282 ), (51.84266, -1.87047 ),
    (52.32209,  3.271671), (52.22631,  2.1483  ), (52.17919,  0.458765),
    (52.14845,  1.091358), (52.29695,  1.681535), (52.65615,  3.164477),
    (52.13287,  1.186956), (52.19713,  0.052245), (52.85175, -1.74943 ),
    (52.98735,  0.296592), (52.68782,  3.358744), (52.67473,  4.383683),
    (52.41917,  4.093512), (52.51428,  3.582494), (52.44482,  3.515402),
    (52.47698,  3.829205), (52.46947,  4.746665), (52.40592,  4.393268),
    (52.37406,  4.570991), (52.43313,  3.32205),  (52.39657,  3.302281),
    (52.34543,  3.638763), (52.45495,  4.447601), (52.34317,  3.644572),
    (52.35402,  4.803036), (52.4026 ,  4.896817), (52.46815,  4.738608),
    (52.36552,  4.097582), (52.11678,  3.730362), (52.9878 ,  6.72993 ),
    (52.55287,  5.514886), (52.58836,  6.272805), (52.11357,  4.959362),
    (52.63933,  6.397523), (51.95124,  5.30471 ), (52.87718,  7.639933)
]
lats = [ll[1] for ll in latlongs]
longs = [ll[2] for ll in latlongs]
scaled_lats = (lats .- mean(lats)) ./ std(lats)
scaled_longs = (longs .- mean(longs)) ./ std(longs)
scaled_years = (years .- mean(years)) ./ std(years)
```

And this is the sampler code that we have so far.

```julia
# Sampler setup
struct SimpleMHSampler <: AbstractMCMC.AbstractSampler
    σ::Float64   # standard deviation of the proposal distribution
end
struct Transition
    value::Vector{Float64}
    lp::Float64   # log probability density up to additive constant
end

# Method 1: Initial step
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::SimpleMHSampler;
    kwargs...
)
    α = rand(rng, α_prior_dist)
    β = rand(rng, β_prior_dist)
    lp = LogDensityProblems.logdensity(model.logdensity, (α = α, β = β))
    transition = Transition([α, β], lp)
    return (transition, transition)
end

# Method 2: Subsequent steps
# Includes a state parameter which is passed from the previous step.
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::SimpleMHSampler,
    state::Transition;
    kwargs...
)
    # Generate a new proposal
    α, β = state.value
    α_proposal_dist = Normal(α, sampler.σ)
    β_proposal_dist = Normal(β, sampler.σ)
    α = rand(rng, α_proposal_dist)
    β = rand(rng, β_proposal_dist)
    lp = LogDensityProblems.logdensity(model.logdensity, (α = α, β = β))
    proposal = Transition([α, β], lp)
    # Determine whether to accept
    if log(rand(Float64)) < proposal.lp - state.lp
        return (proposal, proposal)
    else
        return (state, state)
    end
end
```

## The returned state

If we look at the setup above, there is one parameter that we arguably aren't using well enough.
This is the second return value of `AbstractMCMC.step`.
Recall that the first element of the returned tuple is the sample itself; and the second element is some kind of state that isn't given to the user, but is instead passed as the `state` parameter into the next call to `AbstractMCMC.step`.

In our original implementation, we needed to return this state because the next step has to generate a new proposal based on the previous step.

```julia
    α, β = state.value
```

We also stored the log probability to avoid having to recalculate it.

```julia
    if log(rand(Float64)) < proposal.lp - state.lp
```

And on top of this, we also want to store the names of the parameters, as well as their distributions.
I'll sneakily say that we would like to also store a boolean to indicate whether a parameter is observed or not.
(In our swan model, the latitudes/longitudes `D` and the years `t` are observed, but the parameters `α` and `β` are not.)
Putting it together, this gives us the following structure:

```julia
struct VarInfo
    names::Vector{Symbol}
    is_observed::Dict{Symbol, Bool}
    values::Dict{Symbol, Union{Float64,<:AbstractVector{Float64}}}
    logprobs::Dict{Symbol, Float64}
    distributions::Dict{Symbol, Function}
end
```

Note that for the parameters `α` and `β`, their distributions are static, because they just have a prior distribution of `Normal(0, 1)`.
However, the data variables `D[i]` have a distribution that depends on these values of `α` and `β`, as well as the time `t[i]` (the scaled years).
Thus, we have to store their distributions as a function that returns a distribution.

What should this function look like?
Well, its inputs are some combination of parameters, but we don't know _what_ combination of parameters they are.
Luckily, we've just defined a struct that tells us all about the parameters!
So, for this function, we can pass in the entire `VarInfo` object, and let the function extract the parameters that it needs.

For example, our priors can just look like this, and ignore the `VarInfo` object entirely.

```julia
function α_dist(vi::VarInfo)
    return Normal(0, 1)
end
function β_dist(vi::VarInfo)
    return Normal(0, 1)
end
```

For our data variable `D`, we previously wrote that `D[i] ~ Normal(α + β * t[i], 1)`.
However, let's condense this by using a multivariate distribution for the entire vector `D`.
That allows us to not need to create one variable for each element of `D`.[^turing-arrays]

```julia
function D_dist(vi::VarInfo)
    # Now we need to use `vi` to extract the values of α, β, and t
    α = vi.values[:α]
    β = vi.values[:β]
    t = vi.values[:t]
    return MvNormal(α .+ β .* t, 1)
end
```

Finally, we don't actually need a distribution for `t`, because it's not a random variable in our model.
However, we do need to keep track of it in our `VarInfo` because `D` needs its value.

Let's write a function to initialise our VarInfo object:

```julia
function setup_varinfo(t, D)
    return VarInfo(
        # names
        [:α, :β, :t, :D],
        # is_observed
        Dict(:α => false, :β => false, :t => true, :D => true),
        # values
        Dict(:α => 0.0, :β => 0.0, :t => t, :D => D),
        # logprobs -- we'll fill these in later
        Dict(),
        # distributions
        Dict(:α => α_dist, :β => β_dist, :D => D_dist)
    )
end
```

and let's make one:

```julia
lats_vi = setup_varinfo(scaled_years, scaled_lats)
```


## Using our `VarInfo`

Having gone through the effort of defining this, what do we do with it?

Clearly, one of the things we need to do is to sample from the distributions that we have.
We need to treat the observed variables `D` differently from the unobserved variables `α` and `β`, because the latter must be sampled, whereas `D` is fixed.

We'll write two different functions for each of these.

```julia
# For unobserved variables like α and β
function assume!(rng::Random.AbstractRNG, name::Symbol, vi::VarInfo; value=nothing)
    dist = vi.distributions[name](vi)
    if value === nothing
        value = rand(rng, dist)
    end
    logprob = logpdf(dist, value)
    vi.values[name], vi.logprobs[name] = value, logprob
end
```

Why do we need the `value` keyword argument here?
Well, if we don't ever pass in a value, then the value will be sampled from the distribution; but here the distribution refers to the _prior_ distribution.
In the MH algorithm, we will want to sample from the _proposal_ distribution.
We will have to let our sampler generate that value before passing it on to this function.

```julia
# For observed variables like D
function observe!(name::Symbol, vi::VarInfo)
    dist = vi.distributions[name](vi)
    value = vi.values[name]
    logprob = logpdf(dist, value)
    vi.values[name], vi.logprobs[name] = value, logprob
end
```

Notice that these functions _mutate_ the `VarInfo` object; that's why we use the `!` in the function names.
We could accomplish the same with an immutable struct, but it would be more awkward and potentially also less performant.
(Of course, the tradeoff is that it becomes harder to reason about mutable code.)

Having written these, we can now write a function to sample all of the variables:

```julia
function sample!(rng::Random.AbstractRNG, vi::VarInfo; values=nothing)
    for name in vi.names
        # This part deals with α and β
        if !vi.is_observed[name]
            value = values === nothing ? nothing : get(values, name, nothing)
            assume!(rng, name, vi; value=value)
        # This part deals with D. Note the `name in keys(vi.distributions)`
        # check -- this avoids t.
        elseif vi.is_observed[name] && name in keys(vi.distributions)
            observe!(name, vi)
        end
    end
end
```

By the way, there's a slightly insidious subtlety here, which is the order of sampling the variables.
Conveniently, we have defined `vi.names = [:α, :β, :t, :D]`, which works correctly with the `for name in vi.names` loop because we want to sample `α` and `β` first before `D`.
However, if we had defined `vi.names = [:D, :α, :β, :t]`, this could easily silently give us the wrong answer by sampling `D` based on the previous iteration's values of `α` and `β`.

Anyway, we'll assume that we carefully define the array of variable names.
Having sampled values for all of these, the other thing we can do is to calculate the log density.
Since the `assume!` and `observe!` functions update `vi.logprobs` for us, this is remarkably easy:

```julia
function logdensity(vi::VarInfo)
    return sum(values(vi.logprobs))
end
```

Let's give it a spin just to see what's happening:

```julia
sample!(Random.default_rng(), lats_vi)
lats_vi.values[:α], lats_vi.values[:β]
```

```
(0.47192922283010164, -0.22246548674416422)
```

And if we pass the values in, we want to make sure that they are fixed:

```julia
sample!(Random.default_rng(), lats_vi; values=Dict(:α => 1.5, :β => 2.5))
lats_vi.values[:α], lats_vi.values[:β]
```

```
(1.5, 2.5)
```

Great!

## Modifying the sampler

Now that we have all of this code, it will allow us to get rid of all the hardcoded parameter names in the sampler.

These structs are the same as before, with one small change: we'll update the types of the `value` field in `Transition` to match those in `VarInfo`.
This just means we don't have to faff with marshalling data between the two types.

```julia
struct SimpleMHSampler <: AbstractMCMC.AbstractSampler
    σ::Float64   # standard deviation of the proposal distribution
end

struct Transition
    value::Dict{Symbol, Union{Float64, <:AbstractVector{Float64}}}
    lp::Float64   # log probability density up to additive constant
end
```

As before, we need to implement two methods, one for the initial step and one for subsequent steps.
In the initial step, we need to construct the `VarInfo` and sample from it.
Instead of hardcoding this inside the sampler, we'll actually pass it in inside the model.
To do so, we'll have to define our own subtype of `AbstractModel` and instantiate it with a `VarInfo`.

```julia
struct SwanModelWithVarInfo <: AbstractMCMC.AbstractModel
    vi::VarInfo
end

swan_model_lats = SwanModelWithVarInfo(setup_varinfo(scaled_years, scaled_lats))
```

Now we have access to this inside the sampler.
Notice how little code is left here: most of it has been moved to the `VarInfo` functions.

```julia
# Method 1: Initial step
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::SimpleMHSampler;
    kwargs...
)
    sample!(rng, model.vi)
    transition = Transition(model.vi.values, logdensity(model.vi))
    return (transition, model.vi)
end
```

The subsequent steps are a bit more tricky because we need to sample from the proposal distribution for the unobserved variables.
This means we need to peek inside the `VarInfo` a little bit more.
Thankfully, we did encode this information in the `is_observed` field, so we can use that to our advantage.
By the way, we changed the name of the `state` parameter to `vi` here just to make the names clearer; this doesn't affect the code at all because the names of positional arguments are not used in Julia.

```julia
# Method 2: Subsequent steps
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::SimpleMHSampler,
    vi::VarInfo;
    kwargs...
)
    # Make a copy of the previous state
    prev_vi = deepcopy(vi)
    # Generate a new proposal value for each of the unobserved variables
    proposal_values = Dict()
    for name in vi.names
        if !vi.is_observed[name]
            proposal_dist = Normal(vi.values[name], sampler.σ)
            proposal_values[name] = rand(rng, proposal_dist)
        end
    end
    # Then sample with these values being fixed. This call mutates the object,
    # which is why we needed to save the previous state above.
    sample!(rng, vi; values=proposal_values)
    # Determine whether to accept
    if log(rand(Float64)) < logdensity(vi) - logdensity(prev_vi)
        successful_transition = Transition(vi.values, logdensity(vi))
        return (successful_transition, vi)
    else
        failed_transition = Transition(prev_vi.values, logdensity(prev_vi))
        return (failed_transition, prev_vi)
    end
end
```

*Et voilà:*

```julia
chain = sample(swan_model_lats, SimpleMHSampler(1.0), 1000)
mean([transition.value[:β] for transition in chain]) * std(lats) / std(years)
```

```
0.015264797714150671
```

And with longitudes:

```julia
swan_model_longs = SwanModelWithVarInfo(setup_varinfo(scaled_years, scaled_longs))
chain = sample(swan_model_longs, SimpleMHSampler(1.0), 1000)
mean([transition.value[:β] for transition in chain]) * std(longs) / std(years)
```

```
0.1918038250009541
```

## What we've achieved

Notice that now our definition of `AbstractMCMC.step` has absolutely zero reference to any of the parameter names or distributions.
The only things we absolutely require are that:

- the model must have a field called `vi` that contains a `VarInfo` object
- the `sample!` and `logdensity` methods must be defined for this `VarInfo` object appropriately

In fact, note that even the way we've defined these two methods is completely general.
So the only part which is hardcoded is the _setup_ of the `VarInfo` object, namely, the functions `α_dist`, `β_dist`, `D_dist`; and the `setup_varinfo` function.

To prove that our sampler is completely general, let's have a go at constructing a different `VarInfo` from a different model, say this one from the [Turing.jl coinflip tutorial](https://turinglang.org/docs/tutorials/00-introduction/):

```julia
# Unconditioned coinflip model with `N` observations.
@model function coinflip(y)
    # Our prior belief about the probability of heads in a coin toss.
    p ~ Beta(1, 1)

    # Heads or tails of a coin are drawn from `N` independent and identically
    # distributed Bernoulli distributions with success rate `p`.
    y ~ filldist(Bernoulli(p), length(y))
end
```

Translating this into a `VarInfo` object, and again being careful to declare the `names` array in the right order (`p` before `y`), we get:

```julia
function coinflip_vi(y)
    return VarInfo(
        # names
        [:p, :y],
        # is_observed
        Dict(:p => false, :y => true),
        # values
        Dict(:p => 0.5, :y => y),
        # logprobs -- we'll fill these in later
        Dict(),
        # distributions
        Dict(
            :p => (vi) -> Beta(1, 1),
            :y => (vi) -> filldist(Bernoulli(min(max(vi.values[:p], 0), 1)), length(vi.values[:y]))
        )
    )
end
```

We had to do one hack here, which is to ensure that the parameter passed to the Bernoulli distribution is fixed to be between 0 and 1.
If we don't do this, `Bernoulli()` will throw an error if it gets a value of `p` outside of this range, which can easily happen with the MH proposal distribution.
(I actually don't know how Turing.jl gets around this, I have to look into the source code a bit more!)

Let's setup some fake data with 80% heads and sample from this:

```julia
data = Float64.(rand(Bernoulli(0.8), 200))

struct CoinFlipModel <: AbstractMCMC.AbstractModel
    vi::VarInfo
end

cf = CoinFlipModel(coinflip_vi(data))
chain = sample(cf, SimpleMHSampler(1.0), 1000)
mean([transition.value[:p] for transition in chain])
```

```
0.8052172649299042
```

## Wait! What happened to `LogDensityProblems`?

Well, we just _don't need it any more_.
Instead of having a `LogDensityProblems` interface-compliant struct, we've changed it so that we have a struct that exposes a `VarInfo` which we can directly calculate the log density of.

In the last post, we said that there was a way to wrap Turing models into something that satisfies the `LogDensityProblems` interface.
It turns out that this function, `DynamicPPL.LogDensityFunction`, wraps it in a struct that contains a `VarInfo` object _anyway_ (see the [documentation](https://turinglang.org/DynamicPPL.jl/stable/api/#DynamicPPL.LogDensityFunction)).
It then implements the `LogDensityProblems` interface precisely by calling a function on that `VarInfo` object.

In general, what we did last time with the `LogDensityProblems` interface alone is not good enough for writing a completely general MCMC sampler, because of the issue of storing parameter names and distributions.
That's not to say that the `LogDensityProblems` interface is the problem, because you _can_ get generality while still obeying the interface (which is precisely what `DynamicPPL.LogDensityFunction` does).
It's just that the interface on its own does not provide enough power, and if you want to write something general, you have to extend it with what's essentially a `VarInfo`.

## How close are we to Turing.jl?

In terms of the level of detail, we're obviously quite far off.
Our `SimpleMHSampler` only lets you control one parameter, which is the standard deviation of the proposal distribution.
The `AdvancedMH` package, which is the implementation of the MH algorithm in the Turing framework, lets you specify completely different proposal distributions for each parameter, which may be static (i.e. it never changes, even when the parameter value changes) or dynamic.

Furthermore, the `VarInfo` struct that we have is rather simplified.
It does capture the key details (you can compare it with [the actual definition in `DynamicPPL`](https://github.com/TuringLang/DynamicPPL.jl/blob/24a73809b1a9f0f5b4e5f907f737b57c6eaf801a/src/varinfo.jl#L39-L45)), but it doesn't do quite as much as what Turing does behind the scenes, which includes (for example) separating out arrays or structs of parameters into their constituent parts.

The biggest difference, though, is that Turing.jl uses metaprogramming to generate model functions.
Actually, if we wanted to, we could probably use a macro to generate the `VarInfo` setup function, too!
We just need to turn this:

```julia
@our_model function coinflip(y)
    p ~ Beta(1, 1)
    y ~ filldist(Bernoulli(p), length(y))
end
```

into this.
For example, we can get the `is_observed` dictionary by checking whether the symbol is one of the function parameters (which are the observed variables):

```julia
struct CoinFlipModel <: AbstractMCMC.AbstractModel
    vi::VarInfo
    function CoinFlipModel(data)
        return CoinFlipModel(VarInfo(
            [:p, :y],                       # names
            Dict(:p => false, :y => true),  # is_observed
            Dict(:p => 0.5, :y => data),    # values
            Dict(),                         # logprobs
            Dict(
                :p => (vi) -> Beta(1, 1),
                :y => (vi) -> filldist(Bernoulli(min(max(vi.values[:p], 0), 1)), length(vi.values[:y]))
            )
        ))
    end
end
```

Although Turing uses similar tactics such as checking the function parameters, it also operates at a slightly different level.
Instead of generating the setup for the `VarInfo` object, it generates the entire `sample!` function, which stripped to its very core looks something like this:

```julia
function sample!(rng::Random.AbstractRNG, vi::VarInfo, y)
    p = assume!(rng, Beta(1, 1), :p, vi)
    observe!(filldist(Bernoulli(p), length(y)), y, :y, vi)
end
```

This is usually called the 'model evaluation function'.
Notice how the variable `p` is assigned to as the result of `assume!`.
This means that the next line can directly call `Bernoulli(p)`, meaning that distributions don't have to be stored as functions.
It also means that if we switch the order of these two lines, we will get an `UndefVarError` because `p` will not have been defined yet.
True enough:

```julia
@model function coinflip(y)
    # Wrong way round
    y ~ filldist(Bernoulli(p), length(y))
    p ~ Beta(1, 1)
end
sample(coinflip(data), MH(), 1000)
```

```
ERROR: UndefVarError: `p` not defined [...]
```

## What's left for this series?

In this post we managed to completely generalise our sampler to work with any model, so long as we defined the `VarInfo` struct correctly.
This is a big accomplishment! 🥳

However, we did this by constructing our own homemade `VarInfo` struct, though, which means that our samplers are still not Turing-compatible.
So, the only thing left to do is to explain what Turing's own `VarInfo` looks like and how we can use it.
Of course, it's not going to look exactly the same because of the differences described above.
We also won't discuss exactly what every function does.
But hopefully this post has given a little bit of context as to why we need certain things to be the way they are—for example, having `assume` and `observe` be different functions.

It's unlikely that I'll go into the metaprogramming aspects.
Firstly, I genuinely find them mind-boggling.
But also, there is already [some decent documentation on that](https://turinglang.org/docs/tutorials/docs-05-for-developers-compiler/), which the interested reader can check out.
So, the next post might be the last in this series... unless I am suddenly inspired to write more.


## The code

As usual, the code used in this post is [on GitHub](https://github.com/penelopeysm/penelopeysm.github.io/blob/main/src/content/posts/2024-09-19-mcmc5/mh_turing.jl).


[^turing-arrays]: This is actually _not_ how Turing.jl works, though.
Turing generates one variable for each element of `D`.
That's why if you were to run this model in Turing, the returned chain would contain one variable per element of `D`.
This also allows Turing's `VarInfo` to not use the awkward Union of `Float64` and `AbstractVector{Float64}`: it just uses `<:Real`.
