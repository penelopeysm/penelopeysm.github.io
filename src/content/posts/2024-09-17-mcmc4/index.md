---
title: MCMC... with LogDensityProblems; part 4
publishDate: "2024-09-17"
tags: ["inference"]
---

In [Part 3 of this series](/posts/2024-09-02-mcmc3), we explored how the Metropolis–Hastings (MH) algorithm can be implemented in a manner that conforms to the expected AbstractMCMC API.
However, we ran into a hitch: the calculation of the log probability density, `lp`, was hardcoded into the sampler itself.

## Passing `nothing` as a model

In the previous post, we wrote the following code, which used the `lp_swan()` function to calculate `lp`; this means that the `AbstractMCMC.step` calls didn't use the `model` parameter at all.

```julia
using Distributions
using AbstractMCMC
using Random
using Turing

# Calculation of `lp`
α_prior_dist = Normal(0, 1)
β_prior_dist = Normal(0, 1)

function lp_swan(α, β, D)
    lp_prior = logpdf(α_prior_dist, α) + logpdf(β_prior_dist, β)
    lp_likelihood = sum(
        logpdf(Normal(α + (β * t_i), 1), D_i)
        for (D_i, t_i) in zip(D, scaled_years)
    )
    return lp_prior + lp_likelihood
end

# Initial step
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    sampler::SimpleMHSampler;
    kwargs...
)
    α = rand(rng, α_prior_dist)
    β = rand(rng, β_prior_dist)
    lp = lp_swan(α, β, scaled_lats)

    transition = Transition([α, β], lp)
    return (transition, transition)
end

# Subsequent steps
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model,
    sampler::SimpleMHSampler,
    state::Transition;
    kwargs...
)
    # Generate a new transition
    α, β = state.value
    α_proposal_dist = Normal(α, sampler.σ)
    β_proposal_dist = Normal(β, sampler.σ)
    α = rand(rng, α_proposal_dist)
    β = rand(rng, β_proposal_dist)
    lp = lp_swan(α, β, scaled_lats)
    proposal = Transition([α, β], lp)

    # Decide whether to accept it
    if log(rand(Float64)) < proposal.lp - state.lp
        return (proposal, proposal)
    else
        return (state, state)
    end
end
```

Because of this, we saw that we can pass _any_ model we like to the call to `sample`, and still get the same results.

```julia
@model function not_the_right_model()
    a ~ Normal(0, 1)
end

chain = sample(not_the_right_model(), SimpleMHSampler(1.0), 1000)
mean([transition.value[2] for transition in chain]) * std(lats) / std(years)
```

```
0.015967697749133865
```

Surely, that means we can just pass _nothing_ as the model?

```julia
chain = sample(nothing, SimpleMHSampler(1.0), 1000)
```

```
ERROR: ArgumentError: the log density function does not support the LogDensityProblems.jl interface. Please implement the interface or provide a model of type `AbstractMCMC.AbstractModel`
```

## LogDensityProblems

It turns out that the `model` parameter can't just be anything; it has to be something which satisfies the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface, or a model of type `AbstractMCMC.AbstractModel`.

We'll look first at LogDensityProblems, because this is somewhat simpler.
This is another package that specifies a standard interface for calculating log densities (again, up to an additive constant).
Fundamentally, the `model` must be some object `m` for which the following methods are implemented:[^derivatives]

```julia
using LogDensityProblems

# This tells us how much of the interface `m` implements
LogDensityProblems.capabilities(::Type(m))
# This tells us the size of `m`'s inputs
LogDensityProblems.dimension(m)
# This calculates the log density for a given set of parameters `θ`, and should return a float
LogDensityProblems.logdensity(m, θ)
```

More information on this can be found in the [package's API documentation](https://www.tamaspapp.eu/LogDensityProblems.jl/dev/#log-density-api).

we need to create a specific type for our model.
Let's create a `SwanModel` struct that holds the years as well as the data (which can be either longitudes or latitudes).

```julia
struct SwanModel
    scaled_years::Vector{Float64}
    scaled_data::Vector{Float64}
end
```

Then we implement the required methods:

```julia
function LogDensityProblems.capabilities(::Type{SwanModel})
    # This means no gradient/Hessian available
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(problem::SwanModel)
    # The number of parameters in the model
    return 2
end

function LogDensityProblems.logdensity(problem::SwanModel, params)
    α, β = params.α, params.β
    lp_prior = logpdf(α_prior_dist, α) + logpdf(β_prior_dist, β)
    lp_likelihood = sum(
        logpdf(Normal(α + (β * t_i), 1), D_i)
        for (D_i, t_i) in zip(problem.scaled_data, problem.scaled_years)
    )
    return lp_prior + lp_likelihood
end
```

Notice how the implementation of `logdensity` is almost completely the same as the `lp_swan` function we had before.

## Modifying the sampler

We now need to make a small tweak to the sampler to use the new model.
Instead of calling the hardcoded `lp_swan()` function, we will instead call the `LogDensityProblems.logdensity()` method.

There is one key detail to note here, though:
when a LogDensityProblems-compliant model is passed to AbstractMCMC, it gets wrapped in an `AbstractMCMC.LogDensityModel` type.
Thus, in the calls to `logdensity()` below, we need to access the underlying model with `model.logdensity` rather than just `model`.

Note that `AbstractMCMC.LogDensityModel` is a subtype of `AbstractMCMC.AbstractModel`, so we can still use the type signature below.
(In fact, this is precisely the reason why it gets wrapped: if the `model` parameter was just 'something that obeyed the LogDensityProblems interface', we would not be able to specify a type for it.)

```julia
struct SimpleMHSampler <: AbstractMCMC.AbstractSampler
    σ::Float64   # standard deviation of the proposal distribution
end
struct Transition
    value::Vector{Float64}
    lp::Float64   # log probability density up to additive constant
end

# Initial step
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::SimpleMHSampler;
    kwargs...
)
    α = rand(rng, α_prior_dist)
    β = rand(rng, β_prior_dist)
    # Note `model.logdensity` rather than `model`
    lp = LogDensityProblems.logdensity(model.logdensity, (α = α, β = β))
    transition = Transition([α, β], lp)
    return (transition, transition)
end

# Subsequent steps
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

## Using our `SwanModel`

Let's set up the data again.

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

Then we can simply call:

```julia
chain = sample(SwanModel(scaled_years, scaled_lats), SimpleMHSampler(1.0), 1000)
mean([transition.value[2] for transition in chain]) * std(lats) / std(years)
```

```
0.015109294465923706
```

And if we want to use the longitudes instead, we can simply swap this out:

```julia
chain = sample(SwanModel(scaled_years, scaled_longs), SimpleMHSampler(1.0), 1000)
mean([transition.value[2] for transition in chain]) * std(longs) / std(years)
```

```
0.18589111754290985
```

## Turing models

Earlier, we said that the wrapped `AbstractMCMC.LogDensityModel` is a subtype of `AbstractMCMC.AbstractModel`.
What other types are there?
How about Turing's own models?

```julia
@model function demo()
    x ~ Normal(0, 1)
end
typeof(demo())
```

```
DynamicPPL.Model{typeof(demo), (), (), (), Tuple{}, Tuple{}, DynamicPPL.DefaultContext}
```

It turns out that any model that is defined with Turing is also a subtype of `AbstractMCMC.AbstractModel`.

```julia
typeof(demo()) <: AbstractMCMC.AbstractModel
```

```
true
```

If we try to run our sampler with this model, though, it will fail.
This is because our sampler expects `model.logdensity` to be something that obeys the LogDensityFunctions interface:

```julia
chain = sample(demo(), SimpleMHSampler(1.0), 1000)
```

```
ERROR: type Model has no field logdensity [...]
```

As it turns out, there is a way to wrap Turing models such that they become something that satisfies the LogDensityProblems interface.
Specifically, they need to be wrapped inside `DynamicPPL.LogDensityFunction`:

```julia
using DynamicPPL: LogDensityFunction
demo_wrapped = LogDensityFunction(demo())
LogDensityProblems.capabilities(demo_wrapped)
```

```
LogDensityProblems.LogDensityOrder{0}()
```

Let's try with that!

```julia
chain = sample(demo_wrapped, SimpleMHSampler(1.0), 1000)
```

```
ERROR: MethodError: no method matching logdensity(::LogDensityFunction{…}, ::@NamedTuple{…})
```

How anticlimactic.

## It's because of parameter names

The eagle-eyed reader will notice that, although we have made the calculation of the log density more flexible, we have still hardcoded the parameter names as well as their prior distributions.

Recall that this is our definition of the initial step:

```julia
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::SimpleMHSampler;
    kwargs...
)
    α = rand(rng, α_prior_dist)
    β = rand(rng, β_prior_dist)
    lp = LogDensityProblems.logdensity(model.logdensity, (α = α, β = β))
    # ...
end
```

It is the last line above that causes the `MethodError` seen above.
`demo_wrapped` is a model that takes no parameters, but we have hardcoded a named tuple with `α` and `β`.

To truly generalise the sampler, we need to embed this information in one of the parameters to `step`.
While we're at it, we should also avoid hardcoding the prior distributions.
Essentially, what we need is a structure that holds information about the names of the variables as well as their distributions.
We will look at this in the next post.

## Code

As always, the full code from this post is [on GitHub](https://github.com/penelopeysm/penelopeysm.github.io/blob/main/src/content/posts/2024-09-17-mcmc4/mh_turing.jl).

[^derivatives]: If your sampler makes use of gradients or Hessians, then there are also other methods to implement, but this is the minimal interface.
