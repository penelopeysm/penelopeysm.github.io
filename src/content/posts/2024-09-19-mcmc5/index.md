---
title: MCMC... with a simplified VarInfo; part 5
publishDate: "2024-09-19"
tags: ["inference"]
draft: true
---

In [Part 4 of this series](/posts/2024-09-17-mcmc4), we saw how our samplers can be generalised to accept a model that satisfies the LogDensityProblems interface.
This allowed us to abstract away the calculation of the log density into the `model` parameter, instead of hardcoding it as part of the sampler.

However, we still haven't achieved full generality, because the sampler still hardcodes information about what the parameters are called, how many of them there are, and how they are distributed.

In this post, we will abstract all of these out of our sampler so that we can have a fully generalised sampler.
The end result will conceptually be quite close to what Turing.jl does, except that it is heavily simplified.

## The existing code

Before we begin, let's set up the code again.

```julia
using AbstractMCMC
using Distributions
using LogDensityProblems
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
Putting it together, this gives us the following structure:

```julia
struct VarInfo
    names::Vector{Symbol}
    values::Dict{Symbol, Float64}
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

For our data variable, we previously wrote that `D[i] ~ Normal(α + β * t[i], 1)`.
However, let's condense this by using a multivariate distribution for the entire vector `D`.
