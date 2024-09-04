---
title: MCMC... with AbstractMCMC; part 3
publishDate: "2024-09-02"
tags: ["inference"]
---

In [the previous post in this series](/posts/2024-08-22-mcmc2), we tidied up our implementation of the Metropolis–Hastings (MH) algorithm, and applied it to an actual Bayesian inference problem (on migratory patterns of Bewick's swans).

Here, we'll redo the swan analysis using [the Turing.jl package](https://turinglang.org/), and then look at how we can tie our original MH implementation into Turing's framework.


## Turing

Turing provides what is known as a _probabilistic programming language_ (PPL).
This is a language which allows users to specify a probabilistic model in a high-level way, and perform inference on it.
It heavily draws on Julia's metaprogramming abilities to allow users to provide a model using syntax that resembles how it would be notated with mathematics.
So, the model specification isn't _really_ Julia code, but rather something that is transformed by the `@model` macro into Julia code that can be run.

Recall in our swan analysis that we had latitudes and longitudes, as well as years.
This is our input data.

```julia
using Statistics: mean, std

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

Our model assumes that:

 - the latitude (or longitude, depending on which analysis we are performing) is distributed normally with mean $\alpha + \beta t$ and standard deviation 1: $y \sim \mathcal{N}(\alpha + \beta t, 1)$.

 - the parameters $\alpha$ and $\beta$ are distributed normally with mean 0 and standard deviation 1: $\alpha, \beta \sim \mathcal{N}(0, 1)$.

In Turing's syntax, this is written as:

```julia
using Turing

@model function swan_model(t, y)
    α ~ Normal(0, 1)
    β ~ Normal(0, 1)
    for i in eachindex(y)
        y[i] ~ Normal(α + β * t[i], 1)
    end
end
```

(The `eachindex` function gives an iterator over all the indices of `y`.)
Note that the data, or observations, are passed in as arguments to the model.
This is important, because Turing needs to know which of the variables are observed and which are latent parameters that must be inferred, and it uses the function signature to determine that.
It's also smart enough to recognise that each of the individual `y[i]` are observed, even though the variable name passed in was `y`.

In a similar way, you can also have parameters that are arrays, and Turing will understand that each element of the array is a single parameter.
So, for example, we could have written $\alpha, \beta$ as a single parameter $\theta$:

```julia
@model function swan_model_alternate(t, y)
    θ ~ MvNormal([0.0, 0.0], 1.0)
    for i in eachindex(y)
        y[i] ~ Normal(θ[1] + θ[2] * t[i], 1)
    end
end
```

For now, we'll stick with the original `swan_model`.
We can perform the inference by first passing in the data as the parameters, and then using the `sample` function.
Here, `MH()` is the Metropolis–Hasting sampler that [comes with Turing](https://github.com/TuringLang/Turing.jl/blob/07cc40beb0c6caa60e945e204f0fbc88cd3d4362/src/mcmc/mh.jl), not our own.
We'll see later how we can define our own sampler that can be plugged in here.

```julia
chain = sample(swan_model(scaled_years, scaled_lats), MH(), 1000)
```

This gives us an object that looks like this.
Notice how it understands that $\alpha$ and $\beta$ are the parameters of the model.

```
Chains MCMC chain (1000×3×1 Array{Float64, 3}):

Iterations        = 1:1:1000
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 0.05 seconds
Compute duration  = 0.05 seconds
parameters        = α, β
internals         = lp

Summary Statistics
  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Float64

           α   -0.0319    0.1937    0.0578     8.1012    12.2190    1.2014      158.8470
           β    0.5859    0.1758    0.0290    27.9212    20.1204    1.0349      547.4739

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           α   -0.3081   -0.1308   -0.0870    0.0757    0.2544
           β    0.2521    0.4969    0.5568    0.7089    0.8869

```

If we call `chain[:β]`, we get an array of the samples of $\beta$ so far.
(You can index `chain` with strings too, so `chain["β"]` would work as well.)
We can then unscale this to get the same value that we did in the previous post.

```julia
mean(chain[:β]) * std(lats) / std(years)
```

```
0.014031702378438407
```

## AbstractMCMC's API

In the above, we used the MH sampler that is built into Turing.
In order to use our own sampler, we need to reorganise our code into a form that obeys Turing's—or specifically, AbstractMCMC's—interface, [documented here](https://turinglang.org/AbstractMCMC.jl/dev/design/).

In summary, AbstractMCMC expects you to define a new sampler type as well as associated methods that dispatch on that type.
The sampler type must be a subtype of [`AbstractMCMC.AbstractSampler`](https://turinglang.org/AbstractMCMC.jl/dev/api/#AbstractMCMC.AbstractSampler):

```julia
using AbstractMCMC

struct SimpleMHSampler <: AbstractMCMC.AbstractSampler
    σ::Float64
end
```

In our case, this struct will contain a single field, `σ`, which is the standard deviation of the proposal distribution.
This is the only parameter that we need to specify for the simplistic MH algorithm that we are using.
In a more sophisticated version, we could (for example) embed more information about the proposal distribution here.

**The key method that needs to be defined is `AbstractMCMC.step`.**
In fact, two separate methods for this function need to be defined.
The first of these is called to generate the initial state of the chain, and the second is called for each subsequent step.

```julia
using Random

# Method 1: Initial step
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::SimpleMHSampler;
    kwargs...
)
    # implement
end

# Method 2: Subsequent steps
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::SimpleMHSampler,
    state;
    kwargs...
)
    # implement
end
```

As can be seen, the difference between the two is a `state` parameter.
This parameter can be of any type the user chooses, and is used to pass information between successive sampling steps.

Let's go through the other function parameters to make sure their purpose is clear.
 - `rng` is a `Random.AbstractRNG` object which can be passed to control the pseudo-random number generation.
   This can be passed to functions such as `rand()` when we are generating a new proposal, for example.
 - The `model` argument is an object that represents the model we are sampling from.
   We need to use this to evaluate the value of `lp` for the new proposal.
   For now, we will actually completely ignore the `model` argument and manually calculate the value of `lp` for the new proposal.
   The magic of Turing's `@model` macro is that it will use the model definition to automatically generate the code which calculates `lp`.
   We'll revisit this in a later section.
 - For our implementation, the `sampler` argument is an object of type `SimpleMHSampler`, with a specified value of `σ`.

While the method signature above tells us about the parameters it expects, it doesn't say anything about the return type.
As it turns out, AbstractMCMC expects both new methods to return a tuple of `(sample, state)`.
Here, `sample` is the value that was sampled at this step; these samples are aggregated and eventually returned to the user when they run the MCMC sampling.
On the other hand, `state` is not (by default) returned to the user: instead, it represents some sort of persistent state that is carried through successive MCMC steps.

AbstractMCMC doesn't specify what types both `sample` and `state` need to be, so we are free to define any structure that is convenient.
In our case, we can reuse our `Transition` struct from the previous post as the _sample_.
We'll abbreviate `log_ptilde` to `lp` to match the terminology used in the Turing source code.
In Turing, this quantity is often called the 'log probability' or 'log density'; though it's worth reiterating here that this is only true up to an additive constant.


```julia
struct Transition
    value::Vector{Float64}
    lp::Float64
end
```

What about `state`?
Well, at each step of the MH algorithm, our proposal distribution is centred on the parameter values in the previous step.
Therefore, at the very minimum, the state needs to provide those parameter values.
Furthermore, we can store the value of `lp` for the previous step as well, so that we avoid recalculating it (similar considerations were discussed in the previous post).
It turns out that we can reuse the `Transition` struct for this purpose as well.

### Calculating `lp` (but with Distributions.jl)

At each step, we need to calculate the value of `lp` for each `Transition`.
We previously did this by explicitly evaluating a mathematical expression, derived from the probability distribution functions:

```julia
function log_ptilde_swan(θ, D)
    α, β = θ
    log_prior = -0.5 * (α^2 + β^2)
    log_likelihood = sum(-0.5 * (D_t - α - (β * t))^2 for (D_t, t) in zip(D, scaled_years))
    return log_prior + log_likelihood
end
```

It will be much easier, and much more generalisable, to use the Distributions.jl package to do this.
The prior distributions are

```math
\alpha, \beta \sim \mathcal{N}(0, 1)
```

If we have a particular value of $\alpha$ and $\beta$ we want to calculate the prior probability for, we can simply call `logpdf(Normal(0, 1), α)` and `logpdf(Normal(0, 1), β)` to get the probability for each.

Likewise, the likelihood for each (scaled) point $\mathcal{D}_i$ is

```math
\mathcal{D}_i \sim \mathcal{N}(\alpha + \beta t_i, 1)
```

where $t_i$ is the (scaled) year.
Putting this all together:

```julia
using Distributions

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
```


### Method 1: Initial step

For the initial step, we need to create a new `Transition`.
In the previous post, we tackled this by setting all the parameters to 0.
A more sensible option may be to instead sample from the prior distributions: we have $\alpha, \beta \sim \mathcal{N}(0, 1)$, which is easy to sample from.
For simplicity, we'll use the sampling functionality in Distributions.jl.


```julia
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::SimpleMHSampler;
    kwargs...
)
    α = rand(rng, α_prior_dist)
    β = rand(rng, β_prior_dist)
    lp = lp_swan(α, β, scaled_lats)   # or scaled_longs

    transition = Transition([α, β], lp)
    return (transition, transition)
end
```

The first `transition` is the sample that is aggregated and returned to the user, and the second `transition` is the state that is passed to the next step.

### Method 2: Subsequent steps

The subsequent steps are very similar, except that our values of `α` and `β` are now sampled from a proposal distribution rather than the prior.
The proposal distribution in turn depends on the state that is passed in.

```julia
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
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

Fundamentally, this is the least amount of work needed to implement a new sampler that is compatible with AbstractMCMC.

### What does this actually let us do?

It means we can call `sample` with our new sampler:

```julia
chain = sample(swan_model(scaled_years, scaled_lats), SimpleMHSampler(1), 1000)
```

This will return a vector with length 1000, where each element is a `Transition` (because that was what was returned as the `sample` from the steps above).
So, we can get the mean value of $\beta$ (and unscale) as follows:

```julia
mean([transition.value[2] for transition in chain]) * std(lats) / std(years)
```

```
0.015130780756628156
```

As a reminder, the original paper said $0.015 \pm 0.003$, so our sampler _is_ working!

However, it's worth pointing out that the way our sampler is written, it _only_ works for the swan model, and specifically only with latitudes.
This is because the calculation of `lp` is hardcoded inside the _sampler_ itself.
We could, in theory, pass an entirely different model and the sampler would still give us the same result as before.

```julia
@model function not_the_right_model()
    a ~ Normal(0, 1)
end

chain = sample(not_the_right_model(), SimpleMHSampler(1), 1000)
mean([transition.value[2] for transition in chain]) * std(lats) / std(years)
```

```
0.015967697749133865
```

If we wanted to use it with a different model, or even different data (such as the longitudes), we would need to rewrite the `lp_swan` function.
This is obviously not ideal; we can't rewrite a new sampler for each different model.
The solution to this lies in the `model` parameter which we have been ignoring so far.
By extracting the calculation of `lp` into something that is tied to `model`, this allows us to decouple the sampler from the model that it is sampling from.
This will be the subject of the next post.

### Other bits of AbstractMCMC

There are a few other methods in AbstractMCMC which can be overloaded to provide custom functionality.
However, AbstractMCMC provides a default implementation for these which means that it is not mandatory for downstream users to define them.
These include:

 - `AbstractMCMC.samples`: a function which takes the first sample and sets up a container (such as an array) to store that sample and the rest of them.
 - `AbstractMCMC.save!!`: a function which saves the sample generated by each step in the aforementioned container.
 - `AbstractMCMC.bundle_samples`: a function which takes the container and transforms it into a format that is more convenient for the user.

For details about the expected method signatures, please refer to the [AbstractMCMC documentation](https://turinglang.org/AbstractMCMC.jl/dev/design/).

## Code

The full code from this post is [on GitHub](https://github.com/penelopeysm/penelopeysm.github.io/blob/main/src/content/posts/2024-09-02-mcmc3/mh_turing.jl).
