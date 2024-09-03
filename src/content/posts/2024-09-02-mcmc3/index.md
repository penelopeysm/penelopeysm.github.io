---
title: MCMC... with Turing; part 3
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

## AbstractMCMC

In the above, we used the MH sampler that is built into Turing.
In order to use our own sampler, we need to reorganise our code into a form that obeys Turing's—or specifically, AbstractMCMC's—interface, [documented here](https://turinglang.org/AbstractMCMC.jl/dev/design/).

In summary, AbstractMCMC expects you to define a new sampler type as well as associated methods that dispatch on that type.
The sampler type must be a subtype of [`AbstractMCMC.AbstractSampler`](https://turinglang.org/AbstractMCMC.jl/dev/api/#AbstractMCMC.AbstractSampler):

```julia
type MHSampler <: AbstractMCMC.AbstractSampler end
```

The key method that needs to be defined is `AbstractMCMC.step`.
In fact, two methods for this function need to be defined:

```julia
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::MHSampler;
    kwargs...
)
```

This one above is used for the first step of the chain, for which there is no previous state to begin from.
This method must therefore generate an initial state.

On the other hand, the following method is used for the second and subsequent steps, where the state at the previous step is provided as an argument.
In our case, the state that we are using is our `Transition` struct.
We'll abbreviate `log_ptilde` to `lp` to match the terminology used in the Turing source code.
In Turing, this quantity is often called the 'log probability' or 'log density'; though it's worth reiterating here that this is only true up to an additive constant.

```julia
struct Transition
    value::Float64   # x
    lp::Float64      # log(ptilde(x))
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::MHSampler;
    state,
    ...
)
```

In both cases, `rng` is a `Random.AbstractRNG` object which can be passed to control the pseudo-random number generation.
This can be passed to functions such as `rand()` when we are generating a new proposal, for example.

The `model` argument is an object that represents the model we are sampling from.
We need to use this to evaluate the value of `lp` for the new proposal.
Turing's `@model` macro does a _lot_ of behind-the-scenes magic to generate such an object.
For now, we will actually ignore the `model` argument and manually calculate the value of `lp` for the new proposal.
We will revisit this in a later post in this series.

The `sampler` argument is used as a way to specify at the type level which sampler we are using, i.e., to choose which method to dispatch to.

### Initial step


