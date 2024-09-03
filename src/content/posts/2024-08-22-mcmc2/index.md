---
title: MCMC... properly; part 2
publishDate: "2024-08-22"
tags: ["inference"]
---

In [the previous post in this series](/posts/2024-08-19-mcmc1), we looked at implementing the Metropolis–Hastings (MH) algorithm for sampling from a distribution.

In this post, we'll do a couple of things: firstly, we'll clean up the implementation a bit, and then we'll apply it to an actual Bayesian inference problem.

## The original implementation

Here's our original sampler from the previous post.

```julia
function ptilde(x)
    # The distribution we want to sample from: gamma distribution, but
    # unnormalised
    return x >= 0 ? x^2 * exp(-5x) : 0
end

function q(x, x_i; σ)
    # Proposal distribution
    return exp(-0.5 * (x - x_i)^2 / σ^2) / sqrt(2π * σ^2)
end

function sample_q(x_i; σ)
    # Sample from the proposal distribution
    return randn() * σ + x_i
end

function A(x, x_i)
    # Acceptance probability
    return min(1, ptilde(x) / ptilde(x_i))
end

function accept(x, x_i)
    # Accept or reject the proposal
    return rand(Float64) < A(x, x_i)
end

function sample(; σ)
    samples = zeros(Float64, 1000)
    samples[1] = 0

    for i in 2:length(samples)
        # x_i is the previous sample
        x_i = samples[i-1]
        # Sample from the proposal distribution
        x_star = sample_q(x_i; σ=σ)
        # Accept or reject the proposal
        samples[i] = accept(x_star, x_i) ? x_star : x_i
    end

    return samples
end
```

### Cleaning up: a `Transition`

Note that on each iteration of the loop, we are calculating $A(x, x_i)$ and hence $\tilde{p}(x)$ and $\tilde{p}(x_i)$.
However, we should _already_ have calculated $\tilde{p}(x_i)$ on the previous iteration.
In fact, if the step is rejected, then this value may well have been calculated many iterations ago.
We can therefore save time by storing the value of $\tilde{p}(x)$ together with the corresponding value of $x$ in a struct.

We'll call this struct a `Transition`.
In fact, it's better to store $\log[\tilde{p}(x)]$ instead of $\tilde{p}(x)$, for numerical stability.
This allows us to represent a much wider range of values without running into issues with floating-point arithmetic, and is especially important here because the probabilities involved can be _very_ small.
(Imagine a dataset $\mathcal{D}$ that contains 1000 independently obtained values; the likelihood $p(\mathcal{D} | \theta)$ is then product of 1000 probabilities, each of which are smaller than 1.)

By the way, just to be absolutely clear, here $\log$ means the natural logarithm.

```julia
struct Transition
    value::Float64         # x
    log_ptilde::Float64    # log(ptilde(x))
end
```

### Cleaning up: `accept`

We can modify the definition of `accept` to use this instead.
The acceptance probability (for our symmetric proposal distribution) was:

```math
A(x^\star | x^{(i)}) = \min\left(1,
\frac{\tilde{p}(x^\star)}{\tilde{p}(x^{(i)})}
\right),
\tag{1}
```

or taking the logarithm on both sides,,

```math
\log A(x^\star | x^{(i)}) = \min\left(0,
\log \tilde{p}(x^\star) - \log \tilde{p}(x^{(i)})
\right).
\tag{2}
```

So we can write:

```julia
function accept(current::Transition, proposal::Transition)
    return log(rand(Float64)) < min(0, proposal.log_ptilde - current.log_ptilde)
end
```

Or really, we don't need to take the `min`, because the left-hand side is always smaller than 0.
(`rand` generates a number between 0 and 1, and taking the logarithm of that always gives a negative number.)

```julia
function accept(current::Transition, proposal::Transition)
    return log(rand(Float64)) < proposal.log_ptilde - current.log_ptilde
end
```

This implementation assumes a symmetric proposal distribution, but can easily be extended to accommodate an asymmetric one by modifying the definition of `accept`.
In particular, we would need to pass `σ` as a parameter, and on the right-hand side of the inequality, we would need to add `log(q(current.value, proposal.value; σ=σ)) - log(q(proposal.value, current.value; σ=σ))`.

### Cleaning up: `sample`

We also define a new `log_ptilde` function, which directly calculates the logarithm instead of first calculating $\tilde{p}(x)$ and then taking the logarithm.
This means that we don't ever deal with true probabilities but rather just the logarithms.
Since $\tilde{p}(x) = x^2 \exp(-5x)$, we have that $\log[\tilde{p}(x)] = 2 \log(x) - 5x$.

```julia
function log_ptilde(x)
    return x >= 0 ? 2log(x) - 5x : -Inf
end
```

Finally, we just need to modify `sample` to use these updated definitions.
We'll also add extra parameters to `sample` so that we can customise its behaviour.

```julia
function sample(log_ptilde_func; σ, x_init, N_samples)
    # Initialise
    samples = Vector{Transition}(undef, N_samples)
    samples[1] = Transition(x_init, log_ptilde_func(x_init))

    for i in 2:N_samples
        # Get x_i and ptilde(x_i) from the previous sample
        current = samples[i - 1]
        # Sample from the proposal distribution and construct a new
        # Transition
        x_star = sample_q(current.value; σ=σ)
        log_ptilde_star = log_ptilde_func(x_star)
        proposal = Transition(x_star, log_ptilde_star)
        # Accept or reject the proposal
        samples[i] = accept(current, proposal) ? proposal : current
    end
    return samples
end
```

Let's test it out to make sure things are fine, by the way.

```julia
using Statistics: mean
samples = sample(log_ptilde; σ=1, x_init=0, N_samples=1000)
values = [s.value for s in samples]
mean(values .^ 2)
```

```
0.46275880759757343
```

As before, this is not _exactly_ the expected value of 0.48 (this could be improved with more samples), but the fact that we're fairly close tells us that the implementation is working as expected.


## Multiple dimensions

Before we move on to actually applying this, we should further generalise this to the case where we have more than a single parameter we are trying to estimate.

First, we need to modify the `Transition` struct to take a vector of floats.

```julia
struct Transition
    value::Vector{Float64}
    log_ptilde::Float64
end
```

In fact, ideally, we would store some kind of information telling us what each float means.
For example, we could store some kind of dictionary that maps parameter names to their values.
For simplicity, we won't do that here, though.

Next, we need to modify our `sample_q` function.
Right now it samples from a 1D normal distribution; we need to adjust it based on the number of parameters we pass in.
For simplicity, we'll keep the standard deviation in every dimension the same, i.e. the covariance matrix will be a multiple of the identity matrix.
Instead of trying to implement this ourselves, we'll just use Distributions.jl.

```julia
using Distributions: MvNormal
using LinearAlgebra: I

function sample_q(x_i::Vector{Float64}; σ)
    return rand(MvNormal(x_i, σ^2 * I))
end
```

`sample` itself doesn't need to be modified beyond what we've already done; it should work as long as we pass in an `x_init` of the right length.


## Let's do some Bayesian inference!

... on swans! 🦢
[_Bewick's swans_](https://www.wwt.org.uk/our-work/projects/bewicks-swans/) are migratory birds: they nest (over the summer) in Russia, but when winter comes round they fly thousands of kilometers in search of warmer places to hang out, such as UK and mainland Europe.
They're very closely associated with the Wildfowl and Wetlands Trust site in Slimbridge, Gloucestershire.
Here's a 2019 picture of a pair in Slimbridge.

![Bewick's swans at Slimbridge](swan3.jpg)

It turns out that over the years, Bewick's swans have started to prefer to _not_ fly quite so far.
In a 2020 paper, Nuijten *et al.* (DOI: [10.1111/gcb.15151](https:/doi.org/10.1111/gcb.15151)) provide some data for the mean latitude and longitude of swan sightings in the winter season over a period of 48 years.

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
```

First, we'll rescale our data so that the mean is 0 and the standard deviation is 1.
This is actually really important; if we don't do this, the sampler will have a very hard time exploring the parameter space.
(Recall how we struggled when we mis-set the width of the proposal distribution, $\sigma$, in the previous post.)
There are more advanced MCMC methods which can help with this, perhaps the subject of future posts, but for now this is one of the quickest and easiest ways to improve the performance of the sampler.

```julia
using Statistics: mean, std

scaled_lats = (lats .- mean(lats)) ./ std(lats)
scaled_longs = (longs .- mean(longs)) ./ std(longs)
scaled_years = (years .- mean(years)) ./ std(years)
```

Let's say we want to fit a linear regression model to these data to see how much it is changing over time.
We assume that the latitude, $x$, in each year is normally distributed, with a mean of $\alpha + \beta t$ ($t$ being the year) and a fixed standard deviation of 1.

```math
x \sim \mathcal{N}(\alpha + \beta t, 1)
\tag{3}
```

Our priors for $\alpha$ (the intercept) and $\beta$ (the slope) can just be normal distributions themselves.
Because we have scaled our data, let's say that $\alpha \sim \mathcal{N}(0, 1)$ and $\beta \sim \mathcal{N}(0, 1)$.

Recall that in Bayesian analysis, the $\tilde{p}$ function that we evaluate is the numerator here in Bayes's theorem:

```math
p(\vec{\theta} | \mathcal{D}) = \frac{p(\mathcal{D} | \vec{\theta}) p(\vec{\theta})}{p(\mathcal{D})},
\tag{4}
```

where here we have a vector of parameters, $\vec{\theta} = (\alpha, \beta)$.
We need to find expressions for our prior probability $p(\vec{\theta})$, as well as the likelihood $p(\mathcal{D} | \vec{\theta})$.

Suppose we have sampled values of $\alpha_i$ and $\beta_i$.
Then the prior probability is the product of two Gaussians:

```math
\begin{align}
p(\alpha = \alpha_i, \beta = \beta_i) &= p(\alpha = \alpha_i) \cdot p(\beta = \beta_i) \\
 &= \frac{1}{2\pi} \cdot \exp\left(-\frac{\alpha_i^2}{2}\right) \cdot \exp\left(-\frac{\beta_i^2}{2}\right),
\tag{5}
\end{align}
```

or more helpfully, the _logarithm_ of this is

```math
\log[p(\vec{\theta})] = \log\left(\frac{1}{2\pi}\right) - \frac{\alpha_i}{2} - \frac{\beta_i^2}{2}.
\tag{6}
```

To calculate the likelihood, let's look at a single data point first, and denote $\mathcal{D}_t$ as the data point for year $t$:

```math
p(\mathcal{D_t} | \alpha_i, \beta_i) = \frac{1}{\sqrt{2\pi}} \exp\left[-\frac{(\mathcal{D_t} - \alpha_i - \beta_i t)^2}{2}\right],
\tag{7}
```

or again, with logarithms,

```math
\log[p(\mathcal{D_t} | \alpha_i, \beta_i)] = \log\left(\frac{1}{\sqrt{2\pi}}\right) - \frac{(\mathcal{D_t} - \alpha_i - \beta_i t)^2}{2}.
\tag{8}
```

The probability of observing _every_ data point is just the product of these probabilities for each individual data point.
Or, equivalently... (you know the drill by now)

```math
\log[p(\mathcal{D} | \alpha_i, \beta_i)] = \sum_t \log[p(\mathcal{D_t} | \alpha_i, \beta_i)].
\tag{9}
```

By the way, recall that $\tilde{p}(x)$ need only be _proportional_ to $p(x)$.
That means that any multiplicative constants can be ignored.
Or, when we convert to logarithms, any _additive_ constants can be ignored.
This means we can just ignore all the $\log(1/\sqrt{2\pi})$ terms above!
Our `log_ptilde` function will then just be the sum of eqs. (6) and (9), sans the constants.

In this function, we pass in $\mathcal{D}$ as an argument as well so that we can run it separately for the latitude and longitude data.

```julia
function log_ptilde_swan(θ, D)
    α, β = θ
    log_prior = -0.5 * (α^2 + β^2)
    log_likelihood = sum(-0.5 * (D_t - α - (β * t))^2 for (D_t, t) in zip(D, scaled_years))
    return prior + likelihood
end
```

That's actually all we need to run our sampler!

```julia
function log_ptilde_swan_lats(θ)
    return log_ptilde_swan(θ, scaled_lats)
end

samples = sample(log_ptilde_swan_lats; σ=1, x_init=[0., 0.], N_samples=500000)
lat_alphas = [s.value[1] for s in samples]
lat_betas = [s.value[2] for s in samples]
mean(lat_alphas), mean(lat_betas)
```

```
(-0.0021677233451007747, 0.6129947641988561)
```

The intercept $\alpha$ isn't particularly interesting, but the slope $\beta$ tells us about a pattern over time.
Let's unscale the value:

```julia
mean(lat_betas) * std(lats) / std(years)
```

```
0.014679713281066304
```

The paper reported a value of $0.015 \pm 0.003$ (I don't think they mentioned how they calculated this, but my guess is that they used R's `lm` function).
So we're doing quite well!
Let's look at the longitudes too:

```julia
function log_ptilde_swan_longs(θ)
    return log_ptilde_swan(θ, scaled_longs)
end

samples = sample(log_ptilde_swan_longs; σ=1, x_init=[0., 0.], N_samples=500000)
long_alphas = [s.value[1] for s in samples]
long_betas = [s.value[2] for s in samples]
mean(long_betas) * std(longs) / std(years)
```

```
0.18752740414797528
```

The paper reported $0.192 \pm 0.013$.
Indeed, after just two posts covering MCMC we are able to use the technique to reproduce _science_! 🎉

More soberingly, there is a lot of evidence to suggest that the trends in Bewick's swan migrations are due to global warming.
The general explanation is that the swans don't need to fly quite so far south because it is comfortable enough for them to stay in, for example, Germany.
Similar patterns have been seen with other migratory birds as well.
The Guardian ran a short article on this in 2023: [Warmer winters keeping Bewick's swans away from Britain](https://www.theguardian.com/news/2023/oct/13/warmer-winters-keep-bewicks-swans-away-britain).
Although this data suggests that the swans are capable of modifying their behaviour as an adaptation to climate change, we definitely can't take it for granted.

In the [next post](/posts/2024-09-02-mcmc3), we'll look at how to implement the same model within [the Turing.jl probabilistic programming framework](https://turinglang.org/).
Turing contains a large number of sub-packages that pertain to different components of Bayesian inference.
In particular, [AbstractMCMC.jl](https://github.com/TuringLang/AbstractMCMC.jl) defines an interface for MCMC sampling, and any sampler that conforms to this can be used with Turing.
Thus, we'll also look at how we can adapt our existing sampling code to work with Turing.

## The code, in full

For convenience, the Julia code used for the swan migration analysis is [also on GitHub](https://github.com/penelopeysm/penelopeysm.github.io/blob/main/src/content/posts/2024-08-22-mcmc2/code.jl).
