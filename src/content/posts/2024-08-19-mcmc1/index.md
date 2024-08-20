---
title: Monte Carlo... from scratch
publishDate: "2024-08-19"
tags: ["inference"]
---

Since I'm working on Bayesian analysis things, I may as well write some articles about the underlying theory to make sure I actually understand what I'm working on.

## The need for sampling

In Bayesian inference, it's a not-uncommon scenario to have a probability distribution that is analytically intractable.
Say that $\mathcal{D}$ is some data (observations) that we want to model, and our model takes a single parameter $\theta$.
Then Bayes' theorem tells us that:

```math
p(\theta | \mathcal{D}) = \frac{p(\mathcal{D} | \theta) p(\theta)}{p(\mathcal{D})}.
\tag{1}
```

Here, $p(\theta)$ is the _prior_, which we choose ourselves; and $p(\mathcal{D} | \theta)$ is the _likelihood_, which can be calculated from the model we have chosen.
For example, if $\mathcal{D}$ is just a single number and our model says that is normally distributed with mean $\theta$ and variance 1:

```math
\mathcal{D} \sim \mathcal{N}(\theta, 1).
\tag{2}
```

then the likelihood is:

```math
p(\mathcal{D} | \theta) = \frac{1}{\sqrt{2\pi}} \exp\left[-\frac{1}{2} (\mathcal{D} - \theta)^2\right].
\tag{3}
```

The problem with this usually comes with the denominator in eq. 1, i.e. $p(\mathcal{D})$, which is called the _evidence_ or _marginal likelihood_.
To calculate this, we need to integrate (or _marginalise_) over all possible values of $\theta$:

```math
p(\mathcal{D}) = \int_{-\infty}^{\infty} p(\mathcal{D} | \theta) p(\theta) \, \mathrm{d}\theta.
\tag{4}
```

This is difficult!
For the numerator, evaluating eq. 3 and then tacking on $p(\theta)$ will give us the value of $p(\mathcal{D} | \theta) p(\theta)$ for _one_ given value of $\theta$.
We can also denote this as $p(\theta, \mathcal{D})$; it is a _joint distribution_.

But for the denominator (eq. 4), we need to evaluate this for _all_ possible values of $\theta$.
Sometimes, if a nice form for the prior is chosen (a _conjugate prior_), this integral can be solved analytically.
In this case, the conjugate prior is itself a normal distribution (see e.g. Bishop, §2.3.6).
Usually, you can't.[^1]

[^1]: As someone with a quantum mechanics background, this denominator looks awfully like a normalisation constant.
Indeed, it is: it's there to make sure that the posterior distribution integrates to 1 over all space.
But can't we just ignore it?
We do that all the time in physics!
Well, the problem is that we actually _do_ want to know the exact value of the posterior distribution for each value of $\theta$.
It's like having to evaluate the electron density at a given point in space; we have to use normalised orbitals, we can't just ignore the normalisation constant.

Instead of trying to find a closed form for this integral, what we can do instead is to directly sample from the posterior distribution.
That is, we generate many samples of $\theta$ which are distributed according to $p(\theta|\mathcal{D})$.

## Sampling

How does this work, given that we don't actually know $p(\theta|\mathcal{D})$?
This is where a sampling algorithm comes in.
The role of a sampling algorithm is to generate a number of samples of some random variable $x$ that are distributed according to some target distribution $p(x)$
(In our case, this means sampling $\theta$ from the posterior.)

The use of random sampling to approximate a probability distribution is generally called a _Monte Carlo_ method.
Of course, it is impossible to reconstruct a full continuous probability distribution from a finite number of samples.
However, we are often not really interested in the shape of a full distribution, but rather some kind of expectation value.
For example, we might want to know the mean of some function $f(x)$, i.e.,

```math
E[f(x)] = \int f(x) p(x) \, \mathrm{d}x.
\tag{5}
```

In this case, if we have $N$ samples $\{x^{(1)}, x^{(2)}, \ldots, x^{(N)}\}$ drawn from the distribution $p(x)$, then we can approximate this expectation value as

```math
\frac{1}{N} \sum_{i=1}^N f(x^{(i)}).
\tag{6}
```

In other words, we apply $f$ to each sample and take the mean of the values.
One can show that this is an _unbiased_ estimator of $E[f(x)]$, in that if we draw a finite number of samples and evaluate the mean of $f(x)$, it will not necessarily be equal to $E[f(x)]$ (for example, you might be very unlucky and draw values of $x$ that are clustered around each other).
However, the expectation value of this over all possible samples is equal to $E[f(x)]$.

So, as long as we have a way of generating samples, we can obtain any expectation value we want.
Crucially, sampling methods do this without knowledge of the full distribution $p(x)$; they generate samples _only_ by evaluating the value of some other distribution that is proportional to $p(x)$, let's say $\tilde{p}(x)$, such that $\tilde{p}(x) = kp(x)$ for all $x$ and some $k$.
(In our case, $\tilde{p}(x)$ is the joint distribution, i.e. the numerator in eq. 1.)

There are a number of ways to do this (see Bishop §11 for an overview); here we focus only on **Markov chain Monte Carlo** (MCMC) methods.
A _Markov chain_ is a sequential series of samples, where the $(i+1)$-th sample $x^{(i+1)}$ is generated based on the $i$-th sample $x^{(i)}$ as well as the value of $\tilde{p}(x)$.
The defining feature of a Markov chain is that this _transition_, from $x^{(i)}$ to $x^{(i+1)}$, is independent of all samples prior to $x^{(i)}$.

## Metropolis–Hastings

The simplest MCMC method is the _Metropolis–Hastings_ algorithm, so we will specifically look at this here, and implement it from scratch in Julia.
Just to keep things very simple, let's start by saying that we want to sample from a Gaussian distribution with mean 2 and variance 4.
There are [more direct ways of sampling from a Gaussian](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform), but this is just to illustrate the principle.

As a motivating example, let's say we want to evaluate $E[x^2]$.
Since we know the variance $E[x^2] - (E[x])^2 = 4$, and the mean $E[x] = 2$, we know that $E[x^2] = 8$; this should be our target value.
Let's also implement $\tilde{p}(x)$ as the Gaussian distribution but without the normalising constant:

```math
\tilde{p}(x) = \exp\left[-(x - 2)^2\right].
\tag{7}
```

```julia
function ptilde(x)
    return exp(-(x - 2)^2)
end
```


## Not from scratch

As a sanity check, let's compare this with a library like Distributions.jl.
You could technically make this even simpler by using [`Base.randn`](https://docs.julialang.org/en/v1/stdlib/Random/#Base.randn), which Distributions [calls under the hood](https://github.com/JuliaStats/Distributions.jl/blob/13029c03fa885a2b4512b954e61f9d5a7dfa0612/src/univariate/continuous/normal.jl#L117-L119).

```julia
using Distributions

function E_x2(N_samples)
    # Note the second parameter of `Normal` is standard deviation, not variance
    x = rand(Normal(2, 2), N_samples)
    return mean(x.^2)
end

E_x2(1000000)
```

```
7.996257112863108
```

As expected, this is close to 8.
