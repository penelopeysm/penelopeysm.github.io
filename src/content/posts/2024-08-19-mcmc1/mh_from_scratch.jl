using Plots
using StatsPlots
using Distributions
using Statistics

function ptilde(x)
    return x >= 0 ? x^2 * exp(-5x) : 0
end

function q(x, x_i; σ)
    return exp(-0.5 * (x - x_i)^2 / σ^2) / sqrt(2π * σ^2)
end

function sample_q(x_i; σ)
    return randn() * σ + x_i
end

function A(x, x_i)
    return min(1, ptilde(x) / ptilde(x_i))
end

function accept(x, x_i)
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
        # Accept or reject the proposal samples[i] = accept(x_star, x_i) ? x_star : x_i
    end

    return samples
end

function plot_samples(xs, fname)
    p1 = plot(xs, label="Samples", dpi=120)
    p2 = histogram(xs, bins=50; normed=true, label="Samples")
    plot!(p2, Gamma(3, 1/5), label="True distribution")
    p = plot(p1, p2, layout=(1, 2))
    png(p, fname)
end

samples = sample(σ=1)
@show mean(samples .^ 2)
plot_samples(samples, "normal_stdev_samples.png")

samples_small = sample(σ=0.001)
@show mean(samples_small .^ 2)
plot_samples(samples_small, "small_stdev_samples.png")

samples_big = sample(σ=0.001)
@show mean(samples_big .^ 2)
plot_samples(samples_big, "big_stdev_samples.png")
