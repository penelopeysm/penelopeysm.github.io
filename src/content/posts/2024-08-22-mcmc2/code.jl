using Distributions: MvNormal
using LinearAlgebra: I
using Statistics: mean, std

struct Transition
    value::Vector{Float64}
    log_density::Float64
end

function sample_q(x_i::Vector{Float64}; σ)
    return rand(MvNormal(x_i, σ^2 * I))
end

function accept(current::Transition, proposal::Transition)
    return log(rand(Float64)) < proposal.log_density - current.log_density
end

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
        logdensity_star = log_ptilde_func(x_star)
        proposal = Transition(x_star, logdensity_star)
        # Accept or reject the proposal
        samples[i] = accept(current, proposal) ? proposal : current
    end
    return samples
end

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



function log_ptilde_swan(θ, D)
    α, β = θ
    log_prior = -0.5 * (α^2 + β^2)
    log_likelihood = sum(-0.5 * (D_t - α - (β * t))^2 for (D_t, t) in zip(D, scaled_years))
    return log_prior + log_likelihood
end

function log_ptilde_swan_lats(θ)
    return log_ptilde_swan(θ, scaled_lats)
end

samples = sample(log_ptilde_swan_lats; σ=1, x_init=[0., 0.], N_samples=500000)
lat_alphas = [s.value[1] for s in samples]
lat_betas = [s.value[2] for s in samples]
@show mean(lat_betas) * std(lats) / std(years)

function log_ptilde_swan_longs(θ)
    return log_ptilde_swan(θ, scaled_longs)
end

samples = sample(log_ptilde_swan_longs; σ=1, x_init=[0., 0.], N_samples=500000)
long_alphas = [s.value[1] for s in samples]
long_betas = [s.value[2] for s in samples]
@show mean(long_betas) * std(longs) / std(years)
