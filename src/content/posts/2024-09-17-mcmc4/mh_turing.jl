using AbstractMCMC
using Distributions
using LogDensityProblems
using Random
using Statistics: mean, std
using Turing

# Set up data
years = range(1970, 2017)
latlongs = [
    (51.74055, -2.40512),
    (51.75693, -2.3889),
    (51.78825, -1.53601),
    (51.77742, -2.24422),
    (51.77258, -2.67131),
    (51.90814, -1.44842),
    (51.93837, -1.12551),
    (51.84574, -1.79373),
    (51.81055, -2.06563),
    (52.02035, -1.18665),
    (52.07196, -0.99282),
    (51.84266, -1.87047),
    (52.32209, 3.271671),
    (52.22631, 2.1483),
    (52.17919, 0.458765),
    (52.14845, 1.091358),
    (52.29695, 1.681535),
    (52.65615, 3.164477),
    (52.13287, 1.186956),
    (52.19713, 0.052245),
    (52.85175, -1.74943),
    (52.98735, 0.296592),
    (52.68782, 3.358744),
    (52.67473, 4.383683),
    (52.41917, 4.093512),
    (52.51428, 3.582494),
    (52.44482, 3.515402),
    (52.47698, 3.829205),
    (52.46947, 4.746665),
    (52.40592, 4.393268),
    (52.37406, 4.570991),
    (52.43313, 3.32205),
    (52.39657, 3.302281),
    (52.34543, 3.638763),
    (52.45495, 4.447601),
    (52.34317, 3.644572),
    (52.35402, 4.803036),
    (52.4026, 4.896817),
    (52.46815, 4.738608),
    (52.36552, 4.097582),
    (52.11678, 3.730362),
    (52.9878, 6.72993),
    (52.55287, 5.514886),
    (52.58836, 6.272805),
    (52.11357, 4.959362),
    (52.63933, 6.397523),
    (51.95124, 5.30471),
    (52.87718, 7.639933),
]
lats = [ll[1] for ll in latlongs]
longs = [ll[2] for ll in latlongs]
scaled_lats = (lats .- mean(lats)) ./ std(lats)
scaled_longs = (longs .- mean(longs)) ./ std(longs)
scaled_years = (years .- mean(years)) ./ std(years)

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
    kwargs...,
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
    kwargs...,
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

α_prior_dist = Normal(0, 1)
β_prior_dist = Normal(0, 1)

struct SwanModel
    scaled_years::Vector{Float64}
    scaled_data::Vector{Float64}
end

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
        logpdf(Normal(α + (β * t_i), 1), D_i) for
        (D_i, t_i) in zip(problem.scaled_data, problem.scaled_years)
    )
    return lp_prior + lp_likelihood
end

chain = sample(SwanModel(scaled_years, scaled_lats), SimpleMHSampler(1), 1000)
mean([transition.value[2] for transition in chain]) * std(lats) / std(years)

chain = sample(SwanModel(scaled_years, scaled_longs), SimpleMHSampler(1), 1000)
mean([transition.value[2] for transition in chain]) * std(longs) / std(years)
