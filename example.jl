## Example

# Setup
using Turing
using StatsBase
using DataFrames
using StatsPlots

## Data

# Lifetimes of parts from manufacturers A and StatsBase
t = [59.6, 37.4, 47.6, 40.6, 48.6, 36.3, 31.5, 31.4, 45.7, 48.9, 48.7, 59.2, 51.9]
# The first nine parts are from A (coded as 1), the last four from B (coded as 2)
m = [repeat([1] ,9); repeat([2], 4)]
# Lifetimes are positive, so we log-transform them
logt = log.(t)
# For easier interpretation and choice of priors, we standardize
zed = StatsBase.fit(ZScoreTransform, logt)
zlogt = StatsBase.transform(zed, logt)

## Model

# Define model structure
@model function gaussians(y, c, α_μ = 0, α_σ = 1, τ_k = 1, τ_θ = 1)
    # Number of intercepts 
    nc = length(unique(c))
    
    # Priors
    α ~ filldist(Normal(α_μ, α_σ), nc)
    τ ~ filldist(Gamma(τ_k, τ_θ), nc)
    
    # Observations
    # y .~ Normal.(α[c], 1 ./ .√τ[c])

    # The above works for inference, but not for predictive sampling.
    # For that to work, we need to use a loop.
    for i in eachindex(y)
        y[i] ~ Normal(α[c[i]], 1 ./ .√τ[c[i]])
    end 
end

## Prior predictive simulation

# Define the model for prior prediction
prior_predictive_design = gaussians(repeat([missing], 8_000), repeat([1], 8_000))

# Sample from the prior 
prior_predictive_sample = sample(prior_predictive_design, Prior(), 8)

# Extract simulated observations into a dataframe
prior_predictive_y = DataFrame(prior_predictive_sample)[:, r"y"]

# Index observations by category (i.e., manufacturer in our example) instead of interpretation
prior_predictive_y = permutedims(prior_predictive_y, [:y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]) 

# Summarize simulated observations
describe(prior_predictive_y)

# Go back from dataframe to matrix
prior_predictive_y = Matrix(prior_predictive_y)

# Visual sanity check
boxplot(prior_predictive_y, ylabel = "Standardized log-lifetime", xticks = (1:8, repeat([""], 8)))
violin(prior_predictive_y, ylabel = "Standardized log-lifetime", xticks = (1:8, repeat([""], 8)))
density(prior_predictive_y, ylabel = "Density", xlabel = "Standardized log-lifetime", linewidth = 2, fill = (0, 0.3))

# Stack for easier handling
prior_predictive_y_stacked = vec(prior_predictive_y)

# Map to observation space
prior_predictive_lifetime_stacked = exp.(StatsBase.reconstruct(zed, prior_predictive_y_stacked))

# Unstack
prior_predictive_lifetime = reshape(prior_predictive_lifetime_stacked, 8_000, 8)

# Visual check
boxplot(prior_predictive_lifetime, ylabel = "Lifetime", xticks = (1:8, repeat([""], 8)))
violin(prior_predictive_lifetime, ylabel = "Lifetime", xticks = (1:8, repeat([""], 8)))
density(prior_predictive_lifetime, ylabel = "Density", xlabel = "Lifetime", linewidth = 2, fill = (0, 0.3))

## Inference

# Initialize data model
data_design = gaussians(zlogt, m, 0, 1)

# Sample from the posterior
posterior_sample = sample(data_design, NUTS(), 8_000, n_chains = 4)

# Diagnostics
plot(posterior_sample, linewidth = 3, color = :darkred, fill = (0, 0.3, :darkred))


# Reconstruct original scale
mean_a = exp.(StatsBase.reconstruct(zed, posterior_sample["α[1]"]))
mean_b = exp.(StatsBase.reconstruct(zed, posterior_sample["α[2]"]))

# Probability that *mean* lifetime from B is greater
sum(mean_b - mean_a .> 0) / length(posterior_sample)
# Probability that *mean* lifetime from B is more than 3 hours greater
sum(mean_b - mean_a .> 3) / length(posterior_sample)

## Posterior predictive simulation

# Posterior predictive design
posterior_predictive_design = gaussians([missing, missing], [1, 2])

# Predict using the posterior
posterior_predictive_sample = predict(posterior_predictive_design, posterior_sample)

# Stack
posterior_predictive_sample_stacked = vec(posterior_predictive_sample.value)

# Map to observation space
posterior_predictive_lifetime_stacked = exp.(StatsBase.reconstruct(zed, posterior_predictive_sample_stacked))

# Unstack
posterior_predictive_lifetime = reshape(posterior_predictive_lifetime_stacked, (8_000, 2))

# Visual check
boxplot(posterior_predictive_lifetime, labels = ["A" "B"], ylabel = "Lifetime", xticks = ([1, 2], ["", ""]))
violin(posterior_predictive_lifetime, labels = ["A" "B"], ylabel = "Lifetime", xticks = ([1, 2], ["", ""]))
density(posterior_predictive_lifetime, labels = ["A" "B"], ylabel = "Density", xlabel = "Lifetime", linewidth = 2, fill = (0, 0.3))

## Comparison of predicted lifetimes

t_a = posterior_predictive_lifetime[:, 1]
t_b = posterior_predictive_lifetime[:, 2]

# Probability that when randomly choosing a part from each manufacturer,
# the lifetime of that from B is greater
sum(t_b - t_a .> 0) / length(posterior_predictive_sample)

# Probability that when randomly choosing a part from each manufacturer,
# the lifetime of that from B is more than 3 hours greater
sum(t_b - t_a .> 3) / length(posterior_predictive_sample)
