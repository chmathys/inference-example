# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,jl:percent
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Julia 1.10.3
#     language: julia
#     name: julia-1.10
# ---

# %% [markdown]
# # Bayesian Inference Example
# **Chris Mathys**
#
# This is the example I use in my introductory lecture to Bayesian inference, developed for UCL's twice-yearly [SPM Courses](https://www.fil.ion.ucl.ac.uk/spm/course/).
#
# We use [Julia](https://julialang.org/) and its probabilistic programming language [Turing.jl](https://turing.ml/).

# %%
# Setup
using Turing
using StatsBase
using DataFrames
using StatsPlots
using HypothesisTests

# %% [markdown]
# ## The data
#
# Two manufacturers, A and B, deliver the same kind of components. Your boss has bought a small sample from each and measured their lifetimes (in, say, days). Now he wants you to decide which manufucturer to buy from. He also tells you that the parts from B are more expensive, but if they live at least three days longer than those from A, buying from B ends up being cheaper. (Dataset created by simulation, adapted from an idea by Jaynes (1976)).

# %% [markdown]
# Lifetimes of parts from manufacturers A and B:
#

# %%
t = [59.6, 37.4, 47.6, 40.6, 48.6, 36.3, 31.5, 31.4, 45.7, 48.9, 48.7, 59.2, 51.9]


# %% [markdown]
# The first nine parts are from A (coded as 1), the last four from B (coded as 2).

# %%
m = [repeat([1] ,9); repeat([2], 4)]


# %% [markdown]
# There are good reasons to analyze data that can only take positive values on a logarithmic scale. For example, if we simply applied a *t*-test to these two groups of observations, the implied model would allow for negative lifetimes, which doesn't make sense. Lifetimes are positive, so we log-transform them.

# %%
logt = log.(t)


# %% [markdown]
# For easier interpretation and choice of priors, we standardize. That is, we subtract the mean of all data points from each point, so that the new mean is 0, then we divide by the standard deviation, so that the new standard deviation is 1. This is also called *Z-Scoring*. It means that the data are now on a scale that makes it easy to understand the meaning and properties of our parameters and their priors.
#
# We save the parameters of our standardization in the variable `zed` so we can later use them to transform the results back to the observation scale.

# %%
zed = StatsBase.fit(ZScoreTransform, logt)
zlogt = StatsBase.transform(zed, logt)

# %% [markdown]
# ## How *not* to analyze these data
#
# First off, let's have a look at what happens when we reflexively apply recipes from classical statistics to this dataset instead of using probability theory (i.e., Bayesian inference). This is not intended as a takedown of classical statistics in general, whose methods - when correclty applied - can be just as valid as Bayesian ones. It is merely an illustration of the dangers of blindly applying conventional recipes. I will however say that the way classical statistics has been practiced encourages such a blind application, while Bayesian approaches force the user to lay open his assumptions and think about them carefully.
#
# What classical recipe would be the most common on to use here? I would say the *t*-test. But which *t*-test? The one for equal variances or the one for unequal ones? The recipe for finding that out is the *F*-test. So let's do that.

# %%
VarianceFTest(logt[1:9], logt[10:13])

# %% [markdown]
# There is no significant difference between variances. Therefore, we use the *t*-test for equal variances.

# %%
EqualVarianceTTest(logt[1:9], logt[10:13])

# %% [markdown]
# So there is also no significant difference in the means of the two samples. Perhaps the scientific thing to do would be to go back to the boss and tell him that there is no difference in average lifetimes between manufacturers A and B, and that we should therefore just buy from A since their parts are cheaper.
#
# Can we do better, though? Yes - we can use probability theory (i.e., Bayesian inference) to answer the relevant question directly. However, in order to do that, we cannot simply apply a recipe. Instead, we need to specify a model, justify it using first principles and prior predictive simulation, determine the posterior distribution, and finally get our answer by looking at the posterior predictive distribution.

# %% [markdown]
# ## The model

# %% [markdown]
# ### Structure
#
# We choose a simple model where log-lifetimes from each manufacturer have their particular mean and standard deviation. This corresponds to the minimally constrainging assumption that manufacturers differ both in the overall quality of their parts and in the consistency of that quality. For example, a manufacturer's parts may be of high quality on average (high mean log-lifetime) but also highly variable (high standard deviation of log-lifetimes).

# %%
@model function gaussians(y, c, α_μ = 0, α_σ = 1, θ = 1)
    # Number of categories
    nc = length(unique(c))
    
    # Priors
    α ~ filldist(Normal(α_μ, α_σ), nc)
    σ ~ filldist(Exponential(θ), nc)
    
    # Observations
    # y .~ Normal.(α[c], σ[c])
    # The above works for inference, but not for predictive sampling.
    # For that to work, we need to use a loop.
    for i in eachindex(y)
        y[i] ~ Normal(α[c[i]], σ[c[i]])
    end 
end

# %% [markdown]
# We call the model `gaussians` because its parameters are the mean and standard deviation of a set of Gaussian distributions, each corresponding to a different category (manufacturers in our example).
#
# The model takes five arguments: a vector of observations (log-lifetimes in our example), a vector of category indices (1 for manufacturer A, 2 for B), and three *hyperparameters*. These are parameters which specify the priors on the parameters. We need two hyperparameters to specify the prior on the means of the categories and one to specify the prior on their standard deviations. The defaults for the hyperparameters are chosen so that they should work well with standardized data. Since our data are standardized, a good first choice is to go with these defaults.
#
# ### Justification
#
# If the following remarks are too technical, just ignore them. Note only that we didn't just choose this model for convenience. We have a substantial justification for all assumptions that went into it.
#
# We can justify our choice of a Gaussian model from first principles. The minimal assumptions we can make are that parts from a given manufacturer will on average have a certain level of quality but will also have some variation around that level. No further assumptions are warranted based on the information we have. Given this, the Gaussian is the *maximum entropy* distribution of log-lifetimes, i.e. the one introducing the least assumptions. Another legitimate choice would have been a gamma model of lifetimes directly, without log-transforming them, because the gamma is the maximum entropy distribution for positive quantities. However, the gamma distribution is less intuitive in its properties, its parameters are harder to interpret, standardization by Z-scoring doesn't work because it leads to negative values, and specifying appropriate priors is therefore much harder.
#
# For the Gaussians' standard deviation, we use an exponential prior. This only has one parameter, $\theta$, with an expectation of $1/\theta$ and a variance of $1/\theta^2$. It is an appropriate prior distribution for positive quantities about which we do not have enough information to make strong assumptions. So the only assumption we make is about their scale $\theta$.

# %% [markdown]
# # Prior predictive simulation

# %% [markdown]
# We can only really understand the implications of the assumptions in our model by simulating data from it. In particular, to understand the implications of our choice of priors, we need to simulate data from them. If the distribution of these simulated data make sense in light of everything we know, then we have an appropriate model, including appropriate priors.
#
# In our case, we test this by creating eight fictitious manufacturers characterized by parameters (mean and precision of standardized log-lifteime) drawn randomly from the model's priors. Then we randomly draw 8,000 observations (standardized log-lifetimes) from each manufacturer's distribution.
#
# We do this by first specifying the *design* for a single simulated manufacturer:

# %%
prior_predictive_design = gaussians(repeat([missing], 8_000), repeat([1], 8_000), 0, 0.5, 0.2)

# %% [markdown]
#
# A remark on terminology: what I've just called a design is conventionally called a model, as in `model1 = gaussians(...)`. However, this terminology is misleading. The name model should be reserved for the specification of how parameters and observations relate to each other, as in `@model function ... end`. Varying structures of datasets to which the model is applied should go by a different name, e.g. design. In the course of this example, we will only use a single model: `gaussians` as defined above, with a choice of hyperparameters based on prior predictive simulation. However, we will use this same model with three different designs. We have one design for prior predictive simulation, one for fitting to the data, and one for posterior predictive simulation.
#
# After defining our prior predictive design, we take eight samples corresponding to eight fictitious manufacturers:

# %%
prior_predictive_sample = sample(prior_predictive_design, Prior(), 8)

# %% [markdown]
# We now need to do a bit of manipulation on our simulated data in order to arrange the eight samples of predicted log-lifetimes as the columns of an 8000-by-8 matrix. 

# %%
# Extract simulated observations into a dataframe
prior_predictive_y = DataFrame(prior_predictive_sample)[:, r"y"]

# %%
# Index observations by category (i.e., manufacturer in our example) instead of interpretation
prior_predictive_y = permutedims(prior_predictive_y, [:y1, :y2, :y3, :y4, :y5, :y6, :y7, :y8]) 

# %%
# Summarize simulated observations
describe(prior_predictive_y)

# %%
# Go back from dataframe to matrix
prior_predictive_y = Matrix(prior_predictive_y)

# %% [markdown]
# ### Visual sanity checks
#
# We are now in a position to apply some sanity checks to the simulated data. What we want to see is a reasonable though not excessive amount of variation of stnadardized log-lifetimes within and between the different manufacturers.

# %%
boxplot(prior_predictive_y, ylabel = "Standardized log-lifetime", xticks = (1:8, repeat([""], 8)))

# %%
violin(prior_predictive_y, ylabel = "Standardized log-lifetime", xticks = (1:8, repeat([""], 8)))

# %%
density(prior_predictive_y, ylabel = "Density", xlabel = "Standardized log-lifetime", linewidth = 2, fill = (0, 0.3))

# %% [markdown]
# This all looks fine, but it's not yet on the scale we're really interested in - the observation scale. We get to that scale by applying the inverse transformations in reverse order: first, we undo the standardization to get log-lifetimes, then we undo the log-scaling by exponentiating.

# %%
# Stack for easier handling
prior_predictive_y_stacked = vec(prior_predictive_y) 

# %%
# Map to observation space
prior_predictive_lifetime_stacked = exp.(StatsBase.reconstruct(zed, prior_predictive_y_stacked))

# %%
# Unstack
prior_predictive_lifetime = reshape(prior_predictive_lifetime_stacked, 8_000, 8)

# %% [markdown]
# Now that we're on the observation scale, we can repeat our sanity checks here.

# %%
boxplot(prior_predictive_lifetime, ylabel = "Lifetime", xticks = (1:8, repeat([""], 8)))


# %%
violin(prior_predictive_lifetime, ylabel = "Lifetime", xticks = (1:8, repeat([""], 8)))


# %%
density(prior_predictive_lifetime, ylabel = "Density", xlabel = "Lifetime", linewidth = 2, fill = (0, 0.3))

# %% [markdown]
# We can see that there is a fair amount of variation of lifetimes both between and within simulated manufacturers while almost none of the simulated lifetimes is entirely outside the plausible range. So we are happy with our choice of priors. What is plausible depends entirely on our background knowledge of how the world works. The data alone cannot tell us this, and as opposed to our choice of model structure, we cannot justify our choice of hyperparameters (i.e., priors, which is what we're checking here) from first principles. All we can do is argue that our prior predictive distribution makes sense. It is the best summary of our modelling assumptions. As an exercise, you can ask yourself what the implied prior predictive distribution of a *t*-test is and whether that distribution makes sense.
#
# (Answer: a *t*-test on lifetimes implies a prior predictive distribution that assigns the same probability to a lifetime of 50, which is plausible, as to one of 50 million, which is absurd, or of -50 million, which is impossible. In other words, almost all of the *t*-test's implied prior probability mass is outside the plausible range and much of it even in an impossible region. That doesn't make sense.)
#
# As a second exercise, you can play around with the priors by choosing different hyperparameters and see how far you can go until the prior predictive distribution becomes unreasonable.

# %% [markdown]
# ## Inference
#
# ### Posterior distribution

# %% [markdown]
# Our data design consists of the model applied to the data we're analyzing: the vector of standardized log-lifetimes and the vector of manufacturer indices.

# %%
data_design = gaussians(zlogt, m, 0, 0.5, 0.2)

# %% [markdown]
# Using this design with our chosen model, we sample from the posterior distribution. This gives us estimates of the mean and precision of the standardized log-lifetimes for manufacturers A and B.

# %%
posterior_sample = sample(data_design, NUTS(), 8_000, n_chains = 4)

# %% [markdown]
# The diagnostics of these samples ("chains" in the jargon) look good. The `rhat` values are close to 1 and the effect sample sizes (`ess_bulk` and `ess_tail`) are reasonably large (chains without any autocorrelation would have a value of 8,000). (Note that for the prior predictive sample, the same diagnostics are not interpretable because they are calculated across instead of within categories - a limitation of how prior predictive simulation is implemented here).
#
# In addition to the numeric diagnostic values above, we plot our chains to get a visual impression: 

# %%
plot(posterior_sample, linewidth = 3, color = :darkred, fill = (0, 0.3, :darkred))


# %% [markdown]
# All chains look healthy.

# %% [markdown]
# ### Comparison of posterior means
#
# When the boss said it would be worth buying from manufacturer B if their parts lived at least three days longer, what did he mean, exactly? Would the *median lifetime* of parts from B have to be three days longer, or would the *predicted lifetime difference* of randomly chosen part from each manufacturer have to be at least three days? He wasn't clear about that, but these are two very different things. The first is a question about the posterior distribution, namely about the difference of the distribtuions of $\alpha[1]$ and $\alpha[2]$. The second is a question about the posterior predictive distribution, which we don't yet have. So let's just answer the first question for now.
#
# We take the samples of $\alpha[1]$ and $\alpha[2]$ and transform them back to the observations scale, where we determine the proportion of times that the sample from $\alpha[2]$ was greater than that from $\alpha[1]$. Then we determine the proportion of times where the difference was greater than three. This gives us the probability that the median lifetime of parts from B is at least three days longer than that of those from A. If this probability is what the boss was after, how high would it need to be? He should have been specific also about that.
#

# %%
# Transform to observation scale
mean_a = exp.(StatsBase.reconstruct(zed, posterior_sample["α[1]"]))
mean_b = exp.(StatsBase.reconstruct(zed, posterior_sample["α[2]"]))

# %%
# Probability that *median* lifetime from B is greater
sum(mean_b - mean_a .> 0) / length(posterior_sample)


# %%
# Probability that *median* lifetime from B is more than 3 hours greater
sum(mean_b - mean_a .> 3) / length(posterior_sample)

# %% [markdown]
# We see that this probability is around 95% *even though the t-test, which tests differences in the means of samples, said that the difference was non-significant!*. This is how far astray we can be led by dichotomizing into significant and non-significant. When the variances of the samples were not significantly different, we proceeded by pretending they were the same because those were the only alternatives we had: same or different - and according to how classical statistics is often practiced, what is not significant is taken not to exist. Then, after deciding that there was no difference in variance between the two samples, the equal-variance *t*-test also found no significant difference in means, which is again too often taken to mean that such a difference doesn't exist *even though probability theory (i.e., Bayesian inference) tells us that, under plausible assumptions, it is about 95%!*
#
# In any case, the question the *t*-test addresses is not even the most relevant. More relevant than whether the means of the posterior distributions differ is the question how much longer we can expect a random part from B to live than a random part from A. More precisely, we can ask: what is the probability that a part from B lives at least three days longer than a part from A. This can be answered by looking at the posterior predictive distribution.

# %% [markdown]
# # Posterior predictive simulation

# %% [markdown]
# We simulate standardized log-lifetimes from the posterior predictive distribution by using a design where the observations are missing and only manufacturers 1 and 2 (i.e., A and B) are represented.

# %%
posterior_predictive_design = gaussians([missing, missing], [1, 2], 0, 0.5, 0.2)

# %% [markdown]
# The `predict` function allows us to sample from the posterior predictive distribution by combining the posterior predictive design with the posterior distribution.

# %%
posterior_predictive_sample = predict(posterior_predictive_design, posterior_sample)

# %% [markdown]
# Next, we transform to the observation scale and apply visual checks:

# %%
# Stack
posterior_predictive_sample_stacked = vec(posterior_predictive_sample.value)

# %%
# Map to observation space
posterior_predictive_lifetime_stacked = exp.(StatsBase.reconstruct(zed, posterior_predictive_sample_stacked))

# %%
# Unstack
posterior_predictive_lifetime = reshape(posterior_predictive_lifetime_stacked, (8_000, 2))

# %%
boxplot(posterior_predictive_lifetime, labels = ["A" "B"], ylabel = "Lifetime", xticks = ([1, 2], ["", ""]))

# %%
violin(posterior_predictive_lifetime, labels = ["A" "B"], ylabel = "Lifetime", xticks = ([1, 2], ["", ""]))

# %%
density(posterior_predictive_lifetime, labels = ["A" "B"], ylabel = "Density", xlabel = "Lifetime", linewidth = 2, fill = (0, 0.3))

# %% [markdown]
# ### Comparison of predicted lifetimes
#
# Finally, we can compare predicted lifetimes.

# %%
t_a = posterior_predictive_lifetime[:, 1]
t_b = posterior_predictive_lifetime[:, 2]

# %%
# Probability that when randomly choosing a part from each manufacturer,
# the lifetime of that from B is greater
sum(t_b - t_a .> 0) / length(posterior_predictive_sample)

# %%
# Probability that when randomly choosing a part from each manufacturer,
# the lifetime of that from B is more than 3 hours greater
sum(t_b - t_a .> 3) / length(posterior_predictive_sample)

# %% [markdown]
# In conclusion, if a probability of about 70% that parts from B will live at least three days longer than parts from A is good enough for the boss, then he should buy from B.
