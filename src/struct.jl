@kwdef struct KenwardRogerMatrices
    fit::MixedModel
    σ2s::Vector{Float64}
    V::Array{Float64,2}
    W::Array{Float64,2}
    P::Vector{Matrix{Float64}}
    Q::Matrix{Matrix{Float64}}
    CovVar::Array{Float64,2}
    CovBeta::Array{Float64,2}
    StdBetas::Array{Float64,1}
end

@kwdef struct CovMatrixVarComponents
    ids::Vector{String}
    values::Vector{String}
end

@kwdef struct FixedEffect
    id::String
    estimate::Float64
    std_error::Float64
    lb_ci_alpha05::Float64
    ub_ci_alpha05::Float64
    num_df::Float64
    den_df::Float64
    t_statistic::Float64
    p_value::Float64
end

@kwdef struct FixedEffectsCovariance
    cov_matrix::Vector{Vector{Float64}}
    ids::Vector{String}
end
