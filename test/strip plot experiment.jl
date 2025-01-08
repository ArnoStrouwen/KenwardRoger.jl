using CSV
using DataFrames
using MixedModels
using LinearAlgebra
using ForwardDiff
using Distributions
using Test

using KenwardRoger

df = DataFrame(CSV.File("Data battery cell Chapter 11.csv"))
rename!(df, "Whole Plots" => :WP)
rename!(df, "Subplots" => :SP)

fm = @formula(
    Y ~
        1 +
        (1 | WP) +
        (1 | SP) +
        X1 +
        X2 +
        X3 +
        X4 +
        X5 +
        X6 +
        (X2 + X3 + X4 + X5 + X6) & (X1) +
        (X3 + X4 + X5 + X6) & (X2) +
        (X4 + X5 + X6) & (X3) +
        (X5 + X6) & (X4) +
        (X6) & (X5)
)
m = fit(MixedModel, fm, df; REML=true)

kr = kenwardroger_matrices(m)
estimates = kenwardroger_estimates(m, kr)

res = DataFrame(CSV.File("Results battery cell.csv"))
@test isapprox(res[!, "Estimate"], getfield.(estimates, :estimate), atol=1e-7, rtol=1e-7)
@test isapprox(res[!, "Std Error"], getfield.(estimates, :std_error), atol=1e-2, rtol=1e-2)
@test isapprox(res[!, "DFDen"], getfield.(estimates, :den_df), atol=1e-2, rtol=1e-2)
