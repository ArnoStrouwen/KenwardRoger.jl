using CSV
using DataFrames
using MixedModels
using Test
using LinearAlgebra

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
kr = adjust_KR(m)

res = DataFrame(CSV.File("Results battery cell.csv"))
@test isapprox(res[!, "Estimate"], kr.m.β, atol=1e-7, rtol=1e-7)
@test_broken isapprox(
    res[!, "Std Error"], sqrt.(diag(kr.varcovar_adjusted)), atol=1e-5, rtol=1e-5
)
@test_broken isapprox(res[!, "DFDen"], kr.v, atol=1e-5, rtol=1e-5)

kr = adjust_KR(m; FIM_σ²=:expected)

res = DataFrame(CSV.File("Results battery cell lmertest.csv"))
@test isapprox(res[!, "coefficients.Estimate"], kr.m.β, atol=1e-10, rtol=1e-7)
@test isapprox(
    res[!, "coefficients.Std..Error"],
    sqrt.(diag(kr.varcovar_adjusted)),
    atol=1e-5,
    rtol=1e-10,
)
@test isapprox(res[!, "coefficients.df"], kr.v, atol=1e-10, rtol=1e-6)
