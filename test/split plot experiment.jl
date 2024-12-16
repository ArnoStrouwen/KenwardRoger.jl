using CSV
using DataFrames
using MixedModels
using LinearAlgebra
using ForwardDiff
using Distributions
using Test

using KenwardRoger

df = DataFrame(CSV.File("Data Wind tunnel Chapter 10.csv"))
rename!(df, "Whole Plots" => :WP)

fm = @formula(
    EFFICIENCY ~
        1 +
        (1 | WP) +
        FRH +
        RRH +
        YA +
        GC +
        FRH & RRH +
        FRH & YA +
        FRH & GC +
        RRH & YA +
        RRH & GC +
        YA & GC +
        FRH & FRH +
        RRH & RRH +
        YA & YA +
        GC & GC
)
m = fit(MixedModel, fm, df; REML=true)

kr = kenwardroger_matrices(m)
estimates = kenwardroger_estimates(m, kr)

res = DataFrame(CSV.File("Results Wind tunnel.csv"))
@test isapprox(res[!, "Estimate"], getfield.(estimates, :estimate), atol=1e-9, rtol=1e-9)
@test isapprox(res[!, "Std Error"], getfield.(estimates, :std_error), atol=1e-6, rtol=1e-6)
@test isapprox(res[!, "DFDen"], getfield.(estimates, :den_df), atol=1e-2, rtol=1e-2)
