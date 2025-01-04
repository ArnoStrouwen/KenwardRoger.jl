using CSV
using DataFrames
using MixedModels
using LinearAlgebra
using ForwardDiff
using Distributions
using Test

using KenwardRoger

df = DataFrame(MixedModels.dataset(:sleepstudy))
fm = @formula(reaction ~ 1 + days + (1 + days | subj))
m = fit(MixedModel, fm, df; REML=true)

kr = kenwardroger_matrices(m)
try
    estimates = kenwardroger_estimates(m, kr)
catch e
    println(e)
end

res = DataFrame(CSV.File("Results sleep study.csv"))
@test isapprox(res[!, "Estimate"], m.β, atol=1e-9, rtol=1e-9)
@test isapprox(
    res[!, "Std Error"],
    [sqrt(kr.CovVar[1, 1]), sqrt(kr.CovVar[2, 2])],
    atol=1e-5,
    rtol=1e-5,
)
@test isapprox(res[!, "DFDen"], getfield.(estimates, :den_df), atol=1e-2, rtol=1e-2)
