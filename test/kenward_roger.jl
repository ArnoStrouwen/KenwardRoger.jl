using CSV
using DataFrames
using MixedModels
using LinearAlgebra
using ForwardDiff
using Distributions
using Test

using KenwardRoger
include("fixture_estimates.jl")

@testset "KenwardRoger" begin
    df = DataFrame(CSV.File("Data Pastry Dough Experiment Chapter 7.csv"))
    rename!(df, "Flow Rate" => :FR, "Moisture Content" => :MC, "Screw Speed" => :SS)
    rename!(df, "Longitudinal Expansion Index" => :LEI)
    function one_minus_one_coding!(x)
        minx = minimum(x)
        maxx = maximum(x)
        span = maxx - minx
        x .-= minx .+ span / 2
        x ./= (span / 2)
        return nothing
    end
    one_minus_one_coding!(df[!, :FR])
    one_minus_one_coding!(df[!, :MC])
    one_minus_one_coding!(df[!, :SS])

    fm = @formula(
        LEI ~
            1 +
            (1 | Day) +
            FR +
            MC +
            SS +
            FR & MC +
            FR & SS +
            MC & SS +
            FR & FR +
            MC & MC +
            SS & SS
    )
    m = fit(MixedModel, fm, df; REML=true)

    kr = kenwardroger_matrices(m)
    estimates = kenwardroger_estimates(m, kr)

    for (i, e) in enumerate(estimates)
        d_expected = Dict(
            fieldnames(FixedEffect) .=>
                getfield.(Ref(expected_estimates[i]), fieldnames(FixedEffect)),
        )
        d_obtained = Dict(
            fieldnames(FixedEffect) .=>
                getfield.(Ref(estimates[i]), fieldnames(FixedEffect)),
        )
        for (key, value) in d_obtained
            if value isa Number
                @test isapprox(value, d_expected[key], atol=1e-2)
            end
        end
    end
end
