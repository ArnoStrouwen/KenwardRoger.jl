using DataFrames
using MixedModels
using Test
using KenwardRoger

df = DataFrame(MixedModels.dataset(:sleepstudy))
fm = @formula(reaction ~ 1 + days + zerocorr(1 + days | subj))
m = fit(MixedModel, fm, df)

let err = nothing
    try
        adjust_KR(m)
    catch err
    end
    @test err isa Exception
    @test sprint(showerror, err) == "model needs to be estimated using REML"
end
