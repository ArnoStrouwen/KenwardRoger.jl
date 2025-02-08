using Test
using SafeTestsets

@testset "KenwardRoger.jl" begin
    @safetestset "Code quality (Aqua.jl)" begin
        using Aqua
        using KenwardRoger
        Aqua.test_all(KenwardRoger)
    end
    @safetestset "Code linting (JET.jl)" begin
        using JET
        using KenwardRoger
        JET.test_package(KenwardRoger; target_defined_modules=true)
    end
    @safetestset "blocked experiment" begin
        include("blocked experiment.jl")
    end
    @safetestset "split plot experiment" begin
        include("split plot experiment.jl")
    end
    @safetestset "random slope" begin
        include("random_slope.jl")
    end
end
