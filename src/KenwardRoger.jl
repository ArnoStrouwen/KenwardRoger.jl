module KenwardRoger

using ForwardDiff
using MixedModels
using Distributions
using LinearAlgebra

include("struct.jl")

export kenwardroger_matrices, kenwardroger_estimates

function kenwardroger_matrices(m::MixedModel)
    β = m.β
    y = m.y
    X = m.X
    σ2sq_gam = [m.sigmas[i][1][1]^2 for i in 1:length(m.sigmas)]
    σ2s = [m.sigma^2, σ2sq_gam...]
    Zs = [I(nobs(m)), m.reterms...]
    ZZs = [Z * Z' for Z in Zs]
    V(σ2s) = sum([σ2s[i] * ZZs[i] for i in eachindex(σ2s)])
    function modified_profile_likelihood(σ2s)
        Vinv = inv(V(σ2s))
        return -1 / 2 * logdet(V(σ2s)) - 1 / 2 * logdet(X' * Vinv * X) -
               1 / 2 * (y - X * β)' * Vinv * (y - X * β)
    end

    FIM_obs = -ForwardDiff.hessian(modified_profile_likelihood, σ2s)
    W = inv(FIM_obs)

    Vinv = inv(V(σ2s))
    P = [-X' * Vinv * ZZs[i] * Vinv * X for i in eachindex(ZZs)]
    Q = [
        X' * Vinv * ZZs[i] * Vinv * ZZs[j] * Vinv * X for i in eachindex(ZZs),
        j in eachindex(ZZs)
    ]

    factor = zeros(size(m.vcov)...)
    for i in eachindex(ZZs)
        for j in eachindex(ZZs)
            factor += W[i, j] * (Q[i, j] - P[i] * m.vcov * P[j])
        end
    end
    varcovar_adjusted = m.vcov + 2 * m.vcov * factor * m.vcov
    adjusted_error = sqrt.([varcovar_adjusted[i, i] for i in 1:size(m.vcov, 1)])

    return KenwardRogerMatrices(
        m, σ2s, V(σ2s), W, P, Q, m.vcov, varcovar_adjusted, adjusted_error
    )
end

function kenwardroger_estimates(m::MixedModel, kr::KenwardRogerMatrices)
    fixed_effects = Vector{FixedEffect}()
    varcovar = kr.CovVar
    for (ic, coef) in enumerate(coefnames(m))
        c = 1
        p = length(coefnames(m))
        C = zeros(p, c)
        C[ic, 1] = 1
        M = C * inv(C' * varcovar * C) * C'
        A1 = 0.0
        A2 = 0.0
        for i in eachindex(kr.σ2s)
            for j in eachindex(kr.σ2s)
                Pi = kr.P[i]
                Pj = kr.P[j]
                Wij = kr.W[i, j]
                A1 +=
                    Wij *
                    tr(M * varcovar * Pi * varcovar) *
                    tr(M * varcovar * Pj * varcovar)
                A2 += Wij * tr(M * varcovar * Pi * varcovar * M * varcovar * Pj * varcovar)
            end
        end

        B = (A1 + 6A2) / (2c)
        g = ((c + 1)A1 - (c + 4)A2) / ((c + 2)A2)
        c1 = g / (3c + 2(1 - g))
        c2 = (c - g) / (3c + 2(1 - g))
        c3 = (c + 2 - g) / (3c + 2(1 - g))
        Estar = inv(1 - A2 / c)
        Vstar = (2 / c) * (1 + c1 * B) / ((1 - c2 * B)^2 * (1 - c3 * B))
        ρ = Vstar / (2 * Estar^2)
        v = 4 + (c + 2) / (c * ρ - 1)
        λ = v / (Estar * (v - 2))

        t = TDist(v)
        tstar = m.beta[ic] / kr.StdBetas[ic]
        p_value = 2 * ccdf(t, abs(tstar))
        α = 0.05
        t_α = quantile(t, 1 - α / 2)
        δ = t_α * kr.StdBetas[ic]

        push!(
            fixed_effects,
            FixedEffect(
                coef,
                m.β[ic],
                kr.StdBetas[ic],
                m.β[ic] - δ,
                m.β[ic] + δ,
                λ,
                v,
                tstar,
                p_value,
            ),
        )
    end
    return fixed_effects
end

end
