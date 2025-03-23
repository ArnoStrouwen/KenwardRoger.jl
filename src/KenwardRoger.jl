module KenwardRoger

using ForwardDiff
using MixedModels
using Distributions
using LinearAlgebra

using StatsAPI: StatsAPI, coeftable
using StatsBase: StatsBase, CoefTable
using Markdown

export adjust_KR
export adjust_SW

export vcov_varpar

struct LinearMixedModelKR{Float64} <: MixedModel{Float64}
    m::LinearMixedModel{Float64}
    varcovar_adjusted::Matrix{Float64}
    W::Matrix{Float64}
    P::Vector{Matrix{Float64}}
    Q::Matrix{Matrix{Float64}}
    v::Vector{Float64}
end
struct LinearMixedModelSW{Float64} <: MixedModel{Float64}
    m::LinearMixedModel{Float64}
    W::Matrix{Float64}
    v::Vector{Float64}
end

function vcov_varpar(m::MixedModel; FIM_σ²=:observed)
    β = m.β
    y = m.y
    X = m.X
    n = length(y)
    Φ = m.vcov

    nθs = [length(sigmas) for sigmas in m.sigmas]
    k(b, r, c) = (nθs[b] * (c - 1) + r - sum(1:c))
    σ²γ = [
        if r == c
            m.sigmas[b][r]^2
        else
            m.sigmarhos[b][2][k(b, r, c)] * m.sigmas[b][r] * m.sigmas[b][c]
        end for (b, r, c) in m.parmap
    ]
    σ²s = [m.sigma^2, σ²γ...]
    Zsγ = [
        if r == c
            m.reterms[b][:, r:length(m.sigmas[b]):end]
        else
            (
                m.reterms[b][:, c:length(m.sigmas[b]):end],
                m.reterms[b][:, r:length(m.sigmas[b]):end],
            )
        end for (b, r, c) in m.parmap
    ]
    Zs = [I(n), Zsγ...]
    ZZs = [Z isa Tuple ? (Z[1] * Z[2]') + (Z[2] * Z[1]') : Z * Z' for Z in Zs]
    V = sum([σ²s[i] * ZZs[i] for i in eachindex(σ²s)])
    Vinv = inv(V)
    P = [-transpose(X) * Vinv * ZZ * Vinv * X for ZZ in ZZs]
    Q = [X' * Vinv * ZZi * Vinv * ZZj * Vinv * X for ZZi in ZZs, ZZj in ZZs]
    if FIM_σ² == :observed
        Pvcov = Vinv - Vinv * X * Φ * X' * Vinv
        FIMσ² = [
            (
                1 / 2 * tr(-Pvcov * ZZs[i] * Pvcov * ZZs[j]) -
                1 / 2 *
                (y - X * β)' *
                Vinv *
                (-2 * ZZs[i] * Vinv * ZZs[j]) *
                Vinv *
                (y - X * β)
            ) for i in eachindex(σ²s), j in eachindex(σ²s)
        ]
    elseif FIM_σ² == :observed_SAS_MATCHING
        Pvcov = Vinv - Vinv * X * Φ * X' * Vinv
        FIMσ² = [
            (
                1 / 2 * tr(-Pvcov * ZZs[i] * Pvcov * ZZs[j]) -
                1 / 2 *
                (y - X * β)' *
                Vinv *
                (-2 * ZZs[i] * Pvcov * ZZs[j]) *
                Vinv *
                (y - X * β)
            ) for i in eachindex(σ²s), j in eachindex(σ²s)
        ]
    elseif FIM_σ² == :expected
        FIMσ² = [
            1 / 2 * tr(Vinv * ZZs[i] * Vinv * ZZs[j]) - tr(Φ * Q[i, j]) +
            1 / 2 * tr(Φ * P[i] * Φ * P[j]) for i in eachindex(σ²s), j in eachindex(σ²s)
        ]
    else
        error("FIM_σ² needs to equal :observed or :expected")
    end
    return W = inv(FIMσ²)
end

function adjust_KR(m::MixedModel; FIM_σ²=:observed)
    let err = nothing
        try
            loglikelihood(m)
        catch err
        end
        if err === nothing
            error("model needs to be estimated using REML")
        end
    end

    β = m.β
    p = length(β)
    y = m.y
    X = m.X
    n = length(y)
    Φ = m.vcov

    nθs = [length(sigmas) for sigmas in m.sigmas]
    k(b, r, c) = (nθs[b] * (c - 1) + r - sum(1:c))
    σ²γ = [
        if r == c
            m.sigmas[b][r]^2
        else
            m.sigmarhos[b][2][k(b, r, c)] * m.sigmas[b][r] * m.sigmas[b][c]
        end for (b, r, c) in m.parmap
    ]
    σ²s = [m.sigma^2, σ²γ...]
    Zsγ = [
        if r == c
            m.reterms[b][:, r:length(m.sigmas[b]):end]
        else
            (
                m.reterms[b][:, c:length(m.sigmas[b]):end],
                m.reterms[b][:, r:length(m.sigmas[b]):end],
            )
        end for (b, r, c) in m.parmap
    ]
    Zs = [I(n), Zsγ...]
    ZZs = [Z isa Tuple ? (Z[1] * Z[2]') + (Z[2] * Z[1]') : Z * Z' for Z in Zs]
    V = sum([σ²s[i] * ZZs[i] for i in eachindex(σ²s)])
    Vinv = inv(V)
    P = [-transpose(X) * Vinv * ZZ * Vinv * X for ZZ in ZZs]
    Q = [X' * Vinv * ZZi * Vinv * ZZj * Vinv * X for ZZi in ZZs, ZZj in ZZs]

    W = vcov_varpar(m; FIM_σ²=FIM_σ²)

    factor = zeros(size(m.vcov)...)
    for i in eachindex(ZZs)
        for j in eachindex(ZZs)
            factor += W[i, j] * (Q[i, j] - P[i] * Φ * P[j])
        end
    end
    varcovar_adjusted = Φ + 2 * Φ * factor * Φ
    error_adjusted = sqrt.(diag(varcovar_adjusted))

    v = zeros(p)
    for k in eachindex(β)
        c = 1
        C = zeros(p, c)
        C[k, 1] = 1
        M = C * inv(C' * Φ * C) * C'
        A1 = 0.0
        A2 = 0.0
        for i in eachindex(σ²s)
            for j in eachindex(σ²s)
                A1 += W[i, j] * tr(M * Φ * P[i] * Φ) * tr(M * Φ * P[j] * Φ)
                A2 += W[i, j] * tr(M * Φ * P[i] * Φ * M * Φ * P[j] * Φ)
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
        v[k] = 4 + (c + 2) / (c * ρ - 1)
        #λ = v / (Estar * (v - 2))
    end
    return LinearMixedModelKR(m, varcovar_adjusted, W, P, Q, v)
end

function StatsAPI.coeftable(m::LinearMixedModelKR)
    t = TDist.(m.v)
    error_adjusted = sqrt.(diag(m.varcovar_adjusted))
    tstar = m.m.β ./ error_adjusted
    p = 2 * ccdf.(t, abs.(tstar))
    α = 0.05
    t_α = quantile.(t, 1 - α / 2)
    δ = t_α .* error_adjusted
    lb_ci_alpha05 = m.m.β .- δ
    ub_ci_alpha05 = m.m.β .+ δ
    num_df = ones(length(m.m.β)) # IS THIS ALWAYS ONE OR LAMBDA?
    return CoefTable(
        hcat(m.m.β, error_adjusted, lb_ci_alpha05, ub_ci_alpha05, num_df, m.v, tstar, p),
        ["Coef.", "Std. Error", "LB coef.", "UB coef.", "NumDF", "DenDF", "t", "Pr(>|t|)"],
        coefnames(m.m),
        8, # pvalcol
        7, # teststatcol
    )
end

Base.show(io::IO, m::LinearMixedModelKR) = show(io, coeftable(m))

function Base.show(io::IO, ::MIME"text/latex", m::LinearMixedModelKR)
    ct = coeftable(m)
    first_row = [vcat("", ct.colnms)]
    body_rows = [
        vcat(ct.rownms[i], [string(round(col[i]; sigdigits=5)) for col in ct.cols]) for
        i in eachindex(ct.cols[1])
    ]
    rows = vcat(first_row, body_rows)
    align = [:l, :r, :r, :r, :r, :r, :r, :r, :r]
    table = Markdown.Table(rows, align)
    return print(io, Markdown.latex(table))
end
function Base.show(io::IO, ::MIME"text/html", m::LinearMixedModelKR)
    ct = coeftable(m)
    first_row = [vcat("", ct.colnms)]
    body_rows = [
        vcat(ct.rownms[i], [string(round(col[i]; sigdigits=5)) for col in ct.cols]) for
        i in eachindex(ct.cols[1])
    ]
    rows = vcat(first_row, body_rows)
    align = [:l, :r, :r, :r, :r, :r, :r, :r, :r]
    table = Markdown.Table(rows, align)
    return print(io, Markdown.html(table))
end
function Base.show(io::IO, ::MIME"text/markdown", m::LinearMixedModelKR)
    ct = coeftable(m)
    first_row = [vcat("", ct.colnms)]
    body_rows = [
        vcat(ct.rownms[i], [string(round(col[i]; sigdigits=5)) for col in ct.cols]) for
        i in eachindex(ct.cols[1])
    ]
    rows = vcat(first_row, body_rows)
    align = [:l, :r, :r, :r, :r, :r, :r, :r, :r]
    table = Markdown.Table(rows, align)
    return print(io, Markdown.MD(table))
end

function adjust_SW(m::MixedModel; FIM_σ²=:observed)
    β = m.β
    p = length(β)
    y = m.y
    X = m.X
    n = length(y)
    Φ = m.vcov

    nθs = [length(sigmas) for sigmas in m.sigmas]
    k(b, r, c) = (nθs[b] * (c - 1) + r - sum(1:c))
    σ²γ = [
        if r == c
            m.sigmas[b][r]^2
        else
            m.sigmarhos[b][2][k(b, r, c)] * m.sigmas[b][r] * m.sigmas[b][c]
        end for (b, r, c) in m.parmap
    ]
    σ²s = [m.sigma^2, σ²γ...]
    Zsγ = [
        if r == c
            m.reterms[b][:, r:length(m.sigmas[b]):end]
        else
            (
                m.reterms[b][:, c:length(m.sigmas[b]):end],
                m.reterms[b][:, r:length(m.sigmas[b]):end],
            )
        end for (b, r, c) in m.parmap
    ]
    Zs = [I(n), Zsγ...]
    ZZs = [Z isa Tuple ? (Z[1] * Z[2]') + (Z[2] * Z[1]') : Z * Z' for Z in Zs]
    V = sum([σ²s[i] * ZZs[i] for i in eachindex(σ²s)])
    Vinv = inv(V)

    W = vcov_varpar(m; FIM_σ²=FIM_σ²)
    v = zeros(p)
    for k in eachindex(β)
        c = 1
        C = zeros(p, c)
        C[k, 1] = 1
        grad = [first(C' * Φ * X' * Vinv * ZZ * Vinv * X * Φ * C) for ZZ in ZZs]
        v[k] = 2 * (first(C' * inv(X' * inv(V) * X) * C))^2 / (grad' * W * grad)
    end
    return LinearMixedModelSW(m, W, v)
end

function StatsAPI.coeftable(m::LinearMixedModelSW)
    t = TDist.(m.v)
    error = m.m.stderror
    tstar = m.m.β ./ error
    p = 2 * ccdf.(t, abs.(tstar))
    α = 0.05
    t_α = quantile.(t, 1 - α / 2)
    δ = t_α .* error
    lb_ci_alpha05 = m.m.β .- δ
    ub_ci_alpha05 = m.m.β .+ δ
    num_df = ones(length(m.m.β)) # IS THIS ALWAYS ONE OR LAMBDA?
    return CoefTable(
        hcat(m.m.β, error, lb_ci_alpha05, ub_ci_alpha05, num_df, m.v, tstar, p),
        ["Coef.", "Std. Error", "LB coef.", "UB coef.", "NumDF", "DenDF", "t", "Pr(>|t|)"],
        coefnames(m.m),
        8, # pvalcol
        7, # teststatcol
    )
end

Base.show(io::IO, m::LinearMixedModelSW) = show(io, coeftable(m))

function Base.show(io::IO, ::MIME"text/latex", m::LinearMixedModelSW)
    ct = coeftable(m)
    first_row = [vcat("", ct.colnms)]
    body_rows = [
        vcat(ct.rownms[i], [string(round(col[i]; sigdigits=5)) for col in ct.cols]) for
        i in eachindex(ct.cols[1])
    ]
    rows = vcat(first_row, body_rows)
    align = [:l, :r, :r, :r, :r, :r, :r, :r, :r]
    table = Markdown.Table(rows, align)
    return print(io, Markdown.latex(table))
end
function Base.show(io::IO, ::MIME"text/html", m::LinearMixedModelSW)
    ct = coeftable(m)
    first_row = [vcat("", ct.colnms)]
    body_rows = [
        vcat(ct.rownms[i], [string(round(col[i]; sigdigits=5)) for col in ct.cols]) for
        i in eachindex(ct.cols[1])
    ]
    rows = vcat(first_row, body_rows)
    align = [:l, :r, :r, :r, :r, :r, :r, :r, :r]
    table = Markdown.Table(rows, align)
    return print(io, Markdown.html(table))
end
function Base.show(io::IO, ::MIME"text/markdown", m::LinearMixedModelSW)
    ct = coeftable(m)
    first_row = [vcat("", ct.colnms)]
    body_rows = [
        vcat(ct.rownms[i], [string(round(col[i]; sigdigits=5)) for col in ct.cols]) for
        i in eachindex(ct.cols[1])
    ]
    rows = vcat(first_row, body_rows)
    align = [:l, :r, :r, :r, :r, :r, :r, :r, :r]
    table = Markdown.Table(rows, align)
    return print(io, Markdown.MD(table))
end
end
