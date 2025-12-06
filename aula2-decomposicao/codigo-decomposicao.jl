using Pkg
Pkg.activate(".")
Pkg.instantiate() # Baixa os pacotes do Project.toml a primeira vez que rodar

using JuMP, HiGHS, Ipopt
using Random, Distributions
using PyPlot

# Parâmetros do problema do jornaleiro
c = 25
q = 40
r = 15

function RF(x, ξ)
    m = Model(HiGHS.Optimizer)
    set_silent(m)

    @variable(m, y >= 0)
    @variable(m, z >= 0)

    @constraint(m, y + z <= x)
    @constraint(m, y <= ξ)

    @objective(m, Max, q*y + r*z)

    optimize!(m)
    return objective_value(m)
end

function V(x, dist)
    return sum(p * RF(x, ξ) for (ξ, p) in dist)
end

function funcao_valor(x, dist)
    m = Model(HiGHS.Optimizer)
    set_silent(m)

    # Parâmetro estocástico, a ser `JuMP.fix`-ada a cada cálculo.
    @variable(m, ξ) # Versão modificando a restrição: Comentar esta
    @variable(m, y >= 0)
    @variable(m, z >= 0)

    @constraint(m, y + z <= x)
    @constraint(m, y <= ξ) # Versão modificando a restrição: Substituir pela linha abaixo
    # @constraint(m, restr_demanda, y <= 0)

    @objective(m, Max, q*y + r*z)

    v = 0.0
    for (ξᵢ, pᵢ) in dist
        JuMP.fix(ξ, ξᵢ) # Versão modificando a restrição: Substituir pela linha abaixo
        # JuMP.set_normalized_rhs(m[:restr_demanda], ξᵢ)
        optimize!(m)
        v += pᵢ * objective_value(m)
    end
    return v
end

function receita_futura(xs, ξ)
    m = Model(HiGHS.Optimizer)
    set_silent(m)

    # Variável do primeiro estágio, a ser `JuMP.fix`-ada a cada cálculo.
    @variable(m, x)
    @variable(m, y >= 0)
    @variable(m, z >= 0)

    @constraint(m, y + z <= x)
    @constraint(m, y <= ξ)

    @objective(m, Max, q*y + r*z)

    rf = []
    for xᵢ in xs # x\_i<TAB> para fazer xᵢ
        JuMP.fix(x, xᵢ)
        optimize!(m)
        push!(rf, objective_value(m))
    end
    return rf
end

function receita_futura_lenta(xs, ξ)
    return [RF(x, ξ) for x in xs]
end

## Dualidade

function RF_2(x, ξ)
    m = Model(HiGHS.Optimizer)
    set_silent(m)

    @variable(m, y >= 0)
    @variable(m, z >= 0)
    @variable(m, x_chapeu)

    @constraint(m, y + z <= x_chapeu)
    @constraint(m, y <= ξ)

    @objective(m, Max, q*y + r*z)

    JuMP.fix(x_chapeu, x)
    optimize!(m)
    return objective_value(m), reduced_cost(x_chapeu)
end

function funcao_valor_2(x, dist)
    m = Model(HiGHS.Optimizer)
    set_silent(m)

    # Parâmetro estocástico, a ser `JuMP.fix`-ada a cada cálculo.
    @variable(m, ξ)
    @variable(m, x_chapeu)
    @variable(m, y >= 0)
    @variable(m, z >= 0)

    @constraint(m, y + z <= x_chapeu)
    @constraint(m, y <= ξ)

    @objective(m, Max, q*y + r*z)

    JuMP.fix(x_chapeu, x)
    v = 0.0
    πₓ = 0.0
    for (ξᵢ, pᵢ) in dist
        JuMP.fix(ξ, ξᵢ)
        optimize!(m)
        v += pᵢ * objective_value(m)
        πₓ += pᵢ * reduced_cost(x_chapeu)
    end
    return v, πₓ
end

function L_shaped(m1, x0, dist; ε=1e-4, max_iter=100, trace=true)
    set_silent(m1)

    x = m1[:x]
    @variable(m1, θ) # Variável "epigráfica" para aproximar V̂(x)

    objfun = objective_function(m1)
    @objective(m1, Max, objfun + θ)

    k = 0
    xₖ = x0
    if trace
        println(m1)
        println()
        xs = [xₖ]
        V̂s = [-Inf]
        gaps = []
    end
    for k in 1:max_iter
        v, πₓ = funcao_valor_2(xₖ, dist)
        gap = abs(V̂s[end] - v)
        if trace
            push!(gaps, gap)
        end
        if gap <= ε
            break
        end
        
        @constraint(m1, θ <= v + πₓ * (x - xₖ))
        optimize!(m1)
        xₖ = value.(x)
        if trace
            println(m1)
            println()

            push!(xs, xₖ)
            push!(V̂s, value(θ))
        end
    end
    if trace
        return xs, V̂s, gaps, k
    else
        return value(x), value(θ), k
    end
end

function jornaleiro_L_shaped(dist)
    m1 = Model(HiGHS.Optimizer)
    set_silent(m1)

    @variable(m1, 0 <= x <= 300)

    @objective(m1, Max, -c*x)

    x0 = 100.0
    xs, θx, gaps, n_iter = L_shaped(m1, x0, dist)
    return xs, θx, gaps, n_iter
end

# Aversão ao risco
function funcao_valor_cvar(x, dist, α=0.15)
    m = Model(HiGHS.Optimizer)
    set_silent(m)

    # Parâmetro estocástico, a ser `JuMP.fix`-ada a cada cálculo.
    @variable(m, ξ)
    @variable(m, y >= 0)
    @variable(m, z >= 0)

    @constraint(m, y + z <= x)
    @constraint(m, y <= ξ)

    @objective(m, Max, q*y + r*z)

    # Até aqui, igual a funcao_valor
    # Primeira diferença: armazenar todos os retornos
    rs = []
    for (ξᵢ, pᵢ) in dist
        JuMP.fix(ξ, ξᵢ)
        optimize!(m)
        push!(rs, objective_value(m))
    end

    # Cálculo do CVaR, acumulando os retornos em ordem crescente
    # até atingir o α-quantil
    perm = sortperm(rs)
    v = 0.0
    prob_acum = 0.0
    for i in perm
        pᵢ = dist[i][2]
        if prob_acum + pᵢ > α
            # Se ultrapassar o α-quantil, inclui apenas até ele
            v += (α - prob_acum) * rs[i]
            break
        end
        prob_acum += pᵢ
        v += pᵢ * rs[i]
    end
    # E faz a média dos α piores retornos
    return v / α
end
