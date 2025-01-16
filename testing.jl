#cd(@__DIR___)
using Pkg
Pkg.activate(".")

using DifferentialEquations
using ModelingToolkit
using DataDrivenDiffEq
using DataDrivenSparse
using Plots


# Create a test problem
function lorenz(u, p, t)
    x, y, z = u

    ẋ = 10.0 * (y - x)
    ẏ = x * (28.0 - z) - y
    ż = x * y - (8 / 3) * z
    return [ẋ, ẏ, ż]
end

u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 20.0)
dt = 0.1
prob = ODEProblem(lorenz, u0, tspan)
sol = solve(prob, Tsit5(), saveat = dt)

## Start the automatic discovery
ddprob = DataDrivenProblem(sol)

@variables t x(t) y(t) z(t)
u = [x; y; z]
basis = Basis(polynomial_basis(u, 2), u, iv = t)

@show equations(basis)
n,m = length(basis), size(ddprob)[1]
@parameters Ξ[1:n, 1:m] # Ξ is the coefficient matrix
# ξ is vectorized version of Ξ


# ξ = vec(Ξ)
constraints = [
    Ξ[4,1] ~  10;
    Ξ[2,1] + Ξ[4,1] ~ 0
]
n_c = length(constraints)

#### EXPERIMENTAL

λ = 1e-5

solbasis = solve(ddprob, basis, opt, options = DataDrivenCommonOptions())

# Integrate the equations in solbasis

@named odesys = ODESystem(equations(solbasis), get_iv(solbasis), states(solbasis), parameters(solbasis), tspan = (0.0, 100.0))
odesys = structural_simplify(odesys)

u0map = Dict(zip(states(solbasis), u0))
pmap = get_parameter_map(solbasis)
prob = ODEProblem(odesys, u0map, tspan, pmap)
sol = solve(prob, Tsit5(), saveat = dt)
plot(sol)

model = solbasis(ddprob.X,get_parameter_values(solbasis), ddprob.t)
plot(DX'[1:400,1])
plot!(model'[1:400,1])
using CommonSolve


function rel_error(cache::ConstrainedSTLSQcache)
    return rss(cache)/norm(cache.DX)
end

function StatsBase.residuals(cache::ConstrainedSTLSQcache)
    @unpack Ξ_est, Θ, DX = cache
    return Θ*Ξ_est - DX
end

function StatsBase.rss(cache::ConstrainedSTLSQcache)
    return sum(abs2, residuals(cache))
end

function StatsBase.dof(cache::ConstrainedSTLSQcache)
    @unpack zero_entries = cache
    return length(zero_entries)
end

function create_solution_basis(cache::ConstrainedSTLSQcache)
    @unpack basis, opt, Ξ_est, zero_entries = cache
    @unpack Ξ = opt

    ξ_sym_sol = vec(hcat(Ξ))
    ξ_est = vec(Ξ_est)
    ξ_sym_sol[zero_entries] .= 0
    Ξ_sym_sol = reshape(ξ_sym_sol, size(Ξ))
    D = Differential(basis.iv)

    non_zero_entries = [entry for entry in 1:length(ξ_sym_sol) if entry ∉ zero_entries]

    parameter_values =ξ_est[non_zero_entries]
    # Create a basis with the non-zero entries and the values set to cache.Ξ_est

    p_new = map(eachindex(parameter_values)) do i
        _set_default_val(ξ_sym_sol[non_zero_entries[i]], parameter_values[i])
    end

    u = states(basis)
    solbasis = Basis(D.(u).~Ξ_sym_sol'*rhs.(equations(basis)),
                        u,
                        iv=get_iv(basis),
                        parameters = p_new,
                        controls = controls(basis),
                        #implicits,
                        name = gensym(:Basis),
                        #eval explicits
                        )
    return solbasis
end



opt = STLSQ(exp10.(-5:0.1:-1))
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 1))
println(get_basis(ddsol))
println(get_parameter_map(get_basis(ddsol)))

using Plots

plot(
    plot(ddprob), plot(ddsol), layout = (1,2)
)
plot(ddsol)