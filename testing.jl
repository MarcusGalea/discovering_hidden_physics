#cd(@__DIR___)
using Pkg
Pkg.activate(".")

using DifferentialEquations
using ModelingToolkit
using DataDrivenDiffEq
using DataDrivenSparse
using DocStringExtensions
using DataDrivenConstrained


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

using Symbolics
using LinearAlgebra
using Plots

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




opt = ConstrainedSTLSQ(1e-5, constraints, hcat(Ξ))
cache = ConstrainedSTLSQcache(opt, ddprob, basis)
@show step!(cache)

solbasis = create_solution_basis(cache)
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

opt = STLSQ(exp10.(-5:0.1:-1))
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 1))
println(get_basis(ddsol))
println(get_parameter_map(get_basis(ddsol)))

using Plots

plot(
    plot(ddprob), plot(ddsol), layout = (1,2)
)
plot(ddsol)

