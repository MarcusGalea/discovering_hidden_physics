using Pkg
Pkg.activate(".")
using DataDrivenConstrained

using DifferentialEquations
using ModelingToolkit
using DataDrivenDiffEq
using DataDrivenSparse
using DocStringExtensions


# Create a test problem
function lorenz(u, p, t)
    x, y, z = u

    ẋ = 10.0 * (y - x)
    ẏ = x * (28.0 - z) - y
    ż = x * y - (8 / 3) * z
    return [ẋ, ẏ, ż]
end




u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
dt = 0.1
prob = ODEProblem(lorenz, u0, tspan)
sol = solve(prob, Tsit5(), saveat = dt)

## Start the automatic discovery
ddprob = DataDrivenProblem(sol, use_interpolations = true)
X = hcat(sol.u...) 
ddprob2 = ContinuousDataDrivenProblem(X, sol.t, InterpolationMethod()) 


@independent_variables t
@variables x(t) y(t) z(t)
u = [x; y; z]
basis = Basis([polynomial_basis(u, 2); cos.(u); sin.(u)], u, iv = t)

@show equations(basis)
n,m = length(basis), length(u)
@parameters Ξ[1:n, 1:m] # Ξ is the coefficient matrix


constraints = [
    Ξ[4,1] ~  10;
    Ξ[2,1] + Ξ[4,1] ~ 0
]
#### EXPERIMENTAL

λ = 1e-5



opt = DataDrivenConstrained.ConstrainedSTLSQ(λ, Equation[], hcat(Ξ))
solbasis = solve(ddprob, basis, opt, options = DataDrivenCommonOptions())
get_parameter_map(solbasis)

odesys = ODESystem(solbasis, tspan)
odesys = structural_simplify(odesys)

u0map = Dict(zip(states(solbasis), u0))
pmap = get_parameter_map(solbasis)
basisprob = ODEProblem(odesys, u0map, tspan, pmap)
ddsol = solve(basisprob, Tsit5(), saveat = dt)

optconstrained = DataDrivenConstrained.ConstrainedSTLSQ(λ, constraints, hcat(Ξ))
solbasisconstrained = solve(ddprob, basis, optconstrained, options = DataDrivenCommonOptions())
get_parameter_map(solbasisconstrained)

odesysconstrained = ODESystem(solbasisconstrained, tspan)
odesysconstrained = structural_simplify(odesysconstrained)

u0map = Dict(zip(states(solbasisconstrained), u0))
pmapconstrained = get_parameter_map(solbasisconstrained)
basisprob = ODEProblem(odesys, u0map, tspan, pmapconstrained)
ddsolconstrained = solve(basisprob, Tsit5(), saveat = dt)


using Plots

# Plot data
colors = [:black, :red, :green]
p = plot(layout = (3, 1), size = (800, 600))

# Plot each variable in its own subfigure with thicker lines
linewidth = 2
opacity1 = 1.0
t_interval = collect(200:1:400)
plot!(p, sol.t[t_interval], sol[1,t_interval], label = string(u[1]), color = colors[1], subplot = 1, linewidth = linewidth, alpha = opacity1)
plot!(p, sol.t[t_interval], sol[2,t_interval], label = string(u[2]), color = colors[1], subplot = 2, linewidth = linewidth, alpha = opacity1)
plot!(p, sol.t[t_interval], sol[3,t_interval], label = string(u[3]), color = colors[1], subplot = 3, linewidth = linewidth, alpha = opacity1)

opacity2 = 0.5
plot!(p, ddsol.t[t_interval], ddsol[1,t_interval], label = string(u[1])*"_est", linestyle = :dash, color = colors[2], subplot = 1, linewidth = linewidth, alpha = opacity2)
plot!(p, ddsol.t[t_interval], ddsol[2,t_interval], label = string(u[2])*"_est", linestyle = :dash, color = colors[2], subplot = 2, linewidth = linewidth, alpha = opacity2)
plot!(p, ddsol.t[t_interval], ddsol[3,t_interval], label = string(u[3])*"_est", linestyle = :dash, color = colors[2], subplot = 3, linewidth = linewidth, alpha = opacity2)

plot!(p, ddsolconstrained.t[t_interval], ddsolconstrained[1,t_interval], label = string(u[1])*"_constrained", linestyle = :dot, color = colors[3], subplot = 1, linewidth = linewidth, alpha = opacity2)
plot!(p, ddsolconstrained.t[t_interval], ddsolconstrained[2,t_interval], label = string(u[2])*"_constrained", linestyle = :dot, color = colors[3], subplot = 2, linewidth = linewidth, alpha = opacity2)
plot!(p, ddsolconstrained.t[t_interval], ddsolconstrained[3,t_interval], label = string(u[3])*"_constrained", linestyle = :dot, color = colors[3], subplot = 3, linewidth = linewidth, alpha = opacity2)

display(p)


### MLJ regression
using MLJBase

mutable struct MLJConstrainedSTLSQ <: MLJBase.Deterministic
    """ basis is a vector of basis functions to include in the model """
    basis::Basis
    """ constraints is a list of constraints """
    constraints::Vector{Equation}
    """ λ is the threshold of the iteration """
    λ::Float64
    """ Ξ is the symbolic coefficient matrix """
    Ξ::AbstractMatrix
    """ interpolation is the interpolation method """
    interpolation::InterpolationMethod
end


Xt = vcat(sol.t', X)

function fit(model::MLJConstrainedSTLSQ, Xt)
    X = Xt[2:end,:]
    t = Xt[1,:]
    @unpack interpolation = model
    ddprob = ContinuousDataDrivenProblem(X, t, InterpolationMethod())
    opt = DataDrivenConstrained.ConstrainedSTLSQ(model.λ, model.constraints, model.Ξ)
    solbasis = solve(ddprob, model.basis, opt, options = DataDrivenCommonOptions())
    return solbasis
end

function predict(model::MLJConstrainedSTLSQ, solbasis, Xt)
    X = Xt[2:end,:]
    t = Xt[1,:]
    pred = solbasis(X,[],t)
    return pred
end

# model = MLJConstrainedSTLSQ(basis, constraints, λ, Ξ, InterpolationMethod())
@unpack constraints, Ξ = opt
@unpack X, DX,t = ddprob
get_parameter_values(solbasis)
Θ = basis(X,get_parameter_values(solbasis), t)'