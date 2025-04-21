using Pkg
Pkg.activate(".")
using DataDrivenConstrained

using DifferentialEquations
using ModelingToolkit
using DataDrivenDiffEq
using DataDrivenSparse
using MLJ
using Catalyst
using Plots
using DataFrames
using DataInterpolations
include("plotting.jl") # for plotting functions
export plot_interpolation, plot_data
using Statistics

# Create enzyme dinamyics model

rn = @reaction_network begin
    k1, E + S --> ES
    k2, ES --> E + S
end


u0 = [10.0, 1.0, 0.0]
tspan = (0.0, 10.0)
p = [0.2, 0.1]
prob = ODEProblem(rn, u0, tspan, p)
data = solve(prob, Tsit5(), saveat = 0.1)

plot(data, xlabel = "t", ylabel = "Concentration", title = "Enzyme Kinetics", label = ["E" "S" "ES"])


u = species(rn)
@independent_variables t
basis = Basis([polynomial_basis(u, 2);], u, iv = t)


@show equations(basis)
n,m = length(basis), length(u)
@parameters Ξ[1:n, 1:m] # Ξ is the coefficient matrix


constraints = [Ξ[i,2] + Ξ[i,3] ~ 0 for i in 1:n] # sum of all species should be conserved



#### EXPERIMENTAL

λ = 1e-2
rho = 50
@show u

X = hcat(data.u...)
#append t to X
column_names = Symbol.(["t"; string.(u)])  # Convert species names to symbols for column names
df = DataFrame(vcat(data.t', X)', :auto)  # Transpose X to match DataFrame's column-major format
rename!(df, column_names) 


interp = InterpolationMethod(CubicSpline)


### 
noise = 0.1
#random noise added to df
df[:,2:end] = df[:,2:end] .+ randn(size(df[:,2:end])) .* noise

model = MLJConstrainedSTLSQ(basis, [], rho, λ, hcat(Ξ),interp)
model_constrained = MLJConstrainedSTLSQ(basis, constraints, rho, λ, hcat(Ξ), interp)
fitted_basis,cache = fit(model, df)
fitted_basis_constrained, cache_constrained = fit(model_constrained, df)
df_sol = DataDrivenConstrained.predict(fitted_basis, df)
df_sol_constrained = DataDrivenConstrained.predict(fitted_basis_constrained, df)
println(get_parameter_map(fitted_basis))
println(fitted_basis)
est = cache.Ξ_est
println("conservation 1", est[:,1] + est[:,3]) 
println("conservation 2", est[:,2] + est[:,3])
#substitute the values of the parameters in the equations


p1 = plot_data(df, df_sol ,states(basis); ddsolconstrained = df_sol_constrained, xlabel = "t", ylabel = "Concentration")
#save figure
savefig(p1, "enzyme_kinetics_noisy.png")

# model = DataDrivenConstrained.ConstrainedSTLSQ(λ, constraints, hcat(Ξ))
# ddprob = ContinuousDataDrivenProblem(X, data.t, interp)
# cache = DataDrivenConstrained.ConstrainedSTLSQcache(model, ddprob, basis)
# @unpack opt, zero_entries, C, d, Θ, DX = cache
# @unpack λ, constraints, Ξ = opt

# # Least squares step
# A = vcat(hcat(DataDrivenConstrained.blockdiags(Θ'Θ, size(DX,2)),C'), hcat(C, zeros(length(d),length(d))))
# b = vcat(vec(Θ'*DX),d)
# ξ_est = (A\b)[1:end-length(d)]



#### use inbuilt SINDy
opt = STLSQ(λ, rho)
DX = Array(data(data.t, Val{1}))
#ddprob = ContinuousDataDrivenProblem(X, data.t, interp)
ddprob = ContinuousDataDrivenProblem(X, data.t, DX)

ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions())
println(ddsol) # hide
plot(ddsol) 
println(get_basis(ddsol)) # hide
println(get_parameter_map(get_basis(ddsol))) # hide

using MLJBase
mutable struct MLJSTLSQ <: MLJBase.Deterministic
    """ basis is a vector of basis functions to include in the model """
    basis::DataDrivenDiffEq.Basis
    """ λ is the threshold of the iteration """
    λ::Float64
    """ Ξ is the symbolic coefficient matrix """
    Ξ::AbstractMatrix
    """ interpolation is the interpolation method """
    interpolation::InterpolationMethod
end


# For interpolation do, A(t)
###INTERPOLATION
interval = 1:100
#log scale

plot_interpolation(data, interval; variable_names = hcat(["E"; "S"; "ES"]...))

#exclude first row from dataframe
#rearrance df to have t in the first column
df_sol = df_sol[:,[4,1,2,3]]

import DataDrivenConstrained.fit


