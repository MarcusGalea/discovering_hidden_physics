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

# Create enzyme dinamyics model

rn = @reaction_network begin
    k1, E + S --> ES
    k2, ES --> E + S
    k3, ES --> E + P
end


u0 = [10.0, 10.0, 0.0, 0.0]
tspan = (0.0, 10.0)
p = [0.5, 0.5, 0.5]
prob = ODEProblem(rn, u0, tspan, p)
data = solve(prob, Tsit5(), saveat = 0.1)

plot(data, vars = [2, 3, 4], labels = ["S" "ES" "P"], title = "Enzyme Kinetics", xlabel = "Time", ylabel = "Concentration", legend = :topright)


u = species(rn)

basis = Basis([polynomial_basis(u, 2);], u, iv = t)


@show equations(basis)
n,m = length(basis), length(u)
@parameters Ξ[1:n, 1:m] # Ξ is the coefficient matrix


constraints = [
    #Ξ[4,1] ~  10;
    #Ξ[2,1] + Ξ[4,1] ~ 0
]
#### EXPERIMENTAL

λ = 1e-2
@show u

X = hcat(data.u...)
#append t to X
X = vcat(data.t', X)
column_names = Symbol.(["t"; string.(u)])  # Convert species names to symbols for column names
df = DataFrame(X', :auto)  # Transpose X to match DataFrame's column-major format
rename!(df, column_names) 

model = MLJConstrainedSTLSQ(basis, constraints, λ, hcat(Ξ), InterpolationMethod(CubicSpline))
fitted_model = fit(model, df)
get_parameter_map(fitted_model)
pred = predict(fitted_model, df)