using Pkg
Pkg.activate(".")
using DataDrivenConstrained

using DifferentialEquations
using ModelingToolkit
using DataDrivenDiffEq
using DataDrivenSparse
using DocStringExtensions
using DataFrames 
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
tspan = (0.0, 100.0)
dt = 0.1
prob = ODEProblem(lorenz, u0, tspan)
sol = solve(prob, Tsit5(), saveat = dt)


## Start the automatic discovery
ddprob = DataDrivenProblem(sol, use_interpolations = false)
X = hcat(sol.u...) 
df = DataFrame(:t => sol.t, :x => X[1,:], :y => X[2,:], :z => X[3,:])

thirddegBspline(u,t) = BSplineInterpolation(u, t, 3, :Uniform, :Average)

ddprob = ContinuousDataDrivenProblem(X, sol.t, InterpolationMethod(thirddegBspline)) 
#plot derivative interpolation vs true derivative
interval = 1:1000
#log scale
p1 = plot(layout = (2, 1), size = (800, 600))
plot!(p1,sol.t[interval],ddprob.DX'[interval,:], label = ["ẋ" "ẏ" "ż"].*"_true", linewidth = 2, legend = :bottomright, title = "True vs Interpolated Derivatives", xlabel = "t", ylabel = "Derivative",subplot = 1)
plot!(p1, sol.t[interval],ddprob2.DX'[interval,:], label = ["ẋ" "ẏ" "ż"].*"_interpolated", linestyle = :dash, linewidth = 2, subplot = 1)
display(p1)
#savefig in plots folder
savefig(p1, "plots/true_vs_interpolated_derivatives.png")
#derivative residual
plot!(p1,sol.t[interval],abs.(ddprob.DX'[interval,:] .- ddprob2.DX'[interval,:]), label = ["ẋ" "ẏ" "ż"].*"_residual", linewidth = 2, legend = :bottomright, title = "Residual of True vs Interpolated Derivatives", xlabel = "t", ylabel = "Residual", subplot = 2)

### exclude part of data due to interpolation error
curvature = thirddegBspline(ddprob2.DX,ddprob2.t)
#plot curvature
plot(p1, ddprob2.t,curvature(ddprob2.t)', label = ["x" "y" "z"], linewidth = 2, title = "Curvature", xlabel = "t", ylabel = "Curvature", subplot = 1)
#highlight the part of the part of the data that has high curvature
highcurvetur = abs.(curvature(ddprob2.t)) .>= 0.8*maximum(abs.(curvature(ddprob2.t)))
#plot high curvature
#plot!(p1, ddprob2.t[sum(highcurvetur, dims=1)],curvature(ddprob2.t)[highcurvetur]', label = ["x" "y" "z"], linewidth = 2, title = "Curvature", xlabel = "t", ylabel = "Curvature", subplot = 1)

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

λ = 1e-2



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
t_interval = collect(1:1:200)
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


dfsol = DataFrame(:t => ddsol.t, :x => ddsol[1,:], :y => ddsol[2,:], :z => ddsol[3,:])

function plot_data(df, dfsol, u; ddsolconstrained = nothing, t_interval = collect(1:1:size(dfsol, 1)))
    num_states = length(u)
    colors = [:black, :red, :green]
    linewidth = 2
    opacity1 = 0.7
    opacity2 = 0.5

    p = plot(layout = (num_states, 1), size = (800, 600))

    # Plot each variable in its own subfigure with thicker lines and less opacity
    for i in 1:num_states
        plot!(p, df.t[t_interval], df[t_interval,i], label = string(u[i]), color = colors[1], subplot = i, linewidth = linewidth, alpha = opacity1)
        plot!(p, dfsol.t[t_interval], dfsol[t_interval,i], label = string(u[i]) * "_est", linestyle = :dash, color = colors[2], subplot = i, linewidth = linewidth, alpha = opacity2)
        if !isnothing(ddsolconstrained)
            plot!(p, ddsolconstrained.t[t_interval], ddsolconstrained[i, t_interval], label = string(u[i]) * "_constrained", linestyle = :dot, color = colors[3], subplot = i, linewidth = linewidth, alpha = opacity2)
        end
    end

    display(p)
end

# Example usage
plot_data(df, dfsol, u; ddsolconstrained = ddsolconstrained, t_interval = collect(200:1:400))




using DataFrames
Xt = vcat(sol.t', X)


#split dataframe at time 80
df_train = df[df.t .< 90, :]
df_test = df[df.t .>= 90, :]
#get state from  df 

model = MLJConstrainedSTLSQ(basis, constraints, λ, hcat(Ξ), InterpolationMethod())
fitted_model = fit(model, df)
get_parameter_map(fitted_model)
pred = predict(fitted_model, df)
t_interval = collect(1:1:size(df, 1))
plot_data(df_test, pred, u)

#number of rows in df
