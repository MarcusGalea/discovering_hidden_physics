## Lotka-Volterra equations
using Pkg
Pkg.activate(joinpath(@__DIR__, "..","UDEs"))
using Revise, Optimization, ModelingToolkit,DifferentialEquations,Plots,Lux
## Lotka-Volterra equations
@parameters α β γ δ
@independent_variables  t
@variables x(t) y(t) z(t)
Dt = Differential(t)
eqs = [
    Dt(x) ~ α * x - β * x * y,
    Dt(y) ~ δ * x * y - γ * y,
]
measured_quantities = [z ~ x + y]  # Example of a measured quantity
@named sys = ODESystem(eqs, t, [x, y], [α, β, γ, δ]; observed = measured_quantities)
sys = complete(sys)
params =  Dict([α => 0.1, 
                β => 0.02, 
                δ => 0.01,
                γ => 0.3])

u0 = Dict([x => 40.0, y => 9.0])
tspan = (0.0, 200.0)
dt = 0.1

sys = complete(sys)
odefun = ODEFunction(sys, unknowns(sys), parameters(sys))
prob = ODEProblem(odefun, [40.0, 9.0], tspan, [0.1, 0.02, 0.01, 0.3])
sol = solve(prob, Tsit5(), saveat=dt)
data = hcat(sol.u...)'
plot(sol, vars=(x, y), xlabel="prey", ylabel="predator",
     title="Lotka-Volterra Model", label="Solution",
     legend=:topright, linewidth=2, markersize=4)






@variables z(t)
known_eqs = [
    Dt(x) ~ α * x,
    Dt(y) ~ -γ * y
]
unknown_eqs = [
    Dt(x) ~ -β * x * y,
    Dt(y) ~ δ * x * y,
    
]
deviance = 0.1 # deviance for the unknown equations
params_guess_known = Dict([α => 0.1, # + deviance * randn()
                        γ => 0.3, # + deviance * randn()])
                        ])

params_guess_unknown = Dict([β => 0.02,# + deviance * randn()
                        δ => 0.01 # + deviance * randn()])
                        ])


@named sys_known = ODESystem(known_eqs, t, [x, y, z], [α, γ], defaults = params_guess_known, observed = [z ~ x + y])
@named sys_unknown = ODESystem(unknown_eqs, t, [x, y], [β, δ], defaults = params_guess_unknown)
sys_known = complete(sys_known)
sys_unknown = complete(sys_unknown)


# hmodel = HiddenODE(sys_known, sys_unknown)

# initp = init_params(hmodel)
# odefun = ODEFunction(hmodel)
# prob = ODEProblem(odefun, [40.0, 9.0], tspan, initp)
# sol = solve(prob, Tsit5(), saveat=dt)
# data = hcat(sol.u...)'
# plot(sol.t, data, label=["Prey" "Predator"], xlabel="Time", ylabel="Population", title="Lotka-Volterra Model with Hidden ODE")





#### PETAB
##Convert data to dataframe
using DataFrames
n_data = size(data, 1)
sample_size = 200


sample_idcs = rand(1:n_data, sample_size)
prey_df = DataFrame(simulation_id = "cond1", obs_id = "prey_o", time = sol.t[sample_idcs], measurement= data[sample_idcs, 1],)
predator_df = DataFrame(simulation_id = "cond1", obs_id = "predator_o", time = sol.t[sample_idcs], measurement= data[sample_idcs, 2],)
measurements = vcat(prey_df, predator_df)

#### PETAB Model
using PEtab
#Setup observables
@parameters σ
obs_x = PEtabObservable(x, σ)
obs_y = PEtabObservable(y, σ)
obs = Dict("prey_o" => obs_x, "predator_o" => obs_y)


#setup initial conditions
cond1 = Dict(:x => 40.0, :y => 9.0)
conds = Dict("cond1" => cond1)
# Setup parameters

#model parameters

estimate = true # Set to true if you want to estimate the parameters
p_α = PEtabParameter(α, lb = 1e-6, ub = 1e0, estimate = estimate, scale = :lin, value = 0.1)
p_β = PEtabParameter(β, lb = 1e-6 , ub = 1e0, estimate = estimate, scale = :lin, value = 0.02)
p_γ = PEtabParameter(γ, lb = 1e-6, ub = 1e0, estimate = estimate, scale = :lin, value = 0.3)
p_δ = PEtabParameter(δ, lb = 1e-6, ub = 1e0, estimate = estimate, scale = :lin, value = 0.01)
#noise parameter
p_σ = PEtabParameter(σ, lb = 1e-6, ub = 1e0, estimate = true, scale = :lin, value = 0.2)
pest = [
    p_α, p_β, p_γ, p_δ, p_σ
]
model = PEtabModel(sys,obs, measurements, pest; simulation_conditions  = conds)
petab_prob = PEtabODEProblem(model)

### PETAB
using Bijectors, LogDensityProblems, LogDensityProblemsAD, Distributions,MCMCChains,AdvancedHMC

target = PEtabLogDensity(petab_prob)
sampler = NUTS(0.8)
x = get_x(petab_prob)#res.xmin #Starting point of MCMC is the minimum of the optimization
xprior = to_prior_scale(x, target)
xinference = target.inference_info.bijectors(xprior)
res = sample(target, sampler, 2000; n_adapts = 1000, initial_params = xinference,drop_warmup=true, progress=false)
# using MCMCChains
# chain_hmc = PEtab.to_chains(res, target)
# using Plots, StatsPlots
# p1 = plot(chain_hmc)
# #save the plot
# savefig(p1, "./plots/lotka_volterra_hmc.png")