## Lotka-Volterra equations
using Pkg
using Revise
Pkg.activate(joinpath(@__DIR__))
using Revise, Optimization, ModelingToolkit,DifferentialEquations,Plots,Lux,PEtab
include("../src/hybrid_model.jl")
export HybridModel, init_params, ODEFunction

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
odefun = DifferentialEquations.ODEFunction(sys, unknowns(sys), parameters(sys))
prob = DifferentialEquations.ODEProblem(odefun, [40.0, 9.0], tspan, [0.1, 0.02, 0.01, 0.3])
sol = solve(prob, Tsit5(), saveat=dt)
data = hcat(sol.u...)'
plot(sol, vars=(x, y), xlabel="prey", ylabel="predator",
     title="Lotka-Volterra Model", label="Solution",
     legend=:topright, linewidth=2, markersize=4)




### HIDDEN MODEL SETUP ###

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


@named sys_known = ODESystem(known_eqs, t, [x, y], [α, γ], defaults = params_guess_known, observed = [z ~ x + y])
@named sys_unknown = ODESystem(unknown_eqs, t, [x, y], [β, δ], defaults = params_guess_unknown)
sys_known = complete(sys_known)
sys_unknown = complete(sys_unknown)


hmodel = HybridModel(sys_known, sys_unknown)

initp = init_params(hmodel)
odefun = ODEFunction(hmodel)
u0map = ComponentArray(x = 40.0, y = 9.0)
prob = DifferentialEquations.ODEProblem(odefun, u0map, tspan, initp)
sol = solve(prob, Tsit5(), saveat=dt)
data = hcat(sol.u...)'
plot(sol.t, data, label=["Prey" "Predator"], xlabel="Time", ylabel="Population", title="Lotka-Volterra Model with Hidden ODE")



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
#Setup observables
@parameters σ
obs_x = PEtabObservable(:x, σ)
obs_y = PEtabObservable(:y, σ)
obs = Dict("prey_o" => obs_x, "predator_o" => obs_y)


#setup initial conditions
cond1 = Dict()
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
ptrue = get_x(petab_prob)
#plot true values
scatter(measurements.time, petab_prob.simulated_values(ptrue), label=["Populations"], xlabel="Time", ylabel="Population", title="Lotka-Volterra Model with PETAB", legend=:topright, linewidth=2, markersize=4)


#### DATA
#Convert data to DataFrame  
using ComponentArrays
using DataFrames
using Optimization
# n_data = size(data, 1)
n_data = 200
sample_size = 200


### TRAINING TEST SPLIT ###
sample_idcs = rand(1:n_data, sample_size)
prey_df = DataFrame(simulation_id = "cond1", obs_id = "prey_o", time = sol.t[sample_idcs], measurement= data[sample_idcs, 1],)
predator_df = DataFrame(simulation_id = "cond1", obs_id = "predator_o", time = sol.t[sample_idcs], measurement= data[sample_idcs, 2],)
measurements = vcat(prey_df, predator_df)
#split data into training and validation sets (from time point 180)
train_measurements = measurements[measurements.time .< 180, :]
val_measurements = measurements[measurements.time .>= 180, :]




### KNOWN ODE SYSTEM ###
pvec = ComponentArray(α = 0.1, β = 0.02, δ = 0.01, γ = 0.3)
u0vec = ComponentArray(x = 40.0, y = 9.0)

odefun =  DifferentialEquations.ODEFunction(sys)
odeprob = DifferentialEquations.ODEProblem(odefun, u0vec, tspan, pvec)

conds = Dict("cond1" => Dict(:x => 40.0, :y => 9.0))
model = PEtabModel(odeprob,obs, train_measurements, pest;speciemap = [], simulation_conditions = conds)
petab_prob = PEtabODEProblem(model)
x0 = get_x(petab_prob)
get_u0(x0, petab_prob)


# @unpack probinfo, model_info = petab_prob
# @unpack xindices, model, simulation_info = petab_prob.model_info
# @unpack solver, ss_solver, cache, odeproblem, ml_models_pre_ode = probinfo


# using Plots
# p1 = scatter(train_measurements[!,:time], petab_prob.simulated_values(x0))

# sys_mutated = deepcopy(odeprob)
# sys_mutated, speciemap_use = PEtab._get_speciemap(sys_mutated, model.petab_tables[:conditions], model.petab_tables[:hybridization]
# , Dict{Symbol, MLModel}(), [:x => 40.0, :y => 9.0])
# parametermap_use = PEtab._get_parametermap(sys_mutated, nothing)
# xindices = PEtab.ParameterIndices(model.petab_tables, sys_mutated, parametermap_use, speciemap_use, Dict{Symbol, MLModel}())
# model_SBML = PEtab.SBMLImporter.ModelSBML(model.name)
# hstr, u0!str, u0str, σstr = PEtab.parse_observables(model.name, Dict{Symbol, String}(), sys_mutated, model.petab_tables, xindices, speciemap_use, model_SBML, Dict{Symbol, MLModel}(), false)



#### HYBRID MODEL SETUP ####
rng = Random.default_rng(1234)
#HYBRID MODEL SETUP
known_eqs = [
    Dt(x) ~ α * x
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


@named sys_known = ODESystem(known_eqs, t, [x, y], [α, γ], defaults = params_guess_known, observed = measured_quantities)
@named sys_unknown = ODESystem(unknown_eqs, t, [x, y], [β, δ], defaults = params_guess_unknown)
sys_known = complete(sys_known)
sys_unknown = complete(sys_unknown)

#show known solutio
known_prob = DifferentialEquations.ODEProblem(sys_known, [40.0, 9.0], tspan, [0.1, 0.3])
known_sol = solve(known_prob, Tsit5(), saveat=dt)
plot(known_sol, xlabel="Time", ylabel="Population", title="Known Lotka-Volterra Model", label=["Prey" "Predator"], legend=:topright, linewidth=2, markersize=4, ylim = (0, 60))

# hmodel = HybridModel(sys_known, sys_unknown; rng = rng)
# initp = init_params(hmodel)# .+ 0.02*randn(rng, length(5)) # Initialize parameters with some noise
# hodeprob = ODEProblem(hmodel, u0vec, tspan, initp)
# hsol = solve(hodeprob, Tsit5(), saveat=dt)
# plot(hsol)

# pest = [
#     p_α, p_γ, # Parameters for the known system
#     p_β,p_δ,  # Parameters for the unknown system
#     p_σ # Noise parameter
# ]
# p_values = [0.1, 0.3, 0.02, 0.01, 0.2] # Initial parameter values
# petabhmodel = PEtabModel(hodeprob,obs, train_measurements, pest; simulation_conditions  = conds, speciemap = [:x => 40.0, :y => 9.0], parametermap = p_values, verbose = true)
# osolver = ODESolver(Tsit5();)
# petabhprob = PEtabODEProblem(petabhmodel, odesolver = osolver)#; gradient_method = :ForwardDiff, odesolver = osolver, odesolver_gradient = osolver)
# p = get_x(petabhprob)#get_startguesses(petabhprob, 1)
# get_u0(p,petabhprob)
# @unpack probinfo, model_info = petabhprob
# @unpack xindices, model, simulation_info = petabhprob.model_info
# @unpack solver, ss_solver, cache, odeproblem, ml_models_pre_ode = probinfo
# ps = xindices.xids[:sys]

odeproblem.u0[:]






# #show results
# scatter(train_measurements.time, petabhprob.simulated_values(p), label = "Hybrid Model", color = :blue, xlabel = "Time", ylabel = "Population", title = "Lotka-Volterra Hybrid Model with PETAB", legend = :topright, linewidth = 2, markersize = 4)


# sys_mutated = deepcopy(hodeprob)
# sys_mutated, speciemap_use = PEtab._get_speciemap(sys_mutated, petabhmodel.petab_tables[:conditions], petabhmodel.petab_tables[:hybridization]
# , Dict{Symbol, MLModel}(), [:x => 40.0, :y => 9.0])
# parametermap_use = PEtab._get_parametermap(sys_mutated, nothing)
# xindices = PEtab.ParameterIndices(petabhmodel.petab_tables, sys_mutated, parametermap_use, speciemap_use, Dict{Symbol, MLModel}())
# model_SBML = PEtab.SBMLImporter.ModelSBML(petabhmodel.name)
# hstr, u0!str, u0str, σstr = PEtab.parse_observables(petabhmodel.name, Dict{Symbol, String}(), sys_mutated, petabhmodel.petab_tables, xindices, speciemap_use, model_SBML, Dict{Symbol, MLModel}(), false)



sys =  hodeprob
simulation_conditions = conds
# observables = obs
measurements = train_measurements
# parameters = pest
# speciemap = 
# parametermap::Union{Nothing, AbstractVector},
# events::Union{PEtabEvent, AbstractVector, Nothing}, verbose::Bool,
# ml_models::Union{MLModels, Nothing})::PEtabModel



using Optim
using Plots
using Random
# res = calibrate(petabhprob, x0, BFGS(), save_trace = true)
# p2 = scatter(train_measurements[!,:time], petabhprob.simulated_values(x0guess), label = "Hybrid Model", color = :red)

### TRY AGAIN WITH A NEURAL NETWORK SURROGATE MODEL ###

seed = 1234
rootdir = dirname(@__DIR__)
model_dir = joinpath(rootdir , "models", "lotka_volterra", "nn", "seed_$seed")
#create model dirdata
if !isdir(model_dir)
    mkpath(model_dir)
end
rng = Random.default_rng(seed)
rbf(x) = exp.(-(x.^2))


n_states = length(unknowns(sys_known))
nn1 = Lux.Chain(
    Lux.Dense(2,5,rbf; init_weight =  glorot_uniform),
    Lux.Dense(5,5, rbf, init_weight = glorot_uniform),
    Lux.Dense(5,5, rbf, init_weight = glorot_uniform),
    Lux.Dense(5,2, rbf, init_weight = glorot_uniform),
)
p_nn, st_nn = Lux.setup(rng, nn1) |> ComponentArray |> f64
petab_nn = MLModel(nn1; static = false, dirdata = model_dir, inputs = Symbol.(unknowns(sys_known)), outputs = Symbol.(unknowns(sys_known)))


hmodel = HybridModel(complete(sys_known), petab_nn)
initp = init_params(hmodel)
u0vec = ComponentArray(x = 40.0, y = 9.0)

hodeprob = ODEProblem(hmodel, u0vec, tspan, initp)
hsol = solve(hodeprob, Tsit5(), saveat=dt)
plot(hsol, ylim = (0, 60), xlabel = "Time", ylabel = "Population", title = "Hybrid Model with NN Surrogate", label=["Prey" "Predator"], legend=:topright, linewidth=2, markersize=4)


## PETAB
@parameters σ
obs_x = PEtabObservable(:x, σ)
obs_y = PEtabObservable(:y, σ)
obs = Dict("prey_o" => obs_x, "predator_o" => obs_y)


#setup initial conditions
cond1 = Dict()
conds = Dict("cond1" => cond1)
# Setup parameters

#model parameters

### TRAINING TEST SPLIT ###
sample_idcs = rand(1:n_data, sample_size)
prey_df = DataFrame(simulation_id = "cond1", obs_id = "prey_o", time = sol.t[sample_idcs], measurement= data[sample_idcs, 1],)
predator_df = DataFrame(simulation_id = "cond1", obs_id = "predator_o", time = sol.t[sample_idcs], measurement= data[sample_idcs, 2],)
measurements = vcat(prey_df, predator_df)
#split data into training and validation sets (from time point 180)
train_measurements = measurements[measurements.time .< 180, :]
val_measurements = measurements[measurements.time .>= 180, :]






estimate = true # Set to true if you want to estimate the parameters
p_α = PEtabParameter(α, lb = 1e-6, ub = 1e0, estimate = estimate, scale = :lin, value = 0.1)
p_β = PEtabParameter(β, lb = 1e-6 , ub = 1e0, estimate = estimate, scale = :lin, value = 0.02)
p_γ = PEtabParameter(γ, lb = 1e-6, ub = 1e0, estimate = estimate, scale = :lin, value = 0.3)
p_δ = PEtabParameter(δ, lb = 1e-6, ub = 1e0, estimate = estimate, scale = :lin, value = 0.01)
#noise parameter
p_σ = PEtabParameter(σ, lb = 1e-6, ub = 1e0, estimate = true, scale = :lin, value = 0.2)
p_surrogate = PEtabMLParameter(:surrogate, true, petab_nn.ps)
pest_ml = [
    p_α, p_γ, # Parameters for the known system
    p_σ, # Noise parameter
    p_surrogate # Surrogate model parameter
]

using SciMLSensitivity, Sundials
# rename!(train_measurements, :simulation_id => :simulationConditionId, :obs_id => :observableId) #FOR PETAB WHEN LOADING ODEPROBLEMS 
petabhmodel = PEtabModel(hodeprob,obs, train_measurements, pest_ml; simulation_conditions  = conds, ml_models = hmodel.ml_models)
osolver = ODESolver(Tsit5();  abstol_adj = 1e-3, reltol_adj = 1e-6)
petabhprob = PEtabODEProblem(petabhmodel, odesolver = osolver, gradient_method = :Adjoint, odesolver_gradient = osolver,
                             sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
x0 = get_x(petabhprob)

petabhprob.nllh(x0)
petabhprob.grad(get_x(petabhprob)) # to initialize the gradient

using Base
Base.zero(::Type{Any}) = 0.0
scatter(train_measurements[!,:time], petabhprob.simulated_values(x0), label = "Hybrid Model with NN", color = :red, xlabel = "Time", ylabel = "Population", title = "Lotka-Volterra Hybrid Model with PETAB", legend = :topright, linewidth = 2, markersize = 4)
scatter!(train_measurements[!,:time], train_measurements[!,:measurement], label = "Observed", color = :blue, markersize = 2, linewidth = 1)

using OptimizationOptimJL
using Optim

opt_prob = OptimizationProblem(petabhprob, box_constraints = true)
opt_prob.u0 .= get_x(petabhprob) #.+ 0.1 * randn(rng, length(get_x(petabhprob)))

function cb(state, args...)
    p1 = scatter(measurements[!,:time], measurements[!,:measurement],
     label = "Observed", color = :blue, xlabel = "Time", ylabel = "Population",
     title = "Hybrid Model with NN Surrogate", linewidth=2, markersize=4)
    trajectory = petab_prob.simulated_values(state.u)
    scatter!(p1, measurements[!,:time], trajectory,
             label = "Hybrid Model", color = :red, xlabel = "Time", ylabel = "Population",
             title = "Hybrid Model with NN Surrogate", linewidth=2, markersize=4)
    display(p1)
    return false
end
res = solve(opt_prob,BFGS(), show_trace = true, maxiters = 200, show_every = 10)

p3 = scatter(train_measurements[!,:time], petabhprob.simulated_values(res.u), label = "Hybrid Model with NN", color = :green, ylim = (0, 60))
#plot real data on topright
scatter(p3, train_measurements[!,:time], train_measurements[!,:measurement], label = "Real Data", color = :black, markersize = 2, linewidth = 1)



### CALLBACK TESTING
using Optim
fun(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

function fun_grad!(g, x)
    g[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    g[2] = 200.0 * (x[2] - x[1]^2)
end

function fun_hess!(h, x)
    h[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    h[1, 2] = -400.0 * x[1]
    h[2, 1] = -400.0 * x[1]
    h[2, 2] = 200.0
end;
x0 = [0.0, 0.1]
df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)

lx = [-0.5, -0.5];
ux = [0.5, 0.5];
dfc = TwiceDifferentiableConstraints(lx, ux)

function cb(state, args...)
    println("Current metadata: ", state.metadata["x"])
    return false
end
res = optimize(df, dfc, x0, IPNewton(), Optim.Options(show_trace = true, iterations = 1000, callback = cb, show_every = 10, extended_trace = true))