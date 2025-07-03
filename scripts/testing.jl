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
odefun = ODEFunction(sys, unknowns(sys), parameters(sys))
prob = ODEProblem(odefun, [40.0, 9.0], tspan, [0.1, 0.02, 0.01, 0.3])
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


@named sys_known = ODESystem(known_eqs, t, [x, y, z], [α, γ], defaults = params_guess_known, observed = [z ~ x + y])
@named sys_unknown = ODESystem(unknown_eqs, t, [x, y], [β, δ], defaults = params_guess_unknown)
sys_known = complete(sys_known)
sys_unknown = complete(sys_unknown)


hmodel = HybridModel(sys_known, sys_unknown)

initp = init_params(hmodel)
odefun = ODEFunction(hmodel)
prob = ODEProblem(odefun, [40.0, 9.0], tspan, initp)
sol = solve(prob, Tsit5(), saveat=dt)
data = hcat(sol.u...)'
plot(sol.t, data, label=["Prey" "Predator"], xlabel="Time", ylabel="Population", title="Lotka-Volterra Model with Hidden ODE")





#### PETAB
##Convert data to dataframe
using DataFrames
n_data = size(data, 1)
sample_size = 200


sample_idcs = rand(1:n_data, sample_size)
prey_df = DataFrame(simulation_id = "cond1", obs_id = "prey_o", time = time[sample_idcs], measurement= data[sample_idcs, 1],)
predator_df = DataFrame(simulation_id = "cond1", obs_id = "predator_o", time = time[sample_idcs], measurement= data[sample_idcs, 2],)
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

params =  Dict([α => 0.1, 
                β => 0.02, 
                γ => 0.3,
                δ => 0.01,])

u0 = Dict([x => 40.0, y => 9.0])
tspan = (0.0, 200.0)
dt = 0.1

sys = structural_simplify(sys)
odefun = DifferentialEquations.ODEFunction(sys )
prob = DifferentialEquations.ODEProblem(odefun, [40.0, 9.0], tspan, [0.1, 0.02, 0.01, 0.3])
sol = solve(prob, Tsit5(), saveat=dt)
data = hcat(sol.u...)'


parameters(sys)
p_α = PEtabParameter(Symbol(psym), lb = 1e-6, ub = 1e0, scale = :lin)


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
    p_α, p_β,  p_δ,p_γ, p_σ
]

#### DATA
#Convert data to DataFrame  
using ComponentArrays
using DataFrames
using Optimization
n_data = size(data, 1)
sample_size = 200


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

model = PEtabModel(odeprob,obs, train_measurements, pest; simulation_conditions  = conds)
petab_prob = PEtabODEProblem(model)
x0 = get_x(petab_prob)
using Plots
p1 = scatter(train_measurements[!,:time], petab_prob.simulated_values(x0))


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


hmodel = HybridModel(sys_known, sys_unknown; rng = rng)
initp = init_params(hmodel) .+ 0.02*randn(rng, length(5)) # Initialize parameters with some noise
hodeprob = ODEProblem(hmodel, u0vec, tspan, initp)
hsol = solve(hodeprob, Tsit5(), saveat = dt)
plot(hsol, vars = [x, y], label = ["Prey" "Predator"], title = "Hybrid Model Solution", xlabel = "Time", ylabel = "Population", legend = :topright)

pest = [
    p_α, p_γ, # Parameters for the known system
    p_β, p_δ, # Parameters for the unknown system
    p_σ # Noise parameter
]
petabhmodel = PEtabModel(hodeprob,obs, train_measurements, pest; simulation_conditions  = conds)

osolver = ODESolver(Tsit5(); abstol_adj = 1e-3, reltol_adj = 1e-6)
petabhprob = PEtabODEProblem(petabhmodel; gradient_method = :ForwardDiff, odesolver = osolver, odesolver_gradient = osolver)
x0 = get_x(petabhprob)#get_startguesses(petabhprob, 1)
using Optim
using Plots
res = calibrate(petabhprob, x0, BFGS(), save_trace = true)
res.xmin
x0guess = ComponentArray(initp; σ = 0.2) # Initial guess for the parameters with noise
p2 = scatter(train_measurements[!,:time], petabhprob.simulated_values(x0guess), label = "Hybrid Model", color = :red)

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


n_states = length(unknowns(sys))
nn1 = Lux.Chain(
    Lux.Dense(2,5,rbf; init_weight =  kaiming_normal),
    Lux.Dense(5,5, rbf, init_weight = kaiming_normal),
    Lux.Dense(5,5, rbf, init_weight = kaiming_normal),
    Lux.Dense(5,2, rbf, init_weight = kaiming_normal),
)
p_nn, st_nn = Lux.setup(rng, nn1) |> ComponentArray |> f64
petab_nn = MLModel(nn1; static = false, dirdata = model_dir, inputs = Symbol.(unknowns(sys)), outputs = Symbol.(unknowns(sys)))


hmodel = HybridModel(sys_known, nn1; rng = rng)
initp = init_params(hmodel)
hodeprob = ODEProblem(hmodel, u0vec, tspan, initp)
petabhmodel = PEtabModel(odeprob,obs, train_measurements, pest; simulation_conditions  = conds)
petabhprob = PEtabODEProblem(petabhmodel)
x0 = get_x(petabhprob)
p3 = scatter(train_measurements[!,:time], petabhprob.simulated_values(x0), label = "Hybrid Model with NN", color = :green)




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