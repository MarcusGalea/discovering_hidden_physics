using Pkg
Pkg.activate("scripts\\")
# Pkg.instantiate()

Pkg.status()
using Revise, Optimization,Optim, OptimizationOptimJL,OptimizationPolyalgorithms, ModelingToolkit,DifferentialEquations,Plots, Lux, PEtab, Random, Distributions
seed = 0 #set seed for reproducibility
rng = Random.default_rng(seed) #create a random number generator with the seed
## GENERATE DATA
## Lotka-Volterra equations
@parameters α β γ δ
@independent_variables  t
vars = @variables x(t) y(t) z(t)
Dt = Differential(t)
eqs = [
    Dt(x) ~ α * x - β * x * y,
    Dt(y) ~ δ * x * y - γ * y,
]

params =  Dict([α => 0.1, 
                β => 0.02, 
                δ => 0.01,
                γ => 0.3])
measured_quantities = [z ~ x + y]  # Example of a measured quantity
@named sys = ODESystem(eqs, t, [x, y], [α, β, γ, δ]; observed = measured_quantities, defaults = params)
sys = complete(sys)


u0 = Dict([x => 40.0, y => 9.0])
tspan = (0.0, 80.0)
dt = 0.1

sys = complete(sys)
odefun = ODEFunction(sys, unknowns(sys), parameters(sys))
prob = ODEProblem(odefun, [40.0, 9.0], tspan, [0.1, 0.02, 0.01, 0.3])
sol = solve(prob, Tsit5(), saveat=dt)
data = hcat(sol.u...)'
timedata = sol.t
plot(sol, vars=(x, y), xlabel="prey", ylabel="predator",
     title="Lotka-Volterra Model", label="Solution",
     legend=:topright, linewidth=2, markersize=4)

#Data for data fitting
using DataFrames
n_data = size(data, 1)
sample_size = 200

sample_idcs = rand(1:n_data, sample_size)
scatter(timedata[sample_idcs], data[sample_idcs, :], label=["Prey" "Predator"], xlabel="Time", ylabel="Population", title="Lotka-Volterra Model with Sampled Data")

#split train and test data at 80% of the time series
train_fraction = 0.8
test_time = timedata[1:round(Int, n_data * train_fraction)]

prey_df = DataFrame(simulation_id = "cond1", obs_id = "prey_o", time = timedata[sample_idcs], measurement= data[sample_idcs, 1],)
predator_df = DataFrame(simulation_id = "cond1", obs_id = "predator_o", time = timedata[sample_idcs], measurement= data[sample_idcs, 2],)
measurements = vcat(prey_df, predator_df)

#split dataframe into train and test sets
train_idcs = findall(measurements.time .<= test_time[end])
test_idcs = findall(measurements.time .> test_time[end])
#save train and test data½
train_measurements = measurements[train_idcs, :]
test_measurements = measurements[test_idcs, :]

#### Multiple initial conditions for ensemble learning

# initial_conditions  = [
#     [40.0, 9.0, 0.0],
#     [20.0, 9.0, 0.0],
#     [10.0, 9.0, 0.0],
# ]
# function prob_func(prob, i, repeat)
#     remake(prob, u0 = initial_conditions[i])
# end

# ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
# sim = solve(ensemble_prob, Tsit5(), EnsembleDistributed(), trajectories = length(initial_conditions), saveat = dt)
# scatter(sim[1].t, hcat(sim[1].u...), label=["Prey" "Predator"], xlabel="Time", ylabel="Population", title="Lotka-Volterra Model with Ensemble Data")