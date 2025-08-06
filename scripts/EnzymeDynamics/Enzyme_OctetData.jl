using Revise, Optimization,Optim, OptimizationOptimJL,OptimizationPolyalgorithms, ModelingToolkit,DifferentialEquations,Plots, Lux, PEtab, Random, Distributions


### ENSEMBLE DATA ###

initial_conditions  = [
    [10.0, 1.0, 0.0],
    [5.0, 1.0, 0.0],
    [2.5, 1.0, 0.0], 
]


rn = @reaction_network begin
    k1, E + S --> ES
    k2, ES --> E + S
end
rn = complete(rn)

# Dissociation event
diss_time = 5.0 # Time of the event
diss_condition(u,even_time,integrator) = even_time == diss_time # Set the time of the event to 5.0 seconds
affect!(integrator) = integrator.u[1] = 0.0*integrator.u[1] # Set the concentration of E to 0
cb =DiscreteCallback(diss_condition, affect!)

#Data setup
dt = 0.1 # time step
u0 = [10.0, 1.0, 0.0]
tspan = (0.0, 10.0)
p = Dict(:k1 => 0.4, :k2 => 0.3) # rate constants
time = collect(tspan[1]:dt:tspan[2]) # time vector
prob = ODEProblem(rn, u0, tspan, p)
data = solve(prob, Tsit5(), saveat = time, callback = cb, tstops = [diss_time])
du_actual = data.(data.t, Val{1})


function prob_func(prob, i, repeat)
    remake(prob, u0 = initial_conditions[i])
end

ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)

#add noise to data

# time = data.t
signal =  hcat(data.u...)
noise = 0.02 * randn(size(signal))
signal = signal .+ noise
plot(data.t, signal', xlabel = "t", ylabel = "Concentration", title = "Enzyme Kinetics", label = ["E" "S" "ES"])
plot(data.t, hcat(du_actual...)', xlabel = "t", ylabel = "Concentration Derivative", title = "Enzyme Kinetics", label = ["E" "S" "ES"])


using Pkg
Pkg.activate("scripts\\")
# Pkg.instantiate()

Pkg.status()
seed = 0 #set seed for reproducibility
rng = Random.default_rng(seed) #create a random number generator with the seed


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