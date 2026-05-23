
using Pkg
Pkg.activate("scripts\\")
# Pkg.instantiate()

using Revise, Optimization,Optim, OptimizationOptimJL,Catalyst, OptimizationPolyalgorithms, ModelingToolkit,DifferentialEquations,Plots, Lux, Random, Distributions
seed = 0 #set seed for reproducibility
rng = Random.default_rng(seed) #create a random number generator with the seed


### ENSEMBLE DATA ###

initial_conditions  = [
    [10.0, 0.01, 0.0],
    [5.0, 0.01, 0.0],
    [2.5, 0.01, 0.0], 
]


diss_time = 5.0 # Time of the event

rn = @reaction_network begin
    # @discrete_events 5.0 => [E ~ 0.0]
    @parameters w_S w_ES
    @observables v ~ w_S * S + w_ES * ES
    ka, E + S --> ES
    kd, ES --> E + S
end
rn = complete(rn)

# Dissociation event
diss_condition(u,even_time,integrator) = even_time == diss_time # Set the time of the event to 5.0 seconds
affect!(integrator) = integrator.u[1] = 0.0*integrator.u[1] # Set the concentration of E to 0
cb =DiscreteCallback(diss_condition, affect!)
event = Dict(:callback => cb, :tstops => [diss_time]) # Define the event

odesys = convert(ODESystem, rn)
#Data setup
dt = 0.1 # time step
u0 = [10.0, 0.01, 0.0]
tspan = (0.0, 10.0)
p = Dict(:ka => 0.4, 
          :kd => 0.3, 
        #   :w_E => 0.0, 
          :w_S => 1, 
          :w_ES => 2) # rate constants
time = collect(tspan[1]:dt:tspan[2]) # time vector
prob = ODEProblem(rn, u0, tspan, p)
# data = solve(prob, Tsit5(), saveat = time; event...)
function prob_func(prob, i, repeat)
    remake(prob, u0 = initial_conditions[i])
end

ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
sim = solve(ensemble_prob, Tsit5(), trajectories = length(initial_conditions), saveat = time; event...)
timedata = sim[1].t
#add noise to data

#Data for data fitting
using DataFrames
#split train and test data at 80% of the time series
train_fraction = 0.8
n_data = size(sim, 2)

# test_time = timedata[1:round(Int, n_data * train_fraction)]
noise = 0.00 # noise level for the data
sample_size = length(timedata)
sample_idcs = rand(rng, 2:n_data, sample_size-1)
sample_idcs = [1; sample_idcs]

E_df = [DataFrame(simulation_id = "cond$(i)", obs_id = "E", time = timedata[sample_idcs], measurement = sim[i][sample_idcs][1,:].+noise.*randn(rng, sample_size),) for i in 1:length(initial_conditions)]
S_df = [DataFrame(simulation_id = "cond$(i)", obs_id = "S", time = timedata[sample_idcs], measurement = sim[i][sample_idcs][2,:].+noise.*randn(rng, sample_size),) for i in 1:length(initial_conditions)]
ES_df = [DataFrame(simulation_id = "cond$(i)", obs_id = "ES", time = timedata[sample_idcs], measurement = sim[i][sample_idcs][3,:].+noise.*randn(rng, sample_size),) for i in 1:length(initial_conditions)]
v_df = [DataFrame(simulation_id = "cond$(i)", obs_id = "v", time = timedata[sample_idcs], measurement = sim[i][:v][sample_idcs].+noise.*randn(rng, sample_size),) for i in 1:length(initial_conditions)]
measurements = vcat(E_df..., S_df..., ES_df..., v_df...)

#plot dataframe
label = [x for x in 1:length(initial_conditions)]
colors = [:blue, :green, :orange, :purple, :magenta, :brown]
plot(sim, legend = :topright, color = hcat(colors...))
scatter!(measurements.time, measurements.measurement, group = measurements.obs_id.*"_".* measurements.simulation_id, xlabel="Time", ylabel="Population", 
        title="Enzyme Dynamics ", legend =:topright, marker = [:circle :star5], color = hcat(colors...), markersize = 2)


# split dataframe into train and test sets
# train_idcs = findall(measurements.time .<= test_time[end])
# test_idcs = findall(measurements.time .> test_time[end])
#USE COND3 AS TESTSET
train_idcs = findall(measurements.simulation_id .!= "cond3")
test_idcs = findall(measurements.simulation_id .== "cond3") 
#save train and test data
train_measurements = measurements[train_idcs, :]
test_measurements = measurements[test_idcs, :]

#Create a new dataframe with log transformed time points
train_measurements_log = deepcopy(train_measurements)
#remove initial time points (time = 0) to avoid log(0). Do this for all simulation conditions
train_measurements_log = train_measurements_log[train_measurements_log.time .> 0.1, :]
train_measurements_log.time = log.(train_measurements_log.time) # add small value to avoid log(0)
#use only the v measurements for the log transformed data           
train_measurements_log = train_measurements_log[train_measurements_log.obs_id .== "v", :]
#plot log transformed data
scatter(train_measurements_log.time, train_measurements_log.measurement, group = train_measurements_log.simulation_id, xlabel="Log(Time)", ylabel="Population", 
        title="Enzyme Dynamics (Log Transformed Time)", legend = :topright, marker = [:circle :star5], color = hcat(colors...), markersize = 2)