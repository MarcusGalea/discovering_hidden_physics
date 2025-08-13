
using Pkg
Pkg.activate("scripts\\")
# Pkg.instantiate()

using Revise, Optimization,Optim, OptimizationOptimJL,Catalyst, OptimizationPolyalgorithms, ModelingToolkit,DifferentialEquations,Plots, Lux, PEtab, Random, Distributions
seed = 0 #set seed for reproducibility
rng = Random.default_rng(seed) #create a random number generator with the seed


### ENSEMBLE DATA ###

initial_conditions  = [
    [10.0, 1.0, 0.0, 0.0],
    [5.0, 1.0, 0.0, 0.0],
    [2.5, 1.0, 0.0, 0.0],
]


diss_time = 5.0 # Time of the event

rn = @reaction_network begin
    # @discrete_events 5.0 => [E ~ 0.0]
    @parameters w_S w_ES w_P
    @observables y ~ w_S * S + w_ES * ES + w_P * P
    ka, E + S --> ES
    kd, ES --> E + S
    kc, ES --> E + P
end
rn = complete(rn)

# Dissociation event
diss_condition(u,even_time,integrator) = even_time == diss_time # Set the time of the event to 5.0 seconds
affect!(integrator) = integrator.u[1] = 0.0*integrator.u[1] # Set the concentration of E to 0
dcb = DiscreteCallback(diss_condition, affect!)
event = Dict(:callback => dcb, :tstops => [diss_time]) # Define the event

odesys = convert(ODESystem, rn)
#Data setup
dt = 0.1 # time step
u0 = [10.0, 1.0, 0.0, 0.0]
tspan = (0.0, 10.0)
p = Dict(:ka => 0.4, 
          :kd => 0.3, 
          :kc => 0.05,
        #   :w_E => 0.0, 
          :w_S => 1.0, 
          :w_ES => 2.0,
          :w_P => 1.0
          ) # rate constants
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
noise = 0.02 # noise level for the data
sample_size = length(timedata)
sample_idcs = rand(rng, 2:n_data, sample_size-1)
sample_idcs = [1; sample_idcs]

E_df = [DataFrame(simulation_id = "cond$(i)", obs_id = "E", time = timedata[sample_idcs], measurement = sim[i][sample_idcs][1,:].+noise.*randn(rng, sample_size),) for i in 1:length(initial_conditions)]
S_df = [DataFrame(simulation_id = "cond$(i)", obs_id = "S", time = timedata[sample_idcs], measurement = sim[i][sample_idcs][2,:].+noise.*randn(rng, sample_size),) for i in 1:length(initial_conditions)]
ES_df = [DataFrame(simulation_id = "cond$(i)", obs_id = "ES", time = timedata[sample_idcs], measurement = sim[i][sample_idcs][3,:].+noise.*randn(rng, sample_size),) for i in 1:length(initial_conditions)]
P_df = [DataFrame(simulation_id = "cond$(i)", obs_id = "P", time = timedata[sample_idcs], measurement = sim[i][sample_idcs][4,:].+noise.*randn(rng, sample_size),) for i in 1:length(initial_conditions)]
y_df = [DataFrame(simulation_id = "cond$(i)", obs_id = "y", time = timedata[sample_idcs], measurement = sim[i][:y][sample_idcs].+noise.*randn(rng, sample_size),) for i in 1:length(initial_conditions)]
measurements = vcat(E_df..., S_df..., ES_df..., P_df..., y_df...)

#plot dataframe
# label = [x for x in 1:length(initial_conditions)]
# colors = [:blue, :green, :orange, :purple, :magenta, :brown]
# plot(sim[3].t, sim[3][v], legend = :topright, color = hcat(colors...))
# scatter!(measurements.time, measurements.measurement, group = measurements.obs_id.*"_".* measurements.simulation_id, xlabel="Time", ylabel="Population", 
#         title="Lotka-Volterra Model with Sampled Data", legend =:topright, marker = [:circle :star5], color = hcat(colors...), markersize = 2)


# split dataframe into train and test sets
# train_idcs = findall(measurements.time .<= test_time[end])
# test_idcs = findall(measurements.time .> test_time[end])
#USE COND3 AS TESTSET
train_idcs = findall(measurements.simulation_id .!= "cond3")
test_idcs = findall(measurements.simulation_id .== "cond3") 
#save train and test data
train_measurements = measurements[train_idcs, :]
test_measurements = measurements[test_idcs, :]

### OPTIMIZATION
n_initial_conditions = 3
#CHANGE NUMBER OF INITIAL CONDITIONS HERE

# batch_size = 32 # Batch size for the optimization
@unpack E, S, ES, P, y = sys_known
obs = Dict("y" => y)#"E" => E, "S" => S, "ES" => ES, 
u0map = Dict([E => 10.0, S => 1.0, ES => 0.0, P => 0.0])
ic_vals = Dict(["cond$i" => Dict([var => ic[j] for (j, var) in enumerate(unknowns(sys_known))]) for (i, ic) in enumerate(initial_conditions[1:n_initial_conditions])])
included_exp = (df) -> reduce(.|, [(df.simulation_id .== "cond$i") .& (df.obs_id .== obsvar)
                               for i in 1:n_initial_conditions for obsvar in keys(obs)])
train_measurements_exp = train_measurements[included_exp(train_measurements), :]
test_measurements_exp = test_measurements[included_exp(test_measurements), :]