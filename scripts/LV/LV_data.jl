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
noise = 0.2 # noise level for the data
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
#add noise
data .+= randn(size(data)) .* noise

timedata = sol.t
plot(sol, vars=(x, y), xlabel="prey", ylabel="predator",
     title="Lotka-Volterra Model", label="Solution",
     legend=:topright, linewidth=2, markersize=4)

#Data for data fitting
using DataFrames
n_data = size(data, 1)
sample_size = 200

sample_idcs = rand(rng, 2:n_data, sample_size-1)
sample_idcs = [1; sample_idcs]
scatter(timedata[sample_idcs], data[sample_idcs, :], label=["Prey" "Predator"], xlabel="Time", ylabel="Population", title="Lotka-Volterra Model with Sampled Data")

#split train and test data at 80% of the time series
train_fraction = 0.8
test_time = timedata[1:round(Int, n_data * train_fraction)]

prey_df = DataFrame(simulation_id = "cond1", obs_id = "prey_o", time = timedata[sample_idcs], measurement= data[sample_idcs, 1],)
predator_df = DataFrame(simulation_id = "cond1", obs_id = "predator_o", time = timedata[sample_idcs], measurement= data[sample_idcs, 2],)
measurements = vcat(prey_df, predator_df)

# #split dataframe into train and test sets
# train_idcs = findall(measurements.time .<= test_time[end])
# test_idcs = findall(measurements.time .> test_time[end])
# #save train and test data
# train_measurements = measurements[train_idcs, :]
# test_measurements = measurements[test_idcs, :]

#### Multiple initial conditions for ensemble learning
initial_conditions  = [
    [40.0, 9.0],
    [30.0, 8.0],
    [20.0, 7.0],
]
n_initial_conditions = length(initial_conditions)
function prob_func(prob, i, repeat)
    remake(prob, u0 = initial_conditions[i])
end
#plot dir 
plotdir = "plots/LV/"
if !isdir(plotdir)
    mkpath(plotdir)
end
noise = 0.4
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
sim = solve(ensemble_prob, Tsit5(), EnsembleDistributed(), trajectories = length(initial_conditions), saveat = dt)
prey_df = [DataFrame(simulation_id = "cond$(i)", obs_id = "prey_o", time = timedata[sample_idcs], measurement = sim[i][sample_idcs][1,:].+noise.*randn(rng, sample_size),) for i in 1:length(initial_conditions)]
predator_df = [DataFrame(simulation_id = "cond$(i)", obs_id = "predator_o", time = timedata[sample_idcs], measurement = sim[i][sample_idcs][2,:].+noise.*randn(rng, sample_size),) for i in 1:length(initial_conditions)]
measurements = vcat(prey_df..., predator_df...)
#plot dataframe
label = [x]
colors = [:blue, :green, :orange, :purple, :magenta, :brown]
plot(sim, legend = :topright, color = hcat(colors...))
scatter!(measurements.time, measurements.measurement, group = measurements.obs_id.*"_".* measurements.simulation_id, xlabel="Time", ylabel="Population", 
        title="Lotka-Volterra Model with Sampled Data", legend =:topright, marker = [:circle :star5], color = hcat(colors...), markersize = 2)

# # split dataframe into train and test sets
# train_idcs = findall(measurements.time .<= test_time[end])
# test_idcs = findall(measurements.time .> test_time[end])
# #save train and test data
# train_measurements = measurements[train_idcs, :]
# test_measurements = measurements[test_idcs, :]

train_idcs = findall(measurements.simulation_id .!= "cond3")
test_idcs = findall(measurements.simulation_id .== "cond3") 
#save train and test data
train_measurements = measurements[train_idcs, :]
test_measurements = measurements[test_idcs, :]

obs = Dict("prey_o" => x, "predator_o" => y)
u0map = Dict([x => 40.0, y => 9.0])
ic_vals = Dict(["cond$i" => Dict([var => ic[j] for (j, var) in enumerate(unknowns(sys))]) for (i, ic) in enumerate(initial_conditions[1:n_initial_conditions])])
included_exp = (df) -> reduce(.|, [(df.simulation_id .== "cond$i") .& (df.obs_id .== obsvar)
                               for i in 1:n_initial_conditions for obsvar in keys(obs)])
train_measurements_exp = train_measurements[included_exp(train_measurements), :]
test_measurements_exp = test_measurements[included_exp(test_measurements), :]

# using FFTW
# dfft_vals = abs.(fft(sim[1][1,:]))


# # Assume y is your time series and t is the time vector
# y = sim[1][1, :]           # example: first trajectory, first variable
# dt = sim[1].t[2] - sim[1].t[1]           # sampling interval
# N = length(y)

# Y = abs.(fft(y))

# plot(Y, title="FFT of Prey Population", xlabel="Frequency", ylabel="Magnitude", label = "FFT")
# #create vertical line at dominant frequency (dotted line with low opacity)
# vline!([dominant_freq], color=:red, label="Dominant Frequency $(round(dominant_freq, sigdigits = 2))", linewidth=2, linestyle=:dot, alpha=0.5, legend=:topright)  
# #save plot

# freqs = (0:N-1) / (N*dt)   # frequency vector

# # Ignore the zero frequency (DC component)
# peak_idx = argmax(Y[2:div(N,2)]) + 1  # +1 because we skipped the first element
# dominant_freq = freqs[peak_idx]
# period = 1 / dominant_freq

# println("Estimated period: ", period)