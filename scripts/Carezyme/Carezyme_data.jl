using Pkg
Pkg.activate("scripts//")
using OctetData, ReactionNetworks, Revise, Optimization,Optim, OptimizationOptimJL,OptimizationPolyalgorithms, ModelingToolkit,DifferentialEquations,Plots, Lux, Random, Distributions
include("../../src/hybrid_model.jl")
include("../../src/polyopt.jl")
# using OctetDataFitting

data_path = "C:\\Users\\MGAJ\\OneDrive - Danmarks Tekniske Universitet\\Novozymes\\ReactionNetworkFitting_private\\OctetPEtab\\data"
g80_1 = OctetData.Frds(joinpath(data_path, "exp_raw", "eln20gmml0080_Assay001"); include_StepType = ["ASSOC", "DISASSOC"])

model_dir = joinpath("models", "carnival_evity")
plots_dir = joinpath(model_dir, "plots")
#mkdir
if !isdir(plots_dir)
    mkdir(plots_dir)
end

frdsplits = SplitFrds(g80_1, 
        KineticsDataSplit = [:SampleID,:SampleInfo,:SampleGroup],
        ExperimentInfoSplit = []);

#find split with "Carnivality" in the key
using Dates
datestring = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
idx = findfirst(x -> occursin("CarnivalEvity", x), collect(keys(frdsplits)))
frdsplit = frdsplits[collect(keys(frdsplits))[idx]]
p1 = plot(frdsplit, title = "Raw CarnivalEvity Data", xlabel = "Time [s]", ylabel = "Signal [nm]", label = "Raw Data")
savefig(p1, joinpath(plots_dir, "raw_carnival_evity.png"))

preprocess!(frdsplits,
        baseline_correction = true,
        remove_offset = true,
        shift_time = true,
        sampling_percentage = 0.05,
        moving_avg_window = 3,
        )

frdsplit = frdsplits[collect(keys(frdsplits))[idx]]
p2 = plot(frdsplit, title = "Raw Data Processed",
     xlabel = "Time [s]", ylabel = "Signal [nm]", label = "Processed Data")
savefig(p2, joinpath(plots_dir, "processed_carnival_evity.png"))

plot(frdsplit)


function frds_to_petab_dataframe(frds)
    @variables t E(t)
    diss_time = find_dissociation_time(frds[1])
    sim_info = define_simulation_conditions(frds; column = :MolarConcentration)
    df = frd_to_PEtab_measurements(frds)
    return df, diss_time, sim_info
end
### DATA SPLIT

#find split with "CarnivalEvity" in the key
#get corresponding split. Assume frdsplits is a dictionary


###info 
sim_info = OctetData.define_simulation_conditions(frdsplit; column = :MolarConcentration)
df, diss_time, sim_info = frds_to_petab_dataframe(frdsplit)

# data_constraints, depend_on_last = ReactionNetworks.get_data_constraints(frdsplit, sim_info, u0_constraints)
df = OctetData.frd_to_PEtab_measurements(frdsplit)
df = unique(df)

# model


octet_model,u0_constraints = ReactionNetworks.one_to_N_octet_model(N=3, hydrolysis=false, bs_independence=true)
octet_odesys = convert(ODESystem, octet_model.model)
@unpack E, S_1, ES_1, S_2, ES_2, S_3, ES_3 = octet_odesys
#
@independent_variables t
@variables y(t) #response variable

s0_vals = [1.0, 1.0, 1.0, 1.0]


w_E, w_S_1, w_ES_1 ,w_S_2, w_ES_2, w_S_3, w_ES_3 = octet_model.weights
y_0 = w_S_1*s0_vals[1] + w_S_2*s0_vals[2] + w_S_3*s0_vals[3]
# response = [y ~ w_S_1 * S_1 + (w_E + w_S_1) * ES_1 + w_S_2 * S_2 + (w_E + w_S_2) * ES_2 + w_S_3 * S_3 + (w_E + w_S_3) * ES_3 - y_0]
response = [y ~ w_S_1 * S_1 + w_ES_1 * ES_1 + w_S_2 * S_2 + w_ES_2 * ES_2 + w_S_3 * S_3 + w_ES_3 * ES_3 - y_0]

##### FITTED VALUES
E_weight = 38.5
#s0_vals = [0.1, 0.1, 0.1]
# ka[1],kd[1],ka[2],kd[2],ka[3],kd[3] ka[4],kd[4]
k_vals = [0.00012093141132477341,0.043364283573102444,0.0011891522078911836,0.0014174792626764552,7.055271847120816e-6,1.0e-16 ,0.0001,0.0001]
@unpack ka_1, kd_1 , ka_2, kd_2, ka_3, kd_3= octet_odesys
# ka_1, kd_1, = octet_odesys
k_map = Dict([
    ka_1 => k_vals[1],
    kd_1 => k_vals[2],
    ka_2 => k_vals[3],
    kd_2 => k_vals[4],
    ka_3 => k_vals[5],
    kd_3 => k_vals[6],
])

defaults = Dict(
    ka_1 => k_vals[1],
    kd_1 => k_vals[2],
    ka_2 => k_vals[3],
    kd_2 => k_vals[4],
    ka_3 => k_vals[5],
    kd_3 => k_vals[6],
)
#k_vals = [0.001, 0.01]
#k_vals = [k_vals; k_vals./2; k_vals./4]
w_vals = [100+E_weight,100+E_weight,100+E_weight, 100+E_weight, E_weight, 100, 100, 100, 100]/10

weight_map = Dict([
    w_E => 0.2,
    w_S_1 => 0.2, #w_vals[2],
    w_ES_1 => 0.8,
    w_S_2 => 0.2,
    w_ES_2 => 0.2,
    w_S_3 => 0.0,
    w_ES_3 => 0.0
])


obs = Dict("signal" => y)

println("Plugging values from $sim_info manually cuz I can't change Symbol to Num")
ic_vals = Dict(
    "C1" => Dict(E=>250.0),
    "G1" => Dict(E=>15.62),
    "A1" => Dict(E=>1000.0),
    "B1" => Dict(E=>500.0),
    "E1" => Dict(E=>62.5),
    "F1" => Dict(E=>31.25),
    "D1" => Dict(E=>125.0)
)

u0map = Dict([E => 1.0, S_1 => s0_vals[1], ES_1 => 0.0, S_2 => s0_vals[2], ES_2 => 0.0, S_3 => s0_vals[3], ES_3 => 0.0])

sorted_exps = sort(collect(keys(ic_vals)))

#use the first 7 conditions for training and the last one for validation
train_idcs = reduce(.|, [df.simulation_id .== cond for cond in sorted_exps[1:1]])
# train_idcs = findall( )
val_idcs = reduce(.|, [df.simulation_id .== cond for cond in collect(keys(ic_vals))[6:6]])
test_idcs = reduce(.|, [df.simulation_id .== cond for cond in collect(keys(ic_vals))[7:7]])
#save train and test data
train_measurements = df[train_idcs, :]
val_measurements = df[val_idcs, :]  
test_measurements = df[test_idcs, :]

sampling_percentage = 0.1

known_eqs = Equation[] #none are known for sure
@named empty_sys = ODESystem(known_eqs, t, unknowns(octet_odesys), octet_model.weights[1:end], observed = response, defaults = weight_map)
n_species = length(unknowns(octet_odesys))
rbf(x) = exp.(-(x.^2))

n_nodes = 64
n_h_layers = 3
U = Lux.Chain(
    Lux.Dense(n_species,n_nodes,rbf), [Lux.Dense(n_nodes,n_nodes, rbf) for _ in 1:n_h_layers] , Lux.Dense(n_nodes,n_species)
)

# train_measurements_exp = train_measurements#[train_measurements.simulation_id .== "A1", :]
train_measurements_exp = train_measurements
# Dissociation event
diss_condition(u,even_time,integrator) = even_time == diss_time # Set the time of the event to 5.0 seconds
affect!(integrator) = integrator.u[1] = 0.0*integrator.u[1] # Set the concentration of E to 0
dcb = DiscreteCallback(diss_condition, affect!)
event = Dict(:callback => dcb, :tstops => [diss_time]) # Define the event

hmodel = HybridModel(complete(empty_sys), complete(octet_odesys), events = event)
trainpeprob = HybridPEProblem(hmodel, obs, train_measurements_exp, u0map; 
                   conditions = ic_vals, random_sampling_percentage = sampling_percentage, log_transform = false,
                   ens_alg = EnsembleSplitThreads(), l1_ratio = 0.0, alpha = 0.0, negative_penalty = 1e5,
                   force_dtmin = false)
valpeprob = HybridPEProblem(hmodel, obs, test_measurements, u0map; 
                   conditions = ic_vals, random_sampling_percentage = sampling_percentage, log_transform = false,
                   ens_alg = EnsembleSplitThreads(), l1_ratio = 0.0, alpha = 0.0, 
                   force_dtmin = true)

gt_p = init_params(hmodel)

trainpeprob.obj_func(gt_p, 1.0)


gt_p.surrogate[:ka_1] = k_vals[1]
gt_p.surrogate[:kd_1] = k_vals[2]
gt_p.surrogate[:ka_2] =0.0# k_vals[3]
gt_p.surrogate[:kd_2] = 0.0#*k_vals[4]
gt_p.surrogate[:ka_3] = 0.0#*k_vals[5]
gt_p.surrogate[:kd_3] = 0.0#*k_vals[6]

n_runs = 5# number of runs for the ensemble
max_trials = 10 
maxiters = Int(500/sampling_percentage)
# max_dist_from_gt = 0.5
lb = gt_p*eps(Float64)
ub = gt_p*eps(Float64) .+ 1.0
initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub, rng = Random.default_rng(0))

plot(trainpeprob; included_plots = [:data, :model],
     p = gt_p,
     curve_label = "Initial Guess",
     title = "CarnivalEvity",
     )

using ForwardDiff
ForwardDiff.gradient((p) -> trainpeprob.obj_func(p, 1.0), gt_p)
# initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub, rng = Random.default_rng(0))
opt_prob = Optimization.EnsembleProblem(trainpeprob; initp_samples = gt_p, adalg = Optimization.AutoForwardDiff(),
                                         random_sampling_percentage = sampling_percentage)

trace = Any[]
callback = create_callback(trainpeprob,  plot_every =1, report_every = 10, loss_upper_bound = 1e7,
                       xlabel = "Time", ylabel = "Signal", title = "Octet Simulated Data")
multi_opt_sol = Optimization.solve(opt_prob, ProgressivePolyOpt(lr = 1e-5 , n_partitions = 1), EnsembleThreads(), 
                            trajectories = 1,
                            exponential_decay = true,
                            maxiters = maxiters,
                            maxiter_BFGS = 400,
                           #  show_trace = true, show_every = 10,
                            callback = (state, l) -> callback(state, l; trace = trace))

trace_backup = []
# opt_sol = Optimization.solve(
#     remake(opt_prob; u0 = trace[end].u), Optim.BFGS(alphaguess = 01), EnsembleThreads(), trajectories = 1, maxiters = 300, 
#     callback = (state, l) -> callback(state, l; trace = trace_backup),
# )
opt_sol = Optimization.solve(
    remake(opt_prob; u0 = trace[end].u), ProgressivePolyOpt(lr = 1e-3 , n_partitions = 1), EnsembleThreads(), trajectories = 1, maxiters = 300, 
    callback = (state, l) -> callback(state, l; trace = trace_backup),
)
#save trace as jld2

using JLD2
using Dates
datestring = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

# jldsave(joinpath(plots_dir, "trace_backup_$datestring.jld2"); trace_backup)
#load trace
trace = Any[]

# trace = load(joinpath(plots_dir, "trace_2025-08-15-07-37.jld2"))["trace"]
# jldsave(joinpath(plots_dir, "trace_$datestring.jld2"); trace)


opt_sol = trace[end]
partial_trace = trace[1:end]

p3 = plot_loss([trace[1:end]], trainpeprob; val_prob = valpeprob,
         xlabel = "Iteration", ylabel = "Loss", title = "Loss Trace", legend = :topright, yscale = :log10, size = (1000, 500), margin = 5Plots.mm)
savefig(p3, joinpath(plots_dir, "loss_trace_$datestring.png"))

p4 = param_trace(trace[1:end];
            xlabel = "Iteration", ylabel = "Parameter Value", title = "Parameter Trace",
            legend = Symbol(:outer,:topleft), markersize = 4, size = (1000, 500), margin = 5Plots.mm)
savefig(p4, joinpath(plots_dir, "param_trace_.png"))



p5 = plot(trainpeprob; included_plots = [:data, :model],
     p = opt_sol[1].u#trace[end].u,
     )
savefig(p5, joinpath(plots_dir, "train_peprob_fit_$datestring.png"))


p6 = plot(valpeprob; included_plots = [:data, :model],
     p = trace[end].u,
     )
savefig(p6, joinpath(plots_dir, "val_peprob_fit_$datestring.png"))