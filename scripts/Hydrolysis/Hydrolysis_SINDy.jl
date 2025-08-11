include("Hydrolysis_data.jl")
include("Hydrolysis_hidden_model.jl")
include("../../src/hybrid_model.jl")
include("../../src/SINDy_methods.jl")
include("../../src/polyopt.jl")

hmodel = HybridModel(complete(sys_known), complete(sys_unknown_gt); rng = rng, events = event)

rootdir = dirname(dirname(dirname(@__FILE__)))
model_dir = joinpath(rootdir , "models", "enzyme_dynamics", "Reg", "seed_$seed")


### OPTIMIZATION
n_initial_conditions = 3
#CHANGE NUMBER OF INITIAL CONDITIONS HERE

# batch_size = 32 # Batch size for the optimization
@unpack E, S, ES, P, v = sys_known
obs = Dict("v" => v)#"E" => E, "S" => S, "ES" => ES, 
u0map = Dict([E => 10.0, S => 1.0, ES => 0.0, P => 0.0])
ic_vals = Dict(["cond$i" => Dict([var => ic[j] for (j, var) in enumerate(unknowns(sys_known))]) for (i, ic) in enumerate(initial_conditions[1:n_initial_conditions])])
included_exp = (df) -> reduce(.|, [(df.simulation_id .== "cond$i") .& (df.obs_id .== obsvar)
                               for i in 1:n_initial_conditions for obsvar in keys(obs)])
train_measurements_exp = train_measurements[included_exp(train_measurements), :]
test_measurements_exp = test_measurements[included_exp(test_measurements), :]


alpha = 1e-2
l1_ratio = 0.0
###
using SciMLSensitivity
gt_p = init_params(hmodel)
trainpeprob = HybridPEProblem(hmodel, obs, train_measurements_exp, u0map; 
                   conditions = ic_vals,
                   ens_alg = EnsembleSplitThreads(), l1_ratio = l1_ratio, alpha = alpha,
                #    log_transform = false, 
                sensealg = InterpolatingAdjoint(),
                   force_dtmin = true)

val_peprob = HybridPEProblem(hmodel, obs, test_measurements_exp, ic_vals["cond3"];
                            conditions = ic_vals,
                            ens_alg = EnsembleSplitThreads(), l1_ratio = l1_ratio, alpha = alpha,
                            force_dtmin = true)

plot(val_peprob; included_plots = [:data, :model],
     p = gt_p,
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     
     xlabel = "Time", ylabel = "Signal", title = "Enzyme Dynamics Simulated Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )


n_runs = 10 # number of runs for the ensemble
max_trials = 10 
sampling_percentage = 0.2
maxiters = Int(300/sampling_percentage)
# max_dist_from_gt = 0.5
lb = gt_p*eps(Float64)
ub = gt_p*eps(Float64) .+ 1.0
initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub)
opt_prob = Optimization.EnsembleProblem(trainpeprob; initp_samples = initp_samples, adalg = Optimization.AutoZygote(),
                                         random_sampling_percentage = sampling_percentage)

                                         
cb, trace = create_callback(trainpeprob,  plot_every = 30, report_every = 30, loss_upper_bound = 1e7,
                       xlabel = "Time", ylabel = "Signal", title = "Octet Simuluated Data", save_trace = true)
opt_sol = Optimization.solve(opt_prob, ProgressivePolyOpt(lr = 1e-2, n_partitions = 1), EnsembleThreads(), 
                            trajectories = n_runs,
                            maxiters = maxiters,
                            maxiter_BFGS = 300,
                           #  show_trace = true, show_every = 10,
                            callback = cb)