include("Hydrolysis_data.jl")
include("Hydrolysis_hidden_model.jl")
include("../../src/hybrid_model.jl")
include("../../src/polyopt.jl")

hmodel = HybridModel(complete(sys_known), complete(sys_unknown_gt); rng = rng, events = event)

rootdir = dirname(dirname(dirname(@__FILE__)))
model_dir = joinpath(rootdir , "models", "enzyme_dynamics", "Reg", "seed_$seed")
plot_dir = joinpath(model_dir, "plots")
#make dir if it doesn't exist
if !isdir(plot_dir)
    mkdir(plot_dir)
end


# # conditions = ic_vals
# u0_conditions = overwrite_conditions!(u0map, conditions)
# obs_funs = Dict([obs_fun.lhs =>eval(build_function(obs_fun.rhs, unknowns(hmodel.sys), parameters(hmodel.sys); expression=Val{false})) for obs_fun in observed(hmodel.sys)])

# idx_sim = Dict([cond => i for (i, cond) in enumerate(sort(collect(keys(u0_conditions))))]) #create a map from condition name to ensemble index
# idx_var = Dict([var => i for (i, var) in enumerate(unknowns(hmodel.sys))]) #create a map from variable name to index

alpha = 0.0
l1_ratio = 0.0
###
u0map
gt_p = init_params(hmodel)
trainpeprob = HybridPEProblem(hmodel, obs, train_measurements_exp, u0map; 
                   conditions = ic_vals,
                   ens_alg = EnsembleSplitThreads(), l1_ratio = l1_ratio, alpha = alpha,
                #    log_transform = false, 
                   force_dtmin = true)
trainpeprob.obj_func(gt_p, 1.0) # Test the objective function with ground truth parameters and a proportion of 1.0




loss_function(gt_p, 1.0) # Test the loss function with ground truth parameters and a proportion of 1.0












trainpeprob.obj_func(gt_p, 1.0)


using Dates
date_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

valpeprob = HybridPEProblem(hmodel, obs, test_measurements_exp, ic_vals["cond3"];
                            conditions = ic_vals,
                            ens_alg = EnsembleSplitThreads(), l1_ratio = l1_ratio, alpha = alpha,
                            force_dtmin = true)

p1 = plot(trainpeprob; included_plots = [:data, :model],
     p = gt_p,
     curve_label = "Ground Truth",
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Signal", title = "Octet Dynamics Simulated (Training) Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )

trainpeprob.obj_func(gt_p, 1.0)
     #save plot
savefig(p1, joinpath(plot_dir, "train_peprob_$date_str.png"))
p2 = plot(valpeprob; included_plots = [:data, :model],
     p = gt_p,
     curve_label = "Ground Truth",
     colors = [:blue, :green, :orange, :purple, :magenta, :brown], legend = :topright,
     xlabel = "Time", ylabel = "Signal", title = "Octet Dynamics Simulated (Validation) Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )
savefig(p2, joinpath(plot_dir, "val_peprob_$date_str.png"))

# p2 = plot_hidden_dynamics(trainpeprob; 

n_runs = 5 # number of runs for the ensemble
max_trials = 10 
sampling_percentage = 0.2
maxiters = Int(400/sampling_percentage)
# max_dist_from_gt = 0.5
lb = gt_p*eps(Float64)
ub = gt_p*eps(Float64) .+ 1.0
initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub, rng = Random.default_rng(0))
opt_prob = Optimization.EnsembleProblem(trainpeprob; initp_samples = initp_samples, adalg = Optimization.AutoForwardDiff(),
                                         random_sampling_percentage = sampling_percentage)

trace = Any[]
callback = create_callback(trainpeprob,  plot_every = 20, report_every = 30, loss_upper_bound = 1e7,
                       xlabel = "Time", ylabel = "Signal", title = "Octet Simulated Data")
multi_opt_sol = Optimization.solve(opt_prob, ProgressivePolyOpt(lr = 1e-1, n_partitions = 3), EnsembleThreads(), 
                            trajectories = n_runs,
                            maxiters = maxiters,
                            maxiter_BFGS = 400,
                           #  show_trace = true, show_every = 10,
                            callback = (state, l) -> callback(state, l; trace = nothing))
print(trainpeprob.obj_func(gt_p, 1.0))
best_run = argmin([sol.objective for sol in multi_opt_sol])
p_est = multi_opt_sol[best_run].u
plot(valpeprob; included_plots = [:data, :model],
     p = multi_opt_sol[best_run].u,
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Signal", title = "Enzyme Dynamics Simulated Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )


# multi_opt_sols = []
trace = []
callback = create_callback(trainpeprob,  plot_every = 30, report_every = 30, loss_upper_bound = 1e7,
                       xlabel = "Time", ylabel = "Signal", title = "Octet Simulated Data")
# for (i, trace) in enumerate(traces)
opt_sol = Optimization.solve(remake(opt_prob; u0 = initp_samples[best_run]), ProgressivePolyOpt(lr = 1e-2, n_partitions = 3), EnsembleSerial(),
                            trajectories = 1,
                                maxiters = maxiters,
                                maxiter_BFGS = 300,
                                #  show_trace = true, show_every = 10,
                                callback = (state, l) -> callback(state, l; trace = trace))
# end


p3 = plot_loss([trace], trainpeprob; val_prob = valpeprob,
         xlabel = "Iteration", ylabel = "Loss", title = "Loss Trace", legend = :topright, yscale = :log10)
savefig(p3, joinpath(plot_dir, "loss_trace_$date_str.png"))

p4 = param_trace(trace; ground_truth_values = gt_p,
            xlabel = "Iteration", ylabel = "Parameter Value", title = "Parameter Trace",
            legend = Symbol(:outer, :topright), markersize = 4)
savefig(p4, joinpath(plot_dir, "param_trace_$date_str.png"))

#show training fit and validation fit
p5 = plot(trainpeprob; included_plots = [:data, :model],
     p = multi_opt_sol[best_run].u,
     curve_label = "Estimate",
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Signal", title = "Enzyme Dynamics Simulated (Training) Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )

savefig(p5, joinpath(plot_dir, "train_fit_$date_str.png"))
p6 = plot(valpeprob; included_plots = [:data, :model],
     p = multi_opt_sol[best_run].u,
     curve_label = "Estimate",
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Signal", title = "Enzyme Dynamics Simulated (Validation) Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )
savefig(p6, joinpath(plot_dir, "val_fit_$date_str.png"))

p_est = opt_sol.u