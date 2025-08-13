using Pkg
Pkg.activate("scripts//")
include("LV_hidden_model.jl")
include("../../src/hybrid_model.jl")
include("../../src/polyopt.jl")
include("LV_sampling.jl")
# hmodel = HybridModel(sys_known, sys_unknown_gt; rng = rng)
hmodel = HybridModel(sys_known, sys_unknown_gt; rng = rng)

rootdir = dirname(dirname(dirname(@__FILE__)))
# using Dates
# datestring = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
# #make it a possible folder
# safedatestring = replace(datestring, r"[^a-zA-Z0-9]" => "_")
# model_dir = joinpath(rootdir , "models", "lotka_volterra", "Reg", "seed_$seed", safedatestring)
# plot_dir = joinpath(model_dir, "plots")
# if !isdir(model_dir)
#     mkdir(model_dir)
# end
# if !isdir(plot_dir)
#     mkdir(plot_dir)
# end
# using IterTools: ncycle
# using OptimizationOptimisers

alpha = 0.0 # Lasso penalty
l1_ratio = 0.2 # Lasso/Ridge ratio
maxiters = 300 # maximum number of iterations for the optimization
random_sampling_percentage = 0.5
# batch_size = 32 # Batch size for the optimization
#shuffle dataframe rows
#try only with cond 1
# train_measurements_exp_2 = train_measurements_exp[train_measurements_exp.simulation_id .== "cond1", :]
trainpeprob = HybridPEProblem(hmodel, obs, train_measurements_exp, u0map; random_sampling_percentage = random_sampling_percentage,
                   conditions = ic_vals, log_transform = false)


gt_p = init_params(hmodel)
trainpeprob.obj_func(p_est, 1.0)




valpeprob = HybridPEProblem(hmodel, obs, test_measurements_exp, ic_vals["cond3"];
                            conditions = ic_vals,
                            ens_alg = EnsembleSplitThreads(), l1_ratio = l1_ratio, alpha = alpha,
                            force_dtmin = true)

# gt_p = init_params(hmodel)
p1 = plot(trainpeprob; included_plots = [:data, :model],
     p = gt_p,
     curve_label = "Ground Truth",
     legend = Symbol(:outer, :topright),
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Population", title = "Lotka-Volterra Simulated (Training) Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     size = (1000, 500)
     )
savefig(p1, joinpath(plot_dir, "train_peprob.png"))


p2 = plot(valpeprob; included_plots = [:data, :model],
     p = gt_p,
     curve_label = "Ground Truth",
     legend = Symbol(:outer, :topright),
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Population", title = "Lotka-Volterra Simulated (Validation) Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     size = (1000, 500)
     )
savefig(p2, joinpath(plot_dir, "val_peprob.png"))


# equilibrium_prey = mean(train_measurements_exp[train_measurements_exp.obs_id .== "prey_o", :].measurement)
# equilibrium_predator = mean(train_measurements_exp[train_measurements_exp.obs_id .== "predator_o", :].measurement)
# equilibrium = Dict([x => Int(round(equilibrium_prey)), y => Int(round(equilibrium_predator))])



n_runs = 1 # number of runs for the ensemble




max_dist_from_gt = 0.1
lb = gt_p .- max_dist_from_gt #*0.0 #.
ub = gt_p .+ max_dist_from_gt #.+ max_dist_from_gt

initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub) * 0.0 .+ 0.1
# centered_manifold_updated_params!(hmodel, initp_samples, equilibrium)


p1 = plot(trainpeprob; included_plots = [:data, :model],
     p = initp_samples,
     curve_label = "Ground Truth",
     legend = Symbol(:outer, :topright),
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Population", title = "Lotka-Volterra Simulated (Training) Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     size = (1000, 500)
     )



# #plot the first to parameters for all samples

# # u0map = trainpeprob.u0map
# # u0_conditions = trainpeprob.conditions
# # tspan = trainpeprob.tspan
# # u0_conditions = overwrite_conditions!(u0map, conditions)
# # #create a vector of vector of u0_conditions instead
# # idx_sim = Dict([cond => i for (i, cond) in enumerate(collect(keys(u0_conditions)))]) #create a map from condition name to ensemble index

# # u0_vec = convert_to_vector_conditions(u0_conditions, idx_sim, hmodel)



# # ensemble_prob = EnsembleProblem(hmodel, u0_conditions, tspan, p_est)
# # ens_sol = solve(ensemble_prob, trajectories = 2)
# # plot!(ens_sol)
opt_prob = Optimization.EnsembleProblem(trainpeprob; initp_samples = initp_samples, 
                        adalg = Optimization.AutoForwardDiff())
# callback = create_callback(trainpeprob, plot_every = 1, report_every = 30, loss_upper_bound = 1e7,
#                        xlabel = "Time", ylabel = "Population", title = "Lotka-Volterra Simulated Data")
# ensembleoptsol = Optimization.solve(opt_prob, ProgressivePolyOpt(lr = 1e-3, n_partitions = 10), EnsembleThreads(), trajectories = n_runs;
#                                     callback = callback, maxiters = Int(2000/random_sampling_percentage),)

# # best_run = argmin([opt.objective for opt in ensembleoptsol])
# # best_p = ensembleoptsol[best_run].u
# best_run = 1

# multi_opt_sols = []
trace = []
callback = create_callback(trainpeprob,  plot_every = 50, report_every = 50, loss_upper_bound = 1e7,
                       xlabel = "Time", ylabel = "Signal", title = "Octet Simulated Data")
# for (i, trace) in enumerate(traces)
opt_sol = Optimization.solve(remake(opt_prob; u0 = initp_samples), ProgressivePolyOpt(lr = 1e-3, n_partitions = 1), EnsembleSerial(),
                            trajectories = 1,
                                maxiters = 1500,
                                maxiter_BFGS = 300,
                                exponential_decay = true,
                                #  show_trace = true, show_every = 10,
                                callback = (state, l) -> callback(state, l; trace = trace))

# # #save ensembleoptsol to file


# save_path = joinpath(model_dir, "ensembleoptsol.jld2")
# using JLD2
# @info "Saving ensemble optimization solution to $save_path"
# save(save_path, "ensembleoptsol", ensembleoptsol)

datestring = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
p3 = plot_loss([trace], trainpeprob; val_prob = valpeprob,
         xlabel = "Iteration", ylabel = "Loss", title = "Loss Trace", legend = :topright, yscale = :log10, size = (1000, 500), margin = 5Plots.mm)
savefig(p3, joinpath(plot_dir, "loss_trace_$datestring.png"))

p4 = param_trace(trace; ground_truth_values = gt_p,
            xlabel = "Iteration", ylabel = "Parameter Value", title = "Parameter Trace",
            legend = :topleft, markersize = 4, size = (1000, 500), margin = 5Plots.mm)
savefig(p4, joinpath(plot_dir, "param_trace_.png"))

#show training fit and validation fit
p5 = plot(trainpeprob; included_plots = [:data, :model],
     p = opt_sol[1].u,
     curve_label = "Estimate",
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Signal", title = "Enzyme Dynamics Simulated (Training) Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )

savefig(p5, joinpath(plot_dir, "train_fit_$datestring.png"))
p6 = plot(valpeprob; included_plots = [:data, :model],
     p = opt_sol[1].u,
     curve_label = "Estimate",
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Signal", title = "Enzyme Dynamics Simulated (Validation) Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )
savefig(p6, joinpath(plot_dir, "val_fit_$datestring.png"))
