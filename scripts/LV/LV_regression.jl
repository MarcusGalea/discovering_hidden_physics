include("LV_hidden_model.jl")
include("../../src/hybrid_model.jl")
include("../../src/polyopt.jl")
# hmodel = HybridModel(sys_known, sys_unknown_gt; rng = rng)
hmodel = HybridModel(sys_known, sys_unknown_gt; rng = rng)

rootdir = dirname(dirname(dirname(@__FILE__)))
model_dir = joinpath(rootdir , "models", "lotka_volterra", "Reg", "seed_$seed")
plot_dir = joinpath(model_dir, "plots")
if !isdir(plot_dir)
    mkdir(plot_dir)
end
# using IterTools: ncycle
# using OptimizationOptimisers

alpha = 0.0 # Lasso penalty
l1_ratio = 0.5 # Lasso/Ridge ratio
# batch_size = 32 # Batch size for the optimization
#shuffle dataframe rows
trainpeprob = HybridPEProblem(hmodel, obs, train_measurements_exp, u0map; 
                   conditions = ic_vals )
valpeprob = HybridPEProblem(hmodel, obs, test_measurements_exp, ic_vals["cond3"];
                            conditions = ic_vals,
                            ens_alg = EnsembleSplitThreads(), l1_ratio = l1_ratio, alpha = alpha,
                            force_dtmin = true)

gt_p = init_params(hmodel)
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
n_runs = 1 # number of runs for the ensemble

gt_p = init_params(hmodel)
max_dist_from_gt = 1e-1
lb = gt_p .- max_dist_from_gt
ub = gt_p .+ max_dist_from_gt
initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub)
#plot the first to parameters for all samples

trainpeprob.obj_func(gt_p, 1.0)
opt_prob = Optimization.EnsembleProblem(trainpeprob; initp_samples = initp_samples, 
                          alpha = alpha, l1_ratio = l1_ratio, adalg = Optimization.AutoForwardDiff())
callback = create_callback(trainpeprob, plot_every = 20, report_every = 30, loss_upper_bound = 1e7,
                       xlabel = "Time", ylabel = "Population", title = "Lotka-Volterra Simulated Data")
ensembleoptsol = Optimization.solve(opt_prob, ProgressivePolyOpt(lr = 1e-2, n_partitions = 1), EnsembleThreads(), trajectories = n_runs;
                                    show_trace = true, show_every = 50, callback = callback)

best_run = argmin([opt.objective for opt in ensembleoptsol])
sim = simulate_solution(trainpeprob, gt_p, saveat = collect(0:0.1:100.0))
scatter(timedata[sample_idcs], data[sample_idcs, :], label=["Prey_data" "Predator_data"], xlabel="Time", ylabel="Population", title="Lotka-Volterra Model with Sampled Data", legend =:topright)
plot!(sim, label = ["Prey_model" "Predator_model"], linewidth=2, markersize=4, legend =:topright)

#save ensembleoptsol to file
save_path = joinpath(model_dir, "ensembleoptsol.jld2")
using JLD2
@info "Saving ensemble optimization solution to $save_path"
save(save_path, "ensembleoptsol", ensembleoptsol)