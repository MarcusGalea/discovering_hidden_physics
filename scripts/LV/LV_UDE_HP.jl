using Pkg
Pkg.activate("scripts//")
include("../../src/polyopt.jl")
include("LV_hidden_model.jl")
include("../../src/hybrid_model.jl")
using Optimisers
using JLD2



n_species = length(unknowns(sys_known))
rbf(x) = exp.(-(x.^2))
h = 10
U = Lux.Chain(
    Lux.Dense(n_species,h,rbf), 
    Lux.Dense(h,h,rbf),
    Lux.Dense(h,n_species)
)

hmodel = HybridModel(complete(sys_known), U; rng = rng)

rootdir = dirname(dirname(dirname(@__FILE__)))
model_dir = joinpath(rootdir , "models", "lotka_volterra", "NN", "seed_$seed")
# makedir if not exists(model_dir)
if !isdir(model_dir)
    mkpath(model_dir)
end

# xmean = mean(train_measurements_exp[train_measurements_exp.obs_id .== "prey_o", :measurement])
# ymean = mean(train_measurements_exp[train_measurements_exp.obs_id .== "predator_o", :measurement])
# equilibrium_guess = Dict(x => Int(round(xmean)), y => Int(round(ymean)))

n_runs = 1 # number of runs for the ensemble
max_trials = 10 

gt_p = init_params(hmodel)
# max_dist_from_gt = 0.5
# lb = gt_p .- max_dist_from_gt
# ub = gt_p .+ max_dist_from_gt
# initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub)*0.0
initp_samples = gt_p
initp_samples.sys .*= 0.0 
initp_samples.sys .+= 0.01 # initialize with small perturbation around 0.1
# centered_manifold_updated_params!(hmodel, initp_samples, equilibrium_guess; digits = 4)

# peprob.obj_func(gt_p, train_measurements_exp)

# Convert Dual numbers to Float64 by taking the value part


alpha = 0.0#, 1e-4
l1_ratio = 0.0 # L1 regularization ratio
random_sampling_percentage = 0.2 # Percentage of data to use for random sampling

peprob = HybridPEProblem(hmodel, obs, train_measurements_exp, u0map;
                conditions = ic_vals, force_dtmin = true, alpha = alpha, l1_ratio = l1_ratio)


plot(peprob; included_plots = [:data, :model], 
            p = initp_samples,
            )

valpeprob = HybridPEProblem(hmodel, obs, test_measurements_exp, u0map; 
                   conditions = ic_vals, force_dtmin = false, alpha = alpha, l1_ratio = l1_ratio,
                   random_sampling_percentage = random_sampling_percentage,
                   ens_alg = EnsembleSplitThreads())

opt_prob = Optimization.EnsembleProblem(peprob; initp_samples = initp_samples,
                        adalg = Optimization.AutoForwardDiff(),
                        )
trace = []
callback = create_callback(peprob,  plot_every = 50, report_every = 50, loss_upper_bound = 1e7,
                       xlabel = "Time", ylabel = "Population", title = "Lotka Volterra  Dynamics")
# for (i, trace) in enumerate(traces)

opt_sol = Optimization.solve(remake(opt_prob; u0 = initp_samples), ProgressivePolyOpt(lr = 1e-4, initial_stepnorm = 0.0001, n_partitions = 10), EnsembleSerial(),
                            trajectories = n_runs,
                                maxiters = 3000,
                                maxiter_BFGS = 0,
                                exponential_decay = true,
                                #  show_trace = true, show_every = 10,
                                callback = (state, l) -> callback(state, l; trace = trace))


plot_dir = joinpath(model_dir, "plots")
datestring = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
p3 = plot_loss([trace], trainpeprob; val_prob = valpeprob,
         xlabel = "Iteration", ylabel = "Loss", title = "Loss Trace", legend = :topright, yscale = :log10, size = (1000, 500), margin = 5Plots.mm)
savefig(p3, joinpath(plot_dir, "loss_trace_$datestring.png"))
display(p3)

p4 = param_trace(trace; ground_truth_values = gt_p,
            xlabel = "Iteration", ylabel = "Parameter Value", title = "Parameter Trace",
            legend = Symbol(:outer, :topleft), markersize = 4, size = (1000, 500), margin = 5Plots.mm)
savefig(p4, joinpath(plot_dir, "param_trace_$datestring.png"))
display(p4)
#show training fit and validation fit
p5 = plot(trainpeprob; included_plots = [:data, :model],
     p = opt_sol_bfgs[1].u,
     curve_label = "Estimate",
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Population", title = "Lotka-Volterra Simulated (Training) Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )

savefig(p5, joinpath(plot_dir, "train_fit_$datestring.png"))
p6 = plot(valpeprob; included_plots = [:data, :model],
     p = opt_sol_bfgs[1].u,
     curve_label = "Estimate",
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Population", title = "Lotka-Volterra Simulated (Validation) Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )
savefig(p6, joinpath(plot_dir, "val_fit_$datestring.png"))