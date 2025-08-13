using Pkg
Pkg.activate("scripts//")
include("../../src/polyopt.jl")
include("LV_hidden_model.jl")
include("../../src/hybrid_model.jl")
include("LV_sampling.jl")
using Optimisers
using JLD2


rootdir = dirname(dirname(dirname(@__FILE__)))
model_dir = joinpath(rootdir , "models", "lotka_volterra", "SINDy", "seed_$seed")
#makedir if not exists(model_dir)
if !isdir(model_dir)
    mkpath(model_dir)
end

### Create SINDy system
species = unknowns(sys_known)
unknown_basis = polynomial_basis(species, 2) # Create the unknown equations
@parameters Ξ[1:length(known_eqs), 1:length(unknown_basis)]
default_params = Dict([Ξ[i, j] => 0.0 for i in 1:length(known_eqs), j in 1:length(unknown_basis)])
default_params[Ξ[1,5]] = - params[β]
default_params[Ξ[2,5]] = params[δ] # Set the default parameters for the unknown equations

uk_eqs = create_unknown_eqs(sys_known, unknown_basis; Ξ = hcat(Ξ))
@named sys_SINDy = ODESystem(uk_eqs, t, defaults = default_params)
sys_SINDy = complete(sys_SINDy)


hmodel = HybridModel(sys_known, sys_SINDy; rng = rng)


# ### OPTIMIZATION
# n_initial_conditions = 1
# #CHANGE NUMBER OF INITIAL CONDITIONS HERE
# train_measurements_exp = train_measurements[sum([train_measurements.simulation_id .== "cond$i" for i in 1:n_initial_conditions]), :]
# test_measurements_exp = test_measurements[sum([test_measurements.simulation_id .== "cond$i" for i in 1:n_initial_conditions]), :]
# # batch_size = 32 # Batch size for the optimization
# obs = Dict("prey_o" => x, "predator_o" => y)
# u0map = Dict([x => 40.0, y => 9.0])
# ic_vals = Dict(["cond$i" => Dict([var => ic[j] for (j, var) in enumerate(unknowns(sys_known))]) for (i, ic) in enumerate(initial_conditions[1:1])])






# xmean = mean(train_measurements_exp[train_measurements_exp.obs_id .== "prey_o", :measurement])
# ymean = mean(train_measurements_exp[train_measurements_exp.obs_id .== "predator_o", :measurement])
# equilibrium_guess = Dict(x => Int(round(xmean)), y => Int(round(ymean)))

n_runs = 1 # number of runs for the ensemble
max_trials = 10 
batch_size = 32 # Batch size for the optimization
gt_p = init_params(hmodel)
max_dist_from_gt = 0.5
lb = gt_p .- max_dist_from_gt
ub = gt_p .+ max_dist_from_gt
initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub)*0.0
# centered_manifold_updated_params!(hmodel, initp_samples, equilibrium_guess; digits = 4)

# peprob.obj_func(gt_p, train_measurements_exp)

# Convert Dual numbers to Float64 by taking the value part


alpha = 1e-1
l1_ratio = 0.5 # L1 regularization ratio

peprob = HybridPEProblem(hmodel, obs, train_measurements_exp, u0map;
                conditions = ic_vals, force_dtmin = true, alpha = alpha, l1_ratio = l1_ratio)

valpeprob = HybridPEProblem(hmodel, obs, test_measurements_exp, u0map; 
                   conditions = ic_vals, force_dtmin = true, alpha = alpha, l1_ratio = l1_ratio,
                   ens_alg = EnsembleSplitThreads())

# label = hcat(string.(unknowns(peprob.model.sys))...)
# data = peprob.measurements.measurement
# timedata = peprob.measurements.time
# loss_upper_bound = 1e7
# report_every = 10
# plot_every = 30
# plotsdir = joinpath(model_dir, "plots")

opt_prob = Optimization.EnsembleProblem(peprob; initp_samples = initp_samples,
                        adalg = Optimization.AutoForwardDiff(),
                        )
trace = []
callback = create_callback(trainpeprob,  plot_every = 50, report_every = 50, loss_upper_bound = 1e7,
                       xlabel = "Time", ylabel = "Signal", title = "Octet Simulated Data")
# for (i, trace) in enumerate(traces)
opt_sol = Optimization.solve(remake(opt_prob; u0 = initp_samples), ProgressivePolyOpt(lr = 1e-4, n_partitions = 10), EnsembleSerial(),
                            trajectories = 1,
                                maxiters = 3000,
                                maxiter_BFGS = 0,
                                exponential_decay = false,
                                #  show_trace = true, show_every = 10,
                                callback = (state, l) -> callback(state, l; trace = trace))




plot_dir = joinpath(model_dir, "plots")
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



using Dates
date_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
println("Date string: $date_str")

save_path = joinpath(model_dir, "ensembleoptsol_$(date_str).jld2")
save_path_hyperparams = joinpath(model_dir, "hyperparams_$(date_str).jld2")
save_path_allruns = joinpath(model_dir, "all_runs_$(date_str)") 
#make directory if it does not exist
if !isdir(save_path_allruns)
    mkpath(save_path_allruns)
end

using HyperTuning
scenario = Scenario(lr_exponent = (-7.0..1.0),
                    alpha_exponent = (-7.0..3.0),
                    l1_ratio = (0.0..1.0),
                    max_trials = max_trials,
                    sampler = GridSampler(),
)


peprob = HybridPEProblem(hmodel, obs, train_measurements_exp, u0map; 
                conditions = ic_vals
                )#, batch_size = batch_size,)# OPTIONS HERE ARE UNECESSARY
cb = create_callback(peprob; plot_every = 0, report_every = 0, loss_upper_bound = 1e-7,)
options = Dict(:maxiters => 2000, :show_every => 1, :callback => cb, :trajectories => n_runs)


function hypertuning_objective(trial)
    @unpack lr_exponent, alpha_exponent, l1_ratio = trial
    alpha = 10.0 ^ alpha_exponent
    lr = 10.0 ^ lr_exponent
    
    peprob = HybridPEProblem(hmodel, obs, train_measurements_exp, u0map; 
                   conditions = ic_vals, alpha = alpha, l1_ratio = l1_ratio,
                   )#, batch_size = batch_size,)# OPTIONS HERE ARE UNECESSARY

    opt_prob = Optimization.EnsembleProblem(peprob; initp_samples = initp_samples,
                            adalg = Optimization.AutoForwardDiff())
    ensembleoptsol = Optimization.solve(opt_prob, ProgressivePolyOpt(lr = lr, n_partitions = 10), EnsembleThreads(); options...)

    best_run = argmin([valpeprob.obj_func(sol.u, 1.0) for sol in ensembleoptsol])
    val_loss = valpeprob.obj_func(ensembleoptsol[best_run].u, 1.0)    
    #save each run to a file
    trial_string = join(["$(hyperp)_$(round(val,sigdigits = 4))_" for (hyperp, val) in trial.values])
    current_save_path = joinpath(save_path_allruns, "val_loss_$(round(val_loss, sigdigits = 3))_$(trial_string).jld2")
    save(current_save_path, "ensembleoptsol", (initp = initp_samples[best_run], final_p = ensembleoptsol[best_run].u, val_loss = val_loss, train_loss = ensembleoptsol[best_run].objective))
    return val_loss
end

# trial = hypertune_res.best_trial.trials[1]
# loading_path = joinpath(save_path_allruns, "loss_1.0e16_lr_exponent_-3.558_l1_ratio_0.6935_alpha_exponent_-3.26_.jld2")
# loaded_sol = load(loading_path, "ensembleoptsol")


hypertune_res = HyperTuning.optimize(hypertuning_objective, scenario)
@info "Best loss: $(hypertune_res.best_trial)"
save(save_path_hyperparams, "hypertune_res", hypertune_res)

#load hypertune_res
# ensemble_opt = load(current_save_path, "ensembleoptsol")


# # #REDO BEST RUN WITH BEST PARAMETERS
# # @unpack lr_exponent, alpha_exponent, l1_ratio = hypertune_res
# # alpha_exponent = 1e-1
# # lr_exponent = -3.0
# # alpha = 10.0 ^ alpha_exponent
# # lr = 10.0 ^ lr_exponent
# # l1_ratio = 0.5
# # opt_prob = Optimization.EnsembleProblem(peprob; initp_samples = initp_samples,
# #                           alpha = alpha, l1_ratio = l1_ratio, adalg = Optimization.AutoForwardDiff())
# # ensembleoptsol = Optimization.solve(opt_prob, PolyOptAdamBFGS(lr=lr), EnsembleThreads(), trajectories = n_runs; options...)
# # #save ensembleoptsol to file
# # #get date

# @info "Saving ensemble optimization solution to $save_path and hyperparameter tuning results to $save_path_hyperparams"
# save(save_path, "ensembleoptsol", ensembleoptsol)
