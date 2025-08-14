include("Hydrolysis_data.jl")
include("Hydrolysis_hidden_model.jl")
include("../../src/hybrid_model.jl")
include("../../src/polyopt.jl")

using Lux


n_species = length(unknowns(sys_known))
rbf(x) = exp.(-(x.^2))
n_nodes = 5
n_h_layers = 2
U = Lux.Chain(
    Lux.Dense(n_species,n_nodes,rbf), [Lux.Dense(n_nodes,n_nodes, rbf) for _ in 1:n_h_layers] , Lux.Dense(n_nodes,n_species)
)

hmodel = HybridModel(complete(sys_known), U; rng = rng, events = event)

rootdir = dirname(dirname(dirname(@__FILE__)))
model_dir = joinpath(rootdir , "models", "Hydrolysis", "NN", "seed_$seed")
plot_dir = joinpath(model_dir, "plots")
### OPTIMIZATION
#if no plotdir exists, create it
if !isdir(plot_dir)
    mkdir(plot_dir)
end

alpha = 1e-7 #0.0#
l1_ratio = 0.0
random_sampling_percentage = 0.2
sampling_percentage = 0.2   


initp_samples = init_params(hmodel)
###

trainpeprob = HybridPEProblem(hmodel, obs, train_measurements_exp, u0map; 
                   conditions = ic_vals, random_sampling_percentage = random_sampling_percentage,
                   ens_alg = EnsembleSplitThreads(), l1_ratio = l1_ratio, alpha = alpha,
                #    log_transform = false, 
                   force_dtmin = true)

valpeprob = HybridPEProblem(hmodel, obs, test_measurements_exp, ic_vals["cond3"];
                            conditions = ic_vals,
                            ens_alg = EnsembleSplitThreads(), l1_ratio = l1_ratio, alpha = alpha,
                            force_dtmin = true)

plot(trainpeprob; included_plots = [:data, :model],
     p = initp_samples,
     curve_label = "Ground Truth",
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Signal", title = "Enzyme Dynamics Simulated Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )


n_runs = 10 # number of runs for the ensemble
# max_trials = 10 
# sampling_percentage = 0.2
gt_p = init_params(hmodel)
maxiters = Int(300/sampling_percentage)
# # max_dist_from_gt = 0.5
lb = gt_p*eps(Float64)
ub = gt_p*eps(Float64) .+ 1.0
initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub)



opt_prob = Optimization.EnsembleProblem(trainpeprob; initp_samples = initp_samples,)
opt_sol = Optimization.solve(opt_prob, ProgressivePolyOpt(lr = 1e-1, n_partitions = 3, initial_stepnorm = 1e-4), EnsembleSerial(),
                            trajectories = n_runs,
                                maxiters = maxiters,
                                maxiter_BFGS = 300,
                                #  show_trace = true, show_every = 10,
                                callback = (state, l) -> callback(state, l; trace = trace))




# function param_trace(res, param_idx; ground_truth_value = nothing, run_idcs = collect(1:length(res.runs)),kwargs...)
#     p1 = plot(; kwargs...)
#     runs = res.runs[run_idcs]
#     colors = [:blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :brown]
#     markers = [:v, :cross, :star, :triangle, :x]
#     if ground_truth !== nothing
#         hline!(p1, [ground_truth_value], label = "Ground Truth", color = :red, linestyle = :dash, linewidth = 6, alpha = 0.5)
#     end
#     for (i, run) in enumerate(runs)
#         p0 = run.x0
#         color = colors[i % length(colors) + 1]
#         marker = markers[i % length(markers) + 1]
#         param_trace = hcat(run.xtrace ...)[param_idx,:]
#         scatter!(p1, param_trace, label = "Run $(run_idcs[i])", markersize = 4, color = color, marker = marker, alpha = 0.8)
#         #plot the line connecting the points
#         plot!(p1, param_trace, label = "", linewidth = 1, markersize = 2, color = color, alpha = 0.8)
#     end
#     return p1
# end                                         


best_run = argmin([sol.objective for sol in opt_sol])
trace = []

callback = create_callback(trainpeprob,  plot_every = 30, report_every = 30, loss_upper_bound = 1e7,
                       xlabel = "Time", ylabel = "Signal", title = "Octet Simulated Data")
# for (i, trace) in enumerate(traces)
opt_sol = Optimization.solve(remake(opt_prob; u0 = initp_samples[best_run]), ProgressivePolyOpt(lr = 1e-1, n_partitions = 3, initial_stepnorm = 1e-4), EnsembleSerial(),
                            trajectories = 1,
                                maxiters = maxiters,
                                maxiter_BFGS = 0,
                                #  show_trace = true, show_every = 10,
                                callback = (state, l) -> callback(state, l; trace = trace))

# opt_sols = []
# traces = Any[[] for _ in range(1, n_runs)]
# cb = create_callback(trainpeprob,  plot_every = 30, report_every = 30, loss_upper_bound = 1e7,
#                        xlabel = "Time", ylabel = "Signal", title = "Octet Simuluated Data")
# for (i, trace) in enumerate(traces)
#     opt_sol = Optimization.solve(opt_prob, ProgressivePolyOpt(lr = 1e-2, n_partitions = 3), EnsembleSerial(),
#                                  maxiters = maxiters,
#                                  maxiter_BFGS = 300,
#                                  #  show_trace = true, show_every = 10,
#                                  callback = (state, l) -> cb(state, l; trace = trace))
#     push!(opt_sols, opt_sol)
# end

using Dates
plot_dir = joinpath(model_dir, "plots")
datestring = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
p3 = plot_loss([trace], trainpeprob; val_prob = valpeprob,
         xlabel = "Iteration", ylabel = "Loss", title = "Loss Trace", legend = :topright, yscale = :log10, size = (1000, 500), margin = 5Plots.mm)
savefig(p3, joinpath(plot_dir, "loss_trace_$datestring.png"))
display(p3)

p4 = param_trace(trace; ground_truth_values = gt_p,param_idcs = [1,2],
            xlabel = "Iteration", ylabel = "Parameter Value", title = "Parameter Trace",
            legend = Symbol(:outer, :topleft), markersize = 4, size = (1000, 500), margin = 5Plots.mm)
savefig(p4, joinpath(plot_dir, "param_trace_$datestring.png"))
display(p4)
#show training fit and validation fit
p5 = plot(trainpeprob; included_plots = [:data, :model],
     p = opt_sol[1].u,
     curve_label = "Estimate",
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Population", title = "Lotka-Volterra Simulated (Training) Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )

savefig(p5, joinpath(plot_dir, "train_fit_$datestring.png"))
p6 = plot(valpeprob; included_plots = [:data, :model],
     p = opt_sol[1].u,
     curve_label = "Estimate",
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Population", title = "Lotka-Volterra Simulated (Validation) Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )
savefig(p6, joinpath(plot_dir, "val_fit_$datestring.png"))

datestring = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
save_path_allruns = joinpath(model_dir, "all_runs_$(datestring)")
#make directory if it does not exist
if !isdir(save_path_allruns)
    mkpath(save_path_allruns)
end




n_species = length(unknowns(sys_known))
rbf(x) = exp.(-(x.^2))
n_nodes = 5
n_h_layers = 2
U = Lux.Chain(
    Lux.Dense(n_species,n_nodes,rbf), [Lux.Dense(n_nodes,n_nodes, rbf) for _ in 1:n_h_layers] , Lux.Dense(n_nodes,n_species)
)

hmodel = HybridModel(complete(sys_known), U; rng = rng, events = event)
save_path_allruns = joinpath(model_dir, "all_runs_$(datestring)")
#

max_trials = 81
using HyperTuning
scenario = Scenario(lr_exponent = (-4.0.. -1.0),
                    alpha_exponent = (-7.0.. -1.0),
                    n_h_layer = (1.. 4),
                    n_nodes_exponent = (3.. 6),
                    max_trials = max_trials,
                    sampler = GridSampler(),
)


function hypertuning_objective(trial)
    @unpack lr_exponent, alpha_exponent, n_h_layer, n_nodes_exponent = trial
    alpha = 10.0 ^ alpha_exponent
    lr = 10.0 ^ lr_exponent
    n_nodes = 2 ^ n_nodes_exponent
    U = Lux.Chain(
    Lux.Dense(n_species,n_nodes,rbf), [Lux.Dense(n_nodes,n_nodes, rbf) for _ in 1:n_h_layer] , Lux.Dense(n_nodes,n_species)
    )

    hmodel = HybridModel(complete(sys_known), U; rng = rng, events = event)

    gt_p = init_params(hmodel)
    maxiters = Int(300/sampling_percentage)
    # # max_dist_from_gt = 0.5
    lb = gt_p*eps(Float64)
    ub = gt_p*eps(Float64) .+ 1.0
    initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub)


    trainpeprob = HybridPEProblem(hmodel, obs, train_measurements_exp, u0map; 
                   conditions = ic_vals, random_sampling_percentage = random_sampling_percentage,
                   ens_alg = EnsembleSplitThreads(), l1_ratio = 0.0, alpha = alpha,
                #    log_transform = false, 
                   force_dtmin = true)

                   
    valpeprob = HybridPEProblem(hmodel, obs, test_measurements_exp, ic_vals["cond3"];
                                conditions = ic_vals,
                                ens_alg = EnsembleSplitThreads(), l1_ratio = 0.0, alpha = alpha,
                                force_dtmin = true)
    opt_prob = Optimization.EnsembleProblem(trainpeprob; initp_samples = initp_samples, adalg = Optimization.AutoForwardDiff(),
                                         random_sampling_percentage = random_sampling_percentage)
    opt_sol = Optimization.solve(opt_prob, ProgressivePolyOpt(lr = lr, n_partitions = 4, initial_stepnorm = 1e-4), EnsembleSerial(),
                            trajectories = 1,
                                maxiters = maxiters,
                                maxiter_BFGS = 0,
                                #  show_trace = true, show_every = 10,
                            )

    train_loss = round(minimum([sol.objective for sol in opt_sol]),sigdigits = 3)
    val_loss = round(minimum([valpeprob.obj_func(sol.u, 1.0) for sol in opt_sol]),sigdigits = 3)
    config_str = "val_$(val_loss)_train_$(train_loss)_alpha_$(alpha)_lr_$(lr)_n_h_layer_$(n_h_layer)_n_nodes_$(n_nodes)_seed_$(seed)"
    #save config string to file in savepath all file
    open(joinpath(save_path_allruns, "hyperparams.txt"), "a") do file
        write(file, config_str * "\n")
    end
    return val_loss
end

hypertune_res = HyperTuning.optimize(hypertuning_objective, scenario)

