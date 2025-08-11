include("Hydrolysis_data.jl")
include("Hydrolysis_hidden_model.jl")
include("../../src/hybrid_model.jl")
include("../../src/polyopt.jl")

using Lux


n_species = length(unknowns(sys_known))
rbf(x) = exp.(-(x.^2))
U = Lux.Chain(
    Lux.Dense(n_species,5,rbf), Lux.Dense(5,5, rbf),Lux.Dense(5,5, rbf), Lux.Dense(5,n_species)
)

hmodel = HybridModel(complete(sys_known), U; rng = rng, events = event)

rootdir = dirname(dirname(dirname(@__FILE__)))
model_dir = joinpath(rootdir , "models", "Hydrolysis", "NN", "seed_$seed")

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


alpha = 1e-1
l1_ratio = 0.0
###
gt_p = init_params(hmodel)
trainpeprob = HybridPEProblem(hmodel, obs, train_measurements_exp, u0map; 
                   conditions = ic_vals,
                   ens_alg = EnsembleSplitThreads(), l1_ratio = l1_ratio, alpha = alpha,
                   
                #    log_transform = false, 
                   force_dtmin = true)

valpeprob = HybridPEProblem(hmodel, obs, test_measurements_exp, ic_vals["cond3"];
                            conditions = ic_vals,
                            ens_alg = EnsembleSplitThreads(), l1_ratio = l1_ratio, alpha = alpha,
                            force_dtmin = true)

plot(trainpeprob; included_plots = [:data, :model],
     p = gt_p,
     curve_label = "Ground Truth",
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Signal", title = "Enzyme Dynamics Simulated Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )


n_runs = 2 # number of runs for the ensemble
max_trials = 10 
sampling_percentage = 0.2
maxiters = Int(300/sampling_percentage)
# max_dist_from_gt = 0.5
lb = gt_p*eps(Float64)
ub = gt_p*eps(Float64) .+ 1.0
initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub)
opt_prob = Optimization.EnsembleProblem(trainpeprob; initp_samples = initp_samples,
                                         random_sampling_percentage = sampling_percentage)





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


opt_sol = Optimization.solve(opt_prob, ProgressivePolyOpt(lr = 1e-2, n_partitions = 3), EnsembleSerial(), 
                            trajectories = 1,
                            maxiters = maxiters,
                            maxiter_BFGS = 300,
                           #  show_trace = true, show_every = 10,
                            callback = (state, l) -> cb(state, l; trace = trace))


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
