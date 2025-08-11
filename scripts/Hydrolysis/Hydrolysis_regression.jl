include("Hydrolysis_data.jl")
include("Hydrolysis_hidden_model.jl")
include("../../src/hybrid_model.jl")
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
     colors = [:blue, :green, :orange, :purple, :magenta, :brown], legend = Symbol(:outer, :topright),
     xlabel = "Time", ylabel = "Signal", title = "Enzyme Dynamics Simulated Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )



# p2 = plot_hidden_dynamics(trainpeprob; p_est = gt_p, size = (1200, 500), bottom_margin= 10Plots.mm)


n_runs = 20 # number of runs for the ensemble
max_trials = 10 
sampling_percentage = 0.2
maxiters = Int(400/sampling_percentage)
# max_dist_from_gt = 0.5
lb = gt_p*eps(Float64)
ub = gt_p*eps(Float64) .+ 1.0
initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub)
opt_prob = Optimization.EnsembleProblem(trainpeprob; initp_samples = initp_samples, adalg = Optimization.AutoForwardDiff(),
                                         random_sampling_percentage = sampling_percentage)

trace = Any[]
cb = create_callback(trainpeprob,  plot_every = 1, report_every = 30, loss_upper_bound = 1e7,
                       xlabel = "Time", ylabel = "Signal", title = "Octet Simulated Data")
opt_sol = Optimization.solve(opt_prob, ProgressivePolyOpt(lr = 1e-1, n_partitions = 3), EnsembleThreads(), 
                            trajectories = n_runs,
                            maxiters = maxiters,
                            maxiter_BFGS = 400,
                           #  show_trace = true, show_every = 10,
                            callback = (state, l) -> cb(state, l; trace = trace))

best_run = argmin([sol.objective for sol in opt_sol])

plot(val_peprob; included_plots = [:data, :model],
     p = opt_sol[best_run].u,
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Signal", title = "Enzyme Dynamics Simulated Data",
     conds_ids = ["cond1, cond2"],
     data_proportion = 1.0,
     )


opt_sols = []
traces = Any[[] for _ in 1:n_runs]
cb = create_callback(trainpeprob,  plot_every = 30, report_every = 30, loss_upper_bound = 1e7,
                       xlabel = "Time", ylabel = "Signal", title = "Octet Simulated Data")
for (i, trace) in enumerate(traces)
    opt_sol = Optimization.solve(remake(opt_prob; u0 = initp_samples[i]), ProgressivePolyOpt(lr = 1e-2, n_partitions = 3), EnsembleSerial(),
                                trajectories = 1,
                                 maxiters = maxiters,
                                 maxiter_BFGS = 300,
                                 #  show_trace = true, show_every = 10,
                                 callback = (state, l) -> cb(state, l; trace = trace))
end


p1 = plot_loss(traces[1:1], trainpeprob; val_prob = val_peprob,
         xlabel = "Iteration", ylabel = "Loss", title = "Loss Trace", legend = :topright, yscale = :log10)



function param_trace(trace; param_idcs = collect(1:length(trace[1].u)), ground_truth_values = nothing, kwargs...)
    p1 = plot(; kwargs...)
    param_matrix = zeros(length(trace), length(param_idcs))
    colors = [:red, :blue, :green, :orange, :purple, :magenta, :brown]
    label = last.(split.(labels(trace[1].u), "."))
    for idx in param_idcs
        if !isnothing(ground_truth_values)
            hline!(p1, [ground_truth_values[idx]], label = label[idx].*"_true", color = colors[idx % length(colors) + 1], linestyle = :dash, linewidth = 2, alpha = 0.5)
        end
        for (i, state) in enumerate(trace)
            param_matrix[i, idx] = state.u[idx]
        end
        plot!(p1, param_matrix[:, idx], label = label[idx], color = colors[idx % length(colors) + 1], linewidth = 2, markersize = 4, alpha = 0.8)
    end
    return p1
end

p1 = param_trace(trace; ground_truth_values = gt_p,
            xlabel = "Iteration", ylabel = "Parameter Value", title = "Parameter Trace",
            legend = Symbol(:outer, :topright), markersize = 4)