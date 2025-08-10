include("Enzyme_OctetData.jl")
include("Enzyme_hidden_model.jl")
include("../../src/hybrid_model.jl")
include("../../src/polyopt.jl")

hmodel = HybridModel(complete(sys_known), complete(sys_unknown_gt); rng = rng, events = event)

rootdir = dirname(dirname(dirname(@__FILE__)))
model_dir = joinpath(rootdir , "models", "enzyme_dynamics", "Reg", "seed_$seed")

alpha = 0.0 # Lasso penalty
l1_ratio = 0.5 # Lasso/Ridge ratio


### OPTIMIZATION
n_initial_conditions = 3
#CHANGE NUMBER OF INITIAL CONDITIONS HERE

# batch_size = 32 # Batch size for the optimization
@unpack E, S, ES, v = sys_known
obs = Dict("E" => E, "S" => S, "ES" => ES, "v" => v)
u0map = Dict([E => 10.0, S => 1.0, ES => 0.0])
ic_vals = Dict(["cond$i" => Dict([var => ic[j] for (j, var) in enumerate(unknowns(sys_known))]) for (i, ic) in enumerate(initial_conditions[1:n_initial_conditions])])
included_exp = (df) -> reduce(.|, [(df.simulation_id .== "cond$i") .& (df.obs_id .== obsvar)
                               for i in 1:n_initial_conditions for obsvar in keys(obs)])
train_measurements_exp = train_measurements[included_exp(train_measurements), :]
test_measurements_exp = test_measurements[included_exp(test_measurements), :]

###
trainpeprob = HybridPEProblem(hmodel, obs, train_measurements, u0map; 
                   conditions = ic_vals,
                   ens_alg = EnsembleSplitThreads(),
                #    log_transform = false, 
                   force_dtmin = true)
sim = simulate_solution(trainpeprob, init_params(hmodel))
plot(sim)


initp_samples = init_params(hmodel)

perturbation = 1e+2
plot(trainpeprob; included_plots = [:data, :model],
     p = initp_samples,
     colors = [:blue, :green, :orange, :purple, :magenta, :brown],
     xlabel = "Time", ylabel = "Signal", title = "Enzyme Dynamics Simulated Data",
     conds_ids = ["cond1"],
     obs_ids = ["E", "S", "ES", "v"]
     )
    
plot(sim, legend = :topright)

valpeprob = HybridPEProblem(hmodel, obs, test_measurements_exp, u0map; 
                   conditions = ic_vals,
                   ens_alg = EnsembleSplitThreads())



# function create_callback(peprob::HybridPEProblem;  plot_every = 30, report_every = 10, loss_upper_bound = 1e7,
#                         save_trace = false,
#                        xlabel = "Time", ylabel = "Population", title = "Lotka-Volterra", kwargs...)
peprob = trainpeprob
save_trace = true
loss_upper_bound = 1e7
label = hcat(string.(unknowns(peprob.model.sys))...)
data = peprob.measurements.measurement
saved_states = Array{Optimization.OptimizationState}(undef, 0)
timedata = peprob.measurements.time
report_every = 1
plot_every = 1
function callback(state, l;) #callback function to observe training
    # if save_trace
    #     push!(saved_states, state)
    # end
    if l > loss_upper_bound
        println("Loss exceeded upper bound at iteration $(state.iter). Stopping optimization.")
        return true # Stop the optimization if loss exceeds upper bound
    end
    if prod([SciMLBase.successful_retcode(trajectory) for trajectory in sim]) != 1
        println("Simulation failed at iteration $(state.iter). Stopping optimization.")
        return true # Stop the optimization if simulation fails
    end
    if report_every > 0
        if state.iter % report_every == 0
            println("Iteration: $(state.iter), Loss: $(l), sample_percentage: $(peprob.model.data_proportion), Parameters: $(state.u)")
        end
    end
    if plot_every > 0
        if state.iter % plot_every == 0
            ps = peprob.log_transform ? exp.(state.u) : state.u # Exponentiate the parameters if log transformation is enabled
            p1 = plot(peprob; included_plots = [:data, :model],
                data_proportion = peprob.model.data_proportion,
                p = ps)
            display(p1)
        end
    end
    return false
end
#     return callback, saved_states
# end



n_runs = 1 # number of runs for the ensemble
max_trials = 10 
gt_p = init_params(hmodel)
# max_dist_from_gt = 0.5
lb = gt_p*eps(Float64)
ub = gt_p*eps(Float64) .+ 1.0
initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub)
opt_prob = Optimization.EnsembleProblem(trainpeprob; initp_samples = initp_samples,
                                         alpha = alpha, l1_ratio = l1_ratio,
                                         random_sampling_percentage = 0.2)

                                         
# cb, trace = create_callback(trainpeprob,  plot_every = 30, report_every = 30, loss_upper_bound = 1e7,
#                        xlabel = "Time", ylabel = "Signal", title = "Octet Simuluated Data", save_trace = true)
opt_sol = Optimization.solve(opt_prob, ProgressivePolyOpt(lr = 1e-2, n_partitions = 1), EnsembleThreads(), 
                            trajectories = n_runs,
                            maxiters = 1000,
                            maxiter_BFGS = 300,
                            show_trace = true, show_every = 10,
                            callback = callback)
