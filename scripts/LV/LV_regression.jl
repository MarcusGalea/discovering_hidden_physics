include("LV_hidden_model.jl")
include("../../src/hybrid_model.jl")
# hmodel = HybridModel(sys_known, sys_unknown_gt; rng = rng)
hmodel = HybridModel(sys_known, sys_unknown_gt; rng = rng)

using IterTools: ncycle
using OptimizationOptimisers

alpha = 0.0 # Lasso penalty
l1_ratio = 0.5 # Lasso/Ridge ratio
batch_size = 32 # Batch size for the optimization
obs = Dict("prey_o" => x, "predator_o" => y)
u0map = Dict([x => 40.0, y => 9.0])
initial_conditions = Dict(
    "cond1" => Dict([x => 40.0, y => 9.0]), #remember to define variables before using them
)



#shuffle dataframe rows
train_measurements = train_measurements[shuffle(rng, 1:nrow(train_measurements)), :]
peprob = HybridPEProblem(hmodel, obs, train_measurements, u0map; 
                   conditions = initial_conditions, batch_size = batch_size,)


function callback(state, l; plot_every = 30, report_every =10) #callback function to observe training
    sim = simulate_solution(peprob, state.u)
    if prod(Int.([trajectory.retcode for trajectory in sim])) != 1
        println("Simulation failed at iteration $(state.iter). Stopping optimization.")
        return true # Stop the optimization if simulation fails
    end
    if state.iter % report_every == 0
        println("Iteration: $(state.iter), Loss: $(l), Parameters: $(state.u)")

    end
    if plot_every > 0 && state.iter % plot_every == 0
        p1 = plot(sim, label = ["Prey_model" "Predator_model"], linewidth=2, markersize=4, legend =:topright)
        scatter!(p1, timedata[sample_idcs], data[sample_idcs, :], label=["Prey_data" "Predator_data"], xlabel="Time", ylabel="Population", title="Lotka-Volterra Model with Sampled Data", legend =:topright)
        display(p1)
    end
    return false
end

n_runs = 100 # number of runs for the ensemble
initp_samples = init_params(hmodel; n = n_runs)
# initp_samples .+= randn(size(initp_samples)) + 1e-06 * randn(size(initp_samples)) # Add some noise to the initial parameters
#plot the first to parameters for all samples


opt_prob = Optimization.EnsembleProblem(peprob; initp_samples = initp_samples, 
                          alpha = alpha, l1_ratio = l1_ratio,)

# ensembleoptsol = Optimization.solve(opt_prob, PolyOpt(),
    # EnsembleThreads(), trajectories = n_runs ,epochs = 100)#maxiters = 200, show_trace = true, show_every = 5

ensembleoptsol = Optimization.solve(opt_prob, Optimisers.Adam(), EnsembleThreads(), trajectories = n_runs; epochs = 1000, 
                                    show_trace = true, show_every = 50, callback = callback)
ensembleoptsol.u


sim = simulate_solution(peprob,ensembleoptsol[1].u)# ensembleoptsol[1].u)
scatter(timedata[sample_idcs], data[sample_idcs, :], label=["Prey_data" "Predator_data"], xlabel="Time", ylabel="Population", title="Lotka-Volterra Model with Sampled Data", legend =:topright)
plot!(sim, label = ["Prey_model" "Predator_model"], linewidth=2, markersize=4, legend =:topright)