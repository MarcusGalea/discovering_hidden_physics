include("LV_hidden_model.jl")
include("../../src/hybrid_model.jl")
include("../../src/polyopt.jl")
# hmodel = HybridModel(sys_known, sys_unknown_gt; rng = rng)
hmodel = HybridModel(sys_known, sys_unknown_gt; rng = rng)

rootdir = dirname(dirname(dirname(@__FILE__)))
model_dir = joinpath(rootdir , "models", "lotka_volterra", "Reg", "seed_$seed")
# using IterTools: ncycle
# using OptimizationOptimisers

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


function callback(state, l; plot_every = 30, report_every =10, loss_upper_bound = 1e7) #callback function to observe training
    sim = simulate_solution(peprob, state.u)
    if l > loss_upper_bound
        println("Loss exceeded upper bound at iteration $(state.iter). Stopping optimization.")
        return true # Stop the optimization if loss exceeds upper bound
    end
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

n_runs = 20 # number of runs for the ensemble

gt_p = init_params(hmodel)
max_dist_from_gt = 1e-1
lb = gt_p .- max_dist_from_gt
ub = gt_p .+ max_dist_from_gt
initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub)
#plot the first to parameters for all samples


opt_prob = Optimization.EnsembleProblem(peprob; initp_samples = initp_samples, 
                          alpha = alpha, l1_ratio = l1_ratio, adalg = Optimization.AutoForwardDiff())

# ensembleoptsol = Optimization.solve(opt_prob, PolyOpt(),
    # EnsembleThreads(), trajectories = n_runs ,epochs = 100)#maxiters = 200, show_trace = true, show_every = 5

ensembleoptsol = Optimization.solve(opt_prob, PolyOptAdamBFGS(), EnsembleThreads(), trajectories = n_runs;
                                    show_trace = true, show_every = 50, callback = callback)

best_run = argmin([opt.objective for opt in ensembleoptsol])
sim = simulate_solution(peprob, gt_p, saveat = collect(0:0.1:100.0))
scatter(timedata[sample_idcs], data[sample_idcs, :], label=["Prey_data" "Predator_data"], xlabel="Time", ylabel="Population", title="Lotka-Volterra Model with Sampled Data", legend =:topright)
plot!(sim, label = ["Prey_model" "Predator_model"], linewidth=2, markersize=4, legend =:topright)

#save ensembleoptsol to file
save_path = joinpath(model_dir, "ensembleoptsol.jld2")
using JLD2
@info "Saving ensemble optimization solution to $save_path"
save(save_path, "ensembleoptsol", ensembleoptsol)