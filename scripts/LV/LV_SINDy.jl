using Pkg
Pkg.activate("scripts//")
Pkg.instantiate()
include("../../src/polyopt.jl")
include("LV_hidden_model.jl")
include("../../src/hybrid_model.jl")
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
default_params = Dict([Ξ[i, j] => eps(Float64) for i in 1:length(known_eqs), j in 1:length(unknown_basis)])
default_params[Ξ[1,5]] = - params[β]
default_params[Ξ[2,5]] = params[δ] # Set the default parameters for the unknown equations

uk_eqs = create_unknown_eqs(sys_known, unknown_basis; Ξ = Ξ)
@named sys_SINDy = ODESystem(uk_eqs, t, defaults = default_params)
sys_SINDy = complete(sys_SINDy)


hmodel = HybridModel(sys_known, sys_SINDy; rng = rng)


### OPTIMIZATION

# batch_size = 32 # Batch size for the optimization
obs = Dict("prey_o" => x, "predator_o" => y)
u0map = Dict([x => 40.0, y => 9.0])
initial_conditions = Dict(
    "cond1" => Dict([x => 40.0, y => 9.0]), #remember to define variables before using them
)


peprob = HybridPEProblem(hmodel, obs, train_measurements, u0map; 
                   conditions = initial_conditions)#, batch_size = batch_size,)
valpeprob = HybridPEProblem(hmodel, obs, test_measurements, u0map; 
                   conditions = initial_conditions)



n_runs = 1 # number of runs for the ensemble
max_trials = 1
gt_p = init_params(hmodel)
max_dist_from_gt = 0.5
lb = gt_p .- max_dist_from_gt
ub = gt_p .+ max_dist_from_gt
initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub)

# peprob.obj_func(gt_p, train_measurements)



cb = create_callback(peprob; plot_every = 0, report_every = 500, loss_upper_bound = 1e7,
                       xlabel = "Time", ylabel = "Population", title = "Lotka-Volterra SINDy")

#DANGEROUS CONVSERION INCOMING! MIGHT RUIN OPTIMIZATION
using Base
import Base: convert
using ForwardDiff
Base.convert(::Type{T}, x::ForwardDiff.Dual)  where T <: Number = x.value
# Convert Dual numbers to Float64 by taking the value part

options = Dict(:maxiters => 1000, :show_every => 1, :callback => cb, :trajectories => n_runs)

# ensembleoptsol = Optimization.solve(opt_prob, PolyOptAdamBFGS(lr = 1e-4), EnsembleThreads(); options...)

# sim = simulate_solution(peprob, initp_samples, saveat = collect(0:0.1:100.0))
# scatter(timedata[sample_idcs], data[sample_idcs, :], label=["Prey_data" "Predator_data"], xlabel="Time", ylabel="Population", title="Lotka-Volterra SINDy", legend =:topright)
# plot!(sim, label = ["Prey_model" "Predator_model"], linewidth=2, markersize=4, legend =:topright)


using Dates
date_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

save_path = joinpath(model_dir, "ensembleoptsol_$(date_str).jld2")
save_path_hyperparams = joinpath(model_dir, "hyperparams_$(date_str).jld2")
save_path_allruns = joinpath(model_dir, "all_runs_$(date_str)") 
#make directory if it does not exist
if !isdir(save_path_allruns)
    mkpath(save_path_allruns)
end

using HyperTuning
scenario = Scenario(lr_exponent = (-6.0..1.0),
                    alpha_exponent = (-6.0..2.0),
                    l1_ratio = (0.0..1.0),
                    max_trials = max_trials,
)

function hypertuning_objective(trial)
    @unpack lr_exponent, alpha_exponent, l1_ratio = trial
    alpha = 10.0 ^ alpha_exponent
    lr = 10.0 ^ lr_exponent
    opt_prob = Optimization.EnsembleProblem(peprob; initp_samples = initp_samples,
                          alpha = alpha, l1_ratio = l1_ratio, adalg = Optimization.AutoForwardDiff())
    ensembleoptsol = Optimization.solve(opt_prob, PolyOptAdamBFGS(lr=lr), EnsembleThreads(), trajectories = n_runs; options...)
    best_run = argmin([valpeprob.obj_func(sol.u) for sol in ensembleoptsol])
    val_loss = valpeprob.obj_func(ensembleoptsol[best_run].u)    
    #save each run to a file
    run_save_path = joinpath(save_path_allruns, "run_$(replace(string(res.best_trial.trials[1]), " " => "_")).jld2")
    save(run_save_path, "ensembleoptsol", ensembleoptsol, "val_loss", val_loss)
    return val_loss
end

hypertune_res = HyperTuning.optimize(hypertuning_objective, scenario)
@info "Best parameters: $(hypertune_res.params), Best loss: $(hypertune_res.loss)"



#REDO BEST RUN WITH BEST PARAMETERS
@unpack lr_exponent, alpha_exponent, l1_ratio = hypertune_res
alpha = 10.0 ^ alpha_exponent
lr = 10.0 ^ lr_exponent
opt_prob = Optimization.EnsembleProblem(peprob; initp_samples = initp_samples,
                          alpha = alpha, l1_ratio = l1_ratio, adalg = Optimization.AutoForwardDiff())
ensembleoptsol = Optimization.solve(opt_prob, PolyOptAdamBFGS(lr=lr), EnsembleThreads(), trajectories = n_runs; options...)
#save ensembleoptsol to file
#get date

@info "Saving ensemble optimization solution to $save_path and hyperparameter tuning results to $save_path_hyperparams"
save(save_path, "ensembleoptsol", ensembleoptsol)

# using OptimizationOptimisers
# using IterTools: ncycle
# function PolyOptHM()
# end
# ensembleoptsol = Optimization.solve(opt_prob, Optimisers.Adam(;eta = 1e-4), EnsembleThreads(), trajectories = n_runs; epochs = 100,
#                                     show_trace = true, show_every = 50, callback = callback)
# #save ensembleoptsol to file
# save_path = joinpath(model_dir, "ensembleoptsol.jld2")
# @info "Saving ensemble optimization solution to $save_path"
# save(save_path, "ensembleoptsol", ensembleoptsol)
