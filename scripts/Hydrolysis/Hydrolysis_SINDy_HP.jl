include("Hydrolysis_data.jl")
include("Hydrolysis_hidden_model.jl")
include("../../src/hybrid_model.jl")
include("../../src/SINDy_methods.jl")
include("../../src/polyopt.jl")



# #SINDy MODEL
# rn_SINDy = create_full_reaction_network(unknowns(sys_known), sys_known.t; number_reactants = 2)
# rn__SINDy = remove_known_reactions(rn_SINDy, known_rn)


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


valpeprob = HybridPEProblem(hmodel, obs, test_measurements_exp, u0map; 
                   conditions = ic_vals
                   )#, batch_size = batch_size,)# OPTIONS HERE ARE UNECESSARY

###
gt_p = init_params(hmodel)
n_runs = 2 # number of runs for the ensemble
max_trials = 2 


# max_dist_from_gt = 0.5
lb = gt_p*eps(Float64)
ub = gt_p*eps(Float64) .+ 1.0
initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub)
#save samples in jld2
using Dates
using JLD2
date_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
println("Date string: $date_str")

initp_samples_path = joinpath(model_dir, "initp_samples_$(date_str).jld2")
save(initp_samples_path, "initp_samples", initp_samples)
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
                    sampling_percentage = (0.1..0.9),
                    sampler = GridSampler(),
)


function hypertuning_objective(trial)
    @unpack lr_exponent, alpha_exponent, l1_ratio,sampling_percentage = trial
    maxiters = Int(400/sampling_percentage)
    alpha = 10.0 ^ alpha_exponent
    lr = 10.0 ^ lr_exponent
    
    peprob = HybridPEProblem(hmodel, obs, train_measurements_exp, u0map; 
                   conditions = ic_vals, alpha = alpha, l1_ratio = l1_ratio,
                    random_sampling_percentage = sampling_percentage)

    opt_prob = Optimization.EnsembleProblem(peprob; initp_samples = initp_samples,
                            adalg = Optimization.AutoForwardDiff())
    ensembleoptsol = Optimization.solve(opt_prob, ProgressivePolyOpt(lr = lr, n_partitions = 10), EnsembleThreads(); maxiters = maxiters, trajectories = n_runs)

    best_run = argmin([valpeprob.obj_func(sol.u, 1.0) for sol in ensembleoptsol])
    val_loss = valpeprob.obj_func(ensembleoptsol[best_run].u, 1.0)    
    train_loss = ensembleoptsol[best_run].objective
    #save each run to a file
    println("Trial: $(trial)")
    trial_string = join(["$(hyperp)_$(round(val,sigdigits = 4))_" for (hyperp, val) in trial.values])
    safe_trial_string = replace(trial_string, r"[\\/:*?\"<>|]" => "_")
    current_save_path = joinpath(save_path_allruns, "val_loss_$(round(val_loss, sigdigits = 3))_train_loss_$(round(train_loss, sigdigits = 3)).jld2")#_$(safe_trial_string).jld2")

    save(current_save_path, "info", safe_trial_string)
    #save txt file with all hyperparameters
    return val_loss
end

# trial = hypertune_res.best_trial.trials[1]
# loading_path = joinpath(save_path_allruns, "loss_1.0e16_lr_exponent_-3.558_l1_ratio_0.6935_alpha_exponent_-3.26_.jld2")
# loaded_sol = load(loading_path, "ensembleoptsol")

hypertune_res = HyperTuning.optimize(hypertuning_objective, scenario)
@info "Best loss: $(hypertune_res.best_trial)"
save(save_path_hyperparams, "hypertune_res", hypertune_res)
