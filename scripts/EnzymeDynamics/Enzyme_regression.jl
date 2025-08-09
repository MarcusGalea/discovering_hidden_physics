include("Enzyme_OctetData.jl")
include("Enzyme_hidden_model.jl")
include("../../src/hybrid_model.jl")
include("../../src/polyopt.jl")

hmodel = HybridModel(complete(sys_known), complete(sys_unknown_gt); rng = rng)

rootdir = dirname(dirname(dirname(@__FILE__)))
model_dir = joinpath(rootdir , "models", "enzyme_dynamics", "Reg", "seed_$seed")

alpha = 0.0 # Lasso penalty
l1_ratio = 0.5 # Lasso/Ridge ratio


### OPTIMIZATION
n_initial_conditions = 1
#CHANGE NUMBER OF INITIAL CONDITIONS HERE
train_measurements_exp = train_measurements[sum([train_measurements.simulation_id .== "cond$i" for i in 1:n_initial_conditions]), :]
test_measurements_exp = test_measurements[sum([test_measurements.simulation_id .== "cond$i" for i in 1:n_initial_conditions]), :]
# batch_size = 32 # Batch size for the optimization
@unpack E, S, ES, v = sys_known
obs = Dict("E" => E, "S" => S, "ES" => ES, "v" => v)
u0map = Dict([E => 10.0, S => 1.0, ES => 0.0])
ic_vals = Dict(["cond$i" => Dict([var => ic[j] for (j, var) in enumerate(unknowns(sys_known))]) for (i, ic) in enumerate(initial_conditions[1:n_initial_conditions])])


###
valpeprob = HybridPEProblem(hmodel, obs, test_measurements_exp, u0map; 
                   conditions = ic_vals,
                   ens_alg = EnsembleSplitThreads())
n_runs = 1 # number of runs for the ensemble
max_trials = 10 
batch_size = 32 # Batch size for the optimization
gt_p = init_params(hmodel)
max_dist_from_gt = 0.5
lb = gt_p .- max_dist_from_gt
ub = gt_p .+ max_dist_from_gt
initp_samples = init_params(hmodel, n = n_runs, lb = lb, ub = ub)


sim = simulate_solution(valpeprob, initp_samples)
sim[1][v(t)]


opt_prob = Optimization.EnsembleProblem(valpeprob; initp_samples = initp_samples,
                                         max_trials = max_trials,
                                         alpha = alpha, l1_ratio = l1_ratio)

opt_sol = Optimization.solve(opt_prob, ProgressivePolyOpt(lr = 1e-3, n_partitions = 10), EnsembleThreads(), trajectories = n_runs)


obs_funs = Dict([obs_fun.lhs =>eval(build_function(obs_fun.rhs, unknowns(hmodel.sys); ps = parameters(hmodel.sys), expression=Val{false})) for obs_fun in observed(hmodel.sys)])

@unpack w_E, w_S, w_ES = hmodel.sys
obs_dict = Dict{ModelingToolkit.BasicSymbolic, Vector}()

for obs in observables(hmodel.sys)
    obs_dict[obs] = sim[1][obs]
end
obs_val_dict = [observable_dict(sim[i], initp_samples, hmodel, obs_funs) for i in 1:1]
obs_val_dict[]