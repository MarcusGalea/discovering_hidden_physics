using PEtab
using DifferentialEquations
using Lux
using ModelingToolkit
using Random
import ModelingToolkit:has_observed,observables
using Optimization
import Optimization:solve
using SciMLSensitivity
using DataFrames
using ComponentArrays
using LatinHypercubeSampling
using MLUtils


mutable struct NullModel <: ModelingToolkit.AbstractTimeDependentSystem
    """ A null model that does nothing. Used as a placeholder for surrogate models."""
    name::String

    """ Create a NullModel with a given name."""
    NullModel(name::String = "NullModel") = new(name)

    """ Get the parameters of the NullModel. Returns an empty NamedTuple."""
end

""" HybridModel is a system that combines an ODE system with a surrogate model (SINDy or neural network)."""
mutable struct HybridModel
    """ The known underlying ODE system."""
    sys::ODESystem
    """ The surrogate model, which can be a normal ODESystem, SINDy model, or a Lux neural network."""
    surrogate::Union{ModelingToolkit.AbstractTimeDependentSystem, Lux.Chain, PEtab.MLModel}
    """ Discrete events that trigger during the simulation."""
    events::Vector
    """ Observables that are computed during the simulation. (Defaults to unknowns of sys)"""
    observables::Vector
    """ Random number generator for reproducibility."""
    rng ::Random.AbstractRNG
    """ODEFunction for the HybridModel system."""
    ode_fun::Function
    """Machine learning model for the HybridModel system."""
    ml_models::Dict
    
    """ Construct a HybridModel system with a SINDy/ODE surrogate model. """
    HybridModel(sys::ODESystem, surrogate::T;
                events::Vector = [], 
                observables::Vector = has_observed(sys) || has_observed(surrogate) ?  union([observables(sys);observables(surrogate)]) : unknowns(sys), #choose observables from sys or surrogate if they have them. else use state variables
                rng::Random.AbstractRNG = Random.default_rng(1234),
                ode_fun::Function = ODEFunction(sys, surrogate; rng = rng),
                ml_models::Dict = Dict(),
                ) where 
                T <: ModelingToolkit.AbstractTimeDependentSystem = 
        new(sys, surrogate, events, observables, rng, ode_fun, ml_models)


    """   Construct a HybridModel system with a Lux neural network surrogate model."""
    HybridModel(sys::ODESystem, surrogate::T; 
               events::Vector = [], 
               observables::Vector = has_observed(sys) || has_observed(surrogate) ?  union([observables(sys);observables(surrogate)]) : unknowns(sys), #choose observables from sys or surrogate if they have them. else use state variables
               rng::Random.AbstractRNG = Random.default_rng(1234),
               ode_fun::Function = ODEFunction(sys, surrogate; rng = rng),
               ml_models::Dict = Dict(:surrogate => surrogate),
               ) where 
               T <: Lux.Chain =    
        new(sys, surrogate, events, observables, rng, ode_fun, ml_models)

    """ Construct a HybridModel system with a PEtab MLModel surrogate model."""
    HybridModel(sys::ODESystem, surrogate::T; 
               events::Vector = [], 
               observables::Vector = has_observed(sys) || has_observed(surrogate) ?  union([observables(sys);observables(surrogate)]) : unknowns(sys), #choose observables from sys or surrogate if they have them. else use state variables
               rng::Random.AbstractRNG = Random.default_rng(1234),
               ode_fun::Function = ODEFunction(sys, surrogate; rng = rng),
               ml_models::Dict = Dict(:surrogate => surrogate),
               ) where 
               {T <: PEtab.MLModel} =    
        new(sys, surrogate, events, observables, rng, ode_fun, ml_models)
end

# Convert a dictionary to a NamedTuple (for use with ComponentArrays with ModelingToolkit)
NamedTuple(dict::Dict) = (; (Symbol(string(k)) => v for (k, v) in dict)...) 

#### HIDDEN ODE METHODS ####

function init_params(model::HybridModel; 
    lb = ComponentArray(ode = -ones(length(parameters(model.sys))), 
                        surrogate = -ones(length(parameters(model.surrogate)))),
    ub = ComponentArray(ode = ones(length(parameters(model.sys))), 
                        surrogate = ones(length(parameters(model.surrogate)))),
    n = 1)
    @unpack sys, surrogate, rng = model
    
    # Initialize parameters for the ODE system
    ode_ps = init_params(sys; rng = rng, n = n, lb = lb.ode, ub = ub.ode)
    # Initialize parameters for the surrogate model
    surrogate_ps = init_params(surrogate; rng = rng, n = n, lb = lb.surrogate, ub = ub.surrogate)

    # Combine the parameters into a NamedTuple
    # combined_ps = merge(ode_ps, surrogate_ps)
    if n > 1
        # If n > 1, return a vector of sampled parameters
        combined_ps = [ComponentArray(merge((;sys = ode_ps[i]), (;surrogate = surrogate_ps[i]))) for i in 1:n]
        return ComponentArray(combined_ps)
    end
    combined_ps = merge((;sys = ode_ps), (;surrogate = surrogate_ps))
    return ComponentVector{Float64}(combined_ps)
end

function DifferentialEquations.ODEFunction(model::HybridModel)
        @unpack sys, surrogate, rng = model
        return DifferentialEquations.ODEFunction(sys, surrogate; rng = rng)
end

#create the ODE function for the HybridModel system
function DifferentialEquations.ODEFunction(sys::ODESystem, surrogate::T; rng = Random.default_rng(1234)) where {T <: Union{ModelingToolkit.AbstractTimeDependentSystem, Lux.Chain, PEtab.MLModel}}
    # Get the derivative function for the ODE system
    ode_fun! = DifferentialEquations.ODEFunction(sys, unknowns(sys), parameters(sys))
    # Get the surrogate derivative function
    surrogate_fun! = derivative_function!(surrogate; rng = rng)

    du1 = arr = Any[0.0 for _ in 1:length(unknowns(sys))]
    du2 = copy(du1) # Initialize du2 for the surrogate model
    function update_du!(du, u, p, t)
        # Compute the ODE derivatives
        ode_fun!(du1, u, p.sys, t) 
        # Compute the surrogate derivatives
        surrogate_fun!(du2, u, p.surrogate, t) 
        # Combine the derivatives   
        du.= du1 .+ du2
        return du  
    end
    odefun! = remake(ode_fun!, f = update_du!) # Create a new ODEFunction with the combined derivatives
    return odefun!
end
function DifferentialEquations.ODEProblem(model::HybridModel, u0::Union{Vector, ComponentArray}, tspan, p = init_params(model))
    # @unpack ode_fun = model
    ode_fun = DifferentialEquations.ODEFunction(model)
    prob = DifferentialEquations.ODEProblem(ode_fun, u0, tspan, p)
end

function DifferentialEquations.ODEProblem(model::HybridModel, u0::Dict, tspan, p = init_params(model))
    # @unpack ode_fun, sys= model
    ode_fun = DifferentialEquations.ODEFunction(model)
    sys = model.sys
    # Convert the dictionary to a vector
    u0_vec = [u0[var] for var in unknowns(sys)]
    prob = DifferentialEquations.ODEProblem(ode_fun, u0_vec, tspan, p)
end


function DifferentialEquations.EnsembleProblem(model::HybridModel, u0s::Vector{Vector{T}}, tspan, p = init_params(model)) where {T <: Any}
    prob = ODEProblem(model, u0s[1], tspan, p)
    function prob_func(prob, i, repeat)
        remake(prob, u0 = u0s[i])
    end
    return EnsembleProblem(prob, prob_func = prob_func)
end

function DifferentialEquations.EnsembleProblem(model::HybridModel, u0s::Dict{String, Dict}, tspan, p = init_params(model))
    #Assuming u0s is a Dictionary where keys are simulation IDs and values are dictionaries of initial conditions
    #get sorted keys
    conds =  sort(collect(keys(u0s)))
    prob = DifferentialEquations.ODEProblem(model, u0s[conds[1]], tspan, p)
    function prob_func(prob, i, repeat)
        if u0s[conds[i]] isa Dict
            u0 = [u0s[conds[i]][var] for var in unknowns(model.sys)]
        else
            u0 = u0s[conds[i]]
        end
        remake(prob, u0 = u0)
    end
    return DifferentialEquations.EnsembleProblem(prob, prob_func = prob_func)
end

function DifferentialEquations.EnsembleProblem(model::HybridModel, u0s::Any, tspan, p = init_params(model)) #
    conds =  sort(collect(keys(u0s)))
    prob = ODEProblem(model, u0s[conds[1]], tspan, p)
    function prob_func(prob, i, repeat)
        if u0s[conds[i]] isa Dict
            u0 = [u0s[conds[i]][var] for var in unknowns(model.sys)]
        else
            u0 = u0s[conds[i]]
        end
        remake(prob, u0 = u0)
    end
    return DifferentialEquations.EnsembleProblem(prob, prob_func = prob_func)
end

function solve(model::HybridModel, u0s, time, p = init_params(model); alg = Tsit5, tspan = (time[1], time[end]), kwargs...)
    prob = EnsembleProblem(model, u0s, tspan, p)
    # if prob isa EnsembleProblem
    return solve(prob, alg(), EnsembleDistributed(), trajectories = length(u0s), saveat = time, kwargs...)
    # elseif prob isa ODEProblem
    #     return solve(prob, alg(), saveat = time, kwargs...)
    # end
end

function observed_values(model::HybridModel, sol)
    @unpack observables = model
    return observed_values(observables, sol)
end

function observed_values(observed::Vector, sol)
    # Get the observed values from the solution
    return ComponentArray(; (Symbol(string("iv$i")) => hcat([sim[var] for var in observed]...) for (i, sim) in enumerate(sol))...)
end


### ODE SYSTEM METHODS ###
function init_params(sys::ODESystem; randfun = rand, rng = Random.default_rng(1234), n = 1, lb = -ones(length(parameters(sys))), ub = ones(length(parameters(sys))), 
                        generations = 1000)
    if n > 1
        # If n > 1, return a vector of sampled parameters
        n_parameters = length(parameters(sys))
        samples, _ = LHCoptim(n,n_parameters, generations, rng = rng)
        scaled_samples = scaleLHC(samples, [(lb[i], ub[i]) for i in 1:n_parameters])
        return [NamedTuple(Dict(Symbol(string(p)) => scaled_samples[i, j] for (j, p) in enumerate(parameters(sys)))) for i in 1:n]
    end
    if ModelingToolkit.has_defaults(sys) & any([p in keys(ModelingToolkit.get_defaults(sys)) for p in parameters(sys)])
        defaults = ModelingToolkit.get_defaults(sys)
        # Initialize parameters with defaults (NamedTuple))
        return (; (Symbol(string(p)) => defaults[p] for p in parameters(sys) if p in keys(defaults))...)
    else
        # If no defaults, return an empty random values
        return (; (Symbol(string(p)) => randfun() for p in parameters(sys))...)
    end
end


function derivative_function!(sys::ODESystem; rng = Random.default_rng(1234))
    return DifferentialEquations.ODEFunction(sys, unknowns(sys), parameters(sys))
end

@independent_variables t
function merge_systems(sys::AbstractTimeDependentSystem, surrogate::AbstractTimeDependentSystem)
    # Merge the ODE system and the surrogate model into a single system (Given they are both symbolic systems)
    D = Differential(t)
    #TODO
end

### NEURAL NETWORK METHODS ###
function init_params(nn::Lux.Chain; rng = Random.default_rng(1234), 
                     n = 1,lb = nothing, ub = nothing)
    # Get the parameters of the Lux model
    if n > 1
        return [Lux.initialparameters(rng, nn) for _ in 1:n] # Return a vector of initial parameters
    end
    return Lux.initialparameters(rng, nn)
end

function derivative_function!(nn::Lux.Chain; rng = Random.default_rng(1234))
    # Create a function that computes the derivatives of the Lux model
    st = Lux.initialstates(rng, nn)
    #NeuralODE. The output of the neural network is a vector of derivatives. Network only depends on state and network parameters.
    du = (du, u, p, t) -> first(nn(u, p, st)) #the NN only depends on the state 
    return deepcopy(du)
end

function has_observed(nn::Lux.Chain)
    # Check if the Lux model has observed quantities
    return false # Lux models do not have observed quantities by default
end

function observables(nn::Lux.Chain)
    # Lux models do not have observed quantities by default
    return []
end

#### PETAB ML MODEL METHODS ####
function init_params(model::PEtab.MLModel; rng = Random.default_rng(1234))
    # retrieve the parameters of the PEtab ML model
    return model.ps
end

function derivative_function!(model::PEtab.MLModel; rng = Random.default_rng(1234))
    # Create a function that computes the derivatives of the PEtab ML model
    
    function _ode!(du, u, p, t, model)
        du_nn, st = model.model(u,model.ps, model.st)
        du .= du_nn
        return nothing
    end

    ode! = let _ml_model = model
        (du, u, p, t) -> _ode!(du, u, p, t, _ml_model)
    end
    return deepcopy(ode!)
end

function has_observed(model::PEtab.MLModel)
    # Check if the PEtab ML model has observed quantities
    return false
end

function observables(model::PEtab.MLModel)
    # PEtab ML models do not have observed quantities by default
    return []
end

### NULL MODEL METHODS ###
function init_params(model::NullModel; rng = Random.default_rng(1234))
    # Null model has no parameters (empty NamedTuple)
    return ()
end

function derivative_function!(model::NullModel; rng = Random.default_rng(1234))
    # Null model does nothing, so return a function that does nothing
    return (du, u, p, t) -> du .= 0.0
end

function has_observed(model::NullModel)
    # Null model has no observed quantities
    return false
end

function observables(model::NullModel)
    # Null model has no observed quantities
    return []
end

import ModelingToolkit: parameters

parameters(nn::Lux.Chain) = ComponentArrays.labels(ComponentArray(init_params(nn)))

parameters(model::HybridModel) = (:sys => parameters(model.sys)), (:surrogate => parameters(model.surrogate))

"""
Parameter Estimation Problem for HybridModel.
    This problem involves estimating the parameters of a HybridModel that combines
    both ODE and machine learning components.
"""
mutable struct HybridPEProblem
    """ A HybridModel that uses a Surrogate model as the surrogate."""
    model::HybridModel
    """ The initial conditions for the ODE system."""
    u0map
    """ measurements for the hybrid model."""
    measurements::DataFrame
    """ Conditions. This is a dictionary that maps the conditions to overwrite values in u0map."""
    conditions::Dict
    """observations for the hybrid model."""
    observations::Dict
    """ The time span for the simulation."""
    tspan::Tuple
    """ Create an objective function for the HybridPEProblem."""
    obj_func::Function
    """Upper bounds for the parameters. This is a component array that maps the parameter names to their upper bounds."""
    ub
    """Lower bounds for the parameters. This is a component array that maps the parameter names to their lower bounds."""
    lb
    """ Batch size for the data loader. This is used to create a DataLoader for the measurements."""
    batch_size::Int

    #create constructor for HybridModelPE
    HybridPEProblem(model::HybridModel, observables::Dict, measurements::DataFrame, u0map; 
                  conditions::Dict = overwrite_conditions!(u0map, Dict()), 
                  tspan::Tuple = (0.0, maximum(measurements.time)),
                  ub = nothing,
                  lb = nothing,
                  batch_size = size(measurements, 1),
                  kwargs...
                    ) = new(model, u0map, measurements, conditions, observables, tspan,
                    define_loss_function(model, observables, measurements, u0map; 
                                         conditions = conditions, tspan = tspan, kwargs...),
                    ub, lb, batch_size)
end

function simulate_solution(prob::HybridPEProblem, p;
                            u0map = prob.u0map,
                            conditions = prob.conditions,
                            tspan = prob.tspan,
                            alg = Tsit5(),
                            ens_alg = EnsembleDistributed(),
    )
    # Overwrite the initial conditions with the conditions dictionary
    u0_conditions = overwrite_conditions!(u0map, conditions)
    # Create an EnsembleProblem with the initial conditions and parameters
    ens_prob = DifferentialEquations.EnsembleProblem(prob.model, u0_conditions, tspan, p)
    # Solve the EnsembleProblem
    sim = solve(ens_prob, alg, ens_alg, trajectories = length(u0_conditions), saveat = prob.measurements.time)
end

function define_loss_function(model::HybridModel, obs::Dict, measurements::DataFrame, u0map; 
                              conditions::Dict = Dict(), 
                              tspan::Tuple = (0.0, maximum(measurements.time)),
                              alg = Tsit5(),
                              ens_alg = EnsembleDistributed(),
                              include_plot = false,
                              alpha = 0.0,
                              l1_ratio= 0.0,
                              assume_stable_data = true,
                              unsolved_penalty = 1e+8, # penalty for unsolved simulations,
                              sensealg = ForwardDiffSensitivity()
                              )
    # Define
    u0_conditions = overwrite_conditions!(u0map, conditions)
    obs_funs = Dict([obs_fun.lhs =>eval(build_function(obs_fun.rhs, unknowns(model.sys); ps = parameters(model.sys), expression=Val{false})) for obs_fun in observed(model.sys)])
    
    idx_sim = Dict([cond => i for (i, cond) in enumerate(sort(collect(keys(u0_conditions))))]) #create a map from condition name to ensemble index
    idx_var = Dict([var => i for (i, var) in enumerate(unknowns(model.sys))]) #create a map from variable name to index
    idx_time = Dict(entry.time => i for (i, entry) in enumerate(eachrow(measurements))) #create a map from time to index in the measurements
    # ensemble_prob = EnsembleProblem(model, u0_conditions, tspan)
    max_t_gt = maximum(measurements.time) # Get the maximum time from the measurements

    function loss_function(p, measurements_batch)
        # ensemble_prob = remake(ensemble_prob, p = p) #SUPER SMART!! This allows us to change the parameters of the ensemble problem without creating a new one
        ensemble_prob = EnsembleProblem(model, u0_conditions, tspan, p)
        sim = solve(ensemble_prob, alg, ens_alg, trajectories = length(u0_conditions), saveat = measurements.time, sensealg = sensealg) #Array
        max_t = maximum(vec([traj.t for traj in sim]...))
        idx_time = Dict(time => i for (i, time) in enumerate(sort(measurements.time))) #create a map from time to index in the measurements
        obs_val_dict = [observable_dict(sim[i], p, model, obs_funs) for i in 1:length(u0_conditions)]
        loss = 0.0 
        n = 0
        # println(sim)
        for i in 1:size(measurements_batch, 1)
            entry = measurements_batch[i, :]
            if assume_stable_data && idx_time[entry.time] > length(sim[idx_sim[entry.simulation_id]].t)
                continue # Skip this entry if the time is greater than the maximum time in the simulation
            end
            #make sure the simulation has the same time as the measurement
            @assert sim[idx_sim[entry.simulation_id]].t[idx_time[entry.time]] ≈ entry.time "The time in the measurement $(entry.time) does not match the simulation time $(sim[idx_sim[entry.simulation_id]].t[idx_time[entry.time]])"
            meas_value = entry.measurement
            sim_value = obs_val_dict[idx_sim[entry.simulation_id]][obs[entry.obs_id]][idx_time[entry.time]]
            # sim[idx_var[obs[entry.obs_id]],idx_time[i],idx_sim[entry.simulation_id]]
            loss += (sim_value - meas_value)^2
            n+= 1
        end
        if max_t < max_t_gt
            # If the simulation did not reach the maximum time, add a penalty
            loss += unsolved_penalty * (max_t_gt - max_t)/max_t_gt
        end
        
        # Add Lasso and Ridge penalties if specified (Only if the surrogate model is not a NullModel)
        if alpha > 0.0
            loss += alpha * l1_ratio * sum(abs, p.surrogate)  + # Lasso penalty
                    alpha * (1 - l1_ratio) * sum(x -> x^2, p.surrogate) # Ridge penalty
        end

        return loss/n # Return the average loss
    end
    return deepcopy(loss_function)
end


function overwrite_conditions!(u0map, conditions)
    # Overwrite the initial conditions with the conditions dictionary
    for (cond, condu0map) in conditions
        for u0 in keys(u0map)
            if !haskey(condu0map, u0)
                condu0map[u0] = u0map[u0] # if the condition does not have the variable, use the original value
            end
        end
    end
    return conditions
end

function observable_dict(odesol, p, model::HybridModel, obs_funs::Dict)
    simvals = Array(odesol)
    obs_dict = Dict{ModelingToolkit.BasicSymbolic, Vector}()
    for (i,sym) in enumerate(unknowns(model.sys))
        obs_dict[sym] = simvals[i, :]
    end
    for (obs_sym, obs_fun) in obs_funs
        obs_dict[obs_sym] = [obs_fun(vals, p.sys) for vals in eachcol(simvals)]
    end
    return obs_dict
end

import Optimization: OptimizationProblem, OptimizationFunction
function Optimization.OptimizationFunction(problem::HybridPEProblem; adalg = Optimization.AutoForwardDiff(),
                                                    kwargs...)
    loss_func = define_loss_function(problem.model, problem.observations, problem.measurements, problem.u0map; 
                        conditions = problem.conditions, tspan = problem.tspan, kwargs...)
    # Create the optimization function
    #create function that ignores second input
    f = (p, data) -> loss_func(p, data)
    opt_func = Optimization.OptimizationFunction(f, adalg)
    return opt_func
end

function Optimization.OptimizationProblem(problem::HybridPEProblem, p = init_params(problem.model);shuffle = false,
                                                    kwargs...)
    # Create the optimization problem
    dataloader = MLUtils.DataLoader(problem.measurements, batchsize = problem.batch_size, shuffle = shuffle)
    opt_func = Optimization.OptimizationFunction(problem; kwargs...)
    return OptimizationProblem(opt_func, p, dataloader; lb = problem.lb, ub = problem.ub)
end

function Optimization.EnsembleProblem(problem::HybridPEProblem;
                                                    n = 10, 
                                                    initp_samples = init_params(problem.model; n = n),
                                                    kwargs...)        
    if initp_samples[1] isa Real # If initp_samples is a single vector of parameters, convert it to a vector of ComponentVectors
        initp_samples = [initp_samples]
    end 
    optprob = Optimization.OptimizationProblem(problem, initp_samples[1]; kwargs...)
    function prob_func(prob, i, repeat)
        remake(prob, u0 = initp_samples[i])
    end
    return Optimization.EnsembleProblem(optprob, prob_func = prob_func)
end

function latin_hypercube(trajectories::Int, problem::HybridPEProblem; generations = 1000)
    n_parameters = length(init_params(problem.model))
    samples, _ = LHCoptim(trajectories,n_parameters,generations)
    scaled_samples = scaleLHC(samples, [(problem.lb[i], problem.ub[i]) for i in 1:n_parameters])
    return scaled_samples
end

