using PEtab
using DifferentialEquations
using Lux
using ModelingToolkit
using Random
import ModelingToolkit:has_observed,observables
import Optimization:solve
using SciMLSensitivity
using DataFrames
using ComponentArrays



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
                ml_models::Dict{String, Any} = Dict{String, Any}(),
                ) where 
                T <: ModelingToolkit.AbstractTimeDependentSystem = 
        new(sys, surrogate, events, observables, rng, ode_fun, ml_models)


    """   Construct a HybridModel system with a Lux neural network surrogate model."""
    HybridModel(sys::ODESystem, surrogate::T; 
               events::Vector = [], 
               observables::Vector = has_observed(sys) || has_observed(surrogate) ?  union([observables(sys);observables(surrogate)]) : unknowns(sys), #choose observables from sys or surrogate if they have them. else use state variables
               rng::Random.AbstractRNG = Random.default_rng(1234),
               ode_fun::Function = ODEFunction(sys, surrogate; rng = rng),
               ml_models::Dict{String, Any} = Dict(:surrogate => surrogate),
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

function init_params(model::HybridModel)
    @unpack sys, surrogate, rng = model
    
    # Initialize parameters for the ODE system
    ode_ps = init_params(sys)
    # Initialize parameters for the surrogate model
    surrogate_ps = init_params(surrogate; rng = rng)

    # Combine the parameters into a NamedTuple
    # combined_ps = merge(ode_ps, surrogate_ps)
    combined_ps = merge((;sys = ode_ps), (;surrogate = surrogate_ps))
    return ComponentVector{Float64}(combined_ps)
end

function ODEFunction(model::HybridModel)
        @unpack sys, surrogate, rng = model
        return ODEFunction(sys, surrogate; rng = rng)
end

#create the ODE function for the HybridModel system
function ODEFunction(sys::ODESystem, surrogate::T; rng = Random.default_rng(1234)) where {T <: Union{ModelingToolkit.AbstractTimeDependentSystem, Lux.Chain, PEtab.MLModel}}
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
function ODEProblem(model::HybridModel, u0::Union{Vector, ComponentArray}, tspan, p = init_params(model))
    # @unpack ode_fun = model
    ode_fun = ODEFunction(model)
    prob = DifferentialEquations.ODEProblem(ode_fun, u0, tspan, p)
end

function ODEProblem(model::HybridModel, u0::Dict, tspan, p = init_params(model))
    # @unpack ode_fun, sys= model
    ode_fun = ODEFunction(model)
    sys = model.sys
    # Convert the dictionary to a vector
    u0_vec = [u0[var] for var in unknowns(sys)]
    prob = DifferentialEquations.ODEProblem(ode_fun, u0_vec, tspan, p)
end


function EnsembleProblem(model::HybridModel, u0s::Vector{Vector{T}}, tspan, p = init_params(model)) where {T <: Any}
    prob = ODEProblem(model, u0s[1], tspan, p)
    function prob_func(prob, i, repeat)
        remake(prob, u0 = u0s[i])
    end
    return EnsembleProblem(prob, prob_func = prob_func)
end

function EnsembleProblem(model::HybridModel, u0s::Dict{String, Dict}, tspan, p = init_params(model))
    #Assuming u0s is a Dictionary where keys are simulation IDs and values are dictionaries of initial conditions
    #get sorted keys
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
    return EnsembleProblem(prob, prob_func = prob_func)
end

function EnsembleProblem(model::HybridModel, u0s::Any, tspan, p = init_params(model)) #
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
function init_params(sys::ODESystem; randfun = rand, rng = Random.default_rng(1234))
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
end

### NEURAL NETWORK METHODS ###
function init_params(nn::Lux.Chain; rng = Random.default_rng(1234))
    # Get the parameters of the Lux model
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

"""
Parameter Estimation Problem for HybridModel.
    This problem involves estimating the parameters of a HybridModel that combines
    both ODE and machine learning components.
"""
struct HybridPEProblem
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

    #create constructor for HybridModelPE
    HybridPEProblem(model::HybridModel, observables::Dict, measurements::DataFrame, u0map; 
                  conditions::Dict = overwrite_conditions!(u0map, Dict()), 
                  tspan::Tuple = (0.0, maximum(measurements.time)),
                  kwargs...
                    ) = new(model, u0map, measurements, conditions, observables, tspan,
                    define_loss_function(model, observables, measurements, u0map; 
                                         conditions = conditions, tspan = tspan, kwargs...))
end

function define_loss_function(model::HybridModel, obs::Dict, measurements::DataFrame, u0map; 
                              conditions::Dict = Dict(), 
                              tspan::Tuple = (0.0, maximum(measurements.time)),
                              alg = Tsit5(),
                              ens_alg = EnsembleDistributed(),
                              include_plot = false,
                              lasso_penalty = 0.0,
                              ridge_penalty = 0.0,
                              )
    # Define
    u0_conditions = overwrite_conditions!(u0map, conditions)
    obs_funs = Dict([obs_fun.lhs =>eval(build_function(obs_fun.rhs, unknowns(model.sys); ps = parameters(model.sys), expression=Val{false})) for obs_fun in observed(model.sys)])
    
    idx_sim = Dict([cond => i for (i, cond) in enumerate(sort(collect(keys(u0_conditions))))]) #create a map from condition name to ensemble index
    idx_var = Dict([var => i for (i, var) in enumerate(unknowns(model.sys))]) #create a map from variable name to index
    idx_time = invperm(sortperm(measurements.time)) # Get the permutation to sort timevals
    # ensemble_prob = EnsembleProblem(model, u0_conditions, tspan)

    function loss_function(p)
        # ensemble_prob = remake(ensemble_prob, p = p) #SUPER SMART!! This allows us to change the parameters of the ensemble problem without creating a new one
        ensemble_prob = EnsembleProblem(model, u0_conditions, tspan, p)
        sim = solve(ensemble_prob, alg, ens_alg, trajectories = length(u0_conditions), saveat = measurements.time, sensealg = ForwardDiffSensitivity()) #Array

        obs_val_dict = [observable_dict(sim[i], p, model, obs_funs) for i in 1:length(u0_conditions)]
        loss = 0.0 
        # println(sim)
        for i in 1:size(measurements, 1)
            entry = measurements[i, :]
            meas_value = entry.measurement
            sim_value = obs_val_dict[idx_sim[entry.simulation_id]][obs[entry.obs_id]][idx_time[i]]
            # sim[idx_var[obs[entry.obs_id]],idx_time[i],idx_sim[entry.simulation_id]]
            loss += (sim_value - meas_value)^2
        end
        #
        # Add Lasso and Ridge penalties if specified (Only if the surrogate model is not a NullModel)
        if lasso_penalty > 0.0
            loss += lasso_penalty * sum(abs, p.surrogate) 
        end
        if ridge_penalty > 0.0
            loss += ridge_penalty * sum(p.surrogate .^ 2)
        end

        return loss
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
        obs_dict[obs_sym] = [obs_fun(vals, p) for vals in eachcol(simvals)]
    end
    return obs_dict
end