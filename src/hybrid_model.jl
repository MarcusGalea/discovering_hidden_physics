using PEtab
using DifferentialEquations
using Lux
using ModelingToolkit
using Random
import ModelingToolkit:has_observed,observables
import Optimization:solve
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
                {T <: ModelingToolkit.AbstractTimeDependentSystem} = 
        new(sys, surrogate, events, observables, rng, ode_fun, ml_models)


    """   Construct a HybridModel system with a Lux neural network surrogate model."""
    HybridModel(sys::ODESystem, surrogate::T; 
               events::Vector = [], 
               observables::Vector = has_observed(sys) || has_observed(surrogate) ?  union([observables(sys);observables(surrogate)]) : unknowns(sys), #choose observables from sys or surrogate if they have them. else use state variables
               rng::Random.AbstractRNG = Random.default_rng(1234),
               ode_fun::Function = ODEFunction(sys, surrogate; rng = rng),
               ml_models::Dict{String, Any} = Dict("surrogate" => surrogate),
               ) where 
               {T <: Lux.Chain} =    
        new(sys, surrogate, events, observables, rng, ode_fun, ml_models)

    """ Construct a HybridModel system with a PEtab MLModel surrogate model."""
    HybridModel(sys::ODESystem, surrogate::T; 
               events::Vector = [], 
               observables::Vector = has_observed(sys) || has_observed(surrogate) ?  union([observables(sys);observables(surrogate)]) : unknowns(sys), #choose observables from sys or surrogate if they have them. else use state variables
               rng::Random.AbstractRNG = Random.default_rng(1234),
               ode_fun::Function = ODEFunction(sys, surrogate; rng = rng),
               ml_models::Dict{String, Any} = Dict("surrogate" => surrogate),
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
    surrogate_ps = init_params(surrogate; rng)
    
    # Combine the parameters into a NamedTuple
    combined_ps = merge(ode_ps, surrogate_ps)
    return ComponentVector{Any}(combined_ps)
end

function ODEFunction(model::HybridModel)
        @unpack sys, surrogate, rng = model
        return ODEFunction(sys, surrogate; rng = rng)
end

#create the ODE function for the HybridModel system
function ODEFunction(sys::ODESystem, surrogate::T; rng = Random.default_rng(1234)) where {T <: Union{ModelingToolkit.AbstractTimeDependentSystem, Lux.Chain}}
    # Get the derivative function for the ODE system
    ode_fun! = DifferentialEquations.ODEFunction(sys)
    # Get the surrogate derivative function
    surrogate_fun! = derivative_function!(surrogate; rng = rng)

    du1 = zeros(length(unknowns(sys))) # Initialize du1 for the ODE system
    du2 = copy(du1) # Initialize du2 for the surrogate model
    function update_du!(du, u, p, t)
        # Compute the ODE derivatives
        ode_fun!(du1, u, p, t) 
        # Compute the surrogate derivatives
        surrogate_fun!(du2, u, p, t) 
        # Combine the derivatives   
        du.= du1 .+ du2 
        return du  
    end
    odefun! = remake(ode_fun!, f = update_du!) # Create a new ODEFunction with the combined derivatives
    return odefun!
end

function ODEProblem(model::HybridModel, u0::Union{Vector, ComponentArray}, tspan, p = init_params(model))
    @unpack ode_fun = model
    prob = DifferentialEquations.ODEProblem(ode_fun, u0, tspan, p)
end

function EnsembleProblem(model::HybridModel, u0s::Vector{Vector{T}}, tspan, p = init_params(model)) where {T <: Any}
    prob = ODEProblem(model, u0s[1], tspan, p)
    function prob_func(prob, i, repeat)
        remake(prob, u0 = u0s[i])
    end
    return EnsembleProblem(prob, prob_func = prob_func)
end

function solve(model::HybridModel, u0s, time, p = init_params(model); alg = Tsit5, kwargs...)
    tspan = (time[1], time[end])
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
    if ModelingToolkit.has_defaults(sys)
        defaults = ModelingToolkit.get_defaults(sys)
        params = parameters(sys)
        # Initialize parameters with defaults (NamedTuple))
        return (; (Symbol(string(p)) => defaults[p] for p in parameters(sys) if p in keys(defaults))...)
    else
        # If no defaults, return an empty random values
        return (; (Symbol(string(p)) => randfun() for p in parameters(sys))...)
    end
end


function derivative_function!(sys::ODESystem; rng = Random.default_rng(1234))
    return DifferentialEquations.ODEFunction(sys)
end

### NEURAL NETWORK METHODS ###
function init_params(nn::Lux.Chain; rng = Random.default_rng(1234))
    # Get the parameters of the Lux model
    return (; surrogate = Lux.initialparameters(rng, nn))
end

function derivative_function!(nn::Lux.Chain; rng = Random.default_rng(1234))
    # Create a function that computes the derivatives of the Lux model
    st = Lux.initialstates(rng, nn)
    #NeuralODE. The output of the neural network is a vector of derivatives. Network only depends on state and network parameters.
    du = (du, u, p, t) -> first(nn(u, p, st))
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
    return (; surrogate = model.ps)
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

