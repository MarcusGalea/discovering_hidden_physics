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
using Surrogates


#DANGEROUS CONVSERION INCOMING! MIGHT RUIN OPTIMIZATION (yet necessary for AD with BFGS)
using Base
import Base: convert
using ForwardDiff
Base.convert(::Type{T}, x::ForwardDiff.Dual)  where T <: Number = x.value

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
    events::Dict
    """ Observables that are computed during the simulation. (Defaults to unknowns of sys)"""
    observables::Vector
    """ Random number generator for reproducibility."""
    rng ::Random.AbstractRNG
    """ODEFunction for the HybridModel system."""
    ode_fun::Function
    """Machine learning model for the HybridModel system."""
    ml_models::Dict
    """Proportion of data used for modeling (Not smart to have here)"""
    data_proportion::Float64

    """ Construct a HybridModel system with a SINDy/ODE surrogate model. """
    HybridModel(sys::ODESystem, surrogate::T;
                events::Dict = Dict(), 
                observables::Vector = has_observed(sys) || has_observed(surrogate) ?  union([observables(sys);observables(surrogate)]) : unknowns(sys), #choose observables from sys or surrogate if they have them. else use state variables
                rng::Random.AbstractRNG = Random.default_rng(1234),
                ode_fun::Function = ODEFunction(sys, surrogate; rng = rng),
                ml_models::Dict = Dict(),
                data_proportion::Float64 = 1.0
                ) where 
                T <: ModelingToolkit.AbstractTimeDependentSystem = 
        new(sys, surrogate, events, observables, rng, ode_fun, ml_models, data_proportion)


    """   Construct a HybridModel system with a Lux neural network surrogate model."""
    HybridModel(sys::ODESystem, surrogate::T; 
               events::Dict = Dict(), 
               observables::Vector = has_observed(sys) || has_observed(surrogate) ?  union([observables(sys);observables(surrogate)]) : unknowns(sys), #choose observables from sys or surrogate if they have them. else use state variables
               rng::Random.AbstractRNG = Random.default_rng(1234),
               ode_fun::Function = ODEFunction(sys, surrogate; rng = rng),
               ml_models::Dict = Dict(:surrogate => surrogate),
               data_proportion::Float64 = 1.0
               ) where 
               T <: Lux.Chain =    
        new(sys, surrogate, events, observables, rng, ode_fun, ml_models, data_proportion)

    """ Construct a HybridModel system with a PEtab MLModel surrogate model."""
    HybridModel(sys::ODESystem, surrogate::T; 
               events::Dict = Dict(), 
               observables::Vector = has_observed(sys) || has_observed(surrogate) ?  union([observables(sys);observables(surrogate)]) : unknowns(sys), #choose observables from sys or surrogate if they have them. else use state variables
               rng::Random.AbstractRNG = Random.default_rng(1234),
               ode_fun::Function = ODEFunction(sys, surrogate; rng = rng),
               ml_models::Dict = Dict(:surrogate => surrogate),
               data_proportion::Float64 = 1.0
               ) where 
               {T <: PEtab.MLModel} =    
        new(sys, surrogate, events, observables, rng, ode_fun, ml_models, data_proportion)
end

# Convert a dictionary to a NamedTuple (for use with ComponentArrays with ModelingToolkit)
NamedTuple(dict::Dict) = (; (Symbol(string(k)) => v for (k, v) in dict)...) 

#### HIDDEN ODE METHODS ####

function init_params(model::HybridModel; 
    lb = ComponentArray(sys = -ones(length(parameters(model.sys))), 
                        surrogate = -ones(length(parameters(model.surrogate)))),
    ub = ComponentArray(sys = ones(length(parameters(model.sys))), 
                        surrogate = ones(length(parameters(model.surrogate)))),
    n = 1, # number of samples to generate
    method = :latin_hypercube_sampling, # method to use for sampling parameters
    )
    @unpack sys, surrogate, rng = model
    
    # Initialize parameters for the ODE system
    ode_ps = init_params(sys; rng = rng, n = n, lb = lb.sys, ub = ub.sys,
                        method = method) # 
    # Initialize parameters for the surrogate model
    surrogate_ps = init_params(surrogate; rng = rng, n = n, lb = lb.surrogate, ub = ub.surrogate, 
                               method = method)

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
        # @unpack sys, surrogate, rng = model
        @unpack ode_fun = model
        # return DifferentialEquations.ODEFunction(sys, surrogate; rng = rng)
        return ode_fun
end

#create the ODE function for the HybridModel system
function DifferentialEquations.ODEFunction(sys::ODESystem, surrogate::T; rng = Random.default_rng(1234)) where {T <: Union{ModelingToolkit.AbstractTimeDependentSystem, Lux.Chain, PEtab.MLModel}}
    # Get the derivative function for the ODE system
    ode_fun! = DifferentialEquations.ODEFunction(sys, unknowns(sys), parameters(sys))
    # Get the surrogate derivative function
    surrogate_fun! = derivative_function!(surrogate; rng = rng)

    if surrogate isa ModelingToolkit.AbstractTimeDependentSystem
        @assert isequal(unknowns(surrogate), unknowns(sys)) "Surrogate model variables are not in the same order as the known system."
    end
    du1 = Any[0.0 for _ in 1:length(unknowns(sys))]
    du2 = copy(du1) # Initialize du2 for the surrogate model
    function update_du!(du, u, p, t)
        # Compute the ODE derivatives
        ode_fun!(du1, u, p.sys, t)
        # Compute the surrogate derivatives
        surrogate_fun!(du2, u, p.surrogate, t) 
        # Combine the derivatives
        du .= du1 .+ du2
        return du
    end
    odefun! = remake(ode_fun!, f = update_du!) # Create a new ODEFunction with the combined derivatives
    return odefun!
end

findperm(a, b) = [findfirst(x -> isequal(x, y), b) for y in a]

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
    return solve(prob, alg(), EnsembleDistributed(), trajectories = length(u0s), saveat = time; model.events..., kwargs...)
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
                        generations = 1000, method = :latin_hypercube_sampling, radius = 1.0)
    if n > 1
        # If n > 1, return a vector of sampled Parameters
        if method == :latin_hypercube_sampling
            samples = latin_hypercube_sampling(sys, n, lb, ub; rng = rng, generations = generations)
            return samples
        elseif method == :spherical_sampling
            # return spherical_sampling(sys, n, radius; rng = rng)
            #TODO: Implement spherical sampling
        else
            @error "Unknown sampling method: $method"
        end 
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

    

function latin_hypercube_sampling(sys::ODESystem, n, lb, ub; rng = Random.default_rng(1234),
                                generations = 1000)
        n_parameters = length(parameters(sys))
        samples, _ = LHCoptim(n,n_parameters, generations, rng = rng)
        scaled_samples = scaleLHC(samples, [(lb[i], ub[i]) for i in 1:n_parameters])
        return [NamedTuple(Dict(Symbol(string(p)) => scaled_samples[i, j] for (j, p) in enumerate(parameters(sys)))) for i in 1:n]
end

function derivative_function!(sys::ODESystem; rng = Random.default_rng(1234))
    return DifferentialEquations.ODEFunction(sys, unknowns(sys), parameters(sys))
end

# @independent_variables t
function merge_systems(sys::AbstractTimeDependentSystem, surrogate::AbstractTimeDependentSystem)
    # Merge the ODE system and the surrogate model into a single system (Given they are both symbolic systems)
    D = Differential(t)
    #TODO
end

### NEURAL NETWORK METHODS ###
function init_params(nn::Lux.Chain; rng = Random.default_rng(1234), 
                     n = 1,lb = nothing, ub = nothing, method = nothing)
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
    """log transform parameters before optimization"""
    log_transform::Bool

    """ Create a HybridPEProblem for the HybridModel."""
    HybridPEProblem(model::HybridModel, observables::Dict, measurements::DataFrame, u0map; 
                  conditions::Dict = Dict("cond1" => u0map), 
                  tspan::Tuple = (0.0, maximum(measurements.time)),
                  ub = nothing,
                  lb = nothing,
                  batch_size = size(measurements, 1),
                  log_transform::Bool = false,
                  kwargs...
                    ) = new(model, u0map, measurements, 
                    conditions_in_data(conditions, measurements), 
                    observables, tspan,
                    define_loss_function(model, observables, measurements, u0map; 
                                         conditions = conditions_in_data(conditions, measurements), 
                                         log_transform = log_transform,
                                         tspan = tspan, kwargs...),
                    ub, lb, batch_size, log_transform)
end

function conditions_in_data(conditions::Dict, measurements::DataFrame)
    #remove conditions that don't exist in the measurements
    ids = unique(measurements.simulation_id)
    return Dict(k => v for (k, v) in conditions if k in ids)
end


function simulate_solution(prob::HybridPEProblem, p;
                            u0map = prob.u0map,
                            conditions = prob.conditions,
                            tspan = prob.tspan,
                            alg = Tsit5(),
                            ens_alg = EnsembleDistributed(),
                            saveat =  prob.measurements.time,
                            kwargs...
    )
    # Overwrite the initial conditions with the conditions dictionary
    u0_conditions = overwrite_conditions!(u0map, conditions)
    # Create an EnsembleProblem with the initial conditions and parameters
    ens_prob = DifferentialEquations.EnsembleProblem(prob.model, u0_conditions, tspan, p)
    # Solve the EnsembleProblem
    sim = solve(ens_prob, alg, ens_alg, trajectories = length(u0_conditions); prob.model.events..., saveat = saveat, kwargs...)
end

function ModelingToolkit.unknowns(model::HybridModel)
    # Get the unknowns of the HybridModel system
    return unknowns(model.sys)
end

function define_loss_function(model::HybridModel, obs::Dict, measurements::DataFrame, u0map; 
                              conditions::Dict = Dict("cond1" => u0map), 
                              tspan::Tuple = (0.0, maximum(measurements.time)),
                              alg = Tsit5(),
                              ens_alg = EnsembleDistributed(),
                              include_plot = false,
                              alpha = 0.0,
                              l1_ratio= 0.0,
                              assume_stable_data = true,
                              unsolved_penalty = 1e+16, # penalty for unsolved simulations,
                              log_transform = false,
                              sensealg = ForwardDiffSensitivity(),
                              force_dtmin = true,
                              random_sampling_percentage = 1.0,
                              kwargs...
                              )
    # Define
    u0_conditions = overwrite_conditions!(u0map, conditions)
    obs_funs = Dict([obs_fun.lhs =>eval(build_function(obs_fun.rhs, unknowns(hmodel.sys), parameters(hmodel.sys); expression=Val{false})) for obs_fun in observed(hmodel.sys)])
    
    idx_sim = Dict([cond => i for (i, cond) in enumerate(sort(collect(keys(u0_conditions))))]) #create a map from condition name to ensemble index
    idx_var = Dict([var => i for (i, var) in enumerate(unknowns(model.sys))]) #create a map from variable name to index
    ensemble_prob = EnsembleProblem(model, u0_conditions, tspan)
    max_t_gt = maximum(measurements.time) # Get the maximum time from the measurements
    unique_times = unique(measurements.time)

    function loss_function(p, second_argument)
        if log_transform
            p = exp.(p) # Exponentiate the parameters if log transformation is enabled
        end
        ensemble_prob = remake(ensemble_prob, p = p) #SUPER SMART!! This allows us to change the parameters of the ensemble problem without creating a new one
        t_upper_bound = max_t_gt
        if isnothing(second_argument)
            measurements_batch = measurements # If no measurements are provided, use the full measurements DataFrame
        elseif second_argument isa DataLoader
            measurements_batch = second_argument.data
        elseif second_argument isa DataFrame
            measurements_batch = second_argument
        elseif second_argument isa Float64
            #if second_argument is a Float64, we assume it's the proportion of data to use for the optimization function
            t_upper_bound = second_argument * max_t_gt
            ensemble_prob = remake(ensemble_prob, tspan = (0.0, t_upper_bound))
            measurements_batch = measurements[measurements.time .<= t_upper_bound, :] # Filter measurements to only include those within the time bounds
            model.data_proportion = second_argument # Update the data proportion in the model
            unique_times = unique(measurements_batch.time) # Update the unique times based on the filtered measurements
        else
            @error "No option for second argument"
        end
        # println("Proportion of data used for optimization: $(second_argument) with length $(size(measurements_batch, 1)) measurements.")
        sim = solve(ensemble_prob, alg, ens_alg, trajectories = length(u0_conditions); 
                    saveat = unique_times, 
                    sensealg = sensealg,
                    force_dtmin = force_dtmin,
                    model.events...,
                    kwargs...
                    ) #simulate data

                    if prod([SciMLBase.successful_retcode(trajectory) for trajectory in sim]) != 1
            @warn "Some simulations did not converge. Returning a high loss value."
            return Inf # Return a high loss value if any simulation did not converge
        end
        if random_sampling_percentage < 1.0
            # Randomly sample a percentage of the measurements
            n_samples = Int(floor(random_sampling_percentage * size(measurements_batch, 1)))
            rand_indices = randperm(size(measurements_batch, 1))[1:n_samples]
            measurements_batch = measurements_batch[rand_indices, :]
        end
        idx_time = Dict(time => i for (i, time) in enumerate(sim[1].t))
        obs_val_dict = [observable_dict(sim[i], p, model, obs_funs) for i in 1:length(u0_conditions)]
        loss = 0.0 
        n = size(measurements_batch, 1) # Number of measurements
        # some algorithms don't support DataLoaders, so we access dataframe directly

        for i in 1:size(measurements_batch, 1)
            entry = measurements_batch[i, :]
            @assert isapprox(sim[idx_sim[entry.simulation_id]].t[idx_time[entry.time]], entry.time, atol = 0.1) "The time in the measurement $(entry.time) does not match the simulation time $(sim[idx_sim[entry.simulation_id]].t[idx_time[entry.time]])"
            meas_value = entry.measurement
            sim_value = obs_val_dict[idx_sim[entry.simulation_id]][obs[entry.obs_id]][idx_time[entry.time]]
            loss += (sim_value - meas_value)^2
            n += 1
        end
        loss /= n

        # Add Lasso and Ridge penalties if specified (Only if the surrogate model is not a NullModel)
        if alpha > 0.0
            loss += alpha * l1_ratio * sum(abs, p.surrogate)  + # Lasso penalty
                    alpha * (1 - l1_ratio) * sum(x -> x^2, p.surrogate) # Ridge penalty
        end

        return loss
    end
    return deepcopy(loss_function)
end



function create_callback(peprob::HybridPEProblem;  plot_every = 30, report_every = 10, loss_upper_bound = 1e7, dt = 0.05,
                       xlabel = "Time", ylabel = "Population", title = "Lotka-Volterra", kwargs...)
    label = hcat(string.(unknowns(peprob.model.sys))...)
    data = peprob.measurements.measurement
    timedata = peprob.measurements.time
    function callback(state, l; trace = nothing) #callback function to observe training
        sim = simulate_solution(peprob, state.u)
        if !isnothing(trace)
            push!(trace, state)
        end
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
                println("Iteration: $(state.iter), Loss: $(l), sample_percentage$(peprob.model.data_proportion),Parameters: $(state.u)")
            end
        end
        if plot_every > 0
            if state.iter % plot_every == 0
            ps = peprob.log_transform ? exp.(state.u) : state.u # Exponentiate the parameters if log transformation is enabled
            p1 = plot(peprob; included_plots = [:data, :model],
                saveat = dt,
                data_proportion = peprob.model.data_proportion,
                obs_ids = keys(peprob.observations),
                xlabel = xlabel,
                ylabel = ylabel,
                title = title,
                kwargs...,
                p = ps)
            display(p1)
            end
        end
        return false
    end
    return deepcopy(callback)
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
    obs_dict = Dict{Union{ModelingToolkit.BasicSymbolic, SymbolicUtils.BasicSymbolic}, Vector}()
    for (i,sym) in enumerate(unknowns(model.sys))
        obs_dict[sym] = simvals[i, :]
    end
    # for obs in observables(hmodel.sys)
    #     obs_dict[obs] = simvals[obs]
    # end
    for (obs_sym, obs_fun) in obs_funs
        obs_dict[obs_sym] = [obs_fun(vals, p.sys) for vals in eachcol(simvals)]
    end
    return obs_dict
end

import Optimization: OptimizationProblem, OptimizationFunction
function Optimization.OptimizationFunction(problem::HybridPEProblem; adalg = Optimization.AutoForwardDiff(), use_batches = true,
                                                    kwargs...)
    # loss_func = define_loss_function(problem.model, problem.observations, problem.measurements, problem.u0map; 
                        # conditions = problem.conditions, tspan = problem.tspan, kwargs...)
    # Create the optimization function
    #create function that ignores second input
    # f = (p, data) -> loss_func(p, data)
    f = (params, extra) -> problem.obj_func(params, extra)
    opt_func = Optimization.OptimizationFunction(f, adalg)
    return opt_func
end

function Optimization.OptimizationProblem(problem::HybridPEProblem, p = init_params(problem.model);shuffle = false,
                                                    proportion_of_data = 1.0,
                                                    kwargs...)
    # Create the optimization problem
    if problem.log_transform
        @warn "Log transforming parameters"
        p = log.(p) # Log transform the parameters if specified
    end
    batchsize = problem.batch_size < size(problem.measurements, 1) ? problem.batch_size : size(problem.measurements, 1)
    if batchsize < size(problem.measurements, 1) #Dataloader with batch_size is used if < size of measurements
        opt_func = Optimization.OptimizationFunction(problem; use_batches = true, kwargs...)
        dataloader = MLUtils.DataLoader(problem.measurements, batchsize = batchsize, shuffle = shuffle)
        return OptimizationProblem(opt_func, p, dataloader; lb = problem.lb, ub = problem.ub)
    else
        opt_func = Optimization.OptimizationFunction(problem; use_batches = false, kwargs...)
        return OptimizationProblem(opt_func, p, proportion_of_data; lb = problem.lb, ub = problem.ub)
    end
    return
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

#### SINDy METHODS ####
function polynomial_basis(x::Array, degree::Int = 1)
    @assert degree > 0
    n_x = length(x)
    n_c = binomial(n_x + degree, degree)
    eqs = Array{Num}(undef, n_c)
    _check_degree(x) = sum(x) <= degree ? true : false
    itr = Base.Iterators.product([0:degree for i in 1:n_x]...)
    itr_ = Base.Iterators.Stateful(Base.Iterators.filter(_check_degree, itr))
    filled = false
    @inbounds for i in 1:n_c
        eqs[i] = 1
        filled = true
        for (xi, ci) in zip(x, popfirst!(itr_))
            if !iszero(ci)
                filled ? eqs[i] = xi^ci : eqs[i] *= xi^ci
                filled = false
            end
        end
    end
    eqs
end



species = unknowns(sys_known)
unknown_basis = polynomial_basis(species, 2) 

Base.occursin(needle::Num, haystack::Int64) = false # Override occursin for Num types to always return false whenever haystack is an Int64
function create_unknown_eqs(sys_known::AbstractTimeDependentSystem, unknown_basis::Vector{T}; Ξ = nothing) where T <: Union{Num}
    #remove basis functions that are already in the known system
    if isnothing(Ξ)
        @parameters Ξ[1:length(unknown_basis), 1:length(unknowns(sys_known))]
    end 
    known_species = unknowns(sys_known)
    known_eqs = equations(sys_known)
    Ξ = hcat(Ξ)
    #create empty vector to hold the unknown equations
    unknown_eqs = Vector{Any}(undef, length(known_eqs))
    #if equation is already in the known system, remove it from the unknown basis
    for (i,k_eq) in enumerate(known_eqs)
        eq_basis_funs = copy(unknown_basis)
        for (j, basis_fun) in enumerate(unknown_basis)
            if occursin(basis_fun, k_eq.rhs)
                eq_basis_funs[j] = 0
            end
        end
        unknown_eqs[i] = Equation(k_eq.lhs, sum(eq_basis_funs .* Ξ[i, :]))
    end 
    return unknown_eqs
end

function process_trace(trace; alg = ProgressivePolyOpt)
    # Process the trace data
    runs = [Dict{String, Vector{Any}}("ADAM_1" => [])]
    if alg == ProgressivePolyOpt
        # Apply specific processing for ProgressivePolyOpt
        n_partitions = 1
        @assert typeof(trace[1].original) <: Optimisers.Leaf #ADAM STATE OBJECT
        alg_prev = "ADAM"
        iter_prev = 0
        for (i, state) in enumerate(trace)
            run = runs[end]
            if typeof(state.original) <: Optimisers.Leaf #ADAM current
                if state.iter > iter_prev
                    push!(run["ADAM_$n_partitions"], state)
                else
                    if alg_prev == "ADAM" #we're in the same run, but it's a new partition
                        n_partitions += 1
                        #add new key to existing dict
                        run["ADAM_$n_partitions"] = [state]
                    elseif alg_prev == "BFGS"#same run. Different algorithm
                        push!(runs, Dict{String, Vector{Any}}("ADAM_1" => []))
                        n_partitions = 1
                    else
                        error("Unknown algorithm: $(alg_prev)")
                    end
                end
                alg_prev = "ADAM"
            elseif typeof(state.original) <: Optim.OptimizationState #BFGS
                if alg_prev == "ADAM" #We're in the same run, but different algorithm
                    run["BFGS"] = [state]
                elseif alg_prev == "BFGS"
                    push!(run["BFGS"], state)
                end
            iter_prev = state.iter
        end
    end
end
return runs
end

function plot_loss(traces, train_prob; val_prob = nothing, kwargs...)
    p1 = plot(; xlabel = "Iteration", ylabel = "Loss", title = "Loss Trace", kwargs...)
    for (i, trace) in enumerate(traces)
        iters = collect(1:length(trace))
        ADAM_idcs = [typeof(state.original) <: Optimisers.Leaf for state in trace]
        BFGS_idcs = .!ADAM_idcs
        loss_trace = [train_prob.obj_func(state.u,1.0) for state in trace]
        n_ADAM = sum(ADAM_idcs)
        switch_points = findall([[trace[i].iter < trace[i-1].iter for i in 2:n_ADAM]; false])
        n_switches = length(switch_points)
        plot!(p1, iters[ADAM_idcs], loss_trace[ADAM_idcs], label = "Train Loss (ADAM)", color = :blue, linewidth = 2)
        plot!(p1, iters[BFGS_idcs], loss_trace[BFGS_idcs], label = "Train Loss (BFGS)", color = :blue, linewidth = 2, linestyle = :dash)
        if val_prob !== nothing
            val_loss_trace = [val_prob.obj_func(state.u, 1.0) for state in trace]
            plot!(p1, iters[ADAM_idcs], val_loss_trace[ADAM_idcs], label = "Val Loss (ADAM)", color = :red, linewidth = 2)
            plot!(p1, iters[BFGS_idcs], val_loss_trace[BFGS_idcs], label = "Val Loss (BFGS)", color = :red, linewidth = 2, linestyle = :dash)
        end
        if !isempty(switch_points)
            data_proportion = 1/(n_switches+1)
            data_proportion_values = zeros(Float64, maximum(switch_points))
            for (j, sp) in enumerate(switch_points)
                data_proportion_values[end+1-sp:end] .+= data_proportion
            end
            vline!(p1, switch_points, label = "",  color = :black, linestyle = :dot, linewidth = 1.5, alpha = 0.5)
            partial_loss = [train_prob.obj_func(trace[i].u, data_proportion_values[i]) for i in 1:length(data_proportion_values)]
            plot!(p1, iters[1:length(partial_loss)], partial_loss, label = "Partial Train Loss", color = :green, linewidth = 2)

        end
    end
    return p1
end



""" Plotting Recipe """
@recipe function f(peprob::HybridPEProblem; included_plots = [:data, :model],
        p = init_params(peprob.model),
        data_proportion = peprob.model.data_proportion,
        colors = [:blue, :green, :orange, :purple, :magenta, :brown],
        curve_label = "fit",
        obs_ids = keys(peprob.observations),
        saveat = (peprob.tspan[2]-peprob.tspan[1]) / 200, # Default saveat is 200 points over the tspan
        cond_ids = sort(collect(keys(peprob.conditions))),
        opacity = 0.33,
        )

    t_cutoff = data_proportion * peprob.tspan[2] # Cutoff time for the data proportion
    if :model in included_plots
        sim = simulate_solution(peprob, p; saveat = saveat)
    end

    idx_sim = Dict([cond => i for (i, cond) in enumerate(cond_ids)]) #create a map from condition name to ensemble index

        for (i, cond_id) in enumerate(cond_ids)
            for (j, obs_id) in enumerate(obs_ids)
                # Filter measurements for the current condition and observation ID
                color = colors[mod1(i + (j-1)*length(obs_ids), length(colors))]

                if :data in included_plots
                    meas = peprob.measurements[peprob.measurements.obs_id .== obs_id .&& 
                                                peprob.measurements.simulation_id .== cond_id, :]
                    if !isempty(meas)

                        x = meas.time
                        y = meas.measurement
                        x_normal = x[x .<= t_cutoff] # Filter x values to only include those within the cutoff time
                        y_normal = y[x .<= t_cutoff] # Filter y values to only include those within the cutoff time
                        @series begin
                            label --> "$cond_id - $obs_id"
                            color --> color
                            seriestype --> :scatter
                            x_normal, y_normal
                        end
                        if data_proportion < 1.0
                            x_opaque = x[x .> t_cutoff]
                            y_opaque = y[x .> t_cutoff]
                            @series begin
                                label --> ""
                                color --> color
                                seriestype --> :scatter
                                alpha --> opacity # Make the points semi-transparent
                                x_opaque, y_opaque
                            end
                        end
                    end
                    if :model in included_plots
                        x = sim[idx_sim[cond_id]].t
                        y = sim[idx_sim[cond_id]][peprob.observations[obs_id]]
                        x_normal = x[x .<= t_cutoff] # Filter x values to only include those within the cutoff time
                        y_normal = y[x .<= t_cutoff] # Filter y values to only include those within the cutoff time
                        @series begin
                            label --> "$cond_id - $(obs_id)_$curve_label"
                            color --> color
                            seriestype --> :line
                            x_normal, y_normal
                        end
                        if data_proportion < 1.0
                            x_opaque = x[x .> t_cutoff]
                            y_opaque = y[x .> t_cutoff]
                            @series begin
                                label --> ""
                                color --> color
                                seriestype --> :line
                                alpha --> opacity # Make the line semi-transparent
                                x_opaque, y_opaque
                            end
                        end
                    end
                end
            end
        end
end



function plot_hidden_dynamics(peprob::HybridPEProblem;
                            use_measurements = false,
                            p_est = init_params(peprob.model),
                            p_true = init_params(peprob.model),
                            dt = (peprob.tspan[2] - peprob.tspan[1]) / 200,
                            colors = [:blue, :green, :orange, :purple, :magenta, :brown],
                            xlabel = "Time", ylabel = "Dynamics", title = "Hidden Dynamics of ODE System",
                            seriestype = :line, color = :blue, linewidth = 2,
                            kwargs...
                            )
    @unpack sys, surrogate,rng = peprob.model
    p1 = plot(layout = (1,2);kwargs...)
    ode_fun! = DifferentialEquations.ODEFunction(sys, unknowns(sys), parameters(sys))
    # Get the surrogate derivative function
    surrogate_fun! = derivative_function!(surrogate; rng = rng)
    du_sys = zeros(length(unknowns(sys)))
    du_surrogate = copy(du_sys)  
    if use_measurements
        all_vars = Set(unknowns(sys))
        obs_vars = Set(peprob.observations)
        if !issubset(obs_vars, all_vars)
            @warn "Not all observation variables are in the system's unknowns."
        end
        #TODO
    else
        sim = simulate_solution(peprob, p_true, saveat = dt)
        for (i, traj) in enumerate(sim)
            dU_sys = zeros(length(unknowns(sys)), length(traj.t))
            dU_surrogate = copy(dU_sys)
            if !isnothing(p_true)
                dU_sys_true = zeros(length(unknowns(sys)), length(traj.t))
                dU_surrogate_true = copy(dU_sys_true)
            end
            for (i,tval) in enumerate(traj.t)
                # Get the state at time tval
                u = traj.u[i]
                # Compute the surrogate dynamics
                dU_surrogate[:,i] = surrogate_fun!(dU_surrogate[:,i], u, p_est.surrogate, tval)
                # Compute the ODE dynamics
                dU_sys[:,i] = ode_fun!(dU_sys[:,i], u, p_est.sys, tval)
                if !isnothing(p_true)
                    dU_surrogate_true[:,i] = surrogate_fun!(dU_surrogate_true[:,i], u, p_true.surrogate, tval)
                    dU_sys_true[:,i] = ode_fun!(dU_sys_true[:,i], u, p_true.sys, tval)
                end
            end
            plot!(p1, traj.t, dU_sys', label = "d".*hcat(string.(unknowns(sys))...).*"/dt_est_$i", subplot = 1, 
                    color = hcat(colors...), legend = Symbol(:outer, :left), title = "Known System Dynamics", xlabel = xlabel, ylabel = ylabel)
            plot!(p1, traj.t, dU_surrogate', label = "d".*hcat(string.(unknowns(sys))...).*"/dt_est_$i", subplot = 2, 
                    color = hcat(colors...), legend = Symbol(:outer, :right), title = "Surrogate Dynamics", xlabel = xlabel, ylabel = ylabel)
            if !isnothing(p_true)
                plot!(p1, traj.t, dU_sys_true', label = "d".*hcat(string.(unknowns(sys))...).*"/dt_true_$i", subplot = 1, color = hcat(colors...), legend = Symbol(:outer, :left), xlabel = xlabel, ylabel = ylabel)
                plot!(p1, traj.t, dU_surrogate_true', label = "d".*hcat(string.(unknowns(sys))...).*"/dt_true_$i", subplot = 2, color = hcat(colors...), legend = Symbol(:outer, :right), xlabel = xlabel, ylabel = ylabel)
            end
        end
    end
    return p1
end