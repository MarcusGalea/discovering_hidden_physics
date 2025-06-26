## Lotka-Volterra equations
using Pkg
Pkg.activate(joinpath(@__DIR__, "..","UDEs"))
using Revise, Optimization, ModelingToolkit,DifferentialEquations,Plots,Lux
## Lotka-Volterra equations
@parameters α β γ δ
@independent_variables  t
@variables x(t) y(t) z(t)
Dt = Differential(t)
eqs = [
    Dt(x) ~ α * x - β * x * y,
    Dt(y) ~ δ * x * y - γ * y,
]
measured_quantities = [z ~ x + y]  # Example of a measured quantity
sys = ODESystem(eqs, t, [x, y], [α, β, γ, δ]; observed = measured_quantities)
sys = complete(sys)
params =  Dict([α => 0.1, 
                β => 0.02, 
                δ => 0.01,
                γ => 0.3])

u0 = Dict([x => 40.0, y => 9.0])
tspan = (0.0, 200.0)
dt = 0.1

sys = complete(sys)
odefun = ODEFunction(sys, unknowns(sys), parameters(sys))
prob = ODEProblem(odefun, [40.0, 9.0], tspan, [0.1, 0.02, 0.01, 0.3])
sol = solve(prob, Tsit5(), saveat=dt)
data = hcat(sol.u...)'
plot(sol, vars=(x, y), xlabel="prey", ylabel="predator",
     title="Lotka-Volterra Model", label="Solution",
     legend=:topright, linewidth=2, markersize=4)






@variables z(t)
known_eqs = [
    Dt(x) ~ α * x,
    Dt(y) ~ -γ * y
]
unknown_eqs = [
    Dt(x) ~ -β * x * y,
    Dt(y) ~ δ * x * y,
    
]
deviance = 0.1 # deviance for the unknown equations
params_guess_known = Dict([α => 0.1, # + deviance * randn()
                        γ => 0.3, # + deviance * randn()])
                        ])

params_guess_unknown = Dict([β => 0.02,# + deviance * randn()
                        δ => 0.01 # + deviance * randn()])
                        ])


@named sys_known = ODESystem(known_eqs, t, [x, y, z], [α, γ], defaults = params_guess_known, observed = [z ~ x + y])
@named sys_unknown = ODESystem(unknown_eqs, t, [x, y], [β, δ], defaults = params_guess_unknown)
sys_known = complete(sys_known)
sys_unknown = complete(sys_unknown)


hmodel = HiddenODE(sys_known, sys_unknown)

initp = init_params(hmodel)
odefun = ODEFunction(hmodel)
prob = ODEProblem(odefun, [40.0, 9.0], tspan, initp)
sol = solve(prob, Tsit5(), saveat=dt)
data = hcat(sol.u...)'
plot(sol.t, data, label=["Prey" "Predator"], xlabel="Time", ylabel="Population", title="Lotka-Volterra Model with Hidden ODE")





#### PETAB
##Convert data to dataframe
using DataFrames
n_data = size(data, 1)
sample_size = 200


sample_idcs = rand(1:n_data, sample_size)
prey_df = DataFrame(simulation_id = "cond1", obs_id = "prey_o", time = time[sample_idcs], measurement= data[sample_idcs, 1],)
predator_df = DataFrame(simulation_id = "cond1", obs_id = "predator_o", time = time[sample_idcs], measurement= data[sample_idcs, 2],)
measurements = vcat(prey_df, predator_df)

#### PETAB Model
using PEtab
#Setup observables
@parameters σ
obs_x = PEtabObservable(x, σ)
obs_y = PEtabObservable(y, σ)
obs = Dict("prey_o" => obs_x, "predator_o" => obs_y)


#setup initial conditions
cond1 = Dict(:x => 40.0, :y => 9.0)
conds = Dict("cond1" => cond1)
# Setup parameters

#model parameters

estimate = true # Set to true if you want to estimate the parameters
p_α = PEtabParameter(α, lb = 1e-6, ub = 1e0, estimate = estimate, scale = :lin, value = 0.1)
p_β = PEtabParameter(β, lb = 1e-6 , ub = 1e0, estimate = estimate, scale = :lin, value = 0.02)
p_γ = PEtabParameter(γ, lb = 1e-6, ub = 1e0, estimate = estimate, scale = :lin, value = 0.3)
p_δ = PEtabParameter(δ, lb = 1e-6, ub = 1e0, estimate = estimate, scale = :lin, value = 0.01)
#noise parameter
p_σ = PEtabParameter(σ, lb = 1e-6, ub = 1e0, estimate = true, scale = :lin, value = 0.2)
pest = [
    p_α, p_β, p_γ, p_δ, p_σ
]
model = PEtabModel(sys,obs, measurements, pest; simulation_conditions  = conds)
petab_prob = PEtabODEProblem(model)



import ModelingToolkit:has_observed,observables
import Optimization:solve

""" HiddenODE is a system that combines an ODE system with a surrogate model (SINDy or neural network)."""
mutable struct HiddenODE
    """ The known underlying ODE system."""
    sys::ODESystem
    """ The surrogate model, which can be a normal ODESystem, SINDy model, or a Lux neural network."""
    surrogate::Union{ModelingToolkit.AbstractTimeDependentSystem, Lux.Chain}
    """ Discrete events that trigger during the simulation."""
    events::Vector
    """ Observables that are computed during the simulation. (Defaults to unknowns of sys)"""
    observables::Vector
    """ Random number generator for reproducibility."""
    rng ::Random.AbstractRNG
    """ODEFunction for the HiddenODE system."""
    ode_fun::Function

    
    """ Construct a HiddenODE system with a SINDy/ODE surrogate model. """
    HiddenODE(sys::ODESystem, surrogate::T;
                events::Vector = [], 
                observables::Vector = has_observed(sys) || has_observed(surrogate) ?  union([observables(sys);observables(surrogate)]) : unknowns(sys), #choose observables from sys or surrogate if they have them. else use state variables
                rng::Random.AbstractRNG = Random.default_rng(1234),
                ode_fun::Function = ODEFunction(sys, surrogate; rng = rng)
                ) where 
                {T <: ModelingToolkit.AbstractTimeDependentSystem} = 
        new(sys, surrogate, events, observables, rng, ode_fun)


    """   Construct a HiddenODE system with a Lux neural network surrogate model."""
    HiddenODE(sys::ODESystem, surrogate::T,; 
               events::Vector = [], 
               observables::Vector = has_observed(sys) || has_observed(surrogate) ?  union([observables(sys);observables(surrogate)]) : unknowns(sys), #choose observables from sys or surrogate if they have them. else use state variables
               rng::Random.AbstractRNG = Random.default_rng(1234),
               ode_fun::Function = ODEFunction(sys, surrogate; rng = rng)
               ) where 
               {T <: Lux.Chain} =    
        new(sys, surrogate, events, observables, rng, ode_fun)
end

# Convert a dictionary to a NamedTuple (for use with ComponentArrays with ModelingToolkit)
NamedTuple(dict::Dict) = (; (Symbol(string(k)) => v for (k, v) in dict)...) 

#### HIDDEN ODE METHODS ####

function init_params(model::HiddenODE)
    @unpack sys, surrogate, rng = model
    
    # Initialize parameters for the ODE system
    ode_ps = init_params(sys)
    # Initialize parameters for the surrogate model
    surrogate_ps = init_params(surrogate; rng)
    
    # Combine the parameters into a NamedTuple
    combined_ps = (; ode = ode_ps, surrogate = surrogate_ps)
    return ComponentVector{Float64}(combined_ps)
end

function ODEFunction(model::HiddenODE)
        @unpack sys, surrogate, rng = model
        return ODEFunction(sys, surrogate; rng = rng)
end

#create the ODE function for the HiddenODE system
function ODEFunction(sys::ODESystem, surrogate::T; rng = Random.default_rng(1234)) where {T <: Union{ModelingToolkit.AbstractTimeDependentSystem, Lux.Chain}}
    # Get the derivative function for the ODE system
    ode_fun! = ODEFunction(sys)
    # Get the surrogate derivative function
    surrogate_fun! = derivative_function!(surrogate; rng = rng)

    du1 = zeros(length(unknowns(sys))) # Initialize du1 for the ODE system
    du2 = copy(du1) # Initialize du2 for the surrogate model
    function update_du!(du, u, p, t)
        # Compute the ODE derivatives
        ode_fun!(du1, u, p.ode, t) 
        # Compute the surrogate derivatives
        surrogate_fun!(du2, u, p.surrogate, t) 
        # Combine the derivatives   
        du.= du1 .+ du2 
        return du  
    end
    odefun! = remake(ode_fun!, f = update_du!) # Create a new ODEFunction with the combined derivatives
    return odefun!
end

function Problem(model::HiddenODE, u0::Vector, tspan, p = init_params(model))
    @unpack ode_fun = model
    prob = ODEProblem(ode_fun, u0, tspan, p)
end

function Problem(model::HiddenODE, u0s::Vector{Vector}, tspan, p = init_params(model))
    prob = ODEProblem(model, u0s[1], tspan, p)
    function prob_func(prob, i, repeat)
        remake(prob, u0 = u0s[i])
    end
    return EnsembleProblem(prob, prob_func = prob_func)
end

function solve(model::HiddenODE, u0, time, p = init_params(model); alg = Tsit5, kwargs...)
    tspan = (time[1], time[end])
    prob = Problem(model, u0, tspan, p)
    if prob isa EnsembleProblem
        return solve(prob, alg(), EnsembleDistributed(), trajectories = length(u0), saveat = time, kwargs...)
    elseif prob isa ODEProblem
        return solve(prob, alg(), saveat = time, kwargs...)
    end
end

### ODE SYSTEM METHODS ###
function init_params(sys::ODESystem; randfun = rand, rng = Random.default_rng(1234))
    if ModelingToolkit.has_defaults(sys)
        defaults = ModelingToolkit.get_defaults(sys)
        # Initialize parameters with defaults (NamedTuple))
        return (; (Symbol(string(p)) => defaults[p] for p in parameters(sys) if p in keys(defaults))...)
    else
        # If no defaults, return an empty random values
        return (; (Symbol(string(p)) => randfun() for p in parameters(sys))...)
    end
end


function derivative_function!(sys::ODESystem; rng = Random.default_rng(1234))
    return ODEFunction(sys)
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
