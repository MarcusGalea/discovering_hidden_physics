## Lotka-Volterra equations
using Pkg
Pkg.activate(@__DIR__)
using Revise, Optimization, ModelingToolkit,DifferentialEquations,Plots
@parameters α β γ δ
@independent_variables  t
@variables x(t) y(t) z(t)
Dt = Differential(t)
eqs = [
    Dt(x) ~ α * x - β * x * y,
    Dt(y) ~ δ * x * y - γ * y,
    z ~ x + y
]
@named sys = ODESystem(eqs, t, [x, y, z], [α, β, γ, δ])
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

print(odefun([0.0, 0.0], [40.0, 9.0], [0.1, 0.02, 0.01, 0.3], 0.0))





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


@named sys_known = ODESystem(known_eqs, t, [x, y, z], [α, γ], defaults = params_guess_known)
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

mutable struct HiddenODE
    sys::ODESystem
    surrogate::Union{ModelingToolkit.AbstractTimeDependentSystem, Lux.Chain}
    events::Vector
    observables::Vector
    rng ::Random.AbstractRNG
    
    """ Construct a HiddenODE system with a SINDy/ODE surrogate model. """
    HiddenODE(sys::ODESystem, surrogate::T;
                events::Vector = [], 
                observables::Vector = [],
                rng::Random.AbstractRNG = Random.default_rng(1234)
                ) where 
                {T <: ModelingToolkit.AbstractTimeDependentSystem} = 
        new(sys, surrogate, events, observables, rng)


    """   Construct a HiddenODE system with a Lux neural network surrogate model."""
    HiddenODE(sys::ODESystem, surrogate::T,; 
               events::Vector = [], 
               observables::Vector = [],
               rng::Random.AbstractRNG = Random.default_rng(1234)
               ) where 
               {T <: Lux.Chain} =    
        new(sys, surrogate, events, observables, rng)
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

#create the ODE function for the HiddenODE system
function ODEFunction(model::HiddenODE)
    @unpack sys, surrogate, rng = model
    # Get the derivative function for the ODE system
    ode_fun = derivative_function!(sys)
    # Get the surrogate derivative function
    surrogate_fun = derivative_function!(surrogate; rng = rng)

    du1 = zeros(length(unknowns(sys))) # Initialize du1 for the ODE system
    du2 = copy(du1) # Initialize du2 for the surrogate model
    function update_du!(du, u, p, t)
        # Compute the ODE derivatives
        ode_fun(du1, u, p.ode, t) 
        # Compute the surrogate derivatives
        surrogate_fun(du2, u, p.surrogate, t) 
        # Combine the derivatives   
        du.= du1 .+ du2 
        return du  
    end
    return deepcopy(update_du!)
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
    # Create a function that computes the derivatives of the system
    # params = parameters(sys)
    # vars = unknowns(sys)

    # ode_fun = ODEFunction(sys, vars, params)
    # dfun! = (du, u, p, t) -> ode_fun(du, u, collect(p), t)
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
    #NeuralODE. The output of the neural network is a vector of derivatives
    du = (du, u, p, t) -> first(nn(u, p, st))
    return deepcopy(du)
end

