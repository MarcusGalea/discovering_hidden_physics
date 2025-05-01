# Create enzyme dinamyics model
using Pkg
Pkg.activate(@__DIR__)
using Catalyst
using ModelingToolkit
using Lux
using Optimization
using DifferentialEquations
using Plots
using Statistics
using Random
using SciMLSensitivity
using OptimizationPolyalgorithms
using ComponentArrays
using PhysicsInformedRegression


rn = @reaction_network begin
    k1, E + S --> ES
    k2, ES --> E + S
end
rn = complete(rn)


# Dissociation event
diss_time = 5.0 # Time of the event
diss_condition(u,even_time,integrator) = even_time == diss_time # Set the time of the event to 5.0 seconds
affect!(integrator) = integrator.u[1] = 0.0*integrator.u[1] # Set the concentration of E to 0
cb =DiscreteCallback(diss_condition, affect!)

#Data setup
dt = 0.1 # time step
u0 = [10.0, 1.0, 0.0]
tspan = (0.0, 10.0)
p = [0.4, 0.3] # k1, k2
time = collect(tspan[1]:dt:tspan[2]) # time vector
prob = ODEProblem(rn, u0, tspan, p)
data = solve(prob, Tsit5(), saveat = time, callback = cb, tstops = [diss_time])
du_actual = data.(data.t, Val{1})

#add noise to data

# time = data.t
signal =  hcat(data.u...)
noise = 0.02 * randn(size(signal))
signal = signal .+ noise
plot(data.t, signal', xlabel = "t", ylabel = "Concentration", title = "Enzyme Kinetics", label = ["E" "S" "ES"])
plot(data.t, hcat(du_actual...)', xlabel = "t", ylabel = "Concentration Derivative", title = "Enzyme Kinetics", label = ["E" "S" "ES"])


### ENSEMBLE DATA ###

initial_conditions  = [
    [10.0, 1.0, 0.0],
    [5.0, 1.0, 0.0],
    [2.5, 1.0, 0.0], 
]
function prob_func(prob, i, repeat)
    remake(prob, u0 = initial_conditions[i])
end

ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
sim = solve(ensemble_prob, Tsit5(), EnsembleDistributed(), trajectories = length(initial_conditions), saveat = time, callback = cb, tstops = [diss_time])
ensemble_data = hcat([sim.u[1].u; sim.u[2].u; sim.u[3].u]...)
#ensemble_data = ensemble_data .+ randn(size(ensemble_data)) * 0.02 # Add noise to the ensemble data
ensemble_du = hcat([sim.u[1](sim.u[1].t, Val{1}).u; sim.u[2](sim.u[2].t, Val{1}).u; sim.u[3](sim.u[3].t, Val{1}).u]...)
 # Get the first trajectory
# UDE 
rng = Random.default_rng()
Random.seed!(1234)
# Gaussian RBF as activation
rbf(x) = exp.(-(x.^2))



known_rn = @reaction_network begin
    k1, E + S --> ES
end


# Multilayer FeedForward
U = Lux.Chain(
    Lux.Dense(3,5,rbf), Lux.Dense(5,5, rbf), Lux.Dense(5,5, rbf), Lux.Dense(5,3)
)
# Get the initial parameters and state variables of the model
p_nn, st_nn = Lux.setup(rng, U)


U([0.0;0.0;0.0], p_nn, st_nn) # test the model with a random input



# u0map = zip(Symbol.(species(known_rn)), u0) |> Dict
# pmap = zip(Symbol.(parameters(known_rn)), p) |> Dict
# solve as ODEs
u0map = Dict(:E => u0[1], :S => u0[2], :ES => u0[3])
pmap = Dict(:k1 => p[1], :k2 => p[2])

vars = species(known_rn)
odesys = convert(ODESystem, rn)
odesys = complete(structural_simplify(odesys)) # structural simplify the ODE system
fun = ODEFunction(odesys, vars, parameters(rn)) # convert to ODEFunction
du0 = similar(u0)


p_combined = (;ode = p, nn = p_nn) # combine the parameters into a single dictionary
p_combined = ComponentVector{Float64}(p_combined)
function combined_model!(du, u, p, t)
    # Get the neural network output
    nn_out = first(U(u, p.nn, st_nn))
    # Get the ODE function output
    du = fun(du, u, p.ode, t) # du = f(t,u,p)
    # Add the neural network output to the ODE function output
    du .+= nn_out
end

prob_nn = ODEProblem(combined_model!,u0,tspan, p_combined, saveat = time)
sol_nn = solve(prob_nn, Tsit5(), saveat = time,callback = cb, tstops = [diss_time])
plot(sol_nn, xlabel = "t", ylabel = "Concentration", title = "Enzyme Kinetics with Neural Network", label = ["E" "S" "ES"])



function predict(θ, X = u0, T = time)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Tsit5(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity(),
                callback = cb, tstops = [diss_time]
                ))
end

X̂ = predict(p_combined, u0, time) # test the model with a random input
# Multiple shooting like loss


λ = 1e-3 # regularization parameter

function loss(θ)
    # Start with a regularization on the network
    l = convert(eltype(θ), λ)*sum(abs2, θ[3:end]) ./ length(θ[3:end])
    X̂ = predict(θ, u0, time)
    # Full prediction in x
    l += sum(abs2, signal.- X̂) ./ length(signal)
    return l
end

# Container to track the losses

losses = Float64[]

function cb_nn(p, l)
  push!(losses, l)
  if length(losses)%50==0
      println("Current loss after $(length(losses)) iterations: $(losses[end])")
  end
  return false
end

# First train with ADAM for better convergence -> move the parameters into a
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p_combined)
res1 = Optimization.solve(optprob, PolyOpt(), callback=cb_nn, maxiters = 200)
p_trained = res1.minimizer # trained parameters

# Plot the losses
pl_losses = plot(1:200, losses[1:200], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(201:length(losses), losses[201:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
# Rename the best candidate



## Analysis of the trained network
# Plot the data and the approximation
ts = first(data.t):mean(diff(data.t))/2:last(data.t)
X̂ = predict(p_trained, u0, time)
# Trained on noisy data vs real solution
pl_trajectory = plot(data.t, transpose(X̂), ylabel = "t", xlabel ="x(t), y(t)", label = ["UDE Approximation" nothing])
#scatter!(t, signal[1,:], color = :black, label = "Measurements")
#SKIP BAD CODE
# ymeasurements = unique!(vcat(signal...))
# tmeasurements = unique!(vcat([[time[1], time[end]] for time in eachrow(time)]...))
# scatter!(tmeasurements, ymeasurements, color = :black, label = nothing, legend = :topleft)
## TRY AGAIN
skip = 5
scatter!(pl_trajectory, time[1:skip:end], signal[:,1:skip:end]', labels = ["E" "S" "ES"].*" mesurements", legend = :topleft, markersize = 3)





using Combinatorics


###### REACFTION STUF #####
@independent_variables t
vars = @species E(t), S(t), ES(t)

combs = vcat([[comb for comb in combinations(vars,i)] for i in 1:2]...)
binomial(3,2)



combs_string = [join([first(split(species,"(t)")) for species in string.(comb)],"+") for comb in combs]
n = length(combs)
@parameters k[1:n^2-n]


reactions = Reaction[]
rn_num = 1
for lhs in combs
    for rhs in combs
        if !isequal(lhs,rhs) # identical reactants and products
            lhs_orders = ones(length(lhs)) #change this for higher order reactions
            rhs_orders = ones(length(rhs)) #change this for higher order reactions
            push!(reactions, Reaction(k[rn_num], lhs, rhs))#, lhs_orders, rhs_orders))
            rn_num += 1
        end
    end
end
reactions
@named rs = ReactionSystem(reactions, t, vars, vcat(k...))
ode_rs = convert(ODESystem, complete(rs))

ode_rn = convert(ODESystem, complete(rn)) # convert to ODEFunction


rx = rs
ode = ode_rs
#
#du_approx = PhysicsInformedRegression.finite_diff(data.u, data.t)
function ReactionSINDy(rx, u, du_approx; threshold = 1e-3, lambda =0.0, verbose = false)
    old_rns = equations(rx)
    ode = convert(ODESystem, complete(rx))
    new_rns = Reaction[]
    paramsest = nothing
    while length(new_rns) != length(old_rns)
        # Save the current reactions for comparison
        old_rns = deepcopy(new_rns)  # Use deepcopy to avoid unintended modifications

        # Set up the linear system and perform physics-informed regression
        A, b = PhysicsInformedRegression.setup_linear_system(ode)
        paramsest = physics_informed_regression(ode, u, du_approx, A, b; lambda = lambda, verbose = verbose)

        # Initialize a new list of reactions
        new_rns = Reaction[]

        # Identify parameters above the threshold and retain corresponding reactions
        for reaction in equations(rx)
            if paramsest[reaction.rate] > threshold
                push!(new_rns, reaction)
            elseif verbose
                println("Removing reaction: ", reaction, " with rate: ", paramsest[reaction.rate])
            end
        end
        # Update the reaction system and ODE
        @named rx = ReactionSystem(new_rns, rx.t, species(rx), [reaction.rate for reaction in new_rns])
        ode = convert(ODESystem, complete(rx))  # Convert to ODEFunction
    end
    return rx, paramsest
end

import PhysicsInformedRegression:physics_informed_regression

function physics_informed_regression(ode::AbstractTimeDependentSystem, u::Matrix, du_approx::Matrix, A, b; lambda = 0.0, verbose = false)
    #convert from Matrix to Vector of Vectors
    du_approx = [du_approx[:,i] for i in 1:size(du_approx,2)]
    u = [u[:,i] for i in 1:size(u,2)]
    return PhysicsInformedRegression.physics_informed_regression(ode, u, du_approx, A, b; lambda = lambda, verbose = verbose)
end


threshold = 1e-1 # Set a threshold for parameter pruning
rx, paramsest = ReactionSINDy(rs, ensemble_data, ensemble_du; threshold = threshold , lambda = 0.0, verbose = true)

#sa fdprev_rn = nothing
# while prev_rn != rx
#     ode_rn = convert(ODESystem, complete(rx)) # convert to ODEFunction
#     A,b = PhysicsInformedRegression.setup_linear_system(ode)

for (paramname, paramvalue) in zip(Symbol.(parameters(ode)), paramsest)
    println("$paramname = $paramvalue")
end

###check if the reactions produce the same results as the original system
#Data setup
dt = 0.1 # time step
u0 = [10.0, 1.0, 0.0]
tspan = (0.0, 10.0)
prob = ODEProblem(complete(rx), u0, tspan, paramsest)
data_sindy = solve(prob, Tsit5(), saveat = time, callback = cb, tstops = [diss_time])
plot(data.t, hcat(du_actual...)', xlabel = "t", ylabel = "Concentration", title = "Enzyme Kinetics", label = ["E" "S" "ES"])

#plot data and sindy
plot(data.t, hcat(data_sindy.u...)', label = ["E" "S" "ES"].*" SINDy", markersize = 3)
scatter!(time[1:skip:end], signal[:,1:skip:end]', labels = ["E" "S" "ES"].*" mesurements", legend = :topright, markersize = 3)

params_true = Dict([equations(rs)[13].rate => p[2], equations(rs)[18].rate => p[1]])
p1 = visualize_reaction_network(rs, rx, paramsest; params_true = params_true)
display(p1)


###3# visualization
function visualize_reaction_network(rs, rx, paramsest; params_true = nothing, kwargs..., annotate = true)
    """
    Visualize the reaction network with parameters as circles.

    Args:
        rs: Reaction system. (Original reaction system containing all possible reactions)
        rx: Reaction system. (Reduced reaction system after SINDy)
        paramsest: Estimated parameters. (Estimated parameters from the SINDy algorithm)

        params_true: True parameters. (Optional, for comparison)
    Returns:
        p1: Plot object.
    """
    rate_maps = Dict()
    gt_rate_maps = Dict()
    combs = vcat([[comb for comb in combinations(species(rs),i)] for i in 1:2]...)
    comb_strings_seperated = [Set(string.(comb)) for comb in combs]
    for reaction in equations(rx)
        i = findfirst(isequal(Set(string.(reaction.substrates))), comb_strings_seperated)
        j = findfirst(isequal(Set(string.(reaction.products))), comb_strings_seperated)
        rate_maps[reaction.rate] = (i, j)
        if !isnothing(params_true)
            if reaction.rate in keys(params_true)
                gt_rate_maps[reaction.rate] = (i, j)
            end
        end
    end
    #solve n^2 -n = n_params for n
    n = Int(floor(sqrt(length(parameters(rs)) + 1/4) - 1/2)) +1 
    # Example parameter values (replace with `paramsest` from your code)
    #set params as integers up to 30
    params = vcat(values(paramsest)...)  # Ensure params has 30 elements

    # Create the grid with increased spacing
    spacing = 2  # Adjust this value to control the spacing between grid points
    x = [rate_maps[param][2] for param in keys(paramsest)] .* spacing
    y = [rate_maps[param][1] for param in keys(paramsest)] .* spacing

    # Normalize circle sizes so the maximum radius is 0.5
    max_radius = 0.5
    max_param_value = isnothing(params_true) ? maximum(abs.(params)) : maximum(abs.(vcat(values(params_true)...)))
    sizes = [abs(param) / max_param_value * max_radius * 100 for param in params]  # Scale for visualization

    # Define colors: positive values are blue, negative values are red
    colors = [param >= 0 ? :blue : :red for param in params]

    # Define custom labels for the ticks
    x_labels = combs_string
    y_labels = combs_string

    # Adjust tick positions to match the scaled grid
    xtick_positions = (1:n) .* spacing
    ytick_positions = (1:n) .* spacing

    # Initialize the plot
    p1 = plot()

    # Add ground true parameters as circles
    if !isnothing(params_true)
        x_gt = [gt_rate_maps[param][2] for param in keys(params_true)] .* spacing
        y_gt = [gt_rate_maps[param][1] for param in keys(params_true)] .* spacing

        sizes_gt = [abs(param) / max_param_value * max_radius * 100 for param in values(params_true)]  # Scale for visualization
        scatter!(p1, x_gt, y_gt, m = (7, :white, stroke(1, color), sizes_gt), label="Ground Truth Parameters", legend=:topright, color = :transparent)
    end


    scatter!(p1, x, y, markersize=sizes, color=colors, alpha=0.5, legend=false,
                xlabel="Reactants", ylabel="Products", title="Parameter Visualization",
                xticks=(xtick_positions, x_labels), yticks=(ytick_positions, y_labels),
                xlims=(0, n*2 + spacing),
                ylims=(1 - spacing, n*2 + spacing),
                grid=true, gridcolor=:lightgray, gridalpha=0.5,
                tickfontsize=8, tickfontcolor=:black, size=(800, 600), dpi=300, label = "Estimated Parameters", kwargs...)


    # Annotate the parameter values inside the circles
    if annotate
        for (i, param) in enumerate(params)
            annotate!(p1, x[i], y[i], text(round(param, digits=2), :black, 8, :center))
        end
    end
    return p1
end
# Display the plot
