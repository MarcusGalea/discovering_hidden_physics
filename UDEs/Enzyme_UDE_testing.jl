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
    Lux.Dense(3,5,rbf), Lux.Dense(5,5, rbf),Lux.Dense(5,5, rbf), Lux.Dense(5,3)
)
# Get the initial parameters and state variables of the model
p_nn, st_nn = Lux.setup(rng, U)



# u0map = zip(Symbol.(species(known_rn)), u0) |> Dict
# pmap = zip(Symbol.(parameters(known_rn)), p) |> Dict
# solve as ODEs
u0map = Dict(:E => u0[1], :S => u0[2], :ES => u0[3])
pmap = Dict(:k1 => p[1], :k2 => p[2])

vars = species(known_rn)
odesys = convert(ODESystem, known_rn)
odesys = complete(structural_simplify(odesys)) # structural simplify the ODE system
fun = ODEFunction(odesys, vars, parameters(known_rn)) # convert to ODEFunction
du0 = similar(u0)


p_kass_guess = 0.1 #" initial guess for the association rate constant
p_combined = (;ode = [p_kass_guess], nn = p_nn) # combine the parameters into a single dictionary
p_combined = ComponentVector{Float64}(p_combined)
function combined_model!(du, u, p, t)
    # Get the neural network output
    nn_out = first(U(u, p.nn, st_nn))
    # Get the ODE function output
    du = fun(du, u, p.ode, t) # du = f(t,u,p)
    # Add the neural network output to the ODE function output
    du .+= nn_out
end

neural_model! = (du, u, p, t) -> first(U(u, p.nn, st_nn)) # neural network model

neural_problem = ODEProblem(neural_model!, u0, tspan, p_nn, saveat = time)
combined_problem = ODEProblem(combined_model!,u0,tspan, p_combined, saveat = time)
sol_nn = solve(combined_problem, Tsit5(), saveat = time,callback = cb, tstops = [diss_time])
plot(sol_nn, xlabel = "t", ylabel = "Concentration", title = "Enzyme Kinetics with Neural Network", label = ["E" "S" "ES"])



function predict(θ, X = u0, T = time; problem = combined_problem)
    _prob = remake(problem, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Tsit5(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity(),
                callback = cb, tstops = [diss_time]
                ))
end


λ = 1e-2 # regularization parameter

# Loss function
function loss(θ)
    # Start with a regularization on the network
    l = convert(eltype(θ), λ)*sum(abs2, θ[3:end]) ./ length(θ[3:end])
    X̂ = predict(θ, u0, time)
    # Full prediction in x
    l += sum(abs2, signal.- X̂) ./ length(signal)
    return l
end

# Container to track the losses


maxiters = 200
skipping = 5 # skipping for plotting
xlim = (0, maxiters)
ylim = (1e-4, 1e+1)
layout_setting_1 = @layout [a b; c]
layout_setting_2 = @layout [a b]
layout_setting_3 = @layout [a;b]


function cb_nn(state, l; plot_every = 0, SINDyplot_every = 0, lossplot_every = 0)
  push!(losses, l)
  if length(losses)%50==0
      println("Current loss after $(length(losses)) iterations: $(losses[end])")
  end
  pl_trajectory = nothing
    pl_losses = nothing
    p2 = nothing
  if plot_every > 0 || SINDyplot_every > 0
    X̂ = predict(state.u, u0, time)
  end
  if plot_every > 0 && (length(losses)-1)%plot_every == 0
    pl_trajectory = scatter(time[1:skipping:end], signal[:,1:skipping:end]', labels = ["E" "S" "ES"].*" mesurements", markersize = 3, 
    xlabel = "t", ylabel = "Concentration", title = "Enzyme Kinetics", legend = :topright)
    plot!(pl_trajectory, data.t, transpose(X̂), label = "NN Approximation", markersize = 3)
    # display(pl_trajectory)  
  end
  if SINDyplot_every > 0 && (length(losses)-1)%SINDyplot_every == 0
    DX̂_NN= first(U(X̂, state.u.nn, st_nn))
    rx_nn, paramsest_nn = ReactionSINDy(rs_no_ass, X̂ , DX̂_NN; threshold = threshold , lambda = lambda, verbose = false)
    paramsest_nn[k_ass] = state.u[1]
    p2 = visualize_reaction_network(rs_all_combs, rx_nn, paramsest_nn; params_true = params_true)
    # display(p2)
  end
  if lossplot_every > 0 && (length(losses) -1)%lossplot_every == 0
    i = length(losses)
    if i < maxiters
        pl_losses = plot(1:i, losses[1:i], yaxis = :log10, 
        xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue, xlim = xlim, ylim = ylim)
    else
        pl_losses = plot(1:maxiters, losses[1:maxiters], yaxis = :log10, xlabel = "Iterations", ylabel = "Loss", 
        label = "ADAM", color = :blue)
        plot!(pl_losses, maxiters+1:length(losses), losses[maxiters+1:end], yaxis = :log10, ylim = ylim,
        xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
    end
    # display(pl_losses)
    plot_combined = plot(pl_trajectory, p2, pl_losses, layout = layout_setting_1)
    # display(plot_combined)
    frame(animation, plot_combined)
  end
  return false
end

animation = Plots.Animation()
losses = Float64[]
plot_every = 1

modified_cb(args...) = cb_nn(args...; lossplot_every = plot_every, SINDyplot_every = plot_every, plot_every = plot_every)
# First train with ADAM for better convergence -> move the parameters into a
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p_combined)
res1 = Optimization.solve(optprob, PolyOpt(), callback=modified_cb, maxiters = maxiters)
p_trained = res1.minimizer # trained parameters

# Plot the losses
pl_losses = plot(1:200, losses[1:200], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(201:length(losses), losses[201:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
# Rename the best candidate
#save animation as gif
# destination = "Enzyme_UDE.gif"
gif(animation, destination, fps = 10)


## Analysis of the trained network
# Plot the data and the approximation
ts = first(data.t):mean(diff(data.t))/2:last(data.t)
X̂ = predict(p_trained, u0, time) # test the model with a random input

# # plot(data.t, DX̂_NN', label = "NN Approximation", markersize = 3, xlabel = "t", ylabel = "Concentration", title = "Enzyme Kinetics with Neural Network")
# # Trained on noisy data vs real solution
# plot(pl_trajectory ,data.t, transpose(X̂), ylabel = "Concentration", xlabel ="t", label = ["UDE Approximation" nothing])
# #NEURAL NETWORK PREDICTION
# plot!(pl_trajectory, data.t, transpose(X̂_NN), label = "NN Approximation", markersize = 3)
# skip = 5




# Network prediction
# plot!(pl_trajectory, data.t, transpose(X̂_NN), label = "NN Approximation", markersize = 3)

#scatter!(t, signal[1,:], color = :black, label = "Measurements")
#SKIP BAD CODE
# ymeasurements = unique!(vcat(signal...))
# tmeasurements = unique!(vcat([[time[1], time[end]] for time in eachrow(time)]...))
# scatter!(tmeasurements, ymeasurements, color = :black, label = nothing, legend = :topleft)
## TRY AGAIN


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
@named rs_all_combs = ReactionSystem(reactions, t, vars, vcat(k...))
ode_rs_all_combs = convert(ODESystem, complete(rs_all_combs))

ode_rn = convert(ODESystem, complete(rn)) # convert to ODEFunction


rx = rs_all_combs
ode = ode_rs_all_combs
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
lambda = 0.4 # Set a regularization parameter for the regression
rx, paramsest = ReactionSINDy(rs_all_combs, ensemble_data, ensemble_du; threshold = threshold , lambda = lambda, verbose = true)
# rx, paramsest = ReactionSINDy(rs_all_combs, data.u, du_actual; threshold = threshold , lambda = 0.0, verbose = true)

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
scatter!(time[1:skipping:end], signal[:,1:skipping:end]', labels = ["E" "S" "ES"].*" mesurements", legend = :topright, markersize = 3)


k_ass = equations(rs_all_combs)[18].rate
k_diss= equations(rs_all_combs)[13].rate
params_true = Dict([k_diss => p[2], k_ass => p[1]])
p1 = visualize_reaction_network(rs_all_combs, rx, paramsest; params_true = params_true)
display(p1)


### SINDy ON NN
###
X̂ = predict(p_trained, u0, time)
#Network prediction
DX̂_NN= first(U(X̂, p_trained.nn, st_nn))
rs_no_ass = remove_params(rs_all_combs, k_ass)
rx_nn, paramsest_nn = ReactionSINDy(rs_no_ass, X̂ , DX̂_NN; threshold = threshold , lambda = lambda, verbose = true)
paramsest_nn[k_ass] = p_trained.ode[1]

p2 = visualize_reaction_network(rs_all_combs, rx_nn, paramsest_nn; params_true = params_true)

###3# visualization
function visualize_reaction_network(rs, rx, paramsest; params_true = nothing, annotate = true, kwargs...)
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
    for reaction in equations(rs)
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
        scatter!(p1, x_gt, y_gt, m = (7, :white, stroke(1, color), sizes_gt), label="Ground Truth Parameters", legend=:topright,) 
        #color = :transparent)
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





function remove_params(rn, param_to_remove)
    """
    Remove parameter from the ODE system and return the new system.

    Parameters:
    - `rn`: The original Catalyst Reaction System.
    - `param_to_remove`: A parameter to be removed.

    Returns:
    - `new_rn`: The new Reaction System with parameters removed.
    """
    # Filter reactions where the rate is not in `param_to_remove`
    new_reactions = [reaction for reaction in equations(rn) if !isequal(reaction.rate, param_to_remove)]

    # Filter parameters that are not in `param_to_remove`
    new_params = [param for param in parameters(rn) if !isequal(param, param_to_remove)]

    # Create a new Reaction System with the updated reactions and parameters
    @named new_rn = ReactionSystem(new_reactions, rn.t, species(rn), new_params)
end



######## DOESN*T MATTER ###########

function insert_known_params(ode_rn, known_params)
    """
    insert known parameters from the ODE system and return the new system.

    Parameters:
    - `ode_rn`: The original ODE system.
    - `known_params`: A dictionary of known parameters to be inserted.

    Returns:
    - `new_ode`: The new ODE system with known parameters inserted.
    """
    # Get the equations and parameters of the ODE system
    new_eqs = Equation[]

    new_eqs = Equation[]
    for eq in equations(ode_rn)
        new_eq = substitute(eq, known_params)
        new_eqs = push!(new_eqs, new_eq)  
    end
    #remove parameters that are in known_params
    new_params = [param for param in parameters(ode_rn) if !(param in keys(known_params))]
    @named new_ode = ODESystem(new_eqs, t, ModelingToolkit.get_unknowns(ode_rn), new_params)
    return new_ode
end



