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


include("src/SINDy_methods.jl")
export create_full_reaction_network,ReactionSINDy, visualize_reaction_network, remove_params


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
p = Dict(:k1 => 0.4, :k2 => 0.3) # rate constants
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


rs_all_combs = create_full_reaction_network(species(known_rn), known_rn.iv; number_reactants = 2)
rs_no_ass = remove_known_reactions(rs_all_combs, known_rn)

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
pmap = p

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
combined_model!(du0, u0, p_combined, 0.0)
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
    display(plot_combined)
  end
  return false
end

threshold = 1e-1 # Set a threshold for parameter pruning
lambda = 0.4 # Set a regularization parameter for the regression

"""
    Create SINDy visualization function.
    This function creates a SINDy visualization function that can be used to visualize the results of the SINDy algorithm.

    # Arguments \n
    - `vars`: The variables of the system. \n
    - `t`: The time symbol (Num). \n
    - `threshold`: The threshold for parameter pruning (default: 1e-1). \n	
    - `lambda`: The regularization parameter for the regression (default: 0.4). \n
    - `known_reaction`: The known reaction network (default: nothing). \n
    - `verbose`: Whether to print verbose output (default: false). \n
    - `number_reactants`: The number of reactants in the system (default: 2). \n
    - `params_true`: The true parameters of the system (default: nothing). \n

    # Returns \n
    - `produce_SINDy_plot`: A function that takes the predicted state and its derivative and produces a SINDy plot. \n
"""
function includeSINDy_plot(vars::Vector{<:Num}, t::Num; 
                            threshold = 1e-1, lambda = 0.4, known_reaction = nothing, verbose = false,
                            number_reactants = 2, params_true = nothing)
    rs_all_combs = create_full_reaction_network(vars,t ; number_reactants = number_reactants)
    if !isnothing(known_reaction)
        rs_all_combs = remove_known_reactions(rs_all_combs, known_reaction)
    end
    function produce_SINDy_plot(X̂, DX̂)
        rx, paramsest = ReactionSINDy(rs_all_combs, X̂ , DX̂; threshold = threshold , lambda = lambda, verbose = verbose)
        params
        p2 = visualize_reaction_network(rs_all_combs, rx, paramsest; params_true = params_true)
        return p2 
    end
    return deepcopy(produce_SINDy_plot)
end

# SINDy on the neural network	




animation = Plots.Animation()
losses = Float64[]
plot_every = 5

modified_cb(args...) = cb_nn(args...; lossplot_every = plot_every, SINDyplot_every = plot_every, plot_every = plot_every)
# First train with ADAM for better convergence -> move the parameters into a
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p_combined)
res1 = Optimization.solve(optprob, PolyOpt(), callback=modified_cb, maxiters = maxiters)
p_trained = res1.minimizer # trained parameters

# Plot the losses
pl_losses = plot(1:200, losses[1:200], yaxis = :log10,xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(201:length(losses), losses[201:end], yaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
# Rename the best candidate
#save animation as gif
# destination = "Enzyme_UDE.gif"
# gif(animation, destination, fps = 10)


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



ode_rs_all_combs = convert(ODESystem, complete(rs_all_combs))

ode_rn = convert(ODESystem, complete(rn)) # convert to ODEFunction


rx = rs_all_combs
ode = ode_rs_all_combs
#
#du_approx = PhysicsInformedRegression.finite_diff(data.u, data.t)


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
skipping = 5 # skipping for plotting
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
params_true = Dict([k_diss => p[:k2], k_ass => p[:k1]])
p1 = visualize_reaction_network(rs_all_combs, rx, paramsest; params_true = params_true)
display(p1)


### SINDy ON NN
###
X̂ = predict(p_trained, u0, time)
#Network prediction
DX̂_NN= first(U(X̂, p_trained.nn, st_nn))
rx_nn, paramsest_nn = ReactionSINDy(rs_no_ass, X̂ , DX̂_NN; threshold = threshold , lambda = lambda, verbose = true)
paramsest_nn[k_ass] = p_trained.ode[1]

p2 = visualize_reaction_network(rs_all_combs, rx_nn, paramsest_nn; params_true = params_true)
savefig(p2, "plots/SINDy_res_UDE")


###3# visualization







