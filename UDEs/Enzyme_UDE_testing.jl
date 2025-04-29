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

rn = @reaction_network begin
    k1, E + S --> ES
    k2, ES --> E + S
end
rn = complete(rn)

u0 = [10.0, 1.0, 0.0]
tspan = (0.0, 10.0)
p = [0.4, 0.1]
prob = ODEProblem(rn, u0, tspan, p)
data = solve(prob, Tsit5(), saveat = 0.1)

#add noise to data
time = data.t
signal =  hcat(data.u...)
noise = 0.02 * randn(size(signal))
signal = signal .+ noise
plot(data.t, signal', xlabel = "t", ylabel = "Concentration", title = "Enzyme Kinetics", label = ["E" "S" "ES"])


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

prob_nn = ODEProblem(combined_model!,u0,tspan, p_combined, saveat = 0.1)
sol_nn = solve(prob_nn, Tsit5(), saveat = 0.1)
plot(sol_nn, xlabel = "t", ylabel = "Concentration", title = "Enzyme Kinetics with Neural Network", label = ["E" "S" "ES"])



function predict(θ, X = u0, T = time)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Tsit5(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
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

callback = function (p, l)
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
res1 = Optimization.solve(optprob, PolyOpt(), callback=callback, maxiters = 200)
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
pl_trajectory = plot(time, transpose(X̂), ylabel = "t", xlabel ="x(t), y(t)", color = :red, label = ["UDE Approximation" nothing])
#scatter!(t, signal[1,:], color = :black, label = "Measurements")
ymeasurements = unique!(vcat(signal...))
tmeasurements = unique!(vcat([[time[1], time[end]] for time in eachrow(time)]...))
scatter!(tmeasurements, ymeasurements, color = :black, label = nothing, legend = :topleft)













###### REACFTION STUF #####
@independent_variables t
vars = @species E(t), S(t), ES(t)

combs = vcat([[comb for comb in combinations(vars,i)] for i in 1:2]...)
binomial(3,2)




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
