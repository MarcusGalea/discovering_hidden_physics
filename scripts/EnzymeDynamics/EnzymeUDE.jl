using Pkg
Pkg.activate("scripts\\")

using Optimization, OptimizationOptimisers, Optimisers, ForwardDiff
rosenbrock_ = (x,p) -> (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p = [1.0, 100.0]

optfun = OptimizationFunction(rosenbrock_, AutoForwardDiff())
prob = OptimizationProblem(optfun, x0, p)
function cb(state, l)
    if state.iter % 50 == 0
        println("Iteration: ", state.iter, " has parameters $(state.p)")
    end
    return false
end
sol = solve(prob, Optimisers.Adam(0.1), maxiters=200, callback=cb)
