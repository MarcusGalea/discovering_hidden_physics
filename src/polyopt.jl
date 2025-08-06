using Pkg
Pkg.activate("scripts/")
using Reexport
@reexport using Optimization
using Optimization.SciMLBase, OptimizationOptimJL, OptimizationOptimisers

mutable struct PolyOptAdamBFGS{T}
    lr::T
    beta::Tuple{T, T}
    epsilon::T
    initial_stepnorm::T
end

function PolyOptAdamBFGS(; lr = 0.01, beta = (0.9, 0.999), epsilon = 1e-8, initial_stepnorm = 0.01)
    T = typeof(lr)
    PolyOptAdamBFGS{T}(lr, beta, epsilon, initial_stepnorm)
end

SciMLBase.requiresgradient(opt::PolyOptAdamBFGS) = true

function SciMLBase.__solve(prob::OptimizationProblem,
        opt::PolyOptAdamBFGS,
        args...;
        maxiters = nothing,
        kwargs...)
    # @unpack lr, beta, epsilon, initial_stepnorm = opt
    loss, θ = x -> prob.f(x, prob.p), prob.u0
    deterministic = first(loss(θ)) == first(loss(θ))

    if (!isempty(args) || !deterministic) && maxiters === nothing
        error("Automatic optimizer determination requires deterministic loss functions (and no data) or maxiters must be specified.")
    end

    if isempty(args) && deterministic && prob.lb === nothing && prob.ub === nothing
        # If deterministic then ADAM -> finish with BFGS
        if maxiters === nothing
            res1 = Optimization.solve(prob, Optimisers.ADAM(opt.lr,opt.beta,opt.epsilon), args...; maxiters = 300,
                kwargs...)
        else
            res1 = Optimization.solve(prob, Optimisers.ADAM(opt.lr,opt.beta,opt.epsilon), args...; maxiters,
                kwargs...)
        end

        optprob2 = remake(prob, u0 = res1.u)
        res1 = Optimization.solve(optprob2, BFGS(initial_stepnorm = opt.initial_stepnorm), args...;
            maxiters, kwargs...)
    elseif isempty(args) && deterministic
        res1 = Optimization.solve(prob, BFGS(initial_stepnorm = opt.initial_stepnorm), args...; maxiters,
            kwargs...)
    else
        res1 = Optimization.solve(prob, Optimisers.ADAM(opt.lr,opt.beta,opt.epsilon), args...; maxiters, kwargs...)
    end
end

export PolyOptAdamBFGS