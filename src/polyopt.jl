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


mutable struct ProgressivePolyOpt{T}
    lr::T
    beta::Tuple{T, T}
    epsilon::T
    initial_stepnorm::T
    n_partitions::Int
end

function ProgressivePolyOpt(; lr = 0.01, beta = (0.9, 0.999), epsilon = 1e-8, initial_stepnorm = 1.0, n_partitions = 1)
    T = typeof(lr)
    ProgressivePolyOpt{T}(lr, beta, epsilon, initial_stepnorm, n_partitions)
end

SciMLBase.requiresgradient(opt::ProgressivePolyOpt) = true

function SciMLBase.__solve(prob::OptimizationProblem,
        opt::ProgressivePolyOpt,
        args...;
        maxiters = nothing,
        maxiter_BFGS = nothing,
        exponential_decay = false,
        reduction_factor = 0.5, #reduction factor of iters per partition
        kwargs...)
    # loss, θ = x -> prob.f(x, prob.p), prob.u0
    # deterministic = first(loss(θ)) == first(loss(θ))

    # if (!isempty(args) || !deterministic) && maxiters === nothing
    #     error("Automatic optimizer determination requires deterministic loss functions (and no data) or maxiters must be specified.")
    # end
    maxiters = maxiters === nothing ? 300 : maxiters

    iters_per_partition = exponential_decay ? exponential_decaying_iters(maxiters, opt.n_partitions, reduction_factor) : [Int(round(maxiters / opt.n_partitions)) for _ in 1:opt.n_partitions]
    proportion_per_run = 1.0 / opt.n_partitions
    optprob1 = remake(prob, p = proportion_per_run)
    res1 = nothing
    for i in 1:opt.n_partitions
        println("Running partition $(i) of $(opt.n_partitions) with proportion $(optprob1.p)")
        res1 = Optimization.solve(optprob1, Optimisers.ADAM(opt.lr,opt.beta,opt.epsilon), args...; maxiters = iters_per_partition[i],
            kwargs...)
        optprob1 = remake(optprob1, p = proportion_per_run * (i+1), u0 = res1.u)
    end
    maxiter_BFGS = maxiter_BFGS === nothing ? iters_per_partition : maxiter_BFGS
    try
        res1 = Optimization.solve(optprob1, BFGS(initial_stepnorm = opt.initial_stepnorm), args...;
                            maxiters = maxiter_BFGS, kwargs...)
    catch AssertionError
        @warn "BFGS failed to converge, running Adam instead."
        res1 = Optimization.solve(optprob1, Optimisers.ADAM(opt.lr,opt.beta,opt.epsilon), args...; maxiters = maxiter_BFGS,
            kwargs...)
    end
    return res1
end
export ProgressivePolyOpt

function exponential_decaying_iters(maxiters, n_partitions, r = 0.5)
    """
    Generate a vector of iterations per partition with exponential decay.
    
    # Arguments
    - `maxiters`: Total number of iterations.
    - `n_partitions`: Number of partitions.
    - `r`: Decay factor (0 < r < 1).
    
    # Returns
    A vector of iterations per partition.
    """
    weights = [r^(i-1) for i in 1:n_partitions]
    return round.(Int, maxiters * weights / sum(weights))
end