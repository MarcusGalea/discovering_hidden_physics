#cd(@__DIR___)
using Pkg
Pkg.activate(".")

using DifferentialEquations
using ModelingToolkit
using MLJ
#Pkg.UPDATED_REGISTRY_THIS_SESSION[] = true


## LORENZ ATTRACTOR
@parameters σ ρ β
@variables t x(t) y(t) z(t)
D = Differential(t)

eqs = [D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

# Define the system
@named sys = ODESystem(eqs, t, [x, y, z], [σ, ρ, β])
# sys = structural_simplify(sys)
# Define the initial conditions and parameters
u0 = [
    x => 1.0,
    y => 0.0,
    z => 0.0]

p = [σ => 10.0,
    ρ => 28.0,
    β => 8/3]

# Define the time span
timesteps = collect(0.0:0.01:20.0)

# Simulate the system
prob = ODEProblem(sys, u0,(timesteps[1], timesteps[end]) ,p, saveat = timesteps)
sol = solve(prob)

using Plots
#plot 3d phase space
plot(sol,label = "True", title = "Lorenz Attractor", lw = 2, dpi = 600, idxs = (1,2,3))

using MLJ
using DataDrivenDiffEq
using DataDrivenSparse

# Create a test problem
function lorenz(u, p, t)
    x, y, z = u

    ẋ = 10.0 * (y - x)
    ẏ = x * (28.0 - z) - y
    ż = x * y - (8 / 3) * z
    return [ẋ, ẏ, ż]
end

u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
dt = 0.1
prob = ODEProblem(lorenz, u0, tspan)
sol = solve(prob, Tsit5(), saveat = dt)

## Start the automatic discovery
ddprob = DataDrivenProblem(sol)

@variables t x(t) y(t) z(t)
u = [x; y; z]
basis = Basis(polynomial_basis(u, 5), u, iv = t)
opt = STLSQ2(exp10.(-5:0.1:-1))
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 1))
println(get_basis(ddsol))
println(get_parameter_map(get_basis(ddsol)))


A = ddprob.X
ddprob.DX
DataDrivenSparse.init_cache(opt)


struct STLSQ2{T <: Union{Number, AbstractVector}, R <: Number} <:
    DataDrivenSparse.AbstractSparseRegressionAlgorithm
 """Sparsity threshold"""
 thresholds::T
 """Ridge regression parameter"""
 rho::R

 function STLSQ2(threshold::T = 1e-1, rho::R = zero(eltype(T))) where {T, R <: Number}
     @assert all(threshold .> zero(eltype(threshold))) "Threshold must be positive definite"
     @assert rho>=zero(R) "Ridge regression parameter must be positive definite!"
     return new{T, R}(threshold, rho)
 end
end

Base.summary(::STLSQ2) = "STLSQ2"

struct STLSQ2Cache{usenormal, C <: AbstractArray, A <: BitArray, AT, BT, ATT, BTT} <:
    DataDrivenSparse.AbstractSparseRegressionCache
 X::C
 X_prev::C
 active_set::A
 proximal::SoftThreshold
 A::AT
 B::BT
 # Original Data
 Ã::ATT
 B̃::BTT
end

function init_cache(alg::STLSQ2, A::AbstractMatrix, b::AbstractVector)
 init_cache(alg, A, permutedims(b))
end

function init_cache(alg::STLSQ2, A::AbstractMatrix, B::AbstractMatrix)
 n_x, m_x = size(A)
 @assert size(B, 1)==1 "Caches only hold single targets!"
 @unpack rho = alg
 λ = minimum(get_thresholds(alg))

 proximal = get_proximal(alg)

 if n_x <= m_x && !iszero(rho)
     X = A * A' + rho * I
     Y = B * A'
     usenormal = true
 else
     usenormal = false
     X = A
     Y = B
 end

 coefficients = Y / X

 prev_coefficients = zero(coefficients)

 active_set = BitArray(undef, size(coefficients))

 active_set!(active_set, proximal, coefficients, λ)

 return STLSQ2Cache{usenormal, typeof(coefficients), typeof(active_set), typeof(X),
     typeof(Y), typeof(A), typeof(B)}(coefficients, prev_coefficients,
     active_set, get_proximal(alg),
     X, Y, A, B)
end

function step!(cache::STLSQ2Cache, λ::T) where {T}
 @unpack X, X_prev, active_set, proximal = cache

 X_prev .= X

 step!(cache)

 proximal(X, active_set, λ)
 return
end

function step!(cache::STLSQ2Cache{true})
 @unpack X, A, B, active_set = cache
 p = vec(active_set)
 X[1:1, p] .= /(B[1:1, p], A[p, p])
 return
end

function step!(cache::STLSQ2Cache{false})
 @unpack X, A, B, active_set = cache
 p = vec(active_set)
 X[1:1, p] .= /(B, A[p, :])
 return
end
import CommonSolve

function CommonSolve.solve!(ps::DataDrivenDiffEq.InternalDataDrivenProblem{
                                                          <:DataDrivenSparse.AbstractSparseRegressionAlgorithm
                                                          })
    @unpack alg, basis, testdata, traindata, problem, options, transform = ps

    results = map(traindata) do (X, Y)
        __sparse_regression(ps, X, Y)
    end

    # Get the best result based on test error, if applicable else use testerror
    sort!(results, by = l2error)

    # Convert to basis
    best_res = first(results)

    # Transform the best coefficients
    coefficients = permutedims(copy(get_coefficients(best_res)))
    coefficients = permutedims(StatsBase.transform(transform, coefficients))
    new_basis = DataDrivenDiffEq.__construct_basis(coefficients, basis, problem, options)

    DataDrivenSolution(new_basis, problem, alg, results, ps, best_res.retcode)
end

function __sparse_regression(ps::InternalDataDrivenProblem{
                                                           <:AbstractSparseRegressionAlgorithm
                                                           }, X::AbstractArray,
                             Y::AbstractArray)
    @unpack alg, testdata, options, transform = ps

    coefficients, optimal_thresholds, optimal_iterations = alg(X, Y, options = options)

    trainerror = sum(abs2, Y .- coefficients * X)

    X̃, Ỹ = testdata

    if !isempty(X̃)
        testerror = sum(abs2, Ỹ .- coefficients * X̃)
    else
        testerror = nothing
    end

    retcode = DDReturnCode(1)

    dof = sum(abs.(coefficients) .> 0.0)

    SparseRegressionResult(coefficients, dof, optimal_thresholds,
                           optimal_iterations, testerror, trainerror,
                           retcode)
end

function __sparse_regression(ps::InternalDataDrivenProblem{<:ImplicitOptimizer},
                             X::AbstractArray, Y::AbstractArray)
    @unpack alg, testdata, options, transform, basis, problem, implicit_idx = ps
    @assert DataDrivenDiffEq.is_implicit(basis) "The provided `Basis` does not have implicit variables!"

    candidate_matrix = zeros(Bool, size(implicit_idx))
    idx = ones(Bool, size(candidate_matrix, 2))

    for i in axes(candidate_matrix, 1), j in axes(candidate_matrix, 2)
        idx .= true
        idx[j] = false
        # We want only equations which are either dependent on the variable or on no other
        candidate_matrix[i, j] = implicit_idx[i, j] || sum(implicit_idx[i, idx]) == 0
    end

    opt_coefficients = zeros(eltype(problem), size(candidate_matrix, 2),
                             size(candidate_matrix, 1))
    opt_thresholds = []
    opt_iterations = []

    foreach(enumerate(eachcol(candidate_matrix))) do (i, idx)
        # We enforce that one of the implicit variables is necessary for sucess
        coeff, thresholds, iters = alg(X[idx, :], Y, options = options,
                                       necessary_idx = implicit_idx[idx, i])
        opt_coefficients[i:i, idx] .= coeff
        push!(opt_thresholds, thresholds)
        push!(opt_iterations, iters)
    end

    trainerror = sum(abs2, opt_coefficients * X)

    X̃, Ỹ = testdata

    if !isempty(X̃)
        testerror = sum(abs2, opt_coefficients * X̃)
    else
        testerror = nothing
    end

    retcode = DDReturnCode(1)

    dof = sum(abs.(opt_coefficients) .> 0.0)

    SparseRegressionResult(opt_coefficients, dof, opt_thresholds,
                           opt_iterations, testerror, trainerror,
                           retcode)
end
