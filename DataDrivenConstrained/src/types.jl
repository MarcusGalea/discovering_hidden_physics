
"""
ConstrainedSTLSQ is a type that represents the Constrained Subsequent Thresholded Least Squares algorithm
"""
mutable struct ConstrainedSTLSQ{T}<:DataDrivenDiffEq.AbstractDataDrivenAlgorithm
    """ λ is the threshold of the iteration """
    λ::T
    """ ρ is the regularization parameter """
    ρ::T
    """ constraints is the list of constraints """
    constraints::Vector{Equation}
    """ Ξ is the symbolic coefficient matrix """
    Ξ::AbstractMatrix
end


"""
ConstrainedSTLSQcache is a type that represents the cache for the Constrained Subsequent Thresholded Least Squares algorithm
"""
mutable struct ConstrainedSTLSQcache
    """ opt is the optimization algorithm """
    opt::ConstrainedSTLSQ
    """ Ξ_est is the estimated coefficient matrix """
    Ξ_est::AbstractMatrix
    """ zero_entries is the indices of the zero entries """
    zero_entries::Vector{Int}
    """ C is the matrix of constraint coefficients """
    C::AbstractMatrix
    """ d is the right hand side of the constraint equations """
    d::AbstractVector
    """ Θ is the basis evaluated at the data points """
    Θ::AbstractMatrix
    """ DX is the derivative of the data points """
    DX::AbstractMatrix
    """ basis is the basis of the solution """
    basis::DataDrivenDiffEq.Basis
    ConstrainedSTLSQcache(opt::ConstrainedSTLSQ, Ξ_est, zero_entries, C, d, Θ, DX, basis) = new(opt, Ξ_est, zero_entries, C, d, Θ, DX, basis)
    ConstrainedSTLSQcache(opt::ConstrainedSTLSQ, ddprob::DataDrivenDiffEq.DataDrivenProblem, basis::DataDrivenDiffEq.Basis) = initialize_cache(opt, ddprob, basis)
end


"""
initialize_cache function initializes the cache for the Constrained Subsequent Thresholded Least Squares algorithm
    $(TYPEDSIGNATURES)
# Arguments
- `opt::ConstrainedSTLSQ`: The optimization algorithm
"""
function initialize_cache(opt::ConstrainedSTLSQ, ddprob::DataDrivenDiffEq.DataDrivenProblem, basis::DataDrivenDiffEq.Basis)
    @unpack constraints, Ξ = opt
    @unpack X, DX,t = ddprob
    Θ = basis(X,[], t)'
    C,d = linear_system(constraints, hcat(Ξ))
    zero_entries = []
    return ConstrainedSTLSQcache(opt, hcat(Ξ), zero_entries, C, d, Θ, DX', basis)
end

"""
step! function takes in the cache and updates the cache with the next iteration of the Constrained Subsequent Thresholded Least Squares algorithm
    $(TYPEDSIGNATURES)

# Arguments
- `cache::ConstrainedSTLSQcache`: The cache for the Constrained Subsequent Thresholded Least Squares algorithm
"""