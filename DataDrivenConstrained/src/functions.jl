"""
solve function takes in a DataDrivenProblem, a Basis, and a ConstrainedSTLSQ object and returns the solution basis
    $(TYPEDSIGNATURES)

# Arguments
- `prob::DataDrivenProblem`: The data driven problem
- `basis::Basis`: The basis of the solution
- `opt::ConstrainedSTLSQ`: The optimization algorithm

# Optional arguments
- `options::DataDrivenCommonOptions`: The options for the optimization algorithm

# Returns
- `Basis`: The solution basis
"""
function CommonSolve.solve(prob::DataDrivenProblem, basis::Basis, opt::ConstrainedSTLSQ; options = DataDrivenCommonOptions())
    cache = ConstrainedSTLSQcache(opt, prob, basis)
    @unpack maxiters, abstol, reltol = options

    step!(cache)
    iter = 0
    while true
        iter += 1
        step!(cache)
        if rel_error(cache) < reltol || rss(cache) < abstol  || iter > maxiters
            break
        end
    end
    return create_solution_basis(cache)
end




"""
step! function takes in the cache and updates the cache with the next iteration of the Constrained Subsequent Thresholded Least Squares algorithm
    $(TYPEDSIGNATURES)

# Arguments
- `cache::ConstrainedSTLSQcache`: The cache for the Constrained Subsequent Thresholded Least Squares algorithm
"""
function step!(cache::ConstrainedSTLSQcache)
    @unpack opt, zero_entries, C, d, Θ, DX = cache
    @unpack λ, constraints, Ξ = opt

    # Least squares step
    A = vcat(hcat(blockdiags(Θ'Θ, size(DX,2)),C'), hcat(C, zeros(length(d),length(d))))
    b = vcat(vec(Θ'*DX),d)
    ξ_est = (A\b)[1:end-length(d)]

    # Thresholding step
    threshold = get_threshold(cache, ξ_est)
    zero_entries = findall(x->abs(x)<threshold, ξ_est)
    ξ_est[zero_entries] .= 0

    # Update constraints
    new_zeros = [entry for entry in zero_entries if entry ∉ cache.zero_entries]
    ξ = vec(Ξ)
    new_constraints = [ξ[entry] ~ 0 for entry in new_zeros]
    for constraint in new_constraints
        C,d = add_constraint(constraint, C, d, Ξ)
    end

    # Update cache

    cache.C = C
    cache.d = d
    cache.zero_entries = zero_entries
    cache.Ξ_est = reshape(ξ_est, size(Ξ))
end

function get_threshold(cache::ConstrainedSTLSQcache, ξ_est)
    non_zero_entries = [entry for entry in 1:length(ξ_est) if entry ∉ cache.zero_entries]
    meanabsval = sum(abs.(ξ_est[non_zero_entries]))/length(non_zero_entries)
    return cache.opt.λ*meanabsval
end
    
function _set_default_val(x::Num, val::T) where {T <: Number}
    Num(Symbolics.setdefaultval(Symbolics.unwrap(x), val))
end

"""
rhs function takes in an equation and returns the right hand side of the equation
    $(TYPEDSIGNATURES)
"""
function rhs(equation::Equation)
    return equation.rhs
end



"""
create_block_diag_matrix function takes in a matrix and the number of copies and returns a block diagonal matrix with the given number of copies of the input matrix
    $(TYPEDSIGNATURES)
"""
function blockdiags(block, n)
    # Get the size of each block
    m, p = size(block)
    # Initialize the block diagonal matrix
    result = zeros(m*n, p*n)
    # Fill the block diagonal matrix
    for i in 1:n
        result[(i-1)*m+1:i*m, (i-1)*p+1:i*p] = block
    end
    return result
end

"""
linear_system function takes in the constraints and the coefficient matrix Ξ and returns the linear system of equations
    $(TYPEDSIGNATURES)

# Arguments
- `constraints::Array{Equation}`: An array of equations
- `Ξ::AbstractMatrix`: The coefficient matrix

# Returns
- `C::AbstractMatrix`: The matrix of coefficients
- `d::AbstractVector`: The right hand side of the equations
"""
function linear_system(constraints, Ξ)
    n,m = size(Ξ)
    C = zeros(length(constraints), n*m)
    d = zeros(length(constraints))
    for (k, eq) in enumerate(constraints)
        d[k] = eq.rhs
        lhs = eq.lhs
        for var in get_variables(lhs)
            idx = findfirst(isequal(var), vec(Ξ))
            #assert linear equation
            @assert Symbolics.degree(lhs, var) == 1 "Non-linear equation"
            #get coefficient for variable in the equation
            C[k, idx] = Symbolics.coeff(lhs, var)
        end
    end
    return C, d
end

"""
add_constraint function takes in a constraint, the matrix of coefficients, the right hand side of the equations, and the coefficient matrix Ξ and returns the updated matrix of coefficients and the right hand side of the equations
    $(TYPEDSIGNATURES)

# Arguments
- `constraint::Equation`: An equation
- `C::AbstractMatrix`: The matrix of coefficients
- `d::AbstractVector`: The right hand side of the equations
- `Ξ::AbstractMatrix`: The coefficient matrix

# Returns
- `C::AbstractMatrix`: The updated matrix of coefficients
- `d::AbstractVector`: The updated right hand side of the equations
"""
function add_constraint(constraint::Equation, C::AbstractMatrix, d::AbstractVector, Ξ::AbstractMatrix)
    d = vcat(d, constraint.rhs)
    lhs = constraint.lhs
    for var in get_variables(lhs)
        #assert linear equation
        @assert Symbolics.degree(lhs, var) == 1 "Non-linear equation"

        idx = findfirst(isequal(var), vec(Ξ))
        C = vcat(C, zeros(1, size(C,2)))
        C[end, idx] = Symbolics.coeff(lhs, var)
    end
    return C, d
end