#cd(@__DIR___)
using Pkg
Pkg.activate(".")

using DifferentialEquations
using ModelingToolkit
using DataDrivenDiffEq
using DataDrivenSparse
using DocStringExtensions


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
basis = Basis(polynomial_basis(u, 2), u, iv = t)

using Symbolics
using LinearAlgebra
using Plots
X = ddprob.X
DX = ddprob.DX

@show equations(basis)
n,m = length(basis), size(ddprob)[1]
@parameters Ξ[1:n, 1:m] # Ξ is the coefficient matrix
# ξ is vectorized version of Ξ
ξ = vec(Ξ)
constraints = [
    ξ[4] ~  10;
    ξ[2] + ξ[4] ~ 0
] 
n_c = length(constraints)

#### EXPERIMENTAL

λ = 1e-5

#evaluate basis at the data points
Θ = basis(X,[],ddprob.t)'
C,d = linear_system(constraints, hcat(Ξ))
n_c = length(d)
A = vcat(hcat(blockdiags(Θ'Θ, m),C'), hcat(C, zeros(n_c,n_c)))
b = vcat(vec(Θ'*DX'),d)
ξ_est= (A\b)[1:end-n_c]


#find elements close to zero in ξ_est
zero_entries = []
non_zero_entries = [entry for entry in 1:length(ξ_est) if entry ∉ zero_entries]
all_zero_entries = findall(x->abs(x)<λ, ξ_est) #change this
new_zero_entries = [entry for entry in all_zero_entries if entry ∉ zero_entries]
zero_entries = all_zero_entries
#non zero entries
non_zero_entries = [entry for entry in 1:length(ξ_est) if entry ∉ zero_entries]

mutable struct ConstrainedSTLSQ{T}
    """ λ is the threshold of the iteration """
    λ::T
    """ constraints is the list of constraints """
    constraints::Vector{Equation}
    """ Ξ is the symbolic coefficient matrix """
    Ξ::AbstractMatrix
end

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
    ConstrainedSTLSQcache(opt::ConstrainedSTLSQ, Ξ_est, zero_entries, C, d, Θ, DX) = new(opt, Ξ_est, zero_entries, C, d, Θ, DX)
    ConstrainedSTLSQcache(opt::ConstrainedSTLSQ, ddprob::DataDrivenDiffEq.DataDrivenProblem) = initialize_cache(opt, ddprob)
end


function initialize_cache(opt::ConstrainedSTLSQ, ddprob::DataDrivenDiffEq.DataDrivenProblem)
    @unpack constraints, Ξ = opt
    @unpack X, DX = ddprob
    Θ = basis(X,[],ddprob.t)'
    C,d = linear_system(constraints, hcat(Ξ))
    zero_entries = []
    return ConstrainedSTLSQcache(opt, hcat(Ξ), zero_entries, C, d, Θ, DX')
end

function step!(cache::ConstrainedSTLSQcache)
    @unpack opt, Ξ_est, zero_entries, C, d, Θ, DX = cache
    @unpack λ, constraints, Ξ = opt
    A = vcat(hcat(blockdiags(Θ'Θ, size(DX,2)),C'), hcat(C, zeros(length(d),length(d))))
    b = vcat(vec(Θ'*DX),d)
    ξ_est = (A\b)[1:end-length(d)]

    threshold = get_threshold(cache, ξ_est)
    zero_entries = findall(x->abs(x)<threshold, ξ_est)
    ξ_est[zero_entries] .= 0
    cache.Ξ_est = reshape(ξ_est, size(Ξ))

    new_zeros = [entry for entry in zero_entries if entry ∉ cache.zero_entries]
    new_constraints = [ξ[entry] ~ 0 for entry in new_zeros]
    for constraint in new_constraints
        C,d = add_constraint(constraint, C, d, Ξ)
    end
    cache.C = C
    cache.d = d
    cache.zero_entries = zero_entries
end

function get_threshold(cache::ConstrainedSTLSQcache, ξ_est)
    non_zero_entries = [entry for entry in 1:length(ξ_est) if entry ∉ cache.zero_entries]
    meanabsval = sum(abs.(ξ_est[non_zero_entries]))/length(non_zero_entries)
    return cache.opt.λ*meanabsval
end
    
    
opt = ConstrainedSTLSQ(1e-5, constraints, hcat(Ξ))
cache = ConstrainedSTLSQcache(opt, ddprob)
step!(cache)

for entry in new_zero_entries
    #add constraint to the list of constraints
    C,d = add_constraint(ξ[entry] ~ 0, C, d, hcat(Ξ))
end

ξ_sym_sol = copy(ξ)
ξ_sym_sol[zero_entries] .= 0
Ξ_sym_sol = reshape(ξ_sym_sol, size(Ξ))
Ξ_sym_sol'*rhs.(equations(basis))

D = Differential(t)
D.(u).~Ξ_sym_sol'*rhs.(equations(basis))

Basis(D.(u).~Ξ_sym_sol'*rhs.(equations(basis)),u,iv=t, parameters=ξ)

"""
rhs function takes in an equation and returns the right hand side of the equation
    $(TYPEDSIGNATURES)
"""
function rhs(equation::Equation)
    return equation.rhs
end

#set elements close to zero to zero
ξ_est[zero_entries] .= 0
Ξ_est =reshape(ξ_est, size(Ξ))


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
            #get coefficient for variable in the equation
            C[k, idx] = Symbolics.coeff(lhs, var)
        end
    end
    return C, d
end

function add_constraint(constraint, C, d, Ξ)
    d = vcat(d, constraint.rhs)
    lhs = constraint.lhs
    for var in get_variables(lhs)
        idx = findfirst(isequal(var), vec(Ξ))
        C = vcat(C, zeros(1, size(C,2)))
        C[end, idx] = Symbolics.coeff(lhs, var)
    end
    return C, d
end



C,d = linear_system(constraints, hcat(Ξ))

opt = STLSQ(exp10.(-5:0.1:-1))
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 1))
println(get_basis(ddsol))
println(get_parameter_map(get_basis(ddsol)))

using Plots
plot(ddsol)