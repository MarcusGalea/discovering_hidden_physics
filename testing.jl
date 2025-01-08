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
t = ddprob.t



n,m = length(basis), size(ddprob)[1]
@parameters Ξ[1:n, 1:m] # Ξ is the coefficient matrix
# ξ is vectorized version of Ξ
ξ = vec(Ξ)
constraints = [
    ξ[3] ~  10;
    ξ[2] + ξ[3] ~ 0
] 
n_c = length(constraints)

#### EXPERIMENTAL

λ = 1e-5

#evaluate basis at the data points
Θ = basis(X,[],t)'
C,d = linear_system(constraints, hcat(Ξ)) 
A = vcat(hcat(blockdiags(Θ'Θ, m),C'), hcat(C, zeros(n_c,n_c)))
b = vcat(vec(Θ'*DX'),d)
ξ_est= (A\b)[1:end-n_c]

#find elements close to zero in ξ_est

idx = findall(x->abs(x)<λ, ξ_est)



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

C,d = linear_system(constraints, hcat(Ξ))
C



opt = STLSQ(exp10.(-5:0.1:-1))
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 1))
println(get_basis(ddsol))
println(get_parameter_map(get_basis(ddsol)))

