using Groebner

import Base:rationalize
function Base.rationalize(x;sigdigits=1)
       return Int(round(x*10^(sigdigits-1),digits=0))//10^(sigdigits-1)
end

import ModelingToolkit:calculate_jacobian
function ModelingToolkit.calculate_jacobian(hmodel::HybridModel)
    jacobian = calculate_jacobian(hmodel.sys) + calculate_jacobian(hmodel.surrogate)
    return jacobian
end

function centered_manifold_updated_params!(hmodel::HybridModel, initp::Array{<:ComponentArray}, equilibrium = nothing; kwargs...)
    for i in eachindex(initp)
        initp[i] = centered_manifold_updated_params!(hmodel, initp[i], equilibrium; kwargs...)
    end
    return initp
end

function centered_manifold_updated_params!(hmodel::HybridModel, initp::ComponentArray, equilibrium = nothing; digits = 4)
    jac = calculate_jacobian(hmodel)
    sym_to_val_map = Dict{Num, Float64}()
    string_to_val_map = Dict{String, Float64}()
    for (i, param) in enumerate(parameters(hmodel.surrogate))
        sym_to_val_map[param] = round(initp.surrogate[i], digits = digits)
        string_to_val_map[string(param)] = round(initp.surrogate[i], digits = digits)
    end

    for (i, param) in enumerate(parameters(hmodel.sys))     
        sym_to_val_map[param] = round(initp.sys[i], sigdigits = digits)
        string_to_val_map[string(param)] = round(initp.sys[i], sigdigits = digits)
    end     
    string_to_val_map = centered_manifold_sampling(jac, equilibrium, sym_to_val_map, string_to_val_map)
    initp.sys = map(x -> string_to_val_map[x], labels(initp.sys))
    initp.surrogate = map(x -> string_to_val_map[x], labels(initp.surrogate))
    return initp
end


# params
function centered_manifold_sampling(jac, equilibrium_guess, sym_to_val_map, string_to_val_map; λ_real = 0, λ_imag = 17//100)
    eq1 = 2*λ_real ~ jac[1, 1] + jac[2,2] # Trace of the Jacobian
    eq2 = λ_real^2 + λ_imag^2 ~ jac[1, 1] * jac[2, 2] - jac[1, 2] * jac[2, 1] # Determinant of the Jacobian

    eq1, eq2 = substitute([eq1, eq2], equilibrium_guess)


    for (param, value) in sym_to_val_map
        eq1temp,eq2temp = substitute([eq1, eq2], param => rationalize(value))
        if length(union(arguments(Symbolics.value(eq1temp.rhs)), arguments(Symbolics.value(eq2temp.rhs)))) < 6
            break
        end
        eq1, eq2 = eq1temp, eq2temp
    end


    eigen_sol = first(symbolic_solve([eq1, eq2]))
    for (sym,expr) in eigen_sol
        string_to_val_map[string(sym)] = Symbolics.symbolic_to_float(real(substitute(expr, sym_to_val_map)))
    end
    return string_to_val_map
end
# end

