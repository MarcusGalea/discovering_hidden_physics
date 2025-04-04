


mutable struct MLJConstrainedSTLSQ <: MLJBase.Deterministic
    """ basis is a vector of basis functions to include in the model """
    basis::Basis
    """ constraints is a list of constraints """
    constraints::Vector{Equation}
    """ λ is the threshold of the iteration """
    λ::Float64
    """ Ξ is the symbolic coefficient matrix """
    Ξ::AbstractMatrix
    """ interpolation is the interpolation method """
    interpolation::InterpolationMethod
end



function df_to_Matrix_and_vector(df)
    t = df.t
    columnnames = names(df)
    columnnames = [columnname for columnname in columnnames if columnname != "t"]
    X = hcat([df[!, columnname] for columnname in columnnames]...)
    return X, t
end

function fit(model::MLJConstrainedSTLSQ, df)
    X,t = df_to_Matrix_and_vector(df)
    ddprob = ContinuousDataDrivenProblem(X', t, model.interpolation)
    opt = DataDrivenConstrained.ConstrainedSTLSQ(model.λ, model.constraints, model.Ξ)
    solbasis,cache = solve(ddprob, model.basis, opt, options = DataDrivenCommonOptions())
    return solbasis,cache
end

function predict(solbasis, df)
    t = df.t
    tspan = (t[1], t[end]) 
    odesys = ODESystem(solbasis, tspan)
    odesys = structural_simplify(odesys)
    speciesnames = string.(states(solbasis))
    u0 = [df[!, columnname][1] for columnname in speciesnames]

    u0map = Dict(zip(states(solbasis), u0))
    pmap = get_parameter_map(solbasis)
    basisprob = ODEProblem(odesys, u0map, tspan, pmap)
    ddsol = solve(basisprob, Tsit5(), saveat = t) 
    #return dataframe with column for each state
    map = Dict(zip(speciesnames, [ddsol[i, :] for i in 1:length(speciesnames)]))
    #add t to map
    map["t"] =ddsol.t
    return DataFrame(map)
end