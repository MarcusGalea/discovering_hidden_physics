module DataDrivenConstrained
    using CommonSolve
    using DocStringExtensions
    using LinearAlgebra
    using ModelingToolkit
    using DifferentialEquations
    using DataDrivenDiffEq

    #import types
    include("types.jl")
    export ConstrainedSTLSQ, ConstrainedSTLSQcache, initialize_cache

    #import functions
    include("functions.jl")
    export step!, solve, create_solution_basis, rel_error, rss
end # module DataDrivenConstrained
