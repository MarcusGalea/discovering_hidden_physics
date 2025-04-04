module DataDrivenConstrained
    using CommonSolve
    using DocStringExtensions
    using LinearAlgebra
    using ModelingToolkit
    using DifferentialEquations
    using DataDrivenDiffEq
    using StatsBase
    using DataFrames

    #import types
    include("types.jl")
    export ConstrainedSTLSQ, ConstrainedSTLSQcache, initialize_cache

    #import functions
    include("functions.jl")
    export step!, solve, create_solution_basis, rel_error, rss, ODESystem

    ### MLJ regression
    using MLJBase
    include("MLJmethods.jl")
    export MLJConstrainedSTLSQ, fit, predict

end # module DataDrivenConstrained
