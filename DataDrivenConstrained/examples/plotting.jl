function plot_interpolation(sol, interval; interpolation = InterpolationMethod(CubicSpline), variable_names = hcat(["x$i" for i in 1:size(sol.u[1], 1)]...), kwargs...)
    p1 = plot(layout = (3, 1), size = (800, 600); kwargs...)
    X = hcat(sol.u...)'
    DX = Array(sol(sol.t, Val{1}))
    DX = Array(data(data.t, Val{1}))
    DX_est,X_est,_ = collocate_data(X',data.t,interpolation)
    #plot interpolation vs true data
    plot!(p1,sol.t[interval],X[interval,:], linewidth = 2, legend = :bottomright, title = "True vs Interpolated Data", xlabel = "t", ylabel = "Concentration",subplot = 1, labels = variable_names.*"_true")
    plot!(p1, sol.t[interval],X_est'[interval,:], linestyle = :dash, linewidth = 2, subplot = 1, labels = variable_names.*"_interpolated")
    #plot derivative interpolation vs true derivative
    plot!(p1,sol.t[interval],DX[:,interval]', linewidth = 2, legend = :bottomright, title = "True vs Interpolated Derivatives", xlabel = "t", ylabel = "Derivative",subplot = 2, labels = variable_names.*"_true")
    plot!(p1, sol.t[interval],DX_est'[interval,:], linestyle = :dash, linewidth = 2, subplot = 2, labels = variable_names.*"_interpolated")
    display(p1)
    #savefig in plots folder#derivative residual
    plot!(p1,sol.t[interval],abs.(DX'[interval,:] .- DX_est'[interval,:]), linewidth = 2, legend = :bottomright, title = "Residual of True vs Interpolated Derivatives", xlabel = "t", ylabel = "Residual", subplot = 3, labels = variable_names.*"_residual")
    return p1
end

function plot_data(df, dfsol, u; ddsolconstrained = nothing, t_interval = collect(1:1:size(dfsol, 1)), kwargs...)
    num_states = length(u)
    colors = [:black, :red, :green]
    linewidth = 2
    opacity1 = 0.7
    opacity2 = 0.5

    p = plot(layout = (num_states, 1), size = (800, 600); kwargs...)

    # Plot each variable in its own subfigure with thicker lines and less opacity
    for i in 1:num_states
        variable = string(u[i])
        scatter!(p, df.t[t_interval], df[t_interval,variable], label = variable, color = colors[1], subplot = i, linewidth = linewidth, alpha = opacity1)
        plot!(p, dfsol.t[t_interval], dfsol[t_interval,variable], label = variable * "_est", linestyle = :dash, color = colors[2], subplot = i, linewidth = linewidth, alpha = opacity2)
        if !isnothing(ddsolconstrained)
            plot!(p, ddsolconstrained.t[t_interval], ddsolconstrained[t_interval,variable], label = variable * "_constrained", linestyle = :dot, color = colors[3], subplot = i, linewidth = linewidth, alpha = opacity2)
        end
    end

    display(p)
    return p
end