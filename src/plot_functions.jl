mutable struct PEtabRecipes
    res::PEtab.PEtabMultistartResult

    PEtabRecipes(res) = new(res)
end

logmod(x) = sign(x) * log(abs(x) + 1)



function param_trace(res, param_idx; ground_truth_value = nothing, run_idcs = collect(1:length(res.runs)),kwargs...)
    p1 = plot(; kwargs...)
    runs = res.runs[run_idcs]
    colors = [:blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :brown]
    markers = [:v, :cross, :star, :triangle, :x]
    if ground_truth !== nothing
        hline!(p1, [ground_truth_value], label = "Ground Truth", color = :red, linestyle = :dash, linewidth = 6, alpha = 0.5)
    end
    for (i, run) in enumerate(runs)
        p0 = run.x0
        color = colors[i % length(colors) + 1]
        marker = markers[i % length(markers) + 1]
        param_trace = hcat(run.xtrace ...)[param_idx,:]
        scatter!(p1, param_trace, label = "Run $(run_idcs[i])", markersize = 4, color = color, marker = marker, alpha = 0.8)
        #plot the line connecting the points
        plot!(p1, param_trace, label = "", linewidth = 1, markersize = 2, color = color, alpha = 0.8)
    end
    return p1
end
function param_trace(res, param_idx; ground_truth_value = nothing, run_idcs = collect(1:length(res.runs)),kwargs...)
    p1 = plot(; kwargs...)
    runs = res.runs[run_idcs]
    colors = [:blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :brown]
    markers = [:v, :cross, :star, :triangle, :x]
    if ground_truth !== nothing
        hline!(p1, [ground_truth_value], label = "Ground Truth", color = :red, linestyle = :dash, linewidth = 6, alpha = 0.5)
    end
    for (i, run) in enumerate(runs)
        p0 = run.x0
        color = colors[i % length(colors) + 1]
        marker = markers[i % length(markers) + 1]
        param_trace = hcat(run.xtrace ...)[param_idx,:]
        scatter!(p1, param_trace, label = "Run $(run_idcs[i])", markersize = 4, color = color, marker = marker, alpha = 0.8)
        #plot the line connecting the points
        plot!(p1, param_trace, label = "", linewidth = 1, markersize = 2, color = color, alpha = 0.8)
    end
    return p1
end

function loss_plot(res, petab_prob; 
                    run_idcs = best_runs(res, 10), 
                    transformation = likelogmod, 
                    calc_nllh = true, labels = ["run $(run_idcs[i])" for i in 1:length(run_idcs)],
                    linestyle = :solid,
                    kwargs...)
    runs = res.runs[run_idcs]
    fvals = getfield.(runs, :fmin)
    minval = minimum(fvals)
    minsign = sign(minval)
    maxval = maximum(fvals)
    maxsign = sign(maxval)


    #logmod ticks
    ytick_vals =minsign != maxsign ? sort([minsign*10.0 .^ (0:ceil(log10(abs(minval)))); maxsign* 10.0 .^ (0:ceil(log10(abs(maxval))))]) : [minsign*10.0 .^ (0:ceil(log10(abs(minval))))]
    ytick_labels = string.(ytick_vals)
    ytick_positions = transformation.(ytick_vals)
    colors = [:blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :brown]
    # markers = [:v, :cross, :star, :triangle, :x, :diamond]
    p3 = plot(xlabel = "Iteration", ylabel = "Negative Log-Likelihood", title = "Loss trace",
            yticks = (ytick_positions, ytick_labels), ylim = (transformation(minimum(ytick_vals)),transformation(maximum(ytick_vals))), 
            size = (800, 600))
    for (i, cur_res) in enumerate(runs)

        ftrace = calc_nllh ? petab_prob.nllh.(cur_res.xtrace) : cur_res.xtrace
        color = colors[i % length(colors) + 1]
        # marker = markers[i % length(markers) + 1]
        # scatter!(p3, transformation.(ftrace), markersize = 4, label = "Run $(run_idcs[i])", color = color, marker = marker, alpha = 0.8)
        #plot the line connecting the points
        plot!(p3, transformation.(ftrace), label = label = labels[i], linewidth = 1, markersize = 2, color = color, alpha = 0.8,linestyle = linestyle , kwargs...)
    end
    return p3
end

@recipe function f(res::PEtabRecipes, petab_prob; run_idcs = best_runs(res, 10), transformation = likelogmod, calc_nllh = true)
    runs = res.res.runs[run_idcs]
    fvals = getfield.(runs, :fmin)
    minval = minimum(fvals)
    minsign = sign(minval)
    maxval = maximum(fvals)
    maxsign = sign(maxval)
    # logmod ticks
    ytick_vals = minsign != maxsign ? sort([minsign*10.0 .^ (0:ceil(log10(abs(minval)))); maxsign*10.0 .^ (0:ceil(log10(abs(maxval))))]) : [minsign*10.0 .^ (0:ceil(log10(abs(minval))))]
    ytick_labels = string.(ytick_vals)
    ytick_positions = transformation.(ytick_vals)
    colors = [:blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :brown]
    # Set plot attributes
    xlabel --> "Iteration"
    ylabel --> "Negative Log-Likelihood"
    title --> "Loss trace"
    yticks --> (ytick_positions, ytick_labels)
    ylim --> (transformation(minimum(ytick_vals)), transformation(maximum(ytick_vals)))
    size --> (800, 600)
    # Add each run as a series
    for (i, cur_res) in enumerate(runs)
        ftrace = calc_nllh ? petab_prob.nllh.(cur_res.xtrace) : cur_res.xtrace
        color = colors[i % length(colors) + 1]
        @series begin
            seriestype := :line
            x := 1:length(ftrace)
            y := transformation.(ftrace)
            linewidth := 1
            markersize := 2
            color := color
            alpha := 0.8
            # pass through any extra kwargs
        end
    end
end