using Combinatorics

"""Create a full reaction network with all possible combinations of reactions.

    Args:
        vars: A vector of species names (symbols).
        t: independent variable (time).
        number_reactants: The number of reactants in the reaction. Default is 2.
    Returns:
        A ReactionSystem with all possible reactions.
"""    
function create_full_reaction_network(vars:: Vector, t::Num; 
                                        number_reactants = 2)

    # Create a vector of all possible combinations of reactants
    combs = vcat([[comb for comb in combinations(vars,i)] for i in 1:number_reactants]...)
    # combs_string = [join([first(split(species,"(t)")) for species in string.(comb)],"+") for comb in combs]
    n = length(combs)
    @parameters k[1:n, 1:n] # reaction rates without 
    used_params = [k[i,j] for i in 1:n, j in 1:n if i != j]
    reactions = Reaction[]

    for (i,lhs) in enumerate(combs)
        for (j,rhs) in enumerate(combs)
            if !isequal(lhs,rhs) # identical reactants and products
                # lhs_orders = ones(length(lhs)) #change this for higher order reactions
                # rhs_orders = ones(length(rhs)) #change this for higher order reactions
                push!(reactions, Reaction(k[i,j], lhs, rhs))#, lhs_orders, rhs_orders))
            end
        end
    end
    @named rs_all_combs = ReactionSystem(reactions, t, vars, used_params)
    return rs_all_combs
end


function ReactionSINDy(rx, u, du_approx; threshold = 1e-3, lambda =0.0, verbose = false)
    old_rns = equations(rx)
    ode = convert(ODESystem, complete(rx))
    new_rns = Reaction[]
    paramsest = nothing
    while length(new_rns) != length(old_rns)
        # Save the current reactions for comparison
        old_rns = deepcopy(new_rns)  # Use deepcopy to avoid unintended modifications

        # Set up the linear system and perform physics-informed regression
        A, b = PhysicsInformedRegression.setup_linear_system(ode)
        paramsest = physics_informed_regression(ode, u, du_approx, A, b; lambda = lambda, verbose = verbose)

        # Initialize a new list of reactions
        new_rns = Reaction[]

        # Identify parameters above the threshold and retain corresponding reactions
        for reaction in equations(rx)
            if paramsest[reaction.rate] > threshold
                push!(new_rns, reaction)
            elseif verbose
                println("Removing reaction: ", reaction, " with rate: ", paramsest[reaction.rate])
            end
        end
        # Update the reaction system and ODE
        @named rx = ReactionSystem(new_rns, rx.t, species(rx), [reaction.rate for reaction in new_rns])
        ode = convert(ODESystem, complete(rx))  # Convert to ODEFunction
    end
    return rx, paramsest
end

"""
Visualize the reaction network with parameters as circles.

Args:
    rs: Reaction system. (Original reaction system containing all possible reactions) \n
    rx: Reaction system. (Reduced reaction system after SINDy) \n
    paramsest: Estimated parameters. (Estimated parameters from the SINDy algorithm) \n

    params_true: True parameters. (Optional, for comparison) \n
Returns:
    p1: Plot object.
"""
function visualize_reaction_network(rs, rx, paramsest; params_true = nothing, annotate = true, kwargs...)

    rate_maps = Dict()
    gt_rate_maps = Dict()
    combs = vcat([[comb for comb in combinations(species(rs),i)] for i in 1:2]...)
    comb_strings_seperated = [Set(string.(comb)) for comb in combs]
    for reaction in equations(rs)
        i = findfirst(isequal(Set(string.(reaction.substrates))), comb_strings_seperated)
        j = findfirst(isequal(Set(string.(reaction.products))), comb_strings_seperated)
        rate_maps[reaction.rate] = (i, j)
        if !isnothing(params_true)
            if reaction.rate in keys(params_true)
                gt_rate_maps[reaction.rate] = (i, j)
            end
        end
    end
    #solve n^2 -n = n_params for n
    n = Int(floor(sqrt(length(parameters(rs)) + 1/4) - 1/2)) +1 
    # Example parameter values (replace with `paramsest` from your code)
    #set params as integers up to 30
    params = vcat(values(paramsest)...)  # Ensure params has 30 elements

    # Create the grid with increased spacing
    spacing = 2  # Adjust this value to control the spacing between grid points
    x = [rate_maps[param][2] for param in keys(paramsest)] .* spacing
    y = [rate_maps[param][1] for param in keys(paramsest)] .* spacing

    # Normalize circle sizes so the maximum radius is 0.5
    max_radius = 0.5
    # Adjust tick positions to match the scaled grid
    xtick_positions = (1:n) .* spacing
    ytick_positions = (1:n) .* spacing

    # Initialize the plot
    p1 = plot()

    # Calculate axis limits
    xlims_plot = (0, n*2 + spacing)
    ylims_plot = (1 - spacing, n*2 + spacing)
    axis_range = min(xlims_plot[2] - xlims_plot[1], ylims_plot[2] - ylims_plot[1])

    # Set max_radius to be 0.5 units in axis coordinates
    max_radius_axis_units = 0.5
    # Plots.jl markersize is in points, so we approximate scaling:
    # Use a reference: if axis_range maps to plot size (e.g., 600px), then 0.5 units ~ (0.5/axis_range)*600 px
    # We'll use plot size (800, 600) as in the code
    plot_px = 600  # height in px
    max_radius_px = max_radius_axis_units / axis_range * plot_px

    max_param_value = maximum([abs.(params);abs.(vcat(values(params_true)...))])
    sizes = [abs(param) / max_param_value * max_radius_px * 2 for param in params]  # *2 for diameter

    # Define colors: positive values are blue, negative values are red
    colors = [param >= 0 ? :blue : :red for param in params]

    # Define custom labels for the ticks
    vars = ModelingToolkit.get_unknowns(rs)
    combs = vcat([[comb for comb in combinations(vars,i)] for i in 1:2]...)
    combs_string = [join([first(split(species,"(t)")) for species in string.(comb)],"+") for comb in combs]

    x_labels = combs_string
    y_labels = combs_string

    # Adjust tick positions to match the scaled grid
    xtick_positions = (1:n) .* spacing
    ytick_positions = (1:n) .* spacing

    # Initialize the plot
    p1 = plot()

    # Add ground true parameters as circles
    if !isnothing(params_true)
        x_gt = [gt_rate_maps[param][2] for param in keys(params_true)] .* spacing
        y_gt = [gt_rate_maps[param][1] for param in keys(params_true)] .* spacing

        sizes_gt = [abs(param) / max_param_value * max_radius_px * 2 for param in values(params_true)]  # Scale for visualization
        scatter!(p1, x_gt, y_gt, m = (7, :white, stroke(1, color), sizes_gt), label="Ground Truth Parameters", legend=:topright,) 
        #color = :transparent)
    end


    scatter!(p1, x, y, markersize=sizes, color=colors, alpha=0.5, legend=false,
                xlabel="Reactants", ylabel="Products", title="Parameter Visualization",
                xticks=(xtick_positions, x_labels), yticks=(ytick_positions, y_labels),
                xlims=xlims_plot,
                ylims=ylims_plot,
                grid=true, gridcolor=:lightgray, gridalpha=0.5,
                tickfontsize=8, tickfontcolor=:black, size=(800, 600), dpi=300, label = "Estimated Parameters", kwargs...)


    # Annotate the parameter values inside the circles
    if annotate
        for (i, param) in enumerate(params)
            annotate!(p1, x[i], y[i], text(round(param, digits=2), :black, 8, :center))
        end
    end
    return p1
end



"""
# Function to remove known reactions from the reaction network
# Arguments:
- `reaction_searchspace`: The reaction network to search for reactions.
- `known_reactions`: The known reactions to be removed from the search space.

# Returns:
- `new_reaction_searchspace`: The reaction network with the known reactions removed.
"""
function remove_known_reactions(reaction_searchspace::ReactionSystem, known_reactions::ReactionSystem)

    params_to_remove = Num[]
    for known_reaction in equations(known_reactions)
        # Check if the reaction is in the search space
        for reaction in equations(reaction_searchspace)
            if isequal(known_reaction.netstoich, reaction.netstoich)
                # Remove the known reaction from the search space
                push!(params_to_remove, reaction.rate)
            end
        end
    end
    return remove_params(reaction_searchspace, params_to_remove)
end



"""
Remove parameter from the ODE system and return the new system.

Parameters:
- `rn`: The original Catalyst Reaction System.
- `params_to_remove`: A parameter to be removed.

Returns:
- `new_rn`: The new Reaction System with parameters removed.
"""
function remove_params(rn::ReactionSystem, params_to_remove::Vector{<:Num})

    # Filter reactions where the rate is not in `params_to_remove`
    new_reactions = [reaction for reaction in equations(rn) for param_to_remove in params_to_remove if !isequal(reaction.rate, param_to_remove)]

    # Filter parameters that are not in `param_to_remove`
    new_params = [param for param in parameters(rn) for param_to_remove in params_to_remove if !isequal(param, param_to_remove)]

    # Create a new Reaction System with the updated reactions and parameters
    @named new_rn = ReactionSystem(new_reactions, rn.t, species(rn), new_params)
end



######## DOESN*T MATTER ###########

function insert_known_params(ode_rn, known_params)
    """
    insert known parameters from the ODE system and return the new system.

    Parameters:
    - `ode_rn`: The original ODE system.
    - `known_params`: A dictionary of known parameters to be inserted.

    Returns:
    - `new_ode`: The new ODE system with known parameters inserted.
    """
    # Get the equations and parameters of the ODE system
    new_eqs = Equation[]

    new_eqs = Equation[]
    for eq in equations(ode_rn)
        new_eq = substitute(eq, known_params)
        new_eqs = push!(new_eqs, new_eq)  
    end
    #remove parameters that are in known_params
    new_params = [param for param in parameters(ode_rn) if !(param in keys(known_params))]
    @named new_ode = ODESystem(new_eqs, t, ModelingToolkit.get_unknowns(ode_rn), new_params)
    return new_ode
end
