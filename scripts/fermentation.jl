using Pkg
Pkg.activate(@__DIR__)
# Pkg.instantiate()
# include("../src/hybrid_model.jl")
# # include("../src/SINDy_methods.jl")
# include("../src/polyopt.jl") 
using Revise, DifferentialEquations, ModelingToolkit, Catalyst
using Lux
using ModelingToolkit: t_nounits as t, D_nounits as D
using Plots

fermentation_model = @reaction_network begin
    @parameters begin
        #Initial conditions
        glucose_0 = 130.0   #g/L

        #Growth parameters (assumed time unit: hours, concentrations: g/L)
        μ_DT = 0.35            # 1/h  - death rate of active biomass
        μ_L = 0.005            # 1/h  - activation rate (latent -> active)
        μ_x0 = 0.02            # 1/h  - basal growth rate
        k_x = 5.0              # g/L  - half-saturation constant (for μ_x denominator)
        μ_SD0 = 0.25           # 1/h  - sloughing/death baseline rate
        μ_glucose_max = 1.2  # 1/h  - max glucose consumption rate
        k_glucose = 12.0       # g/L  - Monod constant for glucose
        μ_ethanol_max = 0.8  # 1/h  - max ethanol effect rate
        k_ethanol = 10.0       # g/L  - Monod constant for ethanol effect
        μ_DY = 0.001           # (see equation) empirical coeff for diacetyl formation (per g·L⁻¹·h)
        μ_AB = 0.01            # (see equation) empirical coeff for diacetyl removal (per g·L⁻¹·h)
        Y_EA = 0.05            # g ethyl_acetate / g biomass - yield (dimensionless)
        
        # #Temperature parameters 
        ΔH_total = .5       # J/g - total heat generated per g of glucose consumed (exothermic)
        T_C0 = 4.0             # °C - coolant temperature
        heat_transfer = 0.5    # 1/h - heat transfer coefficient between fermentor and jacket
    end

    @species begin
        #Biomass species
        X_latent(t) = 3.0   # g/L - latent biomass concentration
        X_active(t) = 0.0   # g/L - active biomass concentration
        X_dead(t) = 1.0     # g/L - dead biomass concentration
    end

    @variables begin
        #Substrate and product concentrations
        glucose(t) = glucose_0        # g/L
        ethanol(t) = 0.0        # g/L
        ethyl_acetate(t) = 0.0  # g/L
        diacetyl(t) = 0.0      # g/L

        #Temperature variables
        jacket_temperature(t) = 4.0       # °C
        fermentor_temperature(t) = 10.0   # °C
        coolant_rate(t) = 0.0                # L/min

        #Fed batch feed rate (time-varying input)
        F_feed(t) = 0.0                     # g/L/h - feed rate of glucose (can be time-varying)
        feed_on(t) = 0.0                    # unitless - feed control signal (0 or 1)

        μ_x(t)         # 1/h - instantaneous growth rate
        μ_SD(t)        # 1/h - sloughing/death rate
        μ_glucose(t)   # 1/h - glucose-driven growth contribution
        μ_ethanol(t)   # 1/h - ethanol-driven rate
    end

    @equations begin
        #Growth rates
        μ_x ~ μ_x0 * (glucose / (k_x))    
        μ_SD ~ μ_SD0 * (0.5*glucose_0/(0.5*glucose_0 + ethanol))
        μ_glucose ~ μ_glucose_max * (glucose / (k_glucose + glucose))
        μ_ethanol ~ μ_ethanol_max * (glucose / (k_ethanol + glucose))

        #Feed control logic (simple on/off based on glucose concentration) 
        F_feed ~ feed_on * μ_glucose * X_active
        D(feed_on) ~ 0.0

        #Dynamics of products and by-products
        D(glucose) ~ -μ_glucose * X_active + F_feed
        D(ethanol) ~ μ_ethanol * X_active * (1 - ethanol / (0.5 * glucose_0))
        D(ethyl_acetate) ~ Y_EA * μ_x * X_active
        D(diacetyl) ~ μ_DY * X_active * glucose - μ_AB * diacetyl * ethanol
 
        #Temperature dynamics
        D(coolant_rate) ~ 0.0           # Treat coolant_rate as a control input (constant or time-varying)
        D(fermentor_temperature) ~  μ_glucose * X_active * ΔH_total - heat_transfer * (fermentor_temperature - jacket_temperature)
        D(jacket_temperature) ~ coolant_rate * (T_C0 - jacket_temperature) + heat_transfer * (fermentor_temperature - jacket_temperature )
    end

    #Cell growth and death       
    μ_L,    X_latent --> X_active
    μ_x,    X_active --> 2X_active 
    μ_DT,   X_active --> X_dead 
    μ_SD,   X_dead --> ∅

end 

@unpack X_active, X_latent, X_dead, μ_x, μ_SD, μ_glucose, μ_ethanol = fermentation_model


#Temperature control parameters
condition(u, t, integrator) = u[9] > 15.0 # Trigger when fermentor temperature exceeds 20°C
affect!(integrator) = (integrator.u[4] = 0.1) # Set coolant_rate to 0.1 L/min when triggered
cb = DiscreteCallback(condition, affect!) 
condition_stop(u, t, integrator) = u[9] < 10.0 # Stop when fermentor temperature drops below 10°C
affect_stop!(integrator) = (integrator.u[4] = 0.) # Set coolant_rate back to 0 when stopped
cb_stop = DiscreteCallback(condition_stop, affect_stop!)
#Feed control parameters
condition_feed(u, t, integrator) = u[10] < 100.0 # Trigger when glucose concentration exceeds 0.5 g/L
affect_feed!(integrator) = (integrator.u[8] = 1.0) # Set feed_on to 1.0 when triggered
cb_feed = DiscreteCallback(condition_feed, affect_feed!)
condition_feed_stop(u, t, integrator) = t >= 100.0 # Stop feed after 50 hours
affect_feed_stop!(integrator) = (integrator.u[8] = 0.0) # Set feed_on back to 0 when stopped
cb_feed_stop = DiscreteCallback(condition_feed_stop, affect_feed_stop!)

cb_combined = CallbackSet(cb, cb_stop, cb_feed, cb_feed_stop) 



@named odesys = convert(ODESystem, fermentation_model)
odesys = structural_simplify(odesys) 
@show unknowns(odesys)
prob = ODEProblem(odesys, [], (0.0, 150.0), callback=cb_combined)
sol = solve(prob, Tsit5(), saveat=0.1)      
#plot tempereature dynamics
p_temperature = plot(sol, idxs = [:fermentor_temperature, :jacket_temperature], labels=hcat(["Fermentor temperature", "Jacket temperature"]...), title = "Temperature", xlabel = "Time (h)", ylabel = "Temperature (°C)")
#plot feeding dynamics
p_feeding = plot(sol, idxs = :F_feed, labels="Feed rate", title = "Feeding", xlabel = "Time (h)", ylabel = "Feed rate (g/L/h)")

# p_growth_death = plot(sol, idxs = [μ_x, μ_SD], labels=hcat(["μ_x", "μ_SD"]...), title = "Growth and Death Rates", xlabel = "Time (h)", ylabel = "Rate (1/h)")

p_biomass = plot(sol, idxs = [:X_latent, :X_active, :X_dead, X_latent + X_active + X_dead], labels=hcat(["X_latent", "X_active", "X_dead", "Total"]...), title = "Biomass Concentrations", xlabel = "Time (h)", ylabel = "Concentration (g/L)", )
#plot the substrate and products
p_substrate_products = plot(sol, idxs = [:diacetyl, :ethanol, :ethyl_acetate, :glucose], labels=hcat(["diacetyl", "ethanol", "ethyl_acetate", "glucose"]...), title = "Substrate and Product Concentrations", xlabel = "Time (h)", ylabel = "Concentration (g/L)", )
#plot the temperature variables
# p_temperature = plot(sol, idxs = 8:9, labels=hcat(["fermentor_temperature", "jacket_temperature"]...))
#combine the plots
sol
plot(p_biomass, p_substrate_products, p_temperature, p_feeding, layout=(2,2), size=(900,600))


using Random, Reactant