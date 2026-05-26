using Pkg
Pkg.activate("fermentation_testing")
using Revise, DifferentialEquations, ModelingToolkit, Catalyst, Plots
include("../src/hybrid_model.jl")
export create_callback
include("../src/polyopt.jl")
 
fermentation_model = @reaction_network begin
    @parameters begin
        #Initial conditions
        glucose_0 = 130.0   #g/L

        #Growth parameters (assumed time unit: hours, concentrations: g/L)
        μ_DT = 0.35            # 1/h  - death rate of active biomass
        μ_L = 0.005            # 1/h  - activation rate (latent -> active)
        μ_x0 = 1.0            # 1/h  - basal growth rate
        k_x = 5.0              # g/L  - half-saturation constant (for μ_x denominator)
        μ_SD0 = 0.25           # 1/h  - sloughing/death baseline rate
        μ_glucose_max = 1.2  # 1/h  - max glucose consumption rate
        k_glucose = 12.0       # g/L  - Monod constant for glucose
        μ_metabolite_max = 0.8  # 1/h  - max metabolite effect rate
        k_metabolite = 10.0       # g/L  - Monod constant for metabolite effect
        # μ_DY = 0.001           # (see equation) empirical coeff for diacetyl formation (per g·L⁻¹·h)
        # μ_AB = 0.01            # (see equation) empirical coeff for diacetyl removal (per g·L⁻¹·h)
        # Y_EA = 0.05            # g ethyl_acetate / g biomass - yield (dimensionless)
        
        # #Temperature parameters 
        ΔH_total = .5       # J/g - total heat generated per g of glucose consumed (exothermic)
        T_C0 = 4.0             # °C - coolant temperature
        heat_transfer = 0.5    # 1/h - heat transfer coefficient between fermentor and jacket
        optimal_temperature = 25.0 # °C - optimal temperature for growth 
        sd_temp = 30.0          # °C - standard deviation for temperature effect (for Gaussian distribution)
        
        #Oxygen parameters
        O_solubility = 10.0 # g/L - maximum dissolved oxygen concentration at given conditions 
        kla = 0.1            # 1/h - volumetric mass transfer coefficient for oxygen 
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
        metabolite(t) = 0.0        # g/L
        # ethyl_acetate(t) = 0.0  # g/L
        # diacetyl(t) = 0.0      # g/L

        #Temperature variables
        jacket_temperature(t) = 4.0       # °C
        fermentor_temperature(t) = 10.0   # °C
        coolant_rate(t) = 0.0                # L/min

        #Oxygen concentration 
        # DO(t) = 10.0                     # g/L - dissolved oxygen concentration 
        # OUR(t) = 0.0                    # g/L/h - oxygen uptake rate

        #Fed batch feed rate (time-varying input)
        F_feed(t) = 0.0                     # g/L/h - feed rate of glucose (can be time-varying)
        feed_on(t) = 0.0                    # unitless - feed control signal (0 or 1)

        μ_x(t)         # 1/h - instantaneous growth rate
        μ_SD(t)        # 1/h - sloughing/death rate
        μ_glucose(t)   # 1/h - glucose-driven growth contribution
        μ_metabolite(t)   # 1/h - metabolite-driven rate
        X_total(t)       # g/L - total biomass concentration (latent + active + dead)
    end

    @equations begin
        X_total ~ X_latent + X_active + X_dead

        #Growth rates
        μ_x ~ μ_x0 * (glucose / (k_x + glucose))*exp(-(fermentor_temperature - optimal_temperature)^2/(2*sd_temp^2)) # Temperature effect on growth (Gaussian centered at optimal temperature)
        μ_SD ~ μ_SD0 * (0.5*glucose_0/(0.5*glucose_0 + metabolite))
        μ_glucose ~ μ_glucose_max * (glucose / (k_glucose + glucose))
        μ_metabolite ~ μ_metabolite_max * (glucose / (k_metabolite + glucose))

        #Feed control logic (simple on/off based on glucose concentration) 
        F_feed ~ feed_on * μ_glucose * X_active
        D(feed_on) ~ 0.0

        #Dynamics of products and by-products
        D(glucose) ~ -μ_glucose * X_active + F_feed
        D(metabolite) ~ μ_metabolite * X_active * (1 - metabolite / (0.5 * glucose_0))
        # D(ethyl_acetate) ~ Y_EA * μ_x * X_active
        # D(diacetyl) ~ μ_DY * X_active * glucose - μ_AB * diacetyl * metabolite
 
        #Temperature dynamics
        D(coolant_rate) ~ 0.0           # Treat coolant_rate as a control input (constant or time-varying)
        D(fermentor_temperature) ~  μ_glucose * X_active * ΔH_total - heat_transfer * (fermentor_temperature - jacket_temperature)
        D(jacket_temperature) ~ coolant_rate * (T_C0 - jacket_temperature) + heat_transfer * (fermentor_temperature - jacket_temperature )

        #Oxygen dynamics 
        # D(DO) ~ -OUR + kla * (O_solubility - DO) # Simple oxygen dynamics with uptake and re-aeration
        # OUR ~ 0.5 * μ_x * X_active # Oxygen uptake rate proportional to growth rate and active biomass
    end

    #Cell growth and death       
    μ_L,    X_latent --> X_active
    μ_x,    X_active --> 2X_active 
    μ_DT,   X_active --> X_dead 
    μ_SD,   X_dead --> ∅
end 

@unpack X_active, X_latent, X_dead, X_total, μ_x, μ_SD, μ_glucose, μ_metabolite = fermentation_model



@named odesys = convert(ODESystem, fermentation_model)
odesys = structural_simplify(odesys) 

idx_dict = Dict(zip(string.(unknowns(odesys)), 1:length(unknowns(fermentation_model))))
    



#events
#Temperature control parameters
condition(u, t, integrator) = u[idx_dict["fermentor_temperature(t)"]] > 35.0 # Trigger when fermentor temperature exceeds 30°C
affect!(integrator) = (integrator.u[idx_dict["coolant_rate(t)"]] = 0.5) # Set coolant_rate to 0.5 L/min when triggered
cb = DiscreteCallback(condition, affect!) 
condition_stop(u, t, integrator) = u[idx_dict["fermentor_temperature(t)"]] < 10.0 # Stop when fermentor temperature drops below 10°C
affect_stop!(integrator) = (integrator.u[idx_dict["coolant_rate(t)"]] = 0.) # Set coolant_rate back to 0 when stopped
cb_stop = DiscreteCallback(condition_stop, affect_stop!)
#Feed control parameters
condition_feed(u, t, integrator) = u[idx_dict["glucose(t)"]] < 100.0 # Trigger when glucose concentration exceeds 0.5 g/L
affect_feed!(integrator) = (integrator.u[idx_dict["feed_on(t)"]] = 1.0) # Set feed_on to 1.0 when triggered
cb_feed = DiscreteCallback(condition_feed, affect_feed!)
condition_feed_stop(u, t, integrator) = t >= 100.0 # Stop feed after 50 hours
affect_feed_stop!(integrator) = (integrator.u[idx_dict["feed_on(t)"]] = 0.0) # Set feed_on back to 0 when stopped
cb_feed_stop = DiscreteCallback(condition_feed_stop, affect_feed_stop!)

cb_combined = CallbackSet(cb, cb_stop, cb_feed, cb_feed_stop) 





prob = ODEProblem(odesys, [], (0.0, 150.0), callback=cb_combined)
sol = solve(prob, Tsit5(), saveat=0.1)      
#plot oxygen
#  np_oxygen = plot(sol, idxs = [:DO, :OUR], labels=hcat(["Dissolved Oxygen", "Oxygen Uptake Rate"]...), title = "Oxygen Dynamics", xlabel = "Time (h)", ylabel = "Concentration / Rate (g/L or g/L/h)", color=[:cyan :magenta])


#plot tempereature dynamics
p_temperature = plot(sol, idxs = [:fermentor_temperature, :jacket_temperature], labels=hcat(["Fermentor temperature", "Jacket temperature"]...), title = "Temperature", 
    xlabel = "Time (h)", ylabel = "Temperature (°C)", color = [:red :blue])
#plot feeding dynamics
p_feeding = plot(sol, idxs = :F_feed, labels="Feed rate", title = "Feeding", xlabel = "Time (h)", ylabel = "Feed rate (g/L/h)", color=:green)

# p_growth_death = plot(sol, idxs = [μ_x, μ_SD], labels=hcat(["μ_x", "μ_SD"]...), title = "Growth and Death Rates", xlabel = "Time (h)", ylabel = "Rate (1/h)")

p_biomass = plot(sol, idxs = [:X_latent, :X_active, :X_dead, X_latent + X_active + X_dead], labels=hcat(["X_latent", "X_active", "X_dead", "Total"]...), title = "Biomass Concentrations", xlabel = "Time (h)", ylabel = "Concentration (g/L)", color=[:orange :green :red :black])
#plot the substrate and products
p_substrate_products = plot(sol, idxs = [:diacetyl, :metabolite, :ethyl_acetate, :glucose], labels=hcat(["diacetyl", "metabolite", "ethyl_acetate", "glucose"]...), title = "Substrate and Product Concentrations", xlabel = "Time (h)", ylabel = "Concentration (g/L)", color=[:purple :brown :pink :gray])
#plot the temperature variables
# p_temperature = plot(sol, idxs = 8:9, labels=hcat(["fermentor_temperature", "jacket_temperature"]...))
#combine the plots
final_plot = plot(p_biomass, p_substrate_products, p_temperature, p_feeding, layout=(2,2), size=(900,600))


using DataFrames, Random

# Build a mixed-frequency dataframe.
# Biomass / substrates / products are observed every 24 h. 
# Temperature and feeding are observed every hour.
sample_times = collect(0.0:1.0:150.0)
biomass_times = Set(0.0:24.0:150.0)
rng = MersenneTwister(42)

noise_scale(value, rel_sigma, abs_floor) = max(abs(value) * rel_sigma, abs_floor)
noisy(value, rel_sigma, abs_floor) = value + randn(rng) * noise_scale(value, rel_sigma, abs_floor)

observed_24h(t) = any(isapprox(t, τ; atol = 1e-9) for τ in biomass_times)

df = DataFrame(
    time = sample_times,
    X_latent = [observed_24h(t) ? noisy(sol(t, idxs = :X_latent), 0.05, 0.02) : missing for t in sample_times],
    X_active = [observed_24h(t) ? noisy(sol(t, idxs = :X_active), 0.05, 0.02) : missing for t in sample_times],
    X_dead = [observed_24h(t) ? noisy(sol(t, idxs = :X_dead), 0.05, 0.02) : missing for t in sample_times],
    glucose = [observed_24h(t) ? noisy(sol(t, idxs = :glucose), 0.03, 0.05) : missing for t in sample_times],
    metabolite = [observed_24h(t) ? noisy(sol(t, idxs = :metabolite), 0.03, 0.02) : missing for t in sample_times],
    ethyl_acetate = [observed_24h(t) ? noisy(sol(t, idxs = :ethyl_acetate), 0.03, 0.02) : missing for t in sample_times],
    diacetyl = [observed_24h(t) ? noisy(sol(t, idxs = :diacetyl), 0.03, 0.01) : missing for t in sample_times],
    fermentor_temperature = [noisy(sol(t, idxs = :fermentor_temperature), 0.01, 0.1) for t in sample_times],
    jacket_temperature = [noisy(sol(t, idxs = :jacket_temperature), 0.01, 0.1) for t in sample_times],
    F_feed = [noisy(sol(t, idxs = :F_feed), 0.02, 0.01) for t in sample_times]
)

df.X_total = coalesce.(df.X_latent, 0.0) .+ coalesce.(df.X_active, 0.0) .+ coalesce.(df.X_dead, 0.0)


p_temperature = scatter!(p_temperature, df.time, df.fermentor_temperature, label="Fermentor Temperature", color=:red)
scatter!(p_temperature, df.time, df.jacket_temperature, label="Jacket Temperature", color=:blue)
p_feeding = scatter!(p_feeding, df.time, df.F_feed, label="Feed Rate", color=:green)
p_biomass = scatter!(p_biomass, df.time, df.X_latent, label="X_latent", color=:orange)
scatter!(p_biomass, df.time, df.X_active, label="X_active", color=:green)
scatter!(p_biomass, df.time, df.X_dead, label="X_dead", color=:red)
scatter!(p_biomass, df.time, df.X_total, label="Total Biomass", color=:black)
p_substrate_products = scatter!(p_substrate_products, df.time, df.glucose, label="Glucose", color=:gray)
scatter!(p_substrate_products, df.time, df.metabolite, label="Metabolite", color=:brown)
scatter!(p_substrate_products, df.time, df.ethyl_acetate, label="Ethyl Acetate", color=:pink)
scatter!(p_substrate_products, df.time, df.diacetyl, label="Diacetyl", color=:purple)
final_plot = plot(p_biomass, p_substrate_products, p_temperature, p_feeding, layout=(2,2), size=(900,600))
