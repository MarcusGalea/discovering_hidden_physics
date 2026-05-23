using Pkg
Pkg.activate(@__DIR__)
# Pkg.instantiate()
 
using Revise, DifferentialEquations, ModelingToolkit, Catalyst
using Lux
using ModelingToolkit: t_nounits as t, D_nounits as D
using Plots

fermentation_model = @reaction_network begin
    @parameters begin
        # Initial conditions / process constants
        nutrients_0 = 130.0   # g/L in reactor
        metabolites_0 = 0.0   # g/L
        V0 = 1.0            # L - initial working volume (assume density ~1 kg/L)

        # Feed / control setpoint parameters
        nutrients_feed = 500.0  # g/L - substrate concentration in feed
        F_feed_default = 0.0    # L/h - nominal feed rate (time-varying input in practice)
        pH_setpoint = 5.5
        DO_setpoint = 0.2     # % or fraction, unitless here
        ΔT_setpoint = 0.0     # °C deviation from the desired thermal setpoint

        # Growth parameters (time unit: hours, concentrations: g/L)
        μ_DT = 0.35           # 1/h - death rate of active biomass
        μ_L = 0.005           # 1/h - activation rate (latent -> active)
        μ_SD0 = 0.25          # 1/h - sloughing/death baseline rate
        μ_x_max = 0.03        # 1/h - maximum growth rate driven by nutrients
        k_nutrients = 12.0    # g/L - Monod constant for nutrients
        k_DO = 0.05           # unitless - DO limitation constant
        k_pH = 0.08           # 1/pH^2 - pH sensitivity around the setpoint
        k_T = 0.02            # 1/°C^2 - temperature sensitivity around the setpoint
        Y_MX = 0.15           # g metabolites / g biomass - lumped product yield
        Y_NX = 0.20           # g nutrients / g biomass - lumped nutrient demand

        # Gas / DO related empirical params
        qO2 = 0.02            # mmol O2 / (g biomass · h) - specific O2 uptake rate
        k_DO_transfer = 0.8   # 1/h - aeration control rate toward DO_setpoint
        k_DO_consume = 0.4    # 1/h - DO drawdown from OUR
        k_CER = 0.6           # CO2 evolution per O2 consumed (stoichiometric approx)

        # pH / temperature control proxy parameters
        k_pH_control = 0.5    # 1/h - correction toward setpoint via base/acid additions
        alpha_base = 0.1      # pH change per unit base addition rate
        alpha_acid = 0.1      # pH change per unit acid addition rate
        k_T_gain = 0.08       # °C/h - heating contribution from active biomass
        k_T_loss = 0.12       # 1/h - relaxation toward ΔT_setpoint

        # Default acid/base addition rates (treated as parameters/inputs)
        NH4OH_add = 0.0       # L/h - base addition (default)
        H3PO4_add = 0.0       # L/h - acid addition (default)
    end

    @species begin
        # Biomass species (concentrations, g/L)
        X_latent(t) = 3.0  # g/L - latent biomass
        X_active(t) = 0.0  # g/L - active biomass
        X_dead(t) = 1.0    # g/L - dead biomass
    end

    @variables begin
        # Substrate and product concentrations (g/L)
        nutrients(t) = nutrients_0
        metabolites(t) = metabolites_0

        # Volume
        V(t) = V0               # L - working volume

        # Online sensors / control variables (states)
        DO(t) = DO_setpoint     # dissolved oxygen (fraction)
        pH(t) = pH_setpoint
        ΔT(t) = ΔT_setpoint     # temperature deviation from the setpoint

        # CER and OUR are algebraic proxies (defined below)

        # Algebraic off-gas proxies
        OUR(t) = 0.0            # O2 uptake (arbitrary units)
        CER(t) = 0.0            # CO2 evolution (arbitrary units)

        # Internal rate variables
        μ_x(t)          # 1/h - instantaneous growth rate
        μ_SD(t)         # 1/h - sloughing/death rate
    end

    @equations begin
        # Growth rate is driven by nutrients, DO, pH, and temperature; feed supplies nutrients.
        μ_x ~ μ_x_max * (nutrients / (k_nutrients + nutrients)) * (DO / (k_DO + DO)) * exp(-k_pH * (pH - pH_setpoint)^2) * exp(-k_T * (ΔT - ΔT_setpoint)^2)
        μ_SD ~ μ_SD0

        # Feed -> nutrients -> active biomass -> metabolites
        D(nutrients) ~ (F_feed_default / V) * (nutrients_feed - nutrients) - Y_NX * μ_x * X_active
        D(metabolites) ~ Y_MX * μ_x * X_active - (F_feed_default / V) * metabolites

        # Volume/weight dynamics (simple fed-batch: volume increases by feed parameter)
        D(V) ~ F_feed_default

        # Gas / DO / off-gas dynamics (empirical proxies)
        OUR ~ qO2 * X_active
        CER ~ k_CER * OUR
        D(DO) ~ k_DO_transfer * (DO_setpoint - DO) - k_DO_consume * OUR

        # pH control proxy: base/acid additions (treated as parameters) and control action
        D(pH) ~ k_pH_control*(pH_setpoint - pH) + alpha_base*NH4OH_add - alpha_acid*H3PO4_add

        # Thermal proxy around the setpoint; biomass heating competes with relaxation.
        D(ΔT) ~ k_T_gain * X_active - k_T_loss * (ΔT - ΔT_setpoint)

        # Temperature dynamics left as placeholders (can be expanded using energy balance)
        # D(fermentor_temperature) ~ ...
        # D(jacket_temperature) ~ ...
    end

    # Biomass growth and death (reactions); concentration changes occur alongside dilution
    μ_L,    X_latent --> X_active
    μ_x,    X_active --> 2X_active 
    μ_DT,   X_active --> X_dead 
    μ_SD,   X_dead --> ∅
end 



@named odesys = convert(ODESystem, fermentation_model)
odesys = structural_simplify(odesys) 
prob = ODEProblem(odesys, [], (0.0, 150.0))
sol = solve(prob, Tsit5(), saveat=0.1)      

# Plot biomass and observable process variables
p_biomass = plot(sol, idxs = [:X_latent, :X_active, :X_dead], labels=hcat(["X_latent", "X_active", "X_dead"]...), title = "Biomass Concentrations", xlabel = "Time (h)", ylabel = "Concentration (g/L)")
p_process = plot(sol, idxs = [:nutrients, :metabolites, :DO, :pH, :ΔT], labels=hcat(["nutrients", "metabolites", "DO", "pH", "ΔT"]...), title = "Process Variables", xlabel = "Time (h)", ylabel = "State / Concentration")

plot(p_biomass, p_process, layout = (2,1), size=(900,700))

# using Lux, Random, Reactant, Optimisers, ForwardDiff
# rng = MersenneTwister(1234)
# Random.seed!(rng, 1234) 
# #model growth rate as a function of substrate 
# f(x) = 0.02 * (x / (5.0 + x))
# x = 0:0.1:30  
# y = f.(x) + 0.0001 * randn(rng, length(x)) # add some noise
# plot(x, y, xlabel="Glucose Concentration (g/L)", ylabel="Growth Rate (1/h)", title="Monod-like Growth Rate vs Glucose", legend=false)   
# model = Chain(Dense(1 => 16, relu), Dense(16 => 1))
# const loss_fn = MSELoss()   
# const xdev = gpu_device()   
# ps,st = Lux.setup(rng, model)
# opt = ADAM(0.01)
# tstate = Training.TrainState(model, ps, st, opt)

# using Optimization, OptimizationOptimisers

# optf = OptimizationFunction((ps, x) -> begin
#     model = Lux.reconstruct(model, ps)
#     y_pred = model(x)
#     loss = loss_fn(y_pred, reshape(y, size(y_pred)))
#     return loss
# end, Optimization.AutoZygote())'
# optprob = OptimizationProblem(optf, ps, x)
# sol = solve(optprob, ADAM(0.01), maxiters=100)