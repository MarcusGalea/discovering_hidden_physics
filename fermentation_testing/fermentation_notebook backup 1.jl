### A Pluto.jl notebook ###

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following mock version of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 11111111-1111-1111-1111-111111111111
begin
    using Revise, DifferentialEquations, ModelingToolkit, Plots
    using ModelingToolkit: t_nounits as t, D_nounits as D
    using PlutoUI
    using HypertextLiteral: @htl
    gr()
    default(size = (900, 650), linewidth = 2.5, legend = :topright, background_color = :transparent)
end

# ╔═╡ 22222222-2222-2222-2222-222222222222
function fermentation_css()
    @htl("""
    <style>
        :root {
            --bg: #07111d;
            --bg2: #0b1728;
            --panel: rgba(9, 19, 33, 0.84);
            --line: rgba(156, 193, 255, 0.15);
            --text: #edf4ff;
            --muted: #9eb1ca;
            --accent: #6ee7d8;
            --accent2: #ffb86b;
        }

        body {
            background:
                radial-gradient(circle at top left, rgba(110, 231, 216, 0.15), transparent 28%),
                radial-gradient(circle at 88% 12%, rgba(255, 184, 107, 0.12), transparent 22%),
                linear-gradient(180deg, var(--bg), var(--bg2));
            color: var(--text);
        }

        .hero {
            border: 1px solid var(--line);
            border-radius: 1.4rem;
            padding: 1.2rem 1.35rem;
            background: linear-gradient(135deg, rgba(18, 34, 54, 0.92), rgba(8, 18, 30, 0.9));
            box-shadow: 0 22px 60px rgba(0, 0, 0, 0.28);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
            letter-spacing: 0.02em;
        }

        .hero p {
            margin: 0;
            color: var(--muted);
            line-height: 1.45;
        }

        .panel {
            border: 1px solid var(--line);
            border-radius: 1.1rem;
            background: var(--panel);
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .panel-title {
            margin: 0 0 0.85rem 0;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.8rem;
            color: var(--accent);
            font-weight: 700;
        }

        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
        }

        .slider-card {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 0.95rem;
            background: rgba(255, 255, 255, 0.03);
            padding: 0.8rem 0.85rem 0.9rem;
        }

        .slider-label {
            font-size: 0.85rem;
            color: var(--muted);
            margin-bottom: 0.35rem;
            font-weight: 600;
        }

        .slider-note {
            font-size: 0.74rem;
            color: rgba(237, 244, 255, 0.62);
            margin-top: 0.3rem;
        }

        .plot-card {
            border: 1px solid var(--line);
            border-radius: 1.1rem;
            background: linear-gradient(180deg, rgba(10, 19, 32, 0.85), rgba(8, 15, 26, 0.95));
            padding: 0.9rem;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
        }
    </style>
    """)
end

# ╔═╡ 33333333-3333-3333-3333-333333333333
fermentation_css()

# ╔═╡ 44444444-4444-4444-4444-444444444444
md"""
<div class="hero">
<h1>Fermentation Fed-Batch Explorer</h1>
<p>Interactive controls for initial conditions and key parameters. The plot updates directly from the sliders, with no narrative layer or extra scaffolding.</p>
</div>
"""

# ╔═╡ 55555555-5555-5555-5555-555555555555
begin
    @variables X_latent(t) X_active(t) X_dead(t) glucose(t) ethanol(t) enzyme(t) protein(t) temperature(t) V(t)
    @variables μ_x(t) μ_SD(t) μ_glucose(t) μ_ethanol(t) μ_enzyme(t) μ_protein(t)
    @parameters μ_L μ_DT μ_x0 k_x μ_SD0 μ_glucose_max k_glucose μ_ethanol_max k_ethanol k_enzyme k_protein
    @parameters temp_opt temp_width k_heat k_cool temperature_setpoint F_feed glucose_feed
end

# ╔═╡ 66666666-6666-6666-6666-666666666666
@named fermentation_sys = ODESystem([
    D(X_latent) ~ -μ_L * X_latent,
    D(X_active) ~ μ_L * X_latent + μ_x * X_active - μ_DT * X_active,
    D(X_dead) ~ μ_DT * X_active - μ_SD * X_dead,
    D(glucose) ~ -μ_glucose * X_active + (F_feed / V) * (glucose_feed - glucose),
    D(ethanol) ~ μ_ethanol * X_active - (F_feed / V) * ethanol,
    D(enzyme) ~ μ_enzyme * X_active - (F_feed / V) * enzyme,
    D(protein) ~ μ_protein * X_active - (F_feed / V) * protein,
    D(temperature) ~ k_heat * μ_x * X_active - k_cool * (temperature - temperature_setpoint),
    D(V) ~ F_feed,
    μ_x ~ μ_x0 * (glucose / (k_x + glucose)) * exp(-((temperature - temp_opt)^2) / (2 * temp_width^2)),
    μ_SD ~ μ_SD0 * (1 / (1 + glucose / (k_glucose + 1e-6))),
    μ_glucose ~ μ_glucose_max * (glucose / (k_glucose + glucose)),
    μ_ethanol ~ μ_ethanol_max * (glucose / (k_ethanol + glucose)),
    μ_enzyme ~ k_enzyme * μ_x,
    μ_protein ~ k_protein * μ_x,
], t)

# ╔═╡ 77777777-7777-7777-7777-777777777777
md"""
<div class="panel">
<div class="panel-title">Initial conditions</div>
<div class="grid-3">
  <div class="slider-card">
    <div class="slider-label">Latent biomass, X_latent(0)</div>
    $(@bind X_latent0 Slider(0.0:0.1:10.0, default = 3.0, show_value = true))
    <div class="slider-note">g/L</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Active biomass, X_active(0)</div>
    $(@bind X_active0 Slider(0.0:0.05:5.0, default = 0.0, show_value = true))
    <div class="slider-note">g/L</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Dead biomass, X_dead(0)</div>
    $(@bind X_dead0 Slider(0.0:0.05:5.0, default = 1.0, show_value = true))
    <div class="slider-note">g/L</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Glucose, S(0)</div>
    $(@bind glucose0 Slider(0.0:1.0:250.0, default = 130.0, show_value = true))
    <div class="slider-note">g/L</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Ethanol, P(0)</div>
    $(@bind ethanol0 Slider(0.0:0.1:20.0, default = 0.0, show_value = true))
    <div class="slider-note">g/L</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Working volume, V(0)</div>
    $(@bind V0 Slider(0.5:0.1:20.0, default = 1.0, show_value = true))
    <div class="slider-note">L</div>
  </div>
</div>
</div>
"""

# ╔═╡ 88888888-8888-8888-8888-888888888888
md"""
<div class="panel">
<div class="panel-title">Kinetics and feed</div>
<div class="grid-3">
  <div class="slider-card">
    <div class="slider-label">Activation rate, μ_L</div>
    $(@bind μ_L_slider Slider(0.0:0.001:0.05, default = 0.005, show_value = true))
    <div class="slider-note">1/h</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Growth scale, μ_x0</div>
    $(@bind μ_x0_slider Slider(0.0:0.005:0.2, default = 0.02, show_value = true))
    <div class="slider-note">1/h</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Death rate, μ_DT</div>
    $(@bind μ_DT_slider Slider(0.0:0.01:1.0, default = 0.35, show_value = true))
    <div class="slider-note">1/h</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Feed rate, F_feed</div>
    $(@bind F_feed_slider Slider(0.0:0.01:1.0, default = 0.0, show_value = true))
    <div class="slider-note">L/h</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Feed glucose, S_feed</div>
    $(@bind glucose_feed_slider Slider(0.0:5.0:800.0, default = 500.0, show_value = true))
    <div class="slider-note">g/L</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Temperature setpoint</div>
    $(@bind temperature_setpoint_slider Slider(10.0:0.5:45.0, default = 25.0, show_value = true))
    <div class="slider-note">°C</div>
  </div>
</div>
</div>
"""

# ╔═╡ 99999999-9999-9999-9999-999999999999
md"""
<div class="panel">
<div class="panel-title">Yield and temperature response</div>
<div class="grid-3">
  <div class="slider-card">
    <div class="slider-label">Glucose Monod constant, k_glucose</div>
    $(@bind k_glucose_slider Slider(1.0:1.0:80.0, default = 12.0, show_value = true))
    <div class="slider-note">g/L</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Ethanol Monod constant, k_ethanol</div>
    $(@bind k_ethanol_slider Slider(1.0:1.0:80.0, default = 10.0, show_value = true))
    <div class="slider-note">g/L</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Product yield scale, k_enzyme</div>
    $(@bind k_enzyme_slider Slider(0.0:0.001:0.05, default = 0.01, show_value = true))
    <div class="slider-note">g enzyme / (g biomass · h)</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Protein yield scale, k_protein</div>
    $(@bind k_protein_slider Slider(0.0:0.001:0.08, default = 0.02, show_value = true))
    <div class="slider-note">g protein / (g biomass · h)</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Temperature width, σ_T</div>
    $(@bind temp_width_slider Slider(1.0:0.5:20.0, default = 7.0, show_value = true))
    <div class="slider-note">°C</div>
  </div>
  <div class="slider-card">
    <div class="slider-label">Cooling strength, k_cool</div>
    $(@bind k_cool_slider Slider(0.0:0.01:1.0, default = 0.12, show_value = true))
    <div class="slider-note">1/h</div>
  </div>
</div>
</div>
"""

# ╔═╡ aaaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa
begin
    u0 = [
        X_latent => X_latent0,
        X_active => X_active0,
        X_dead => X_dead0,
        glucose => glucose0,
        ethanol => ethanol0,
        enzyme => 0.0,
        protein => 0.0,
        temperature => 10.0,
        V => V0,
    ]

    p = [
        μ_L => μ_L_slider,
        μ_DT => μ_DT_slider,
        μ_x0 => μ_x0_slider,
        k_x => 5.0,
        μ_SD0 => 0.25,
        μ_glucose_max => 0.006,
        k_glucose => k_glucose_slider,
        μ_ethanol_max => 0.008,
        k_ethanol => k_ethanol_slider,
        k_enzyme => k_enzyme_slider,
        k_protein => k_protein_slider,
        temp_opt => 25.0,
        temp_width => temp_width_slider,
        k_heat => 0.08,
        k_cool => k_cool_slider,
        temperature_setpoint => temperature_setpoint_slider,
        F_feed => F_feed_slider,
        glucose_feed => glucose_feed_slider,
    ]

    sys = structural_simplify(fermentation_sys)
    prob = ODEProblem(sys, u0, (0.0, 150.0), p, warn_initialize_determined = false)
    sol = solve(prob, Tsit5(), saveat = 0.25)
end

# ╔═╡ bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb
begin
    p_biomass = plot(sol,
        idxs = [:X_latent, :X_active, :X_dead],
        labels = ["Latent", "Active", "Dead"],
        title = "Biomass",
        xlabel = "Time (h)",
        ylabel = "g/L",
        color = ["#f4a261", "#2a9d8f", "#e76f51"])

    p_media = plot(sol,
        idxs = [:glucose, :ethanol, :enzyme, :protein],
        labels = ["Glucose", "Ethanol", "Enzyme", "Protein"],
        title = "Substrates and Products",
        xlabel = "Time (h)",
        ylabel = "g/L",
        color = ["#e9c46a", "#f4a261", "#6ee7d8", "#90caf9"])

    p_process = plot(sol,
        idxs = [:temperature, :V],
        labels = ["Temperature", "Volume"],
        title = "Process Signals",
        xlabel = "Time (h)",
        ylabel = "Value",
        color = ["#ffb86b", "#8ecae6"])

    @htl("""
    <div class="plot-card">
    $(plot(p_biomass, p_media, p_process, layout = (3, 1), size = (950, 980)))
    </div>
    """)
end
