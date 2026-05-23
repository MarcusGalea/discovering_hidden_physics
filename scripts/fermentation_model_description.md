# Fed-batch fermentation model — description

## Purpose
This document describes the simplified fed-batch reaction-network model implemented in `scripts/fermentation_fed_batch.jl`. The model is intended as a mechanistic-but-empirical core that can be used with online (1 min) and offline (~24 h) fermentation measurements for parameter estimation, observer design, or batch-level analyses.

## Key assumptions
- Operating mode: fed-batch (no outlet), volume increases with feed. Dilution of concentrations is modelled by feed/volume terms.
- Time units: hours. Concentrations: g/L. Volumes: L.
- Biomass is represented by three pools: `X_latent`, `X_active`, `X_dead` (g/L).
- Substrate/product dynamics include `glucose`, `ethanol`, `ethyl_acetate`, and `diacetyl` (g/L).
- Enzyme and total protein are modelled as observable product pools (`enzyme`, `protein`) produced proportionally to active biomass and diluted by feed.
- Dissolved oxygen (DO), O2 uptake rate (OUR), and CO2 evolution rate (CER) are included as simple empirical proxies suitable for data-driven correction or observer inputs.
- pH dynamics are modelled as a simple control proxy driven toward a setpoint by acid/base additions (`NH4OH_add`, `H3PO4_add`).

## Model structure (high level)
- Monod-like growth: `μ_x`, `μ_glucose`, `μ_ethanol` depend on substrate concentrations with Monod constants.
- Biomass reactions: activation (latent → active), growth (active → 2 active), death (active → dead), and sloughing (dead → ∅).
- Substrate consumption and product formation are coupled to active biomass with empirical yields.
- Feed terms: any concentration `C` is diluted/augmented by `(F_feed/V)*(C_feed - C)`; `V` increases by `D(V)=F_feed`.
- Gas proxies: `OUR = qO2 * X_active`, `CER = k_CER * OUR`. DO dynamics include an aeration control term toward `DO_setpoint`.

## Mapping to available data
- Online (1 min): temperature, DO, off-gas CER/OUR, pH, feed rates, NH4OH/H3PO4 additions, bioreactor weight, setpoints.
  - Feed rate maps to `F_feed(t)` (model input). Use measured feed-rate series directly when simulating or estimating.
  - Off-gas CO2 (CER) and O2 uptake (OUR) map to `CER` and `OUR`; use these for estimation of `qO2` or model correction terms.
  - DO is both a model state and an observed variable; use to tune `k_aer` and aeration control proxy.
  - Bioreactor weight can be converted to volume (assumed density ~1 kg/L) and compared to `V(t)`.

- Offline (~24 h): enzyme activity, protein concentration, nutrient/metabolite measurements, patchy biomass.
  - Use offline enzyme/protein measurements to fit `k_enzyme` and `k_protein`.
  - Patchy biomass measurements (20% of runs) can be used to anchor and validate biomass state scaling.

## Practical notes and extensions
- The current pH and temperature models are proxies; for precise control-modeling, integrate buffer chemistry and energy balances.
- Aeration/oxygen transfer may be improved by adding kLa and gas-liquid mass transfer if you have sparging/OTR data.
- The model exposes time-varying inputs (e.g., `F_feed(t)`, `NH4OH_add(t)`, `H3PO4_add(t)`) so real process data can be injected directly into simulations.
- For parameter estimation with real runs, downsample/align data to consistent time units (hours) and use the measured feed and offline measurements as targets.

## Recommended next steps
1. Plug real `F_feed(t)`, `NH4OH_add(t)`, and `H3PO4_add(t)` time-series from your online dataset into the model inputs and run parameter identification for `k_enzyme`, `qO2`, `k_aer`, and Monod constants.
2. If available, add a `kLa` and explicit oxygen mass transfer model to replace the DO proxy.
3. Use an Extended Kalman Filter or smoothing framework to fuse 1-min online data and 24-h offline enzyme measurements.

---
File: `scripts/fermentation_fed_batch.jl`
