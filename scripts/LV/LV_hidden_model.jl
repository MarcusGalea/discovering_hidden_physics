include("LV_data.jl")

known_eqs = [
    Dt(x) ~ α * x
    Dt(y) ~ -γ * y
]
unknown_gt_eqs = [
    Dt(x) ~ -β * x * y,
    Dt(y) ~ δ * x * y,
]
deviance = 0.1 # deviance for the unknown_gt equations
params_guess_known = Dict([α => 0.1, # + deviance * randn()
                        γ => 0.3, # + deviance * randn()])
                        ])

params_guess_unknown_gt = Dict([β => 0.02,# + deviance * randn()
                        δ => 0.01 # + deviance * randn()])
                        ])


@named sys_known = ODESystem(known_eqs, t, [x, y], [α, γ], defaults = params_guess_known, observed = measured_quantities)
@named sys_unknown_gt = ODESystem(unknown_gt_eqs, t, [x, y], [β, δ], defaults = params_guess_unknown_gt)
sys_known = complete(sys_known)
sys_unknown_gt = complete(sys_unknown_gt)
