known_rn = @reaction_network begin
    @discrete_events 5.0 => [E ~ 0.0] #Dissociation event
    @parameters w_E=0.0 w_S=1.0 w_ES=2.0 k1 = 0.4
    @observables v ~ w_E * E + w_S * S + w_ES * ES
    k1, E + S --> ES
end

rn = complete(rn)
unknowns_rn = @reaction_network begin
    @parameters k2 = 0.3
    k2, ES --> E + S
end

sys_known = convert(ODESystem, known_rn)
sys_unknown_gt = convert(ODESystem, unknowns_rn)
