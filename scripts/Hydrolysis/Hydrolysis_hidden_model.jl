known_rn = @reaction_network begin
    @discrete_events 5.0 => [E ~ 0.0] #Dissociation event
    @species E(t) S(t) ES(t) P(t)
    @parameters w_S=1.0 w_ES=2.0 ka=0.4 kd=0.3  w_P=1.0  #w_E = 0
    @observables v ~  w_S * S + w_ES * ES + w_P * P #w_E * E
    ka, E + S --> ES
    kd, ES --> E + S

end

rn = complete(rn)
unknowns_rn = @reaction_network begin
    @species E(t) S(t) ES(t) P(t)
    @parameters kc=0.05
    kc, ES --> P
end

sys_known = convert(ODESystem, known_rn)
sys_unknown_gt = convert(ODESystem, unknowns_rn)
