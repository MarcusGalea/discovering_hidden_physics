known_rn = @reaction_network begin
    @discrete_events 5.0 => [E ~ 0.0] #Dissociation event
    @species E(t) S(t) ES(t) P(t)
    @parameters w_S=1.0 w_ES=2.0 ka=0.4 kd=0.3  w_P=1.0  #w_E = 0
    @observables y ~  w_S * S + w_ES * ES + w_P * P #w_E * E
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


### OPTIMIZATION
n_initial_conditions = 3
#CHANGE NUMBER OF INITIAL CONDITIONS HERE

# batch_size = 32 # Batch size for the optimization
@unpack E, S, ES, P, y = sys_known
obs = Dict("y" => y)#"E" => E, "S" => S, "ES" => ES, 
u0map = Dict([E => 10.0, S => 1.0, ES => 0.0, P => 0.0])
ic_vals = Dict(["cond$i" => Dict([var => ic[j] for (j, var) in enumerate(unknowns(sys_known))]) for (i, ic) in enumerate(initial_conditions[1:n_initial_conditions])])
included_exp = (df) -> reduce(.|, [(df.simulation_id .== "cond$i") .& (df.obs_id .== obsvar)
                               for i in 1:n_initial_conditions for obsvar in keys(obs)])
train_measurements_exp = train_measurements[included_exp(train_measurements), :]
test_measurements_exp = test_measurements[included_exp(test_measurements), :]