using JuMP
import Ipopt
using Random
using Plots
using Suppressor
using Printf

output = false
bfgs = false

print_level = output ? 5 : 4

include("ipopt_parse.jl")

Δ = 0.01
N = 101
n_ocp = 1

results = Vector{Vector{Any}}()

for seed = 1:n_ocp
    Random.seed!(seed)

    xN_y = 1.0
    xN_v = 0.0
    xN = [xN_y; xN_v]

    if bfgs
        model = Model(
                optimizer_with_attributes(Ipopt.Optimizer, "max_iter" => 3000,
                    "nlp_scaling_method" => "none", "tol" => 1e-5,
                    "print_level" => print_level, "print_timing_statistics" => "yes",
                    "hessian_approximation" => "limited-memory")
                );
    else
        model = Model(
                optimizer_with_attributes(Ipopt.Optimizer, "max_iter" => 1000,
                    "nlp_scaling_method" => "none", "tol" => 1e-7,
                    "max_soc" => 0, "max_filter_resets" => 0, "max_resto_iter" => 0,
                    "print_level" => print_level, "print_timing_statistics" => "yes")
                );
    end

    velocity(acceleration, stage) = stage == 1 ? 0.0 :
        Δ * sum(acceleration[1:stage-1])
    terminal_position(acceleration) =
        Δ * sum(velocity(acceleration, stage) for stage = 1:N-1)

    function cost(acceleration, work)
        J = 0.0
        for k = 1:N-1
            J += Δ * (2.0 * work[k] - acceleration[k] * velocity(acceleration, k))
        end
        J += 500.0 * ((terminal_position(acceleration) - xN_y)^2 +
            (velocity(acceleration, N) - xN_v)^2)
        return J
    end

    @variable(model, -10.0 <= acceleration[1:N-1] <= 10.0)
    @variable(model, work[1:N-1] >= 0.0)

    @objective(model, Min, cost(acceleration, work))

    for k = 1:N-1
        @constraint(model, work[k] - acceleration[k] * velocity(acceleration, k) >= 0.0)
    end
    
    # ## Initialise variables and solve
    
    function set_initial_guess!()
        for k = 1:N-1
            set_start_value(acceleration[k], 0.01)
            set_start_value(work[k], 0.01)
        end
    end
    
    set_attribute(model, "print_level", print_level)
    set_initial_guess!()
    @capture_out optimize!(model)
    set_initial_guess!()
    ipopt_out = ""
    wall_time = 1e3 * @elapsed begin
        ipopt_out = @capture_out optimize!(model)
    end
    objective, constr_viol, n_iter, succ, _, _ = parse_results_ipopt(ipopt_out)
    push!(results, [seed, n_iter, succ, objective, constr_viol, wall_time])
end

fname = bfgs ? "results/bfgs_double_integrator.txt" : "results/double_integrator.txt"
open(fname, "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  \n")
    for i = 1:n_ocp
		@printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f      \n", Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]),
			results[i][4], results[i][5], results[i][6])
    end
end
