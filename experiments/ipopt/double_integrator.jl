using JuMP
import Ipopt
using Random
using Plots
using Suppressor
using Printf
using BenchmarkTools

output = false
benchmark = true
bfgs = false

print_level = output ? 5 : 4

include("ipopt_parse.jl")

Δ = 0.01
N = 101
x1 = [0.0; 0.0]
n_ocp = 1

nx = 2  # num. state
nu = 3  # num. control

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

    f = (x, u) -> x + Δ * [x[2], u[1]]  # forward Euler

    function cost(x, u)
        J = 0.0
        for k = 1:N-1
            J += Δ * (u[k, 2] + u[k, 3])
        end
        J += 500.0 * (x[N, :] - xN)' * (x[N, :] - xN)
        return J
    end

    @variable(model, x[1:N, 1:nx]);
    @variable(model, u[1:N-1, 1:nu]);

    limit = 10.0  # bound is in [5, 10]
    for t = 1:N-1
        set_lower_bound(u[t, 1], -limit)
        set_upper_bound(u[t, 1], limit)
        for i = 2:3
            set_lower_bound(u[t, i], 0.0)
        end
    end

    @objective(model, Min, cost(x, u))

    fix.(x[1, :], x1, force = true)
    for k = 1:N-1
        @constraint(model, x[k+1, :] == f(x[k, :], u[k, :]))
        @constraint(model, u[k, 2] - u[k, 3] == u[k, 1] * x[k, 2])
    end
    
    # ## Initialise variables and solve
    
    ū = [0.01 * ones(3) for k = 1:N-1]
    x̄ = [x1]
    for k in 2:N
        push!(x̄, f(x̄[k-1],  ū[k-1]))
    end
    
    for k = 1:N
        for j = 1:nx
            set_start_value(x[k, j], x̄[k][j])
        end
    end
    
    for k = 1:N-1
        for j = 1:nu
            set_start_value(u[k, j], ū[k][j])
        end
    end
    
    set_attribute(model, "print_level", print_level)
    ipopt_out = @capture_out optimize!(model)
    objective, constr_viol, n_iter, succ, _, _ = parse_results_ipopt(ipopt_out)
    
    if benchmark
		set_attribute(model, "print_level", 0)
		b = @benchmark optimize!($model)
		wall_time = median(b.times) / 1e6
		push!(results, [seed, n_iter, succ, objective, constr_viol, wall_time])
	else
		push!(results, [seed, n_iter, succ, objective, constr_viol])
	end
end

fname = bfgs ? "results/bfgs_double_integrator.txt" : "results/double_integrator.txt"
open(fname, "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  \n")
    for i = 1:n_ocp
        if benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f      \n", Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]),
                            results[i][4], results[i][5], results[i][6])
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n",  Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]), results[i][4], results[i][5])
        end
    end
end
