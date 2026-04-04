using JuMP
import Ipopt
using Random
using Plots
using BenchmarkTools
using Suppressor
using Printf
using BenchmarkTools

output = false
benchmark = false
bfgs = false

print_level = output ? 5 : 4

Δ = 0.04
N = 101

n_ocp = 100

results = Vector{Vector{Any}}()

include("ipopt_parse.jl")

for seed = 1:n_ocp
    Random.seed!(seed)

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

    x1 = [0.0, 0.0, 0.0, 0.0]
    
    block_params = [
        [0.07; 0.12; 0.03711],
        [0.06; 0.12; 0.0355938],
        [0.08; 0.12; 0.0387237],
        [0.07; 0.13; 0.0393039],
        [0.06; 0.13; 0.0378424],
        [0.08; 0.13; 0.0366212],
        [0.07; 0.11; 0.0349493],
        [0.06; 0.11; 0.0333738],
        [0.08; 0.11; 0.0408633]
    ]
    xyc = block_params[rand(1:length(block_params))]

    obstacle = [0.2 + 0.3 * (rand() - 0.5); 0.2 + 0.1 * (rand() - 0.5); 0.05 + 0.02 * (rand() - 0.5)]

    zx = xyc[1]
    zy = xyc[2]
    c = xyc[3]  # ellipsoidal approximation ratio

    xN = [0.3, 0.4, 1.5 * pi, 0.0]
    μ_fric = 0.2 + 0.1 * (rand() - 0.5) # friction coefficient b/w pusher and slider
    force_lim = 0.3
    vel_lim = 3.0
    r_push = 0.01
    r_total = max(zx, zy) + r_push

    nu = 11
    nx = 4

    @variable(model, x[1:N, 1:nx]);
    @variable(model, u[1:N-1, 1:nu]);

    # ## control limits

    ul = [0.0, -force_lim, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9, 0.0, 0.0]
    uu = [force_lim, force_lim, vel_lim, vel_lim, Inf, Inf, Inf, Inf, 0.9, Inf, Inf]

    for t = 1:N-1
        for i = 1:nu
            if !isinf(ul[i])
                set_lower_bound(u[t, i], ul[i])
            end
            if !isinf(uu[i])
                set_upper_bound(u[t, i], uu[i])
            end
        end
    end

    # dynamics

    function R(θ)
        return [[cos(θ); sin(θ); 0] [-sin(θ); cos(θ); 0] [0.0; 0.0; 1.0]]
    end

    L = [1.0; 1.0; c^(-2)]

    function Jc(ϕ)
        return [[1.0; 0.0] [0.0; 1.0] [zx / 2 * tan(ϕ); -zx / 2]]
    end

    function fc(x, u)
        θ = x[3]
        ϕ = x[4]
        return [R(θ) * (L .* (transpose(Jc(ϕ)) * u[1:2])); u[3] - u[4]]
    end

    function f(x, u)
        return x + Δ .* fc(x, u)
    end

    function obs_constr(x, u)
        xdiff = x[1:2] - obstacle[1:2]
        return (obstacle[3] + r_total)^2 - xdiff' * xdiff + u[10] - u[11]
    end

    function constr(x, u)
        [
        μ_fric * u[1] - u[2] - u[5];
        μ_fric * u[1] + u[2] - u[6];
        u[5] * u[3] - u[7];
        u[6] * u[4] - u[8];
        x[4] - u[9]  # bound constraint on ϕ_t
        obs_constr(x, u)
        ]
    end

    for k = 1:N-1
        @constraint(model, x[k+1, :] == f(x[k, :], u[k, :]))
        @constraint(model, constr(x[k, :], u[k, :]) == zeros(6))
    end

    # objective

    stage_cost = (x, u) -> begin
        J = 0.0
        J += 1e-2 * u[1:2]' * u[1:2] + 2. * sum(u[7:8]) + 2. * u[11]
        return J
    end

    term_cost = x -> 20. * (x - xN)' * (x - xN)

    function cost(x, u)
        J = 0.0
        for k = 1:N-1
            J += stage_cost(x[k, :], u[k, :])
        end
        J += term_cost(x[N, :])
        return J
    end
        
    @objective(model, Min, cost(x, u))

    set_attribute(model, "print_level", print_level)
    fix.(x[1, :], x1, force = true)
    
    xs_init = LinRange(x1, xN, N)[2:end]
    ū = [1e-2 .* ones(nu) for k = 1:N-1]
    
    x̄ = [x1]
    for k in 2:N
        push!(x̄, f(x̄[k-1], ū[k-1]))
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

fname = bfgs ? "results/bfgs_manip_pushing.txt" : "results/manip_pushing.txt"
open(fname, "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   \n")
    for i = 1:n_ocp
        if benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f    \n", Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]),
                            results[i][4], results[i][5], results[i][6])
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n",  Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]), results[i][4], results[i][5])
        end
    end
end
