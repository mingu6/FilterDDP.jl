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
visualise = false

print_level = output ? 5 : 4

N = 101
Δ = 0.05
r_car = 0.02
xN = [1.0; 1.0; π / 4; 0.0]

visualise && include("../visualise/concar.jl")

nx = 4  # num. state
nu = 2  # num. control
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

    @variable(model, x[1:N, 1:nx]);
    @variable(model, u[1:N-1, 1:nu]);

    # ## control limits

    F_lim = 1.5 + rand()
    τ_lim = 3.0 + 2.0 * rand()

    ul = [-F_lim; -τ_lim]
    uu = [F_lim; τ_lim]

    for t = 1:N-1
        for i = 1:nu
            set_lower_bound(u[t, i], ul[i])
            set_upper_bound(u[t, i], uu[i])
        end
    end

    # ## obstacles

    obs_1 = [0.25, 0.25, 0.05] + [(rand() - 0.5) * 0.2, (rand() - 0.5) * 0.2, rand() * 0.15]
    obs_2 = [0.75, 0.75, 0.05] + [(rand() - 0.5) * 0.2, (rand() - 0.5) * 0.2, rand() * 0.15]
    obs_3 = [0.25, 0.75, 0.05] + [(rand() - 0.5) * 0.2, (rand() - 0.5) * 0.2, rand() * 0.15]
    obs_4 = [0.75, 0.25, 0.05] + [(rand() - 0.5) * 0.2, (rand() - 0.5) * 0.2, rand() * 0.15]

    xyr_obs = [obs_1, obs_2, obs_3, obs_4]
    num_obstacles = length(xyr_obs)

    @variable(model, s_obs[1:N-1, 1:num_obstacles] .>= 0.0);  # slacks for obstacle constraints
    @variable(model, s[1:N-1, 1:num_obstacles] .>= 0.0);  # upper bound slack variables

    # ## Dynamics - RK2

    # continuous time dynamics
    function g(x, u)
        [x[4] * cos(x[3]); x[4] * sin(x[3]); u[2]; u[1]]
    end

    function RK2(x, u, g)
        k1 = g(x, u)
        k2 = g(x + Δ * 0.5 * k1, u)
        return x + Δ * k2
    end

    # ## constraints

    obs_dist(obs_xy) = (x) -> begin
        x2d = x[1:2]
        xy_diff = x2d - obs_xy
        return xy_diff' * xy_diff
    end

    for k = 1:N-1
        @constraint(model, RK2(x[k, :], u[k, :], g) == x[k+1, :])
        for (i, obs) in enumerate(xyr_obs)
            @constraint(model, (obs[3] + r_car)^2 - obs_dist(obs[1:2])(x[k, :]) + s_obs[k, i] <= s[k, i])
        end
    end

    stage_cost = (x, u, s) -> begin
        J = 0.0
        J += Δ * (u[1:2] .* [5.0, 1.0])' * u[1:2]
        J += 1000.0 * s' * s
        return J
    end

    term_cost = x -> 200.0 * (x - xN)' * (x - xN)

    function cost(x, u, s)
        J = 0.0
        for k = 1:N-1
            J += stage_cost(x[k, :], u[k, :], s[k, :])
        end
        J += term_cost(x[N, :])
        return J
    end
        
    @objective(model, Min, cost(x, u, s))

    # ## Plots

    if visualise
        plot(xlims=(-0.1, 1.1), ylims=(-0.1, 1.1), xtickfontsize=14, ytickfontsize=14, fontfamily="Computer Modern")
        for xyr in xyr_obs
            plotCircle!(xyr[1], xyr[2], xyr[3])
        end
    end

    set_attribute(model, "print_level", print_level)
    x1 = [0.0; 0.0; π / 8; 0.0] + rand(4) .* [0.0; 0.0; π / 4; 0.0]
    fix.(x[1, :], x1, force = true)
    
    ū = [zeros(nu) for k = 1:N-1]
    
    x̄ = [x1]
    for k in 2:N
        push!(x̄, RK2(x̄[k-1], ū[k-1], g))
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
    
    for k = 1:N-1
        for j = 1:num_obstacles
            set_start_value(s_obs[k, j], 0.01)
            set_start_value(s[k, j], 0.01)
        end
    end
    
    ipopt_out = @capture_out optimize!(model)
    objective, constr_viol, n_iter, succ, _, _ = parse_results_ipopt(ipopt_out)

    # visualise trajectories
    
    xv = value.(x)
    x_sol = [xv[k, :] for k in 1:N]
    visualise && plotTrajectory!(x_sol)
    
    if benchmark
		set_attribute(model, "print_level", 0)
		b = @benchmark optimize!($model)
		wall_time = median(b.times) / 1e6
		push!(results, [seed, n_iter, succ, objective, constr_viol, wall_time])
	else
		push!(results, [seed, n_iter, succ, objective, constr_viol])
	end
    visualise && savefig("plots/concar_IPOPT_$seed.pdf")
end

fname = bfgs ? "results/bfgs_concar_quad.txt" : "results/concar_quad.txt"
open(fname, "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  \n")
    for i = 1:n_ocp
        if benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f     \n", Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]),
                            results[i][4], results[i][5], results[i][6])
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n",  Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]), results[i][4], results[i][5])
        end
    end
end
