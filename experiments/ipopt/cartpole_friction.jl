using JuMP
import Ipopt
using Random
using Plots
using MeshCat
using Suppressor
using Printf
using BenchmarkTools

visualise = false
output = false
benchmark = true
bfgs = false
n_benchmark = 10

print_level = output ? 5 : 4

include("ipopt_parse.jl")
include("../models/cartpole.jl")

if visualise
    include("../visualise/visualise_cartpole.jl")
    !@isdefined(vis) && (vis = Visualizer())
    render(vis)
end

Δ = 0.05
N = 101
n_ocp = 100

results = Vector{Vector{Any}}()

for seed = 1:n_ocp
    Random.seed!(seed)

    qN = [0.0; π]

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

    cartpole = Cartpole{Float64}(2, 1, 2,
        0.9 + 0.2 * rand(),
        0.15 + 0.1 * rand(),
        0.45 + 0.1 * rand(),
        9.81,
        0.05 .+ 0.1 * rand(2))

    nq = cartpole.nq
    nF = cartpole.nu
    nc = cartpole.nc
    nx = 2 * nq
    nu = nF + nq + 6 * nc + 6

    # ## Objective

    function stage_obj(x, u)
		F = u[1]
        s = u[(nF + nq + 6 * nc) .+ (1:6)]
		J = 0.01 * Δ * F * F + 2. * sum(s)
		return J
	end

    function term_obj(x, u)
		J = 0.0 
		
		q⁻ = x[1:cartpole.nq] 
		q = x[cartpole.nq .+ (1:cartpole.nq)] 
		q̇ᵐ⁻ = (q - q⁻) ./ Δ

		J += 200.0 * q̇ᵐ⁻' * q̇ᵐ⁻
		J += 700.0 * (q - qN)' * (q - qN)
		return J
	end

    cost = (x, u) -> begin
        J = 0.0
        for k in 1:N-1
            J += stage_obj(x[k, :], u[k, :])
        end
        J += term_obj(x[N, :], 0.0)
        return J
    end

    # ## Dynamics - forward Euler

    f = (x, u) -> [x[nq .+ (1:nq)]; u[nF .+ (1:nq)]]  # forward Euler
    dyn_con = (x, u) -> implicit_contact_dynamics_slack(cartpole, x, u, Δ)

    # ## Constraints

    @variable(model, x[1:N, 1:nx]);
    @variable(model, u[1:N-1, 1:nu]);

    limit = 10.0
    for t = 1:N-1
        for i = 1:nF
            set_lower_bound(u[t, i], -limit)
            set_upper_bound(u[t, i], limit)
        end
        for i = nF + nq .+ (1:6*nc+6)
            set_lower_bound(u[t, i], 0.0)
        end
    end

    @objective(model, Min, cost(x, u))

    for k = 1:N-1
        @constraint(model, x[k+1, :] == f(x[k, :], u[k, :]))
        @constraint(model, dyn_con(x[k, :], u[k, :]) .== 0.0)
    end
    
    # ## Initialise variables and solve
    
    q1 = zeros(2)
	q1_plus = zeros(2)
	x1 = [q1; q1_plus]
    fix.(x[1, :], x1, force = true)

    q_init = [zeros(2) for k = 1:N-1]
    ū = [[zeros(nF); q_init[k]; 0.01 * ones(6 * nc); 0.01 * ones(6)] for k = 1:N-1]
    
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

    if visualise && seed == 1
        xv = value.(x)
        x_sol = [xv[k, :] for k in 1:N]
        q_sol = [x[1:nq] for x in x_sol]
        visualize!(vis, cartpole, q_sol, Δt=Δ);
    end
end

fname = bfgs ? "results/bfgs_cartpole_friction.txt" : "results/cartpole_friction.txt"
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
