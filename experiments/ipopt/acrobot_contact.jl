using JuMP
import Ipopt
using Random
using Plots
using MeshCat
using Suppressor
using Printf
using LaTeXStrings
using BenchmarkTools

visualise = false
output = false
benchmark = false
bfgs = false

print_level = output ? 5 : 4

Δ = 0.05
N = 101
n_ocp = 100

include("../models/acrobot.jl")
include("ipopt_parse.jl")

if visualise
	include("../visualise/visualise_acrobot.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

qN = [π; 0.0]
xN = [qN; qN]

results = Vector{Vector{Any}}()

for seed = 1:n_ocp
	Random.seed!(seed)

	acrobot_impact = DoublePendulum{Float64}(2, 1, 2,
		0.9 + 0.2 * rand(),
		0.333, 
		0.9 + 0.2 * rand(),
		0.5,
		0.9 + 0.2 * rand(),
		0.333, 
		0.9 + 0.2 * rand(),
		0.5,
		9.81, 0.0, 0.0)

	nq = acrobot_impact.nq
	nc = acrobot_impact.nc
	nτ = acrobot_impact.nu
	nx = 2 * nq
	nu = nτ + nq + 3 * nc

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
					"max_soc" => 0, "max_filter_resets" => 0, "max_resto_iter" => 0,
					"nlp_scaling_method" => "none", "tol" => 1e-7,
                    "print_level" => print_level, "print_timing_statistics" => "yes")
                );
    end

	f = (x, u) -> [x[nq .+ (1:nq)]; u[nτ .+ (1:nq)]]

	# ## Objective

	function stage_obj(x, u)
		τ = u[1]
		s = u[(nτ + nq + 2 * nc) .+ (1:nc)]
		J = 0.01 * Δ * τ * τ + 2. * (s[1] + s[2])
		return J
	end

	function term_obj(x, u)
		J = 0.0 
		
		q⁻ = x[1:acrobot_impact.nq] 
		q = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
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
	end

	# ## Constraints

	constr = (x, u) -> implicit_contact_dynamics_slack(acrobot_impact, x, u, Δ)

	@variable(model, x[1:N, 1:nx]);
	@variable(model, u[1:N-1, 1:nu]);

	limit = 8.0
	for t = 1:N-1
		set_lower_bound(u[t, 1], -limit)
		set_upper_bound(u[t, 1], limit)
		for i = (nτ + nq) .+ (1:3*nc)
			set_lower_bound(u[t, i], 0.0)
		end
	end

	@objective(model, Min, cost(x, u))


	q1 = zeros(2)
	q1_plus = zeros(2)
	x1 = [q1; q1_plus]
	fix.(x[1, :], x1, force = true)
	for k = 1:N-1
		@constraint(model, x[k+1, :] == f(x[k, :], u[k, :]))
		@constraint(model, constr(x[k, :], u[k, :]) .== 0.0)
	end
	
	# ## Initialise variables and solve
	
	q_init = [zeros(2) for k = 1:N-1]
	ū = [[zeros(nτ); q_init[k]; 0.01 * ones(nc); 0.01 * ones(2 * nc)] for k = 1:N-1]
	
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

	# ## Plot solution
	if seed == n_ocp && visualise
		xv = value.(x)
		x_sol = [xv[k, :] for k in 1:N]
		uv = value.(u)
		u_sol = [uv[k, :] for k in 1:N-1]

		θe = map(x -> x[4], x_sol[1:end-1])
		s1 = map(θ -> π / 2 - θ, θe)
		s2 = map(θ -> θ + π / 2, θe)
		λ1 = map(u -> u[4], u_sol)
		λ2 = map(u -> u[5], u_sol)
		plot(range(0, Δ * (N-1), N-1), [s1 s2 λ1 λ2], xtickfontsize=16, ytickfontsize=16, xlabel=L"$t$", ylims=(0,6), fontfamily="Computer Modern",
			legendfontsize=16, linewidth=2, xlabelfontsize=16, linestyle=[:solid :solid :dot :dot], linecolor=[1 2 1 2], legend_columns=-1,
			background_color_legend = nothing, label=[L"$s_t^{(1)}$" L"$s_t^{(2)}$" L"$\lambda^{(1)}_t$" L"$\lambda^{(2)}_t$"])
		savefig("plots/acrobot_contact_IPOPT.pdf")
	end

	# ## Visualise trajectory using MeshCat

	if visualise && seed == n_ocp
		q_sol = state_to_configuration(x_sol)
		visualize!(vis, acrobot_impact, q_sol, Δt=Δ);
	end
end

fname = bfgs ? "results/bfgs_acrobot_contact.txt" : "results/acrobot_contact.txt"
open(fname, "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   \n")
    for i = 1:n_ocp
        if benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f      \n", Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]),
                            results[i][4], results[i][5], results[i][6])
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n",  Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]), results[i][4], results[i][5])
        end
    end
end

