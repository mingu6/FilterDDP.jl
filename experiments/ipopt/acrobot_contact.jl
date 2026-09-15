using JuMP
import Ipopt
using Random
using Plots
using Suppressor
using Printf
using LaTeXStrings

visualise = false
output = false
bfgs = false

print_level = output ? 5 : 4

Δ = 0.05
N = 101
n_ocp = 100

include("../models/acrobot.jl")
include("ipopt_parse.jl")
visualise && include("../visualise/visualise_acrobot.jl")

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

	# ## Objective

	function stage_obj(τ, λ, s)
		J = 0.01 * Δ * τ * τ + 2.0 * (λ[1] * s[1] + λ[2] * s[2])
		return J
	end

	function term_obj(q⁻, q)
		J = 0.0 
		q̇ᵐ⁻ = (q - q⁻) ./ Δ

		J += 200.0 * q̇ᵐ⁻' * q̇ᵐ⁻
		J += 700.0 * (q - qN)' * (q - qN)
		return J
	end

	cost = (q, τ, λ, s) -> begin
		J = 0.0
		for k in 1:N-1
			J += stage_obj(τ[k], λ[k, :], s[k, :])
		end
		J += term_obj(q[N, :], q[N+1, :])
	end

	# ## Constraints

	@variable(model, q[1:N+1, 1:nq]);
	@variable(model, -8.0 <= τ[1:N-1] <= 8.0);
	@variable(model, λ[1:N-1, 1:nc] >= 0.0);
	@variable(model, s[1:N-1, 1:nc] >= 0.0);

	@objective(model, Min, cost(q, τ, λ, s))


	q1 = zeros(2)
	q1_plus = zeros(2)
	fix.(q[1, :], q1, force = true)
	fix.(q[2, :], q1_plus, force = true)
	for k = 1:N-1
		@constraint(model, manipulator_fd(acrobot_impact, Δ,
			q[k, :], q[k+1, :], q[k+2, :], τ[k], λ[k, :]) .== 0.0)
		@constraint(model, s[k, :] .- ϕ_func(acrobot_impact, q[k+2, :]) .== 0.0)
		@constraint(model, λ[k, :] .* s[k, :] .>= 0.0)
	end
	
	# ## Initialise variables and solve
	
	function set_initial_guess!()
		for k = 1:N+1
			for j = 1:nq
				set_start_value(q[k, j], 0.0)
			end
		end
		for k = 1:N-1
			set_start_value(τ[k], 0.0)
			for j = 1:nc
				set_start_value(λ[k, j], 0.01)
				set_start_value(s[k, j], 0.01)
			end
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

	# ## Plot solution
	if seed == n_ocp && visualise
		qv = value.(q)
		λv = value.(λ)
		θe = qv[2:N, 2]
		s1 = map(θ -> π / 2 - θ, θe)
		s2 = map(θ -> θ + π / 2, θe)
		λ1 = λv[:, 1]
		λ2 = λv[:, 2]
		plot(range(0, Δ * (N-1), N-1), [s1 s2 λ1 λ2], xtickfontsize=16, ytickfontsize=16, xlabel=L"$t$", ylims=(0,6), fontfamily="Computer Modern",
			legendfontsize=16, linewidth=2, xlabelfontsize=16, linestyle=[:solid :solid :dot :dot], linecolor=[1 2 1 2], legend_columns=-1,
			background_color_legend = nothing, label=[L"$s_t^{(1)}$" L"$s_t^{(2)}$" L"$\lambda^{(1)}_t$" L"$\lambda^{(2)}_t$"])
		isdir("plots") || mkpath("plots")
		savefig("plots/acrobot_contact_IPOPT.pdf")
	end

	# ## Visualise trajectory using MeshCat

	if visualise && seed == n_ocp
		q_sol = [qv[k, :] for k = 1:N]
		animate_acrobot(q_sol, Δ, acrobot_impact; filename="acrobot_contact.gif");
	end
end

fname = bfgs ? "results/bfgs_acrobot_contact.txt" : "results/acrobot_contact.txt"
open(fname, "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   \n")
    for i = 1:n_ocp
		@printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f      \n", Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]),
			results[i][4], results[i][5], results[i][6])
    end
end

