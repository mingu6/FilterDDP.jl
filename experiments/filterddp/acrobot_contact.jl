using FilterDDP
using LinearAlgebra
using Random
using Plots
using MeshCat
using Printf
using LaTeXStrings
using StaticArrays
using BenchmarkTools

visualise = false
benchmark = false
verbose = true

T = Float64
Δ = 0.05
N = 101
n_ocp = 100

const nx::Int64 = 4
const nu::Int64 = 9

include("../models/acrobot.jl")

if visualise
	include("../visualise/visualise_acrobot.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

qN = SA[π; 0.0]

options = Options{T}(verbose=verbose, optimality_tolerance=1e-7)

results = Vector{Vector{Any}}()
params = Vector{Vector{T}}()

for seed = 1:n_ocp
	Random.seed!(seed)
	
	xN = T[qN; qN]

	acrobot_impact = DoublePendulum{T}(2, 1, 2,
		T(0.9) + 0.2 * rand(T),
		T(0.333), 
		T(0.9) + 0.2 * rand(T),
		T(0.5),
		T(0.9) + 0.2 * rand(T),
		T(0.333), 
		T(0.9) + 0.2 * rand(T),
		T(0.5),
		9.81, 0.0, 0.0)

	nq = acrobot_impact.nq
	nτ = acrobot_impact.nu

	# ## Dynamics - implicit variational integrator (midpoint)

	f = (x, u) -> [x[nq .+ (1:nq)]; u[nτ .+ (1:nq)]]
	dyn = Dynamics(f, nx, nu)

	# ## Objective

	function l(x, u)
		τ = u[1]
		s = u[(nτ + nq + 4) .+ (1:2)]
		J = 0.01 * Δ * τ * τ + 2. * sum(s)
		return J
	end
	function lN(x, u)
		q⁻ = x[1:acrobot_impact.nq] 
		q = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
		q̇ᵐ⁻ = (q - q⁻) ./ Δ

		J = 200.0 * dot(q̇ᵐ⁻, q̇ᵐ⁻)
		J += 700.0 * dot(q - qN, q - qN)

		J += 0.01 * u' * u
		return J
	end
	stage_obj = Objective(l, nx, nu)
	term_obj = Objective(lN, nx, nu)

	# ## Constraints

	c = (x, u) -> implicit_contact_dynamics_slack(acrobot_impact, x, u, Δ)
	constraints = EqualityConstraints(c, nx, nu)

	# ## Control Limits

	control_limits = ControlLimits(
		SVector{nu, T}([-T(8.0); -T(Inf) * ones(T, nq);          zeros(T, 6)]),
		SVector{nu, T}([ T(8.0);  T(Inf) * ones(T, nq); T(Inf) * ones(T, 6)])
	)

	ocp = build_ocp(N, stage_obj, term_obj, dyn, constraints, control_limits)
	solver = Solver(ocp; options=options)
	solver.options.verbose = verbose

	# ## Initialise solver and solve
	
	x1 = SVector{4, T}(zeros(T, 4))
	ū = [SVector{nu, T}([zeros(T, nτ + 2); T(0.01) * ones(T, 6)]) for _ = 1:N]
	solve!(solver, x1, ū)

	if benchmark
		solver.options.verbose = false
		b = @benchmark solve!($solver, $x1, $ū)
		wall_time = median(b.times) / 1e6
		push!(results, [seed, solver.data.k, solver.data.status, solver.data.objective, solver.data.primal_inf, wall_time])
	else
		push!(results, [seed, solver.data.k, solver.data.status, solver.data.objective, solver.data.primal_inf])
	end

	push!(params, T[acrobot_impact.m1, acrobot_impact.I1, acrobot_impact.l1, acrobot_impact.lc1,
				acrobot_impact.m2, acrobot_impact.I2, acrobot_impact.l2, acrobot_impact.lc2])

	# ## Plot solution
	if seed == 1 && visualise
		x_sol, u_sol = get_trajectory(solver)
		θe = map(x -> x[4], x_sol[1:end-1])
		s1 = map(θ -> π / 2 - θ, θe)
		s2 = map(θ -> θ + π / 2, θe)
		λ1 = map(u -> u[4], u_sol[1:end-1])
		λ2 = map(u -> u[5], u_sol[1:end-1])
		plot!(range(0, Δ * (N-1), N-1), [s1 s2 λ1 λ2], xtickfontsize=16, ytickfontsize=16, xlabel=L"$t$", ylims=(0,5),
			legendfontsize=16, linewidth=2, xlabelfontsize=16, linestyle=[:solid :solid :dash :dash], linecolor=[1 2 1 2], 
			legend_columns=-1, fontfamily="Computer Modern",
			background_color_legend = nothing, label=[L"$s_t^{(1)}$" L"$s_t^{(2)}$" L"$\lambda^{(1)}_t$" L"$\lambda^{(2)}_t$"])
		savefig("plots/acrobot_contact_FilterDDP.pdf")
	end

	# ## Visualise trajectory using MeshCat

	if visualise && seed == 1
		q_sol = state_to_configuration(x_sol)
		visualize!(vis, acrobot_impact, q_sol, Δt=Δ);
	end
end

open("results/acrobot_contact.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   \n")
    for i = 1:n_ocp
        if benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f      \n", Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0,
                            results[i][4], results[i][5], results[i][6])
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n",  Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0, results[i][4], results[i][5])
        end
    end
end

# save parameters of each experiment for ProxDDP comparison
open("params/acrobot_contact.txt", "w") do io
    for i = 1:n_ocp
        println(io, join(string.(params[i]), " "))
    end
end
