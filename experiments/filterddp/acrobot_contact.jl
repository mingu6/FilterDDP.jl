using FilterDDP
using LinearAlgebra
using Random
using Plots
using MeshCat
using Printf
using LaTeXStrings

visualise = false
benchmark = false
verbose = true
n_benchmark = 10

T = Float64
Δ = 0.05
N = 101
n_ocp = 100

include("../models/acrobot.jl")

if visualise
	include("../visualise/visualise_acrobot.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

qN = T[π; 0.0]

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
	nc = acrobot_impact.nc
	nτ = acrobot_impact.nu
	nx = 2 * nq
	nu = nτ + nq + 3 * nc

	# ## Dynamics - implicit variational integrator (midpoint)

	f = (x, u) -> [x[nq .+ (1:nq)]; u[nτ .+ (1:nq)]]
	dyn_acrobot = Dynamics(f, nx, nu)
	dynamics = [dyn_acrobot for k = 1:N-1]

	# ## Objective

	function stage_obj(x, u)
		τ = u[1]
		s = u[(nτ + nq + 2 * nc) .+ (1:nc)]
		J = 0.01 * Δ * τ * τ + 2. * sum(s)
		return J
	end

	function term_obj(x, u)
		q⁻ = x[1:acrobot_impact.nq] 
		q = x[acrobot_impact.nq .+ (1:acrobot_impact.nq)] 
		q̇ᵐ⁻ = (q - q⁻) ./ Δ

		J = 200.0 * dot(q̇ᵐ⁻, q̇ᵐ⁻)
		J += 700.0 * dot(q - qN, q - qN)
		return J
	end

	stage = Objective(stage_obj, nx, nu)
	objective = [[stage for k = 1:N-1]..., Objective(term_obj, nx, 0)]

	# ## Constraints

	path_constr = Constraint(
		(x, u) -> implicit_contact_dynamics_slack(acrobot_impact, x, u, Δ),
		nx, nu
		)

	constraints = [[path_constr for k = 1:N-1]..., Constraint(nx, 0)]

	# ## Bounds

	limit = T(8.0)
	bound = Bound(
		[-limit; -T(Inf) * ones(T, nq); zeros(T, nc); zeros(T, nc); zeros(T, nc)],
		[limit; T(Inf) * ones(T, nq); T(Inf) * ones(T, nc); T(Inf) * ones(T, 2 * nc)]
	)
	bounds = [[bound for k in 1:N-1]..., Bound(T, 0)]

	solver = Solver(T, dynamics, objective, constraints, bounds, options=options)
	solver.options.verbose = verbose

	# ## Initialise solver and solve
	
	q1 = zeros(T, 2)
	q1_plus = zeros(T, 2)
	x1 = [q1; q1_plus]

	q_init = [zeros(T, 2) for k = 1:N-1]
	ū = [[[zeros(T, nτ); q_init[k]; T(0.01) * ones(T, nc); T(0.01) * ones(T, 2 * nc)] for k = 1:N-1]..., zeros(T, 0)]
	solve!(solver, x1, ū)

	if benchmark
		solver.options.verbose = false
		solver_time = 0.0
		wall_time = 0.0
		for i in 1:n_benchmark
			solve!(solver, x1, ū)
			solver_time += solver.data.solver_time
			wall_time += solver.data.wall_time
		end
		solver_time /= n_benchmark
		wall_time /= n_benchmark
		push!(results, [seed, solver.data.k, solver.data.status, solver.data.objective, solver.data.primal_inf, wall_time, solver_time])
	else
		push!(results, [seed, solver.data.k, solver.data.status, solver.data.objective, solver.data.primal_inf])
	end

	push!(params, T[acrobot_impact.m1, acrobot_impact.I1, acrobot_impact.l1, acrobot_impact.lc1,
				acrobot_impact.m2, acrobot_impact.I2, acrobot_impact.l2, acrobot_impact.lc2])

	# ## Plot solution
	if seed == 1
		x_sol, u_sol = get_trajectory(solver)
		θe = map(x -> x[4], x_sol[1:end-1])
		s1 = map(θ -> π / 2 - θ, θe)
		s2 = map(θ -> θ + π / 2, θe)
		λ1 = map(u -> u[4], u_sol[1:end-1])
		λ2 = map(u -> u[5], u_sol[1:end-1])
		plot(range(0, Δ * (N-1), N-1), [s1 s2 λ1 λ2], xtickfontsize=16, ytickfontsize=16, xlabel=L"$t$", ylims=(0,5),
			legendfontsize=16, linewidth=2, xlabelfontsize=16, linestyle=[:solid :solid :dash :dash], linecolor=[1 2 1 2], 
			legend_columns=-1, fontfamily="Computer Modern",
			background_color_legend = nothing, label=[L"$s_t^{(1)}$" L"$s_t^{(2)}$" L"$\lambda^{(1)}_t$" L"$\lambda^{(2)}_t$"])
		savefig("plots/acrobot_contact_IPDDP.pdf")
	end

	# ## Visualise trajectory using MeshCat

	if visualise && seed == 1
		q_sol = state_to_configuration(x_sol)
		visualize!(vis, acrobot_impact, q_sol, Δt=Δ);
	end
end

open("results/acrobot_contact.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   solver(ms)  \n")
    for i = 1:n_ocp
        if benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f        %5.1f  \n", Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0,
                            results[i][4], results[i][5], results[i][6] * 1000, results[i][7] * 1000)
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
