using FilterDDP
using LinearAlgebra
using Random
using Plots
using MeshCat
using Printf
using LaTeXStrings
using BenchmarkTools

visualise = false
benchmark = false
verbose = true
n_benchmark = 10
integrator = "Euler"
control_limits = false

T = Float64
Δ = 0.02
N = 201
n_ocp = 100

include("../models/acrobot.jl")

if visualise
	include("../visualise/visualise_acrobot.jl")
	!@isdefined(vis) && (vis = Visualizer())
	render(vis)
end

options = Options{T}(verbose=verbose, optimality_tolerance=1e-7)

results = Vector{Vector{Any}}()
params = Vector{Vector{T}}()

for seed = 1:n_ocp
	Random.seed!(seed)
	
	xN = T[π; 0.0; 0.0; 0.0]

	acrobot = DoublePendulum{T}(2, 1, 2,
		T(0.9) + 0.2 * rand(T),
		T(0.333), 
		T(0.9) + 0.2 * rand(T),
		T(0.5),
		T(0.9) + 0.2 * rand(T),
		T(0.333), 
		T(0.9) + 0.2 * rand(T),
		T(0.5),
		9.81, 0.0, 0.0)

	nx = 2 * acrobot.nq
	nu = acrobot.nu

	# ## Dynamics - implicit variational integrator (midpoint)
    function f_RK4(x, u)
        k1 = acrobot_explicit(acrobot, x, u)
        k2 = acrobot_explicit(acrobot, x + (Δ/2) .* k1, u)
        k3 = acrobot_explicit(acrobot, x + (Δ/2) .* k2, u)
        k4 = acrobot_explicit(acrobot, x + Δ .* k3, u)
        return x + Δ / 6 .* (k1 + 2. .* k2 + 2. .* k3 + k4)
    end
    function f_euler(x, u)
        return x + Δ .* acrobot_explicit(acrobot, x, u)
    end
    f = integrator == "RK4" ? f_RK4 : f_euler
	dyn = Dynamics(T, f, nx, nu)

	# ## Objective

    Q = diagm(T[1., 1., 0.1, 0.1])
	l = (x, u) -> Δ * (2. * u' * u + (x - xN)' * Q * (x - xN))
    lN = (x, u) -> 1000.0 * (x - xN)' * Q * (x - xN)

	stage_obj = Objective(T, l, nx, nu)
	term_obj = Objective(T, lN, nx, nu)

	# ## Control Limits
    cl = control_limits ? ControlLimits(T[-8.0], T[8.0]) : nothing

	ocp = OCP(N, stage_obj, term_obj, dyn; control_limits=cl)
	solver = Solver(ocp; options=options)
	solver.options.verbose = verbose

	# ## Initialise solver and solve
	
	x1 = zeros(T, nx)
	ū = [zeros(T, nu) for _ = 1:N]
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

	push!(params, T[acrobot.m1, acrobot.I1, acrobot.l1, acrobot.lc1, acrobot.m2, acrobot.I2, acrobot.l2, acrobot.lc2])

	# ## Visualise trajectory using MeshCat

	if visualise && seed == 1
        x_sol, u_sol = get_trajectory(solver)
		q_sol = state_to_configuration(x_sol)
		visualize!(vis, acrobot, q_sol, Δt=Δ);
	end
end

open("results/acrobot.txt", "w") do io
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
open("params/acrobot.txt", "w") do io
    for i = 1:n_ocp
        println(io, join(string.(params[i]), " "))
    end
end
