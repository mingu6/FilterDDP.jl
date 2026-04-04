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
verbose = false
control_limits = false

T = Float64
Δ = 0.05
N = 101
n_ocp = 100

const nx::Int64 = 4
const nu::Int64 = 1

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

	# ## Dynamics (explicit Euler)
    function f(x, u)
        return x + Δ .* acrobot_explicit(acrobot, x, u)
    end
	dyn = Dynamics(f, nx, nu)

	# ## Objective

    Q = diagm(T[1., 1., 0.1, 0.1])
	l = (x, u) -> Δ * (2. * u' * u + (x - xN)' * Q * (x - xN))
    lN = (x, u) -> 1000.0 * (x - xN)' * Q * (x - xN) + 2. * u' * u

	stage_obj = Objective(l, nx, nu)
	term_obj = Objective(lN, nx, nu)

	constraints = EqualityConstraints(nx, nu)

	# ## Control Limits
	lim = control_limits ? T(8.0) : T(Inf)
    cl = ControlLimits(SVector{nu, T}(T[-lim]), SVector{nu, T}(T[lim]))

	ocp = build_ocp(N, stage_obj, term_obj, dyn, constraints, cl)
	solver = Solver(ocp; options=options)
	solver.options.verbose = verbose

	# ## Initialise solver and solve
	
	x1 = @SVector zeros(T, nx)
	ū = [SVector{nu, T}(zeros(T, nu)) for _ = 1:N]
	solve!(solver, x1, ū)

	if benchmark
		solver.options.verbose = false
		b = @benchmark solve!($solver, $x1, $ū)
		wall_time = median(b.times) / 1e6
		push!(results, [seed, solver.data.k, solver.data.status, solver.data.objective, solver.data.primal_inf, wall_time])
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
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   \n")
    for i = 1:n_ocp
        if benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f        \n", Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0,
                            results[i][4], results[i][5], results[i][6])
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n",  Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0, results[i][4], results[i][5])
        end
    end
end
