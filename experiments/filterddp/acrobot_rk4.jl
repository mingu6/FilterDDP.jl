using FilterDDP
using LinearAlgebra
using Random
using Plots
using Printf
using LaTeXStrings
using StaticArrays

visualise = false
verbose = false
use_control_limits = false

T = Float64
const Δ = 0.05
N = 101
n_ocp = 100

const nx::Int64 = 4
const nu::Int64 = 1

include("../models/acrobot.jl")
visualise && include("../visualise/visualise_acrobot.jl")

options = Options{T}(verbose=verbose, optimality_tolerance=1e-7)

results = Vector{Vector{Any}}()
params = Vector{Vector{T}}()

const npd::Int64 = 11
const xN = @SVector T[π, 0.0, 0.0, 0.0]
const Q = @SMatrix T[1.0 0.0 0.0 0.0;
					  0.0 1.0 0.0 0.0;
					  0.0 0.0 0.1 0.0;
					  0.0 0.0 0.0 0.1]

function double_pendulum(p)
	return DoublePendulum{eltype(p)}(2, 1, 2,
		p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
		p[9], p[10], p[11])
end

function acrobot_dynamics(x, u, p)
	model = double_pendulum(p)
	k1 = acrobot_explicit(model, x, u)
	k2 = acrobot_explicit(model, x + 0.5 * Δ * k1, u)
	k3 = acrobot_explicit(model, x + 0.5 * Δ * k2, u)
	k4 = acrobot_explicit(model, x + Δ * k3, u)
	return x + (Δ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
end
stage_cost = (x, u) ->
	Δ * (2.0 * dot(u, u) + dot(x - xN, Q * (x - xN)))
terminal_cost = x -> 1000.0 * dot(x - xN, Q * (x - xN))

dynamics = Dynamics(acrobot_dynamics, nx, nu, npd; backend=ForwardDiffBackend())
stage_objective = Objective(stage_cost, nx, nu)
terminal_objective = TerminalObjective(terminal_cost, nx)
equalities = EqualityConstraints(nx, nu)

lim = use_control_limits ? T(8.0) : T(Inf)
control_limits = ControlLimits(
	SVector{nu, T}([-lim]), SVector{nu, T}([lim]))

for seed = 1:n_ocp
	Random.seed!(seed)

	dynamics_parameters = @SVector [
		T(0.9) + 0.2 * rand(T),
		T(0.333), 
		T(0.9) + 0.2 * rand(T),
		T(0.5),
		T(0.9) + 0.2 * rand(T),
		T(0.333), 
		T(0.9) + 0.2 * rand(T),
		T(0.5),
		T(9.81), T(0.0), T(0.0)]
	acrobot = double_pendulum(dynamics_parameters)
	empty_parameters = SVector{0, T}()
	ocp_parameters = OCPParameters(
		stage_objective=empty_parameters,
		term_objective=empty_parameters,
		dynamics=dynamics_parameters,
		equality_constraints=empty_parameters,
		inequality_constraints=empty_parameters)

	ocp = build_ocp(N, stage_objective, terminal_objective, dynamics,
		equalities, control_limits; parameters=ocp_parameters)
	solver = Solver(ocp; options=options)
	solver.options.verbose = verbose

	# ## Initialise solver and solve
	
	x1 = @SVector zeros(T, nx)
	ū = [SVector{nu, T}(zeros(T, nu)) for _ = 1:N-1]
	solve!(solver, x1, ū)
	wall_time = 1e3 * @elapsed solve!(solver, x1, ū)
	push!(results, [seed, solver.data.k, solver.data.status, solver.data.objective, solver.data.primal_inf, wall_time])

	push!(params, T[acrobot.m1, acrobot.I1, acrobot.l1, acrobot.lc1, acrobot.m2, acrobot.I2, acrobot.l2, acrobot.lc2])

	# ## Animate trajectory

	if visualise && seed == 1
		x_sol, u_sol = get_trajectory(solver)
		q_sol = [x[1:2] for x in x_sol]
		animate_acrobot(q_sol, Δ, acrobot);
	end
end

open("results/acrobot_rk4.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   \n")
    for i = 1:n_ocp
		@printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f        \n", Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0,
						results[i][4], results[i][5], results[i][6])
    end
end

open("params/acrobot_rk4.txt", "w") do io
	for parameter_set in params
		println(io, join(string.(parameter_set), " "))
	end
end
