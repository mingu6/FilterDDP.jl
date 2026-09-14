using FilterDDP
using LinearAlgebra
using Random
using Plots
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
const nu::Int64 = 7

include("../models/acrobot.jl")
visualise && include("../visualise/visualise_acrobot.jl")

qN = SA[π; 0.0]

options = Options{T}(verbose=verbose, optimality_tolerance=1e-7)

results = Vector{Vector{Any}}()
params = Vector{Vector{T}}()

const npe::Int64 = 11
const nq::Int64 = 2
const nτ::Int64 = 1

function double_pendulum(p)
	return DoublePendulum{eltype(p)}(2, 1, 2,
		p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
		p[9], p[10], p[11])
end

state_dynamics = (x, u) -> [x[nq .+ (1:nq)]; u[nτ .+ (1:nq)]]

function stage_cost(x, u)
	τ = u[1]
	complementarity = u[(nτ + nq) .+ (1:2)] .* u[(nτ + nq + 2) .+ (1:2)]
	return 0.01 * Δ * τ * τ + 2.0 * sum(complementarity)
end

function terminal_cost(x)
	q⁻ = x[1:nq]
	q = x[nq .+ (1:nq)]
	q̇ᵐ⁻ = (q - q⁻) ./ Δ
	return 200.0 * dot(q̇ᵐ⁻, q̇ᵐ⁻) +
		700.0 * dot(q - qN, q - qN)
end

function contact_equalities(x, u, p)
	model = double_pendulum(p)
	q⁻ = x[1:nq]
	q = x[nq .+ (1:nq)]
	q⁺ = u[nτ .+ (1:nq)]
	τ = u[nτ]
	λ = u[(nτ + nq) .+ (1:2)]
	s = u[(nτ + nq + 2) .+ (1:2)]
	return [
		manipulator_fd(model, Δ, q⁻, q, q⁺, τ, λ);
		s .- ϕ_func(model, q⁺)
	]
end

complementarity_constraints = (x, u) ->
	u[(nτ + nq) .+ (1:2)] .* u[(nτ + nq + 2) .+ (1:2)]

dynamics = Dynamics(state_dynamics, nx, nu)
stage_objective = Objective(stage_cost, nx, nu)
terminal_objective = TerminalObjective(terminal_cost, nx)
equalities = EqualityConstraints(contact_equalities, nx, nu, npe)
inequalities = InequalityConstraints(complementarity_constraints, nx, nu,
	SVector{2, T}(zeros(T, 2)), SVector{2, T}(Inf * ones(T, 2)))
control_limits = ControlLimits(
	SVector{nu, T}([-T(8.0); -T(Inf) * ones(T, nq); zeros(T, 4)]),
	SVector{nu, T}([T(8.0); T(Inf) * ones(T, nq); T(Inf) * ones(T, 4)]))

for seed = 1:n_ocp
	Random.seed!(seed)

	equality_parameters = @SVector [
		T(0.9) + 0.2 * rand(T),
		T(0.333), 
		T(0.9) + 0.2 * rand(T),
		T(0.5),
		T(0.9) + 0.2 * rand(T),
		T(0.333), 
		T(0.9) + 0.2 * rand(T),
		T(0.5),
		T(9.81), T(0.0), T(0.0)]
	acrobot_impact = double_pendulum(equality_parameters)
	empty_parameters = SVector{0, T}()
	ocp_parameters = OCPParameters(
		stage_objective=empty_parameters,
		term_objective=empty_parameters,
		dynamics=empty_parameters,
		equality_constraints=equality_parameters,
		inequality_constraints=empty_parameters)

	ocp = build_ocp(N, stage_objective, terminal_objective, dynamics,
		equalities, inequalities, control_limits; parameters=ocp_parameters)
	solver = Solver(ocp; options=options)
	solver.options.verbose = verbose

	# ## Initialise solver and solve
	
	x1 = SVector{4, T}(zeros(T, 4))
	ū = [SVector{nu, T}([zeros(T, nτ + 2); T(0.01) * ones(T, 4)]) for _ = 1:N-1]
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
		λ1 = map(u -> u[4], u_sol)
		λ2 = map(u -> u[5], u_sol)
		plot!(range(0, Δ * (N-1), N-1), [s1 s2 λ1 λ2], xtickfontsize=16, ytickfontsize=16, xlabel=L"$t$", ylims=(0,5),
			legendfontsize=16, linewidth=2, xlabelfontsize=16, linestyle=[:solid :solid :dash :dash], linecolor=[1 2 1 2], 
			legend_columns=-1, fontfamily="Computer Modern",
			background_color_legend = nothing, label=[L"$s_t^{(1)}$" L"$s_t^{(2)}$" L"$\lambda^{(1)}_t$" L"$\lambda^{(2)}_t$"])
		isdir("plots") || mkpath("plots")
		savefig("plots/acrobot_contact_FilterDDP.pdf")
	end

	# ## Animate trajectory

	if visualise && seed == 1
		x_sol, u_sol = get_trajectory(solver)
		q_sol = [x[1:nq] for x in x_sol]
		animate_acrobot(q_sol, Δ, acrobot_impact; filename="acrobot_contact.gif");
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
