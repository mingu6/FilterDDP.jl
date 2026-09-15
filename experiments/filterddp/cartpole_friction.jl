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
const Δ = 0.05
N = 101
n_ocp = 100

const nx::Int64 = 4
const nu::Int64 = 15

include("../models/cartpole.jl")
visualise && include("../visualise/visualise_cartpole.jl")

options = Options{T}(verbose=verbose, optimality_tolerance=1e-7)

results = Vector{Vector{Any}}()
params = Vector{Vector{T}}()

const npe::Int64 = 6
const nq::Int64 = 2
const nF::Int64 = 1
const qN = @SVector T[0.0; π]

function cartpole_model(p)
    return Cartpole{eltype(p)}(2, 1, 2,
    p[1], p[2], p[3], p[4], SVector(p[5], p[6]))
end

state_dynamics = (x, u) -> SVector(x[3], x[4], u[2], u[3])

function stage_cost(x, u)
    F = u[1]
    return 0.01 * Δ * F * F + u[12] * u[14] + u[13] * u[15] +
        u[4] * u[8] + u[5] * u[9] + u[6] * u[10] + u[7] * u[11]
end

function terminal_cost(x)
    q̇₁ = (x[3] - x[1]) / Δ
    q̇₂ = (x[4] - x[2]) / Δ
    return 200.0 * (q̇₁ * q̇₁ + q̇₂ * q̇₂) +
        700.0 * ((x[3] - qN[1])^2 + (x[4] - qN[2])^2)
end

function contact_equalities(x, u, p)
    model = cartpole_model(p)
    q⁻ = SVector(x[1], x[2])
    q = SVector(x[3], x[4])
    q⁺ = SVector(u[2], u[3])
    q̇ᵐ⁺ = (q⁺ - q) / Δ
    F = u[nF]
    λ = SVector(u[4] - u[5], u[6] - u[7])
    γ1 = p[5] * (p[2] + p[1]) * p[4]
    γ2 = p[6] * p[2] * p[4] * p[3]
    residual = manipulator_fd(model, Δ, q⁻, q, q⁺, F, λ)
    return SVector(
        residual[1], residual[2],
        q̇ᵐ⁺[1] + u[12] - u[8], -q̇ᵐ⁺[1] + u[12] - u[9],
        q̇ᵐ⁺[2] + u[13] - u[10], -q̇ᵐ⁺[2] + u[13] - u[11],
        γ1 - u[4] - u[5] - u[14], γ2 - u[6] - u[7] - u[15])
end

function complementarity_constraints(x, u)
    return SVector(
        u[12] * u[14], u[13] * u[15],
        u[4] * u[8], u[5] * u[9],
        u[6] * u[10], u[7] * u[11])
end

dynamics = Dynamics(state_dynamics, nx, nu)
stage_objective = Objective(stage_cost, nx, nu)
terminal_objective = TerminalObjective(terminal_cost, nx)
equalities = EqualityConstraints(contact_equalities, nx, nu, npe)
inequalities = InequalityConstraints(complementarity_constraints, nx, nu,
    SVector{6, T}(zeros(T, 6)), SVector{6, T}(Inf * ones(T, 6)))
control_limits = ControlLimits(
    SVector{nu, T}([-T(10.0) * ones(T, nF); -T(Inf) * ones(T, nq); zeros(T, 12)]),
    SVector{nu, T}([T(10.0) * ones(T, nF); T(Inf) * ones(T, nq); T(Inf) * ones(T, 12)]))

for seed = 1:n_ocp
    Random.seed!(seed)

    equality_parameters = @SVector [
        T(0.9) + T(0.2) * rand(T),
        T(0.15) + T(0.1) * rand(T),
        T(0.45) + T(0.1) * rand(T),
        T(9.81),
        T(0.05) + T(0.1) * rand(T),
        T(0.05) + T(0.1) * rand(T)]
    cartpole = cartpole_model(equality_parameters)
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
    
	x1 = @SVector zeros(T, 4)
    q_init = [@SVector zeros(T, 2) for _ = 1:N-1]
    ū = [SVector{nu, T}([zeros(T, nF); q_init[k]; T(0.01) * ones(T, 12)]) for k = 1:N-1]
    
    solve!(solver, x1, ū)

    if benchmark
        solver.options.verbose = false
        b = @benchmark solve!($solver, $x1, $ū)
        wall_time = median(b.times) / 1e6
        push!(results, [seed, solver.data.k, solver.data.status, solver.data.objective, solver.data.primal_inf, wall_time])
    else
        push!(results, [seed, solver.data.k, solver.data.status, solver.data.objective, solver.data.primal_inf])
    end

    push!(params, [cartpole.mc, cartpole.mp, cartpole.l, cartpole.friction[1], cartpole.friction[2]])

    # ## Animate trajectory

    if visualise && seed == 1
        x_sol, u_sol = get_trajectory(solver)
        q_sol = [x[1:nq] for x in x_sol]
        animate_cartpole(q_sol, Δ, cartpole);
    end

    # ## Plot solution
	if visualise && seed == 1
		x_sol, u_sol = get_trajectory(solver)
        qdotplus = map(x -> (x[3:4] - x[1:2]) / Δ , x_sol[1:end-1])
        qd1 = map(q -> q[1], qdotplus)
        qd2 = map(q -> q[2], qdotplus)
		λ1 = map(u -> (u[4] - u[5]) * 3.5, u_sol)
		λ2 = map(u -> (u[6] - u[7]) * 15, u_sol)
        F = map(u -> u[1], u_sol)
        isdir("plots") || mkpath("plots")
		plot(range(0, Δ * (N-1), N-1), [qd1 qd2 λ1 λ2 F], xtickfontsize=16, ytickfontsize=16, xlabel=L"$t$", ylims=(-10,14),
			legendfontsize=14, linewidth=2, xlabelfontsize=16, linestyle=[:solid :solid :dash :dash :solid], linecolor=[1 2 1 2 3], 
            legendposition=:top, legendtitleposition=:left, legend_columns=-1, fontfamily="Computer Modern",
			background_color_legend = nothing, label=[L"$p_t^{vm+}$" L"$\theta_t^{vm+}$" L"$\lambda^{(1)}_t$" L"$\lambda^{(2)}_t$" L"F_t"])
		savefig("plots/cartpole_friction_FilterDDP.pdf")
	end
end

open("results/cartpole_friction.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  \n")
    for i = 1:n_ocp
        if benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f     \n", Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0,
                            results[i][4], results[i][5], results[i][6])
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n",  Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0, results[i][4], results[i][5])
        end
    end
end

# save parameters of each experiment for ProxDDP comparison
open("params/cartpole_friction.txt", "w") do io
    for i = 1:n_ocp
        println(io, join(string.(params[i]), " "))
    end
end

