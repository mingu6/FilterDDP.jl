using FilterDDP
using LinearAlgebra
using Plots
using Random
using Printf
using StaticArrays

visualise = false
verbose = true

T = Float64
N = 101
const Δ = 0.05
const r_car = 0.02
n_ocp = 100

const nx::Int64 = 4
const nu::Int64 = 6

visualise && include("../visualise/concar.jl")

options = Options{T}(verbose=verbose, optimality_tolerance=1e-7)

results = Vector{Vector{Any}}()
params = Vector{Vector{T}}()

const xN = @SVector T[1.0; 1.0; π / 4; 0.0]
const no = 4
const npi::Int64 = 12

function continuous_dynamics(x, u)
    return SVector(x[4] * cos(x[3]), x[4] * sin(x[3]), u[2], u[1])
end

function midpoint_dynamics(x, u)
    k1 = continuous_dynamics(x, u)
    k2 = continuous_dynamics(x + Δ * 0.5 * k1, u)
    return x + Δ * k2
end

stage_cost = (x, u) ->
    Δ * (5.0 * u[1]^2 + u[2]^2) +
    1000.0 * (u[3]^2 + u[4]^2 + u[5]^2 + u[6]^2)
terminal_cost = x -> 200.0 * dot(x - xN, x - xN)

function obstacle_constraints(x, u, p)
    return SVector(
        (p[3] + r_car)^2 - (x[1] - p[1])^2 - (x[2] - p[2])^2 - u[3],
        (p[6] + r_car)^2 - (x[1] - p[4])^2 - (x[2] - p[5])^2 - u[4],
        (p[9] + r_car)^2 - (x[1] - p[7])^2 - (x[2] - p[8])^2 - u[5],
        (p[12] + r_car)^2 - (x[1] - p[10])^2 - (x[2] - p[11])^2 - u[6])
end

dynamics = Dynamics(midpoint_dynamics, nx, nu)
stage_objective = Objective(stage_cost, nx, nu)
terminal_objective = TerminalObjective(terminal_cost, nx)
equalities = EqualityConstraints(nx, nu)
inequalities = InequalityConstraints(obstacle_constraints, nx, nu,
    SVector{no, T}(-Inf * ones(T, no)), SVector{no, T}(zeros(T, no)), npi)

for seed = 1:n_ocp
    Random.seed!(seed)

    # ## control limits
    F_lim = T(1.5) + rand(T)
    τ_lim = T(3.0) + T(2.0) * rand(T)

    ul = T[-F_lim; -τ_lim]
    uu = T[F_lim; τ_lim]

    # ## obstacles

    obs_1 = T[0.25, 0.25, 0.05] + T[(rand(T) - T(0.5)) * 0.2, (rand(T) - T(0.5)) * 0.2, rand(T) * 0.15]
    obs_2 = T[0.75, 0.75, 0.05] + T[(rand(T) - T(0.5)) * 0.2, (rand(T) - T(0.5)) * 0.2, rand(T) * 0.15]
    obs_3 = T[0.25, 0.75, 0.05] + T[(rand(T) - T(0.5)) * 0.2, (rand(T) - T(0.5)) * 0.2, rand(T) * 0.15]
    obs_4 = T[0.75, 0.25, 0.05] + T[(rand(T) - T(0.5)) * 0.2, (rand(T) - T(0.5)) * 0.2, rand(T) * 0.15]

    xyr_obs = [obs_1, obs_2, obs_3, obs_4]
    inequality_parameters = SVector{npi, T}(vcat(xyr_obs...))
    empty_parameters = SVector{0, T}()
    ocp_parameters = OCPParameters(
        stage_objective=empty_parameters,
        term_objective=empty_parameters,
        dynamics=empty_parameters,
        equality_constraints=empty_parameters,
        inequality_constraints=inequality_parameters)

    # ## Control Limits

    # [control limits; penalized obstacle violation]
    control_limits = ControlLimits(
        SVector{nu, T}([ul; zeros(T, no)]),
        SVector{nu, T}([uu; T(Inf) * ones(T, no)])
    )

    # ## Initialise solver and solve
    
    ocp = build_ocp(N, stage_objective, terminal_objective, dynamics,
        equalities, inequalities, control_limits; parameters=ocp_parameters)
    solver = Solver(ocp; options=options)
    solver.options.verbose = verbose

    # ## Plots

    if visualise
        plot(xlims=(-0.1, 1.1), ylims=(-0.1, 1.1), xtickfontsize=14, ytickfontsize=14, fontfamily="Computer Modern")
        for xyr in xyr_obs
            plotCircle!(xyr[1], xyr[2], xyr[3])
        end
    end
    
    x1 = SVector{nx, T}(T[0.0; 0.0; π / 8; 0.0] + rand(T, nx) .* T[0.0; 0.0; π / 4; 0.0])
    ū = [SVector{nu, T}([zeros(T, 2); T(1e-2) * ones(T, no)]) for _ = 1:N-1]

    solve!(solver, x1, ū)
    
    x_sol, u_sol = get_trajectory(solver)
    visualise && plotTrajectory!(x_sol)
    solver.options.verbose = false
    wall_time = 1e3 * @elapsed solve!(solver, x1, ū)
    push!(results, [seed, solver.data.k, solver.data.status, solver.data.objective, solver.data.primal_inf, wall_time])
    if visualise
        isdir("plots") || mkpath("plots")
        savefig("plots/concar_quad_FilterDDP_$seed.pdf")
    end

    push!(params, [F_lim; τ_lim; obs_1; obs_2; obs_3; obs_4; x1])
end

open("results/concar_quad.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  \n")
    for i = 1:n_ocp
		@printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f      \n", Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0,
			results[i][4], results[i][5], results[i][6])
    end
end
