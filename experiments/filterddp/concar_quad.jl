using FilterDDP
using LinearAlgebra
using Plots
using Random
using Printf
using StaticArrays
using BenchmarkTools

visualise = false
benchmark = false
verbose = true

T = Float64
N = 101
Δ = 0.05
r_car = 0.02
n_ocp = 100

const nx::Int64 = 4
const nu::Int64 = 6

visualise && include("../visualise/concar.jl")

options = Options{T}(verbose=verbose, optimality_tolerance=1e-7)

results = Vector{Vector{Any}}()
params = Vector{Vector{T}}()

xN = T[1.0; 1.0; π / 4; 0.0]
no = 4
const npi::Int64 = 12

function continuous_dynamics(x, u)
    return [x[4] * cos(x[3]); x[4] * sin(x[3]); u[2]; u[1]]
end

function midpoint_dynamics(x, u)
    k1 = continuous_dynamics(x, u)
    k2 = continuous_dynamics(x + Δ * 0.5 * k1, u)
    return x + Δ * k2
end

stage_cost = (x, u) -> begin
    slack = u[2 .+ (1:no)]
    Δ * dot(u[1:2] .* [5.0, 1.0], u[1:2]) + 1000.0 * dot(slack, slack)
end
terminal_cost = (x, u) -> begin
    slack = u[2 .+ (1:no)]
    200.0 * dot(x - xN, x - xN) + 1000.0 * dot(slack, slack)
end

function obstacle_constraints(x, u, p)
    return [
        (p[3 * obstacle] + r_car)^2 -
            (x[1] - p[3 * obstacle - 2])^2 -
            (x[2] - p[3 * obstacle - 1])^2 - u[2 + obstacle]
        for obstacle = 1:no
    ]
end

dynamics = Dynamics(midpoint_dynamics, nx, nu)
stage_objective = Objective(stage_cost, nx, nu)
terminal_objective = Objective(terminal_cost, nx, nu)
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
    ū = [SVector{nu, T}([zeros(T, 2); T(1e-2) * ones(T, no)]) for _ = 1:N]

    solve!(solver, x1, ū)
    
    x_sol, u_sol = get_trajectory(solver)
    visualise && plotTrajectory!(x_sol)

    if benchmark
        solver.options.verbose = false
        b = @benchmark solve!($solver, $x1, $ū)
        wall_time = median(b.times) / 1e6
        push!(results, [seed, solver.data.k, solver.data.status, solver.data.objective, solver.data.primal_inf, wall_time])
    else
        push!(results, [seed, solver.data.k, solver.data.status, solver.data.objective, solver.data.primal_inf])
    end
    if visualise
        isdir("plots") || mkpath("plots")
        savefig("plots/concar_quad_FilterDDP_$seed.pdf")
    end

    push!(params, [F_lim; τ_lim; obs_1; obs_2; obs_3; obs_4; x1])
end

open("results/concar_quad.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  \n")
    for i = 1:n_ocp
        if benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f      \n", Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0,
                            results[i][4], results[i][5], results[i][6])
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n",  Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0, results[i][4], results[i][5])
        end
    end
end
