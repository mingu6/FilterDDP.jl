using InteriorPointDDP
using LinearAlgebra
using Plots
using Random
using Printf

benchmark = false
verbose = true
visualise = true
n_benchmark = 10

T = Float64
N = 101
Δ = 0.05
r_car = 0.02

options = Options{T}(verbose=verbose, optimality_tolerance=1e-6, μ_init=0.5)

visualise && include("../visualise/concar.jl")

num_state = 4
num_control = 2
n_ocp = 100

results = Vector{Vector{Any}}()
params = Vector{Vector{T}}()

for seed = 1:n_ocp
    Random.seed!(seed)
    
    xN = T[1.0; 1.0; π / 4; 0.0]

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
    num_obstacles = length(xyr_obs)
    num_primal = num_control + 2 * num_obstacles

    # ## Dynamics - RK2

    # continuous time dynamics
    function g(x, u)
        [x[4] * cos(x[3]); x[4] * sin(x[3]); u[2]; u[1]]
    end

    function RK2(x, u, g)
        k1 = g(x, u)
        k2 = g(x + Δ * 0.5 * k1, u)
        return x + Δ * k2
    end

    f = (x, u) -> RK2(x, u, g)

    car = Dynamics(f, num_state, num_primal)
    dynamics = [car for k = 1:N-1]

    # ## objective

    stage_obj = (x, u) -> begin
        s = u[num_control .+ (1:num_obstacles)]    
        J = 0.0
        J += Δ * dot(u[1:2] .* [5.0, 1.0], u[1:2])
        J += 1000.0 * s' * s
        return J
    end
    term_obj = (x, u) -> 200.0 * dot(x - xN, x - xN)

    objective = [
        [Objective(stage_obj, num_state, num_primal) for k = 1:N-1]...,
        Objective(term_obj, num_state, 0)
    ]

    # ## constraints

    obs_dist(obs_xy) = (x, u) -> begin
        x2d = x[1:2]
        xy_diff = x2d - obs_xy
        return dot(xy_diff, xy_diff)
    end
    path_constr_fn = (x, u) -> begin
    [
        # obstacle avoidance constraints w/slack variable,
        # i.e., d_obs^2 - d_thresh^2 >= 0 and d_obs^2 - d_thresh^2 + s = 0, s >= 0
        [(obs[3] + r_car)^2 - obs_dist(obs[1:2])(x, u) - u[num_control + i] + u[num_control + num_obstacles + i] 
            for (i, obs) in enumerate(xyr_obs)];
    ]
    end

    obs_constr = Constraint(path_constr_fn, num_state, num_primal)
    constraints = [[obs_constr for k = 1:N-1]..., Constraint(num_state, 0)]

    # ## bounds

    # [control limits; obs slack; bound slack]
    bound = Bound(
        [ul; zeros(T, num_obstacles); zeros(T, num_obstacles)],
        [uu; T(Inf) * ones(T, num_obstacles); T(Inf) * ones(T, num_obstacles)]
    )
    bounds = [[bound for k in 1:N-1]..., Bound(T, 0)]

    # ## Initialise solver and solve
    
    solver = Solver(T, dynamics, objective, constraints, bounds, options=options)
    solver.options.verbose = verbose

    # ## Plots

    if visualise
        plot(xlims=(-0.1, 1.1), ylims=(-0.1, 1.1), xtickfontsize=14, ytickfontsize=14)
        for xyr in xyr_obs
            plotCircle!(xyr[1], xyr[2], xyr[3])
        end
    end
    
    x1 = T[0.0; 0.0; π / 8; 0.0] + rand(T, num_state) .* T[0.0; 0.0; π / 4; 0.0]
    ū = [[[zeros(T, 2); T(1e-2) * ones(T, 2 * num_obstacles)] for k = 1:N-1]..., zeros(T, 0)]

    solve!(solver, x1, ū)
    
    x_sol, u_sol = get_trajectory(solver)
    visualise && plotTrajectory!(x_sol)

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
    visualise && savefig("plots/concar_IPDDP_$seed.svg")

    push!(params, [F_lim; τ_lim; obs_1; obs_2; obs_3; obs_4; x1])
end

open("results/concar_quad.txt", "w") do io
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
