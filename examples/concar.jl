using InteriorPointDDP
using LinearAlgebra
using Plots
using Random
using BenchmarkTools
using Printf

visualise = false
benchmark = false
verbose = true

T = Float64
N = 101
h = 0.05
r_car = 0.02
xN = T[1.0; 1.0; π / 4]
options = Options{T}(quasi_newton=false, verbose=true)

num_state = 3
num_action = 2

# ## control limits

ul = T[-0.1; -5.0]
uu = T[1.0; 5.0]

# ## obstacles

xyr_obs = [
    T[0.05, 0.25, 0.1],
    T[0.45, 0.1, 0.15],
    T[0.7, 0.7, 0.2],
    T[0.35, 0.4, 0.1]
    ]
num_obstacles = length(xyr_obs)
num_primal = num_action + num_obstacles + 2  # 2 slacks for box constraints

include("../examples/visualise/concar.jl")

# ## Dynamics - explicit midpoint for integrator

function car_continuous(x, u)
    [u[1] * cos(x[3]); u[1] * sin(x[3]); u[2]]
end

function car_discrete(x, u)
    x + h * car_continuous(x + 0.5 * h * car_continuous(x, u), u)
end

car = Dynamics(car_discrete, num_state, num_primal)
dynamics = [car for k = 1:N-1]

# ## objective - waypoint constraints -> high cost

stage_cost = (x, u) -> begin
    J = 0.0
    J += 1e-2 * dot(x - xN, x - xN)
    J += 1e-1 * dot(u[1:2] .* [1.0, 0.1], u[1:2])
    return J
end
objective = [
    [Cost(stage_cost, num_state, num_primal) for k = 1:N-1]...,
    Cost((x, u) -> 1e3 * dot(x - xN, x - xN), num_state, 0)
]

# ## constraints

obs_dist(obs_xy) = (x, u) -> begin
    xp = car_discrete(x, u)[1:2]
    xy_diff = xp[1:2]- obs_xy
    return dot(xy_diff, xy_diff)
end
stage_constr_fn = (x, u) -> begin
[
    # obstacle avoidance constraints w/slack variable,
    # i.e., d_obs^2 - d_thresh^2 >= 0 and d_obs^2 - d_thresh^2 + s = 0, s >= 0
    [(obs[3] + r_car)^2 - obs_dist(obs[1:2])(x, u) + u[num_action + i]
        for (i, obs) in enumerate(xyr_obs)];
    # bound constraints, car must stay within [0, 1] x [0, 1] box
    car_discrete(x, u)[1:2] - u[end-1:end]
]
end

obs_constr = Constraint(stage_constr_fn, num_state, num_primal)
constraints = [obs_constr for k = 1:N-1]

# ## bounds

# [control limits; obs slack; bound slack]
bound = Bound(
    [ul; zeros(T, num_obstacles); zeros(T, 2)],
    [uu; T(Inf) * ones(T, num_obstacles); ones(T, 2)]
)
bounds = [bound for k in 1:N-1]

solver = Solver(T, dynamics, objective, constraints, bounds, options=options)

plot()

# ## Initialise solver and solve
open("examples/results/concar.txt", "w") do io
	@printf(io, " seed  iterations  status    objective      primal      time (s)  \n")
    for seed = 1:50
        solver.options.verbose = verbose
        Random.seed!(seed)
        
        x1 = T[0.0; 0.0; 0.0] + rand(T, 3) .* T[0.05, 0.05, π / 2]
        ū = [[T(1.0e-3) .* (rand(T, 2) .- 0.5); T(0.01) * ones(T, num_obstacles + 2)] for k = 1:N-1]
    
        state_diffs = solve!(solver, x1, ū)
        
        plot!(1:solver.data.k+1, state_diffs, yaxis=:log10, yticks=[1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8], ylims=(1e-9, 3e2), legend=false, linecolor=1, xtickfontsize=14, ytickfontsize=14)
        
        # if benchmark
        #     solver.options.verbose = false
        #     solve_time = @belapsed solve!($solver, $x1, $ū)
        #     @printf(io, " %2s     %5s      %5s    %.8f    %.8f    %.5f  \n", seed, solver.data.k, solver.data.status == 0, solver.data.objective, solver.data.primal_inf, solve_time)
        # else
        #     @printf(io, " %2s     %5s      %5s    %.8f    %.8f \n", seed, solver.data.k, solver.data.status == 0, solver.data.objective, solver.data.primal_inf)
        # end
    end
end

savefig("examples/plots/concar_convergence.png")

# # ## Plot solution

if visualise
    x_sol, u_sol = get_trajectory(solver)
    
    plot()
    plotTrajectory!(x_sol)
    for xyr in xyr_obs
        plotCircle!(xyr[1], xyr[2], xyr[3])
    end
    savefig("examples/plots/concar.png")
end
