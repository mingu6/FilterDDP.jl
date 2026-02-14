using FilterDDP
using LinearAlgebra
using Random
using Plots
using Printf
using Revise

benchmark = true
verbose = true
n_benchmark = 10

T = Float64
Δ = 0.01
N = 101
x1 = T[0.0; 0.0]
    
nx = 2  # position and velocity
nu = 3  # pushing force, 2x slacks for + and - components of abs work
n_ocp = 1

options = Options{T}(verbose=verbose, optimality_tolerance=1e-7)

results = Vector{Vector{Any}}()

for seed = 1:n_ocp
    Random.seed!(seed)

    # ## Dynamics - forward Euler

    f = (x, u) -> x + Δ * [x[2], u[1]]
    dyn = Dynamics(T, f, nx, nu)

    # ## Objective

    xN_y = T(1.0)
    xN_v = T(0.0)
    xN = T[xN_y; xN_v]

    l = (x, u) -> Δ * (u[2] + u[3])
    lN = (x, u) -> 500.0 * dot(x - xN, x - xN)
    stage_obj = Objective(T, l, nx, nu)
	term_obj = Objective(T, lN, nx, nu)

    # ## Constraints

    constraints = Constraints(T, (x, u) -> [
        u[2] - u[3] - u[1] * x[2]
    ], nx,  nu)

    # ## Control Limits

    cl = ControlLimits(T[-10.0, 0.0, 0.0], T[10.0, Inf, Inf])

    ocp = OCP(N, stage_obj, term_obj, dyn; constraints=constraints, control_limits=cl)
    solver = Solver(ocp; options=options)
    solver.options.verbose = verbose
    
    # ## Initialise solver and solve
    
    ū = [[0.01; 0.01; 0.01] for k = 1:N]
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
end

open("results/double_integrator.txt", "w") do io
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
