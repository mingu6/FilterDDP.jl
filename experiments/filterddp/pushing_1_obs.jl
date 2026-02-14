using FilterDDP
using LinearAlgebra
using Random
using Plots
using MeshCat
using Printf
using LaTeXStrings
using Revise

visualise = false
benchmark = false
verbose = true
n_benchmark = 10

T = Float64
Δ = 0.04
N = 101
n_ocp = 100

options = Options{T}(verbose=verbose, optimality_tolerance=1e-7)

function circleShape(xc, yc, r)
    θ = LinRange(0, 2*π, 500)
    xc .+ r * sin.(θ), yc .+ r * cos.(θ)
end

function plotCircle!(xc, yc, r, color=:blue)
    plot!(circleShape(xc, yc, r), seriestype = [:shape], lw = 2.0,
            c = color, linecolor = :black,
            legend = false, fillalpha = 1.0, aspect_ratio = 1)
end

results = Vector{Vector{Any}}()
params = Vector{Vector{T}}()

for seed = 1:n_ocp
	Random.seed!(seed)
    
    x1 = T[0.0, 0.0, 0.0, 0.0]

    block_params = [
        T[0.07; 0.12; 0.03711],
        T[0.06; 0.12; 0.0355938],
        T[0.08; 0.12; 0.0387237],
        T[0.07; 0.13; 0.0393039],
        T[0.06; 0.13; 0.0378424],
        T[0.08; 0.13; 0.0366212],
        T[0.07; 0.11; 0.0349493],
        T[0.06; 0.11; 0.0333738],
        T[0.08; 0.11; 0.0408633]
    ]
    xyc = block_params[rand(1:length(block_params))]

    obstacle = [T(0.2) + T(0.3) * (rand(T) - T(0.5)); T(0.2) + T(0.1) * (rand(T) - T(0.5)); T(0.05) + T(0.02) * (rand(T) - T(0.5))]

    zx = xyc[1]
    zy = xyc[2]
    c = xyc[3]  # ellipsoidal approximation ratio

    xN = T[0.3, 0.4, 1.5 * pi, 0.0]
    μ_fric = T(0.2) + T(0.1) * (rand(T) - T(0.5)) # friction coefficient b/w pusher and slider
    force_lim = T(0.3)
    vel_lim = T(3.0)
    r_push = T(0.01)
    r_total = max(zx, zy) + r_push

    nu = 11
    nx = 4

    # ## Dynamics

    function R(θ)
        return [[cos(θ); sin(θ); 0] [-sin(θ); cos(θ); 0] [0.0; 0.0; 1.0]]
    end

    L = [T(1.0); T(1.0); c^(-2)]

    function Jc(ϕ)
        return [[1.0; 0.0] [0.0; 1.0] [zx / 2 * tan(ϕ); -zx / 2]]
    end

    function fc(x, u)
        θ = x[3]
        ϕ = x[4]
        return [R(θ) * (L .* (transpose(Jc(ϕ)) * u[1:2])); u[3] - u[4]]
    end

    function f(x, u)
        return x + Δ .* fc(x, u)
    end

    function xy_pusher(x)
        xl = xyc[1]
        θ = x[3]
        ϕ = x[4]
        return R(θ)[1:2, 1:2] * [-xl / 2 - r_push; -(xl / 2) * tan(ϕ)] + x[1:2]
    end

    dyn = Dynamics(T, f, nx, nu)

    # ## Objective

    l = (x, u) -> 1e-2 * dot(u[1:2], u[1:2]) + 2. * sum(u[7:8]) + 2. * sum(u[11])
    lN = (x, u) -> 20.0 * dot(x - xN, x - xN) + 2. * sum(u[7:8]) + 2. * sum(u[11])
    stage_obj = Objective(T, l, nx, nu)
	term_obj = Objective(T, lN, nx, nu)

    # ## Constraints

    function obs_constr(x, u)
        xdiff = x[1:2] - obstacle[1:2]
        return (obstacle[3] + r_total)^2 - xdiff' * xdiff + u[10] - u[11]
    end

    function c(x, u)
        [
        μ_fric * u[1] - u[2] - u[5];
        μ_fric * u[1] + u[2] - u[6];
        u[5] * u[3] - u[7];
        u[6] * u[4] - u[8];
        x[4] - u[9];  # bound constraint on ϕ_t
        obs_constr(x, u)
        ]
    end

    constraints = Constraints(T, c, nx, nu)

    # ## Control Limits

    cl = ControlLimits(T[0.0, -force_lim, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9, 0.0, 0.0],
                T[force_lim, force_lim, vel_lim, vel_lim, Inf, Inf, Inf, Inf, 0.9, Inf, Inf])

    ocp = OCP(N, stage_obj, term_obj, dyn; constraints=constraints, control_limits=cl)
    solver = Solver(ocp; options=options)
    solver.options.verbose = verbose
        
    # ## Initialise solver and solve
    
    ū = [0.01 .* ones(T, nu) for k = 1:N]
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

    push!(params, [xyc; μ_fric; obstacle])

    # ## Plot solution
	if seed == 4 && visualise
		x_sol, u_sol = get_trajectory(solver)
        phidot = map(u -> u[3] - u[4] , u_sol[1:end-1])
        fric_lim = map(u -> μ_fric * u[1], u_sol[1:end-1])
        negfric_lim = map(u -> -μ_fric * u[1], u_sol[1:end-1])
        ft = map(u -> u[2], u_sol[1:end-1])
        plot(range(0, Δ * (N-1), N-1), phidot, xtickfontsize=14, ytickfontsize=14, xlabel=L"$t$", ylabel="meters per second",
			legendfontsize=14, linewidth=2, xlabelfontsize=14, ylabelfontsize=14, linestyle=:solid, linecolor=1, 
            legendposition=:bottomleft, legendtitleposition=:left, ylims=(-5, 5), fontfamily="Computer Modern",
			background_color_legend = nothing, label=L"$\dot{\phi}_t$")
        plot!(twinx(), range(0, Δ * (N-1), N-1), [ft fric_lim negfric_lim], xtickfontsize=14, ytickfontsize=14, ylabel="Newtons (N)",
			legendfontsize=14, linewidth=2, xlabelfontsize=14, ylabelfontsize=14, linestyle=[:dash :solid :solid], linecolor=[2 3 3], 
            legendposition=:topright, legendtitleposition=:left, ylims=(-0.10, 0.10), alpha=[1. 0.5 0.5], legend_columns=-1, fontfamily="Computer Modern",
			background_color_legend = nothing, label= [L"$f_t^T$" L"$\pm c_f f_t^n$" false])
		savefig("plots/pushing_FilterDDP.pdf")
	end

    # ## Plot state get_trajectory
    if visualise
        indices = [1 35 67 94]
		x_sol, u_sol = get_trajectory(solver)
        xs = map(x -> x[1], x_sol[1:end-1])  # slider position
        ys = map(x -> x[2], x_sol[1:end-1])
        ps = map(x -> x[3], x_sol[1:end-1])
        xyp = map(xy_pusher, x_sol[1:end-1])
        xp = map(x -> x[1], xyp)
        yp = map(x -> x[2], xyp)
        plot(xs, ys, linecolor=:brown4, linestyle=:dash, linewidth=2, fontfamily="Computer Modern",
            ylims=(-0.07, 0.5), xlims=(-0.07, 0.5), xtickfontsize=14, ytickfontsize=14)
        plot!(xp, yp, linecolor=:blue, linestyle=:solid, linewidth=2, linealpha=0.5)
        plotCircle!(obstacle[1], obstacle[2], obstacle[3], :deeppink3)
        for ind in indices
            plotCircle!(xp[ind], yp[ind], r_push, :blue)
            vertices = [
                [-zx / 2,  -zy / 2],
                [zx / 2, -zy / 2],
                [zx / 2,  zy / 2],
                [-zx / 2,  zy / 2]
                ]
            rot = R(ps[ind])
            vertices_rot = [rot[1:2, 1:2] * v + [xs[ind]; ys[ind]] for v in vertices]
            push_vert_x = map(v -> v[1], vertices_rot)
            push_vert_y = map(v -> v[2], vertices_rot)
            plot!(Shape(push_vert_x, push_vert_y), lw = 2.0,
                    c = :yellow, linecolor = :black, 
                    legend = false, fillalpha = 0.2, aspect_ratio = 1)
            scatter!([xN[1]], [xN[2]], markershape=:star, color=:gold, markersize=10)
        end
        savefig("plots/pushing_qual_$seed.pdf")
    end
end

open("results/pushing_1_obs.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   solver(ms)  \n")
    for i = 1:length(results)
        if benchmark
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f        %5.1f  \n", Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0,
                            results[i][4], results[i][5], results[i][6] * 1000, results[i][7] * 1000)
        else
            @printf(io, " %2s     %5s      %5s    %.8e    %.8e \n",  Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0, results[i][4], results[i][5])
        end
    end
end

# save parameters of each experiment for ProxDDP comparison
open("params/pushing_1_obs.txt", "w") do io
    for i = 1:n_ocp
        println(io, join(string.(params[i]), " "))
    end
end
