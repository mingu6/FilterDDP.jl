using FilterDDP
using LinearAlgebra
using Random
using Plots
using Printf
using LaTeXStrings
using StaticArrays

visualise = false
verbose = true

T = Float64
const Δ = 0.04
N = 101
n_ocp = 100

const nu::Int64 = 8
const nx::Int64 = 4

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

const npd::Int64 = 2
const npe::Int64 = 1
const npi::Int64 = 4
const xN = @SVector T[0.3, 0.4, 1.5 * π, 0.0]
force_lim = T(0.3)
vel_lim = T(3.0)
const r_push = T(0.01)

function rotation_matrix(θ)
    return @SMatrix [cos(θ) -sin(θ) 0.0; sin(θ) cos(θ) 0.0; 0.0 0.0 1.0]
end

function contact_jacobian(ϕ, p)
    return @SMatrix [1.0 0.0 p[1] / 2 * tan(ϕ); 0.0 1.0 -p[1] / 2]
end

function pushing_dynamics(x, u, p)
    θ = x[3]
    ϕ = x[4]
    limit_surface = SVector(one(p[2]), one(p[2]), p[2]^(-2))
    velocity = rotation_matrix(θ) *
        (limit_surface .* (transpose(contact_jacobian(ϕ, p)) * SVector(u[1], u[2])))
    return x + Δ * SVector(velocity[1], velocity[2], velocity[3], u[3] - u[4])
end

complementarity = (x, u) -> SVector(u[5] * u[3], u[6] * u[4])
function stage_cost(x, u)
    force = SVector(u[1], u[2])
    complementarity = SVector(u[5] * u[3], u[6] * u[4])
    return 1e-2 * dot(force, force) +
        2.0 * sum(complementarity) + 2.0 * u[8]
end
terminal_cost = x -> 20.0 * dot(x - xN, x - xN)

function contact_equalities(x, u, p)
    return SVector(
        p[1] * u[1] - u[2] - u[5],
        p[1] * u[1] + u[2] - u[6],
        x[4] - u[7])
end

function pushing_inequalities(x, u, p)
    xdiff = SVector(x[1] - p[1], x[2] - p[2])
    obstacle_constraint = (p[3] + p[4])^2 - dot(xdiff, xdiff) - u[8]
    return SVector(u[5] * u[3], u[6] * u[4], obstacle_constraint)
end

dynamics = Dynamics(pushing_dynamics, nx, nu, npd)
stage_objective = Objective(stage_cost, nx, nu)
terminal_objective = TerminalObjective(terminal_cost, nx)
equalities = EqualityConstraints(contact_equalities, nx, nu, npe)
inequalities = InequalityConstraints(pushing_inequalities, nx, nu,
    SVector{3, T}([0.0, 0.0, -Inf]), SVector{3, T}([Inf, Inf, 0.0]), npi)
control_limits = ControlLimits(
    SVector{nu, T}([0.0, -force_lim, 0.0, 0.0, 0.0, 0.0, -0.9, 0.0]),
    SVector{nu, T}([force_lim, force_lim, vel_lim, vel_lim, Inf, Inf, 0.9, Inf]))

function xy_pusher(x, p)
    θ = x[3]
    ϕ = x[4]
    return rotation_matrix(θ)[1:2, 1:2] *
        [-p[1] / 2 - r_push; -(p[1] / 2) * tan(ϕ)] + x[1:2]
end

for seed = 1:n_ocp
	Random.seed!(seed)
    
    x1 = @SVector zeros(T, nx)

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

    μ_fric = T(0.2) + T(0.1) * (rand(T) - T(0.5)) # friction coefficient b/w pusher and slider
    r_total = max(zx, zy) + r_push
    dynamics_parameters = @SVector [zx, c]
    equality_parameters = @SVector [μ_fric]
    inequality_parameters = @SVector [obstacle[1], obstacle[2], obstacle[3], r_total]
    empty_parameters = SVector{0, T}()
    ocp_parameters = OCPParameters(
        stage_objective=empty_parameters,
        term_objective=empty_parameters,
        dynamics=dynamics_parameters,
        equality_constraints=equality_parameters,
        inequality_constraints=inequality_parameters)

    ocp = build_ocp(N, stage_objective, terminal_objective, dynamics,
        equalities, inequalities, control_limits; parameters=ocp_parameters)
    solver = Solver(ocp; options=options)
    solver.options.verbose = verbose
        
    # ## Initialise solver and solve
    
    ū = [SVector{nu, T}(0.01 .* ones(T, nu)) for k = 1:N-1]
    solve!(solver, x1, ū)
    solver.options.verbose = false
    wall_time = 1e3 * @elapsed solve!(solver, x1, ū)
    push!(results, [seed, solver.data.k, solver.data.status, solver.data.objective, solver.data.primal_inf, wall_time])

    push!(params, [xyc; μ_fric; obstacle])

    # ## Plot solution
	if seed == 4 && visualise
		x_sol, u_sol = get_trajectory(solver)
    phidot = map(u -> u[3] - u[4] , u_sol)
    fric_lim = map(u -> μ_fric * u[1], u_sol)
    negfric_lim = map(u -> -μ_fric * u[1], u_sol)
    ft = map(u -> u[2], u_sol)
        plot(range(0, Δ * (N-1), N-1), phidot, xtickfontsize=14, ytickfontsize=14, xlabel=L"$t$", ylabel="meters per second",
			legendfontsize=14, linewidth=2, xlabelfontsize=14, ylabelfontsize=14, linestyle=:solid, linecolor=1, 
            legendposition=:bottomleft, legendtitleposition=:left, ylims=(-5, 5), fontfamily="Computer Modern",
			background_color_legend = nothing, label=L"$\dot{\phi}_t$")
        plot!(twinx(), range(0, Δ * (N-1), N-1), [ft fric_lim negfric_lim], xtickfontsize=14, ytickfontsize=14, ylabel="Newtons (N)",
			legendfontsize=14, linewidth=2, xlabelfontsize=14, ylabelfontsize=14, linestyle=[:dash :solid :solid], linecolor=[2 3 3], 
            legendposition=:topright, legendtitleposition=:left, ylims=(-0.10, 0.10), alpha=[1. 0.5 0.5], legend_columns=-1, fontfamily="Computer Modern",
			background_color_legend = nothing, label= [L"$f_t^T$" L"$\pm c_f f_t^n$" false])
        isdir("plots") || mkpath("plots")
		savefig("plots/pushing_FilterDDP.pdf")
	end

    # ## Plot state get_trajectory
    if visualise
        indices = [1 35 67 94]
		x_sol, u_sol = get_trajectory(solver)
        xs = map(x -> x[1], x_sol[1:end-1])  # slider position
        ys = map(x -> x[2], x_sol[1:end-1])
        ps = map(x -> x[3], x_sol[1:end-1])
        xyp = map(x -> xy_pusher(x, dynamics_parameters), x_sol[1:end-1])
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
            rot = rotation_matrix(ps[ind])
            vertices_rot = [rot[1:2, 1:2] * v + [xs[ind]; ys[ind]] for v in vertices]
            push_vert_x = map(v -> v[1], vertices_rot)
            push_vert_y = map(v -> v[2], vertices_rot)
            plot!(Shape(push_vert_x, push_vert_y), lw = 2.0,
                    c = :yellow, linecolor = :black, 
                    legend = false, fillalpha = 0.2, aspect_ratio = 1)
            scatter!([xN[1]], [xN[2]], markershape=:star, color=:gold, markersize=10)
        end
        isdir("plots") || mkpath("plots")
        savefig("plots/pushing_qual_$seed.pdf")
    end
end

open("results/manip_pushing.txt", "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)   \n")
    for i = 1:length(results)
		@printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f     \n", Int64(results[i][1]), Int64(results[i][2]), Int64(results[i][3]) == 0,
			results[i][4], results[i][5], results[i][6])
    end
end

# save parameters of each experiment for ProxDDP comparison
open("params/manip_pushing.txt", "w") do io
    for i = 1:n_ocp
        println(io, join(string.(params[i]), " "))
    end
end
