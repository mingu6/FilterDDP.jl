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
const nu::Int64 = 21

include("../models/cartpole.jl")
visualise && include("../visualise/visualise_cartpole.jl")

options = Options{T}(verbose=verbose, optimality_tolerance=1e-7)

results = Vector{Vector{Any}}()
params = Vector{Vector{T}}()

for seed = 1:n_ocp
    Random.seed!(seed)

    qN = T[0.0; π]

    cartpole = Cartpole{T}(2, 1, 2,
        T(0.9) + T(0.2) * rand(T),
        T(0.15) + T(0.1) * rand(T),
        T(0.45) + T(0.1) * rand(T),
        T(9.81),
        T(0.05) .+ T(0.1) * rand(T, 2))

    nq = cartpole.nq
    nF = cartpole.nu

    # ## Dynamics - forward Euler

    f = (x, u) -> [x[nq .+ (1:nq)]; u[nF .+ (1:nq)]]
    dyn = Dynamics(f, nx, nu)

    # ## Objective

    function l(x, u)
		F = u[1]
        s = u[(nF + nq + 12) .+ (1:6)]
		J = 0.01 * Δ * F * F + sum(s)
		return J
	end
	function lN(x, u)
        q⁻ = x[1:cartpole.nq]
		q = x[cartpole.nq .+ (1:cartpole.nq)] 
		q̇ᵐ⁻ = (q - q⁻) ./ Δ

		J = 200.0 * dot(q̇ᵐ⁻, q̇ᵐ⁻)
		J += 700.0 * dot(q - qN, q - qN)

        J += 0.01 * u' * u
		return J
	end
	stage_obj = Objective(l, nx, nu)
	term_obj = Objective(lN, nx, nu)

    # ## Constraints

    c = (x, u) -> implicit_contact_dynamics_slack(cartpole, x, u, Δ)
    constraints = EqualityConstraints(c, nx, nu)

    # ## Control Limits

    cl = ControlLimits(
        SVector{nu, T}([-T(10.0) * ones(T, nF); -T(Inf) * ones(T, nq);       zeros(T, 12);         zeros(T, 6)]),
        SVector{nu, T}([ T(10.0) * ones(T, nF);  T(Inf) * ones(T, nq); Inf * ones(T, 12); T(Inf) * ones(T, 6)])
    )

    ocp = build_ocp(N, stage_obj, term_obj, dyn, constraints, cl)
    solver = Solver(ocp; options=options)
    solver.options.verbose = verbose
    
    # ## Initialise solver and solve
    
	x1 = @SVector zeros(T, 4)
    q_init = [@SVector zeros(T, 2) for _ = 1:N]
    ū = [SVector{nu, T}([zeros(T, nF); q_init[k]; T(0.01) * ones(T, 12); T(0.01) * ones(T, 6)]) for k = 1:N]
    
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
		λ1 = map(u -> (u[4] - u[5]) * 3.5, u_sol[1:end-1])
		λ2 = map(u -> (u[6] - u[7]) * 15, u_sol[1:end-1])
        F = map(u -> u[1], u_sol[1:end-1])
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

