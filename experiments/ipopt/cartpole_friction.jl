using JuMP
import Ipopt
using Random
using Plots
using Suppressor
using Printf

visualise = false
output = false
bfgs = false

print_level = output ? 5 : 4

include("ipopt_parse.jl")
include("../models/cartpole.jl")
visualise && include("../visualise/visualise_cartpole.jl")

Δ = 0.05
N = 101
n_ocp = 100

results = Vector{Vector{Any}}()

for seed = 1:n_ocp
    Random.seed!(seed)

    qN = [0.0; π]

    if bfgs
        model = Model(
                optimizer_with_attributes(Ipopt.Optimizer, "max_iter" => 3000,
                    "nlp_scaling_method" => "none", "tol" => 1e-5,
                    "print_level" => print_level, "print_timing_statistics" => "yes",
                    "hessian_approximation" => "limited-memory")
                );
    else
        model = Model(
                optimizer_with_attributes(Ipopt.Optimizer, "max_iter" => 1000,
                    "nlp_scaling_method" => "none", "tol" => 1e-7,
                    "max_soc" => 0, "max_filter_resets" => 0, "max_resto_iter" => 0,
                    "print_level" => print_level, "print_timing_statistics" => "yes")
                );
    end

    cartpole = Cartpole{Float64}(2, 1, 2,
        0.9 + 0.2 * rand(),
        0.15 + 0.1 * rand(),
        0.45 + 0.1 * rand(),
        9.81,
        0.05 .+ 0.1 * rand(2))

    nq = cartpole.nq
    nc = cartpole.nc

    # ## Objective

    function stage_obj(F, β1, β2, η1, η2, ψ, s)
        return 0.01 * Δ * F * F + ψ[1] * s[1] + ψ[2] * s[2] +
            β1[1] * η1[1] + β1[2] * η1[2] +
            β2[1] * η2[1] + β2[2] * η2[2]
    end

    function term_obj(q⁻, q)
        q̇ᵐ⁻ = (q - q⁻) ./ Δ
        return 200.0 * q̇ᵐ⁻' * q̇ᵐ⁻ + 700.0 * (q - qN)' * (q - qN)
    end

    cost = (q, F, β1, β2, η1, η2, ψ, s) -> begin
        J = 0.0
        for k in 1:N-1
            J += stage_obj(F[k], β1[k, :], β2[k, :], η1[k, :],
                η2[k, :], ψ[k, :], s[k, :])
        end
        J += term_obj(q[N, :], q[N+1, :])
        return J
    end

    # ## Constraints

    @variable(model, q[1:N+1, 1:nq])
    @variable(model, -10.0 <= F[1:N-1] <= 10.0)
    @variable(model, β1[1:N-1, 1:nc] >= 0.0)
    @variable(model, β2[1:N-1, 1:nc] >= 0.0)
    @variable(model, η1[1:N-1, 1:nc] >= 0.0)
    @variable(model, η2[1:N-1, 1:nc] >= 0.0)
    @variable(model, ψ[1:N-1, 1:nc] >= 0.0)
    @variable(model, s[1:N-1, 1:nc] >= 0.0)

    @objective(model, Min, cost(q, F, β1, β2, η1, η2, ψ, s))

    q1 = zeros(nq)
    fix.(q[1, :], q1, force=true)
    fix.(q[2, :], q1, force=true)
    for k = 1:N-1
        λ = [β1[k, 1] - β1[k, 2], β2[k, 1] - β2[k, 2]]
        q̇ᵐ⁺ = (q[k+2, :] - q[k+1, :]) ./ Δ
        γ1 = cartpole.friction[1] * (cartpole.mp + cartpole.mc) * cartpole.g
        γ2 = cartpole.friction[2] * cartpole.mp * cartpole.g * cartpole.l
        @constraint(model, manipulator_fd(cartpole, Δ,
            q[k, :], q[k+1, :], q[k+2, :], F[k], λ) .== 0.0)
        @constraint(model, q̇ᵐ⁺[1] + ψ[k, 1] - η1[k, 1] == 0.0)
        @constraint(model, -q̇ᵐ⁺[1] + ψ[k, 1] - η1[k, 2] == 0.0)
        @constraint(model, q̇ᵐ⁺[2] + ψ[k, 2] - η2[k, 1] == 0.0)
        @constraint(model, -q̇ᵐ⁺[2] + ψ[k, 2] - η2[k, 2] == 0.0)
        @constraint(model, γ1 - β1[k, 1] - β1[k, 2] - s[k, 1] == 0.0)
        @constraint(model, γ2 - β2[k, 1] - β2[k, 2] - s[k, 2] == 0.0)
        @constraint(model, ψ[k, 1] * s[k, 1] >= 0.0)
        @constraint(model, ψ[k, 2] * s[k, 2] >= 0.0)
        @constraint(model, β1[k, 1] * η1[k, 1] >= 0.0)
        @constraint(model, β1[k, 2] * η1[k, 2] >= 0.0)
        @constraint(model, β2[k, 1] * η2[k, 1] >= 0.0)
        @constraint(model, β2[k, 2] * η2[k, 2] >= 0.0)
    end
    
    # ## Initialise variables and solve
    
    function set_initial_guess!()
        for k = 1:N+1
            for j = 1:nq
                set_start_value(q[k, j], 0.0)
            end
        end
        for k = 1:N-1
            set_start_value(F[k], 0.0)
            for j = 1:nc
                set_start_value(β1[k, j], 0.01)
                set_start_value(β2[k, j], 0.01)
                set_start_value(η1[k, j], 0.01)
                set_start_value(η2[k, j], 0.01)
                set_start_value(ψ[k, j], 0.01)
                set_start_value(s[k, j], 0.01)
            end
        end
    end
    
    set_attribute(model, "print_level", print_level)
    set_initial_guess!()
    @capture_out optimize!(model)
    set_initial_guess!()
    ipopt_out = ""
    wall_time = 1e3 * @elapsed begin
        ipopt_out = @capture_out optimize!(model)
    end
    objective, constr_viol, n_iter, succ, _, _ = parse_results_ipopt(ipopt_out)
	push!(results, [seed, n_iter, succ, objective, constr_viol, wall_time])

    if visualise && seed == 1
        qv = value.(q)
        q_sol = [qv[k, :] for k in 1:N]
        animate_cartpole(q_sol, Δ, cartpole);
    end
end

fname = bfgs ? "results/bfgs_cartpole_friction.txt" : "results/cartpole_friction.txt"
open(fname, "w") do io
	@printf(io, " seed  iterations  status     objective           primal        wall (ms)  \n")
    for i = 1:n_ocp
		@printf(io, " %2s     %5s      %5s    %.8e    %.8e     %5.1f      \n", Int64(results[i][1]), Int64(results[i][2]), Bool(results[i][3]),
			results[i][4], results[i][5], results[i][6])
    end
end
