function forward_pass!(solver::Solver{T, nx, nu, nh, ng},
            ocp::OCP{T, nx, nu, nh, ng}, data::SolverData{T},
            options::Options{T}; verbose=false) where {T, nx, nu, nh, ng}
    data.l = 0  # line search iteration counter
    data.status = 0
    data.step_size = T(1.0)
    ΔL = data.expected_change_L
    μ = data.μ
    τ = max(options.τ_min, T(1.0) - μ)

    θ_prev = data.primal_1_curr
    L_prev = data.barrier_lagrangian_curr
    
    while data.step_size >= eps(T)
        γ = data.step_size
        rollout!(solver, ocp, data, τ, γ)
        data.status != 0 && (data.step_size *= 0.5, continue)
        
        # used for sufficient decrease from current iterate step acceptance criterion
        θ = data.primal_1_next
        L = data.barrier_lagrangian_next

        # check acceptability to filter
        data.status = !any(x -> (θ >= x[1] && L >= x[2]), data.filter) ? 0 : 3
        data.status != 0 && (data.step_size *= 0.5, data.l += 1, continue)  # failed, reduce step size
        
        # check for sufficient decrease conditions for the barrier objective/constraint violation
        data.switching = (ΔL < 0.0) && 
            ((-γ * ΔL) ^ options.s_L * γ^(1-options.s_L)  > options.δ * θ_prev ^ options.s_θ)
        data.armijo_passed = L - L_prev - 10. * eps(Float64) * abs(L_prev) <= options.η_L * γ * ΔL
        if (θ <= data.min_primal_1) && data.switching
            data.status = data.armijo_passed ? 0 : 4  #  sufficient decrease of barrier objective
        else
            suff = (θ <= (1. - options.γ_θ) * θ_prev) || (L <= L_prev - options.γ_L * θ_prev)
            data.status = suff ? 0 : 5
        end
        data.status != 0 && (data.step_size *= 0.5, data.l += 1, continue)  # failed, reduce step size
        break
    end
    data.step_size < eps(T) && (data.status = 7)
    data.status != 0 && (verbose && (@warn "Line search failed to find a suitable iterate"))
end

function rollout!(solver::Solver{T, nx, nu, nh, ng},
            ocp::OCP{T, nx, nu, nh, ng}, data::SolverData{T},
            τ::T, step_size::T) where {T, nx, nu, nh, ng}
    μ = data.μ
    control_limits = ocp.control_limits
    inequalities = ocp.inequality_constraints
    slack_bounds = inequalities.bounds
    objective_scale = solver.scaling.objective
    parameters = ocp.parameters

    data.status = 0
    data.primal_1_next = T(0.0)
    data.barrier_lagrangian_next = T(0.0)

    x = solver.nominal[1].x
    for t = 1:ocp.N

        x̄ = solver.nominal[t].x
        ū = solver.nominal[t].u
        s̄ = solver.nominal[t].s
        ϕ̄ = solver.nominal[t].ϕ
        ν̄ = solver.nominal[t].ν
        α = solver.update[t].α
        αs = solver.update[t].αs
        ψh = solver.update[t].ψh
        ψg = solver.update[t].ψg
        β = solver.update[t].β
        βs = solver.update[t].βs
        ωh = solver.update[t].ωh
        ωg = solver.update[t].ωg
        zl̄ = solver.nominal[t].zl
        zū = solver.nominal[t].zu
        zsl̄ = solver.nominal[t].zsl
        zsū = solver.nominal[t].zsu
        χl = solver.update[t].χl
        χu = solver.update[t].χu
        ζl = solver.update[t].ζl
        ζu = solver.update[t].ζu
        χsl = solver.update[t].χsl
        χsu = solver.update[t].χsu
        ζsl = solver.update[t].ζsl
        ζsu = solver.update[t].ζsu

        δx = x - x̄

        u = ū + step_size .* α + β * δx
        ϕ = ϕ̄ + step_size .* ψh + ωh * δx
        zl = zl̄ + step_size .* χl + ζl * δx
        zu = zū + step_size .* χu + ζu * δx
        if ng > 0
            s = s̄ + step_size .* αs + βs * δx
            ν = ν̄ + step_size .* ψg + ωg * δx
            zsl = zsl̄ + step_size .* χsl + ζsl * δx
            zsu = zsū + step_size .* χsu + ζsu * δx
        else
            s, ν, zsl, zsu = s̄, ν̄, zsl̄, zsū
        end
        
        # update current trajectory
        solver.current[t] = TrajectoryElement(
            x, u, s, ϕ, ν, zl, zu, zsl, zsu)

        # evaluate constraints and violation
        if nh > 0
            h = evaluate(ocp.equality_constraints.h, x, u,
                parameters.equality_constraints)
            data.primal_1_next += norm(solver.scaling.equality[t] .* h, 1)
            data.barrier_lagrangian_next += dot(h, ϕ)
        end
        if ng > 0
            q = evaluate(inequalities.g, x, u,
                parameters.inequality_constraints) - s
            data.primal_1_next += norm(solver.scaling.inequality[t] .* q, 1)
            data.barrier_lagrangian_next += dot(q, ν)
        end

        # check frac-to-boundary condition
        ul = u - control_limits.l
        uu = control_limits.u - u
        ul̄ = ū - control_limits.l
        uū = control_limits.u - ū

        if any((ul̄ .* (1. - τ) .> ul) .* control_limits.maskl)
            data.status = 2
            return
        end
        if any((uū .* (1. - τ) .> uu) .* control_limits.masku)
            data.status = 2
            return
        end
        if any(zl̄ .* (1. - τ) .> zl)
            data.status = 2
            return
        end
        if any(zū .* (1. - τ) .> zu)
            data.status = 2
            return
        end

        if ng > 0
            sl = s - slack_bounds.l
            su = slack_bounds.u - s
            sl̄ = s̄ - slack_bounds.l
            sū = slack_bounds.u - s̄

            if any((sl̄ .* (1. - τ) .> sl) .* slack_bounds.maskl)
                data.status = 2
                return
            end
            if any((sū .* (1. - τ) .> su) .* slack_bounds.masku)
                data.status = 2
                return
            end
            if any(zsl̄ .* (1. - τ) .> zsl)
                data.status = 2
                return
            end
            if any(zsū .* (1. - τ) .> zsu)
                data.status = 2
                return
            end
        end

        # evaluate barrier lagrangian
        data.barrier_lagrangian_next -= μ * dot(log.(ul), control_limits.maskl)
        data.barrier_lagrangian_next -= μ * dot(log.(uu), control_limits.masku)
        if ng > 0
            data.barrier_lagrangian_next -= μ * dot(log.(sl), slack_bounds.maskl)
            data.barrier_lagrangian_next -= μ * dot(log.(su), slack_bounds.masku)
        end
        if t == ocp.N
            data.barrier_lagrangian_next += objective_scale *
                evaluate(ocp.term_objective.l, x, u,
                    parameters.term_objective)[1]
        else
            data.barrier_lagrangian_next += objective_scale *
                evaluate(ocp.stage_objective.l, x, u,
                    parameters.stage_objective)[1]
        end

        if t < ocp.N
            x = evaluate(solver.ocp.dynamics.f, x, u, parameters.dynamics)
        end
    end
end
