function forward_pass!(solver::Solver{T, nx, nu, nc, nux, ncx},
            ocp::OCP{T, nx, nu, nc}, data::SolverData{T},
            options::Options{T}; verbose=false) where {T, nx, nu, nc, nux, ncx}
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

function rollout!(solver::Solver{T, nx, nu, nc, nux, ncx}, ocp::OCP{T, nx, nu, nc},
            data::SolverData{T}, τ::T, step_size::T) where {T, nx, nu, nc, nux, ncx}
    μ = data.μ
    cl = ocp.control_limits

    data.status = 0
    data.primal_1_next = T(0.0)
    data.barrier_lagrangian_next = T(0.0)

    x = solver.nominal[1].x
    for t = 1:ocp.N

        x̄ = solver.nominal[t].x
        ū = solver.nominal[t].u
        ϕ̄  = solver.nominal[t].ϕ
        α = solver.update[t].α
        ψ = solver.update[t].ψ
        β = solver.update[t].β
        ω = solver.update[t].ω
        zl̄ = solver.nominal[t].zl
        zū = solver.nominal[t].zu
        χl = solver.update[t].χl
        χu = solver.update[t].χu
        ζl = solver.update[t].ζl
        ζu = solver.update[t].ζu

        δx = x - x̄

        u = ū + step_size .* α + β * δx
        ϕ = ϕ̄  + step_size .* ψ + ω * δx
        zl = zl̄ + step_size .* χl + ζl * δx
        zu = zū + step_size .* χu + ζu * δx
        
        # update current trajectory
        solver.current[t] = TrajectoryElement{T, nx, nu, nc}(x, u, ϕ, zl, zu)

        # evaluate constraints and violation
        if nc > 0
            c = ocp.constraints.c(x, u)
            data.primal_1_next += norm(c, 1)
            data.barrier_lagrangian_next += dot(c, ϕ)
        end

        # check frac-to-boundary condition
        ul = u - cl.l
        uu = cl.u - u
        ul̄ = ū - cl.l
        uū = cl.u - ū

        if any((ul̄ .* (1. - τ) .> ul) .* cl.maskl)
            data.status = 2
            return
        end
        if any((uū .* (1. - τ) .> uu) .* cl.masku)
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

        # evaluate barrier lagrangian
        data.barrier_lagrangian_next -= μ * dot(log.(ul), cl.maskl)
        data.barrier_lagrangian_next -= μ * dot(log.(uu), cl.masku)
        if t == ocp.N
            data.barrier_lagrangian_next += ocp.term_objective.l(x, u)[1]
        else
            data.barrier_lagrangian_next += ocp.stage_objective.l(x, u)[1]
        end

        if t < ocp.N
            x = solver.ocp.dynamics.f(x, u)
        end
    end
end
