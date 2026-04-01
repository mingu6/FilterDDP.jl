function forward_pass!(ocp::OCP{T, nx, nu, nc}, ws::FilterDDPWorkspace{T, nx, nu, nc}, data::SolverData{T},
            options::Options{T}; verbose=false) where {T, nx, nu, nc}
    data.l = 0  # line search iteration counter
    data.status = 0
    data.step_size = T(1.0)
    ΔL = T(0.0)
    μ = data.μ
    τ = max(options.τ_min, T(1.0) - μ)

    θ_prev = data.primal_1_curr
    L_prev = data.barrier_lagrangian_curr
    θ = θ_prev
    
    ΔL = expected_change_lagrangian(ws)  # m in paper

    while data.step_size >= eps(T)
        γ = data.step_size
        try
            rollout!(ocp, ws, data, step_size=γ)
        catch e
            # reduces step size if NaN or Inf encountered
            e isa DomainError && (data.step_size *= 0.5, continue)
            rethrow(e)
        end
        
        data.status = check_fraction_boundary(ocp, ws, τ, data.k)
        data.status != 0 && (data.step_size *= 0.5, continue)

        constraints!(ocp.constraints, ws; mode=:current)
        
        # used for sufficient decrease from current iterate step acceptance criterion
        θ = constraint_violation_1norm(ws; mode=:current)
        L = barrier_lagrangian!(ocp, ws, data; mode=:current)

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
        
        data.barrier_lagrangian_next = L
        data.primal_1_next = θ
        break
    end
    data.step_size < eps(T) && (data.status = 7)
    data.status != 0 && (verbose && (@warn "Line search failed to find a suitable iterate"))
end

function check_fraction_boundary(ocp::OCP{T, nx, nu, nc}, ws::FilterDDPWorkspace{T, nx, nu, nc}, τ::T, k) where {T, nx, nu, nc}
    for (t, wse) in enumerate(ws)
        zl, zl̄ = wse.current.zl, wse.nominal.zl
        zu, zū = wse.current.zu, wse.nominal.zu
        ul, ul̄ = wse.current.ul, wse.nominal.ul
        uu, uū = wse.current.uu, wse.nominal.uu

        if any((ul̄ .* (1. - τ) .> ul) .* ocp.control_limits[t].maskl)
            return 2
        end

        if any((uū .* (1. - τ) .> uu) .* ocp.control_limits[t].masku)
            return 2
        end

        # if any(c * (1. - τ) > d for (c, d) in zip(ul̄, ul))
        #     return 2
        # end

        # if any(c * (1. - τ) > d for (c, d) in zip(uū, uu))
        #     return 2
        # end

        if any(c * (1. - τ) > d for (c, d) in zip(zl̄, zl))
            return 2
        end

        if any(c * (1. - τ) > d for (c, d) in zip(zū, zu))
            return 2
        end
    end
    return 0
end

function expected_change_lagrangian(ws::FilterDDPWorkspace{T, nx, nu, nc}) where {T, nx, nu, nc}
    ΔL = T(0.0)
    for wse in ws
        ΔL += dot(wse.Qû, wse.α)
        ΔL += dot(wse.nominal.c, wse.ψ)
    end
    return ΔL
end

function rollout!(ocp::OCP{T, nx, nu, nc}, ws::FilterDDPWorkspace{T, nx, nu, nc}, data::SolverData{T}; step_size::T=1.0) where {T, nx, nu, nc}
    ws[1].current.x = ws[1].nominal.x

    for t in 1:ocp.N
        δx = ws[t].current.x - ws[t].nominal.x

        # u[t] .= ū[t] + β[t] * (x[t] - x̄[t]) + step_size * α[t]
        ws[t].current.u = ws[t].nominal.u + step_size .* ws[t].α + ws[t].β * δx

        # ϕ[t] .= ϕ̄[t] + ω[t] * (x[t] - x̄[t]) + step_size * ψ[t]
        ws[t].current.ϕ = ws[t].nominal.ϕ + step_size .* ws[t].ψ + ws[t].ω * δx

        ws[t].current.zl = ws[t].nominal.zl + step_size .* ws[t].χl + ws[t].ζl * δx
        ws[t].current.zu = ws[t].nominal.zu + step_size .* ws[t].χu + ws[t].ζu * δx
        
        fn_eval_time_ = time()
        if t < ocp.N
            dynamics!(ocp.dynamics[t], ws[t], ws[t+1], mode=:current)
        end
        
        # evaluate inequality constraints
        ws[t].current.ul = ws[t].current.u - ocp.control_limits[t].l
        ws[t].current.uu = ocp.control_limits[t].u - ws[t].current.u
        data.fn_eval_time += time() - fn_eval_time_
    end
end
