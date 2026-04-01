function solve!(solver::Solver{T, nx, nu, nc}, x1::SVector{nx, T}, u::Vector{SVector{nu, T}}; kwargs...) where {T, nx, nu, nc}
    initialize_trajectory!(solver, u, x1)
    status = solve!(solver; kwargs...)
    return status
end

function solve!(solver::Solver{T, nx, nu, nc}) where {T, nx, nu, nc}
    (solver.options.verbose && solver.data.k==0) && solver_info()

	ocp = solver.ocp
    ws = solver.ws
    options = solver.options
	data = solver.data
    
    reset!(data)
    reset_duals!(solver)
    
    time0 = time()
    
    # automatically select initial perturbation. loosely based on bound of CS condition (duality) for LPs
    fn_eval_time_ = time()
    data.objective = objective(ocp.objective, ws; mode=:nominal)
    data.μ = options.μ_init

    constraints!(ocp.constraints, ws; mode=:nominal)
    data.fn_eval_time += time() - fn_eval_time_
    
    # update performance measures for first iterate (req. for sufficient decrease conditions for step acceptance)
    data.primal_1_curr = constraint_violation_1norm(ws; mode=:nominal)
    data.barrier_lagrangian_curr = barrier_lagrangian!(ocp, ws, data; mode=:nominal)
    
    # filter initialization for constraint violation and threshold for switching rule init. (step acceptance)
    data.max_primal_1 = 1e4 * max(1.0, data.primal_1_curr)
    data.min_primal_1 = 1e-4 * max(1.0, data.primal_1_curr)
    reset_filter!(data)
    
    ni = sum(cl.nl + cl.nu for cl in ocp.control_limits)

    while data.k < options.max_iterations
        fn_eval_time_ = time()
        evaluate_derivatives!(ocp, ws; mode=:nominal)
        data.fn_eval_time += time() - fn_eval_time_
        
        backward_pass!(ocp, ws, data, options, verbose=options.verbose)
        data.status != 0 && break
        # check (outer) overall problem convergence

        data.dual_inf = dual_error(ocp, ws, options)
        data.primal_inf = primal_error(ws)
        data.cs_inf = cs_error(ocp, ws, options, T(0.0))
        
        # check (inner) barrier problem convergence and update barrier parameter if so

        cs_inf_μ = cs_error(ocp, ws, options, data.μ)
        opt_err_μ = max(data.dual_inf, cs_inf_μ, data.primal_inf)
        opt_err_0 = max(data.dual_inf, data.cs_inf, data.primal_inf)

        opt_err_0 < options.optimality_tolerance && break

        if opt_err_μ <= options.κ_ϵ * data.μ && ni > 0 && data.μ > options.optimality_tolerance / 10.0
            data.μ = max(options.optimality_tolerance / 10.0, min(options.κ_μ * data.μ, data.μ ^ options.θ_μ))
            reset_filter!(data)
            # performance of current iterate updated to account for barrier parameter change
            fn_eval_time_ = time()
            constraints!(ocp.constraints, ws; mode=:nominal)
            data.fn_eval_time += time() - fn_eval_time_
            data.barrier_lagrangian_curr = barrier_lagrangian!(ocp, ws, data; mode=:nominal)
            
            data.primal_1_curr = constraint_violation_1norm(ws; mode=:nominal)
            data.j += 1
            continue
        end
        
        options.verbose && iteration_status(data, options)

        forward_pass!(ocp, ws, data, options, verbose=options.verbose)
        data.status != 0 && break
        
        update_nominal_trajectory!(ws)
        (!data.armijo_passed && !data.switching) && update_filter!(data, options)
        data.barrier_lagrangian_curr = data.barrier_lagrangian_next
        data.primal_1_curr = data.primal_1_next
        
        data.k += 1
        data.wall_time = time() - time0
        data.solver_time = data.wall_time - data.fn_eval_time
    end
    
    data.k == options.max_iterations && (data.status = 8)
    options.verbose && iteration_status(data, options)
    options.verbose && on_exit(data)
    return data.status
end

function update_filter!(data::SolverData{T}, options::Options{T}) where T
    new_filter_pt = [(1. - options.γ_θ) * data.primal_1_curr,
                        data.barrier_lagrangian_curr - options.γ_L * data.primal_1_curr]
    push!(data.filter, new_filter_pt)
end

function reset_filter!(data::SolverData{T}) where T
    empty!(data.filter)
    push!(data.filter, [data.max_primal_1, T(-Inf)])
    data.status = 0
end

function primal_error(ws::FilterDDPWorkspace{T, nx, nu, nc}) where {T, nx, nu, nc}
    primal_inf::T = 0   # constraint violation (primal infeasibility)
    for wse in ws
        primal_inf = max(primal_inf, norm(wse.nominal.c, Inf))
    end
    return primal_inf
end

function dual_error(ocp::OCP{T, nx, nu, nc}, ws::FilterDDPWorkspace{T, nx, nu, nc}, options::Options{T}) where {T, nx, nu, nc}
    ni = 0              # number of inequality bounds (upper + lower)
    z_norm::T = 0.0
    ϕ_norm::T = 0.0
    dual_inf::T = 0     # dual infeasibility (stationarity of Lagrangian of barrier subproblem)

    for t = ocp.N:-1:1
        Lu = ocp.objective[t].lu_mem + transpose(ocp.constraints[t].cu_mem) * ws[t].nominal.ϕ
        Lu = Lu - ws[t].nominal.zl + ws[t].nominal.zu
        if t < ocp.N
            Lu = Lu + transpose(ocp.dynamics[t].fu_mem) * ws[t+1].nominal.λ
        end
        dual_inf = max(dual_inf, norm(Lu, Inf))
        z_norm += sum(ws[t].nominal.zl)
        z_norm += sum(ws[t].nominal.zu)
        ϕ_norm += norm(ws[t].nominal.ϕ, 1)
        ni += ocp.control_limits[t].nl + ocp.control_limits[t].nu
    end

    scaling = max(options.s_max, (ϕ_norm + z_norm) / max(ni + nc, 1.0))  / options.s_max
    return dual_inf / scaling
end

function cs_error(ocp::OCP{T, nx, nu, nc}, ws::FilterDDPWorkspace{T, nx, nu, nc}, options::Options{T}, μ::T) where {T, nx, nu, nc}
    cl = ocp.control_limits
    ni::T = 0
    z_norm::T = 0
    cs_inf::T = 0     # dual infeasibility (stationarity of Lagrangian of barrier subproblem)

    for t = ocp.N:-1:1
        ni += ocp.control_limits[t].nl + ocp.control_limits[t].nu
        (ocp.control_limits[t].nu == 0 && ocp.control_limits[t].nl == 0) && continue

        cs_err_l = ws[t].nominal.ul .* ws[t].nominal.zl
        cs_err_l = cs_err_l .- μ
        cs_err_l = cs_err_l .* cl[t].maskl
        cs_inf = max(cs_inf, norm(cs_err_l, Inf))

        cs_err_u = ws[t].nominal.uu .* ws[t].nominal.zu
        cs_err_u = cs_err_u .- μ
        cs_err_u = cs_err_u .* cl[t].masku
        cs_inf = max(cs_inf, norm(cs_err_u, Inf))

        z_norm += sum(ws[t].nominal.zl)
        z_norm += sum(ws[t].nominal.zu)
    end
    
    scaling = max(options.s_max, z_norm / max(ni, 1.0))  / options.s_max
    return cs_inf / scaling
end

function reset_duals!(solver::Solver{T, nx, nu, nc}) where {T, nx, nu, nc}
    cl = solver.ocp.control_limits
    for t = 1:solver.ocp.N
        solver.ws[t].current.ϕ = @SVector zeros(T, nc)
        solver.ws[t].current.zl = SVector{nu, T}(ones(T, nu) .* cl[t].maskl)
        solver.ws[t].current.zu = SVector{nu, T}(ones(T, nu) .* cl[t].masku)
        solver.ws[t].current.λ = @SVector zeros(T, nx)

        solver.ws[t].nominal.ϕ = @SVector zeros(T, nc)
        solver.ws[t].nominal.zl = SVector{nu, T}(ones(T, nu) .* cl[t].maskl)
        solver.ws[t].nominal.zu = SVector{nu, T}(ones(T, nu) .* cl[t].masku)
        solver.ws[t].nominal.λ = @SVector zeros(T, nx)
    end
end

function barrier_lagrangian!(ocp::OCP{T, nx, nu, nc}, ws::FilterDDPWorkspace{T, nx, nu, nc},
        data::SolverData{T}; mode=:nominal) where {T, nx, nu, nc}
    fn_eval_time_ = time()

    barrier_lagrangian = 0.
    if mode == :nominal
        for t = 1:ocp.N
            barrier_lagrangian -= dot(log.(ws[t].nominal.ul), ocp.control_limits[t].maskl)
            barrier_lagrangian -= dot(log.(ws[t].nominal.uu), ocp.control_limits[t].masku)
        end
    else
        for t = 1:ocp.N
            barrier_lagrangian -= dot(log.(ws[t].current.ul), ocp.control_limits[t].maskl)
            barrier_lagrangian -= dot(log.(ws[t].current.uu), ocp.control_limits[t].masku)
        end
    end
    
    barrier_lagrangian *= data.μ
    data.objective = objective(ocp.objective, ws, mode=mode)
    barrier_lagrangian += data.objective

    data.fn_eval_time += time() - fn_eval_time_
    
    if mode == :nominal
        for wse in ws
            barrier_lagrangian += dot(wse.nominal.c, wse.nominal.ϕ)
        end
    else
        for wse in ws
            barrier_lagrangian += dot(wse.current.c, wse.current.ϕ)
        end
    end

    return barrier_lagrangian
end

function constraint_violation_1norm(ws::FilterDDPWorkspace{T, nx, nu, nc}; mode=:nominal) where {T, nx, nu, nc}
    constr_violation = 0.
    if mode == :nominal
        for wse in ws
            constr_violation += norm(wse.nominal.c, 1)
        end
    else
        for wse in ws
            constr_violation += norm(wse.current.c, 1)
        end
    end
    return constr_violation
end
