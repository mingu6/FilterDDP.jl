function solve!(solver::Solver{T}, x1::Vector{T}, u::Vector{Vector{T}}; kwargs...) where T
    initialize_trajectory!(solver, u, x1)
    status = solve!(solver; kwargs...)
    return status
end

function solve!(solver::Solver{T}) where T
    (solver.options.verbose && solver.data.k==0) && solver_info()

	ocp = solver.ocp
    ws = solver.ws
    options = solver.options
	data = solver.data
    
    reset_mem!(solver.ocp.dynamics)
    reset_mem!(solver.ocp.objective)
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

function primal_error(ws::FilterDDPWorkspace{T}) where T
    primal_inf::T = 0   # constraint violation (primal infeasibility)
    for wse in ws
        primal_inf = max(primal_inf, norm(wse.nominal.c, Inf))
    end
    return primal_inf
end

function dual_error(ocp::OCP{T}, ws::FilterDDPWorkspace{T}, options::Options{T}) where T
    ni = 0              # number of inequality bounds (upper + lower)
    z_norm::T = 0.0
    ϕ_norm::T = 0.0
    nc = sum(ocp.nc)
    dual_inf::T = 0     # dual infeasibility (stationarity of Lagrangian of barrier subproblem)

    for t = ocp.N:-1:1
        ws[t].u_tmp1 .= ocp.objective[t].lu_mem
        mul!(ws[t].u_tmp1, transpose(ocp.constraints[t].cu_mem), ws[t].nominal.ϕ, 1.0, 1.0)
        ws[t].u_tmp1 .-= ws[t].nominal.zl
        ws[t].u_tmp1 .+= ws[t].nominal.zu
        t < ocp.N && mul!(ws[t].u_tmp1, transpose(ocp.dynamics[t].fu_mem), ws[t+1].nominal.λ, 1.0, 1.0)
        dual_inf = max(dual_inf, norm(ws[t].u_tmp1, Inf))
        z_norm += sum(ws[t].nominal.zl)
        z_norm += sum(ws[t].nominal.zu)
        ϕ_norm += norm(ws[t].nominal.ϕ, 1)
        ni += ocp.control_limits[t].nl + ocp.control_limits[t].nu
    end

    scaling = max(options.s_max, (ϕ_norm + z_norm) / max(ni + nc, 1.0))  / options.s_max
    return dual_inf / scaling
end

function cs_error(ocp::OCP{T}, ws::FilterDDPWorkspace{T}, options::Options{T}, μ::T) where T
    ni::T = 0
    z_norm::T = 0
    cs_inf::T = 0     # dual infeasibility (stationarity of Lagrangian of barrier subproblem)

    for t = ocp.N:-1:1
        ni += ocp.control_limits[t].nl + ocp.control_limits[t].nu
        (ocp.control_limits[t].nu == 0 && ocp.control_limits[t].nl == 0) && continue
        ws[t].u_tmp1 .= ws[t].nominal.ul 
        ws[t].u_tmp1 .*= ws[t].nominal.zl
        ws[t].u_tmp1 .-= μ
        replace!(ws[t].u_tmp1, NaN=>0.0)
        cs_inf = max(cs_inf, norm(ws[t].u_tmp1, Inf))
        ws[t].u_tmp2 .= ws[t].nominal.uu
        ws[t].u_tmp2 .*= ws[t].nominal.zu
        ws[t].u_tmp2 .-= μ
        replace!(ws[t].u_tmp2, NaN=>0.0)
        cs_inf = max(cs_inf, norm(ws[t].u_tmp2, Inf))
        z_norm += sum(ws[t].nominal.zl)
        z_norm += sum(ws[t].nominal.zu)
    end
    
    scaling = max(options.s_max, z_norm / max(ni, 1.0))  / options.s_max
    return cs_inf / scaling
end

function reset_duals!(solver::Solver{T}) where T
    cl = solver.ocp.control_limits
    for t = 1:solver.ocp.N
        fill!(solver.ws[t].current.ϕ, 0.0)
        fill!(solver.ws[t].current.zl, 0.0)
        fill!(solver.ws[t].current.zu, 0.0)
        fill!(solver.ws[t].current.λ, 0.0)
        @views solver.ws[t].current.zl[cl[t].indl] .= 1.0
        @views solver.ws[t].current.zu[cl[t].indu] .= 1.0
        fill!(solver.ws[t].nominal.ϕ, 0.0)
        fill!(solver.ws[t].nominal.zl, 0.0)
        fill!(solver.ws[t].nominal.zu, 0.0)
        fill!(solver.ws[t].nominal.λ, 0.0)
        @views solver.ws[t].nominal.zl[cl[t].indl] .= 1.0
        @views solver.ws[t].nominal.zu[cl[t].indu] .= 1.0
    end
end

function barrier_lagrangian!(ocp::OCP{T}, ws::FilterDDPWorkspace{T}, data::SolverData{T}; mode=:nominal) where T   
    fn_eval_time_ = time()

    barrier_lagrangian = 0.
    for t = 1:ocp.N
        if mode == :nominal
            ws[t].u_tmp1 .= log.(ws[t].nominal.ul)
        else
            ws[t].u_tmp1 .= log.(ws[t].current.ul)
        end
        for i in ocp.control_limits[t].indl
            barrier_lagrangian -= ws[t].u_tmp1[i]
        end
        if mode == :nominal
            ws[t].u_tmp1 .= log.(ws[t].nominal.uu)
        else
            ws[t].u_tmp1 .= log.(ws[t].current.uu)
        end

        for i in ocp.control_limits[t].indu
            barrier_lagrangian -= ws[t].u_tmp1[i]
        end
    end
    
    barrier_lagrangian *= data.μ
    data.objective = objective(ocp.objective, ws, mode=mode)
    barrier_lagrangian += data.objective

    data.fn_eval_time += time() - fn_eval_time_

    for wse in ws
        if mode == :nominal
            barrier_lagrangian += dot(wse.nominal.c, wse.nominal.ϕ)
        else
            barrier_lagrangian += dot(wse.current.c, wse.current.ϕ)
        end
    end

    return barrier_lagrangian
end

function constraint_violation_1norm(ws::FilterDDPWorkspace{T}; mode=:nominal) where T
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
