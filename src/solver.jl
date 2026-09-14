struct TrajectoryElement{T, nx, nu, nh, ng}
    x::SVector{nx, T}
    u::SVector{nu, T}
    s::SVector{ng, T}
    ϕ::SVector{nh, T}
    ν::SVector{ng, T}
    zl::SVector{nu, T}
    zu::SVector{nu, T}
    zsl::SVector{ng, T}
    zsu::SVector{ng, T}
end

function TrajectoryElement(T::T_, nx::Int, nu::Int, nh::Int, ng::Int) where T_
    TrajectoryElement{T, nx, nu, nh, ng}(
        SVector{nx, T}(zeros(T, nx)),
        SVector{nu, T}(zeros(T, nu)),
        SVector{ng, T}(zeros(T, ng)),
        SVector{nh, T}(zeros(T, nh)),
        SVector{ng, T}(zeros(T, ng)),
        SVector{nu, T}(zeros(T, nu)),
        SVector{nu, T}(zeros(T, nu)),
        SVector{ng, T}(zeros(T, ng)),
        SVector{ng, T}(zeros(T, ng))
        )
end

struct UpdateRule{T, nx, nu, nh, ng, nux, nhx, ngx}
    α::SVector{nu, T}
    αs::SVector{ng, T}
    ψh::SVector{nh, T}
    ψg::SVector{ng, T}
    β::SMatrix{nu, nx, T, nux}
    βs::SMatrix{ng, nx, T, ngx}
    ωh::SMatrix{nh, nx, T, nhx}
    ωg::SMatrix{ng, nx, T, ngx}
    χl::SVector{nu, T}
    χu::SVector{nu, T}
    ζl::SMatrix{nu, nx, T, nux}
    ζu::SMatrix{nu, nx, T, nux}
    χsl::SVector{ng, T}
    χsu::SVector{ng, T}
    ζsl::SMatrix{ng, nx, T, ngx}
    ζsu::SMatrix{ng, nx, T, ngx}
end

function UpdateRule(T::T_, nx::Int, nu::Int, nh::Int, ng::Int) where T_
    nux = nu * nx
    nhx = nh * nx
    ngx = ng * nx
    UpdateRule{T, nx, nu, nh, ng, nux, nhx, ngx}(
        SVector{nu, T}(zeros(T, nu)),
        SVector{ng, T}(zeros(T, ng)),
        SVector{nh, T}(zeros(T, nh)),
        SVector{ng, T}(zeros(T, ng)),
        SMatrix{nu, nx, T, nux}(zeros(T, nu, nx)),
        SMatrix{ng, nx, T, ngx}(zeros(T, ng, nx)),
        SMatrix{nh, nx, T, nhx}(zeros(T, nh, nx)),
        SMatrix{ng, nx, T, ngx}(zeros(T, ng, nx)),
        SVector{nu, T}(zeros(T, nu)),
        SVector{nu, T}(zeros(T, nu)),
        SMatrix{nu, nx, T, nux}(zeros(T, nu, nx)),
        SMatrix{nu, nx, T, nux}(zeros(T, nu, nx)),
        SVector{ng, T}(zeros(T, ng)),
        SVector{ng, T}(zeros(T, ng)),
        SMatrix{ng, nx, T, ngx}(zeros(T, ng, nx)),
        SMatrix{ng, nx, T, ngx}(zeros(T, ng, nx))
    )
end

mutable struct NLPScaling{T, nh, ng}
    objective::T
    equality::Vector{SVector{nh, T}}
    inequality::Vector{SVector{ng, T}}
end

function NLPScaling(::Type{T}, N::Int, nh::Int, ng::Int) where T
    return NLPScaling(
        one(T),
        [SVector{nh, T}(ones(T, nh)) for _ = 1:N],
        [SVector{ng, T}(ones(T, ng)) for _ = 1:N])
end

struct Solver{T, nx, nu, nh, ng, nux, nhx, ngx, O}
    ocp::O
    nominal::Vector{TrajectoryElement{T, nx, nu, nh, ng}}
    current::Vector{TrajectoryElement{T, nx, nu, nh, ng}}
    nominal_terminal_state::Base.RefValue{SVector{nx, T}}
    current_terminal_state::Base.RefValue{SVector{nx, T}}
    update::Vector{UpdateRule{T, nx, nu, nh, ng, nux, nhx, ngx}}
	data::SolverData{T}
    options::Options{T}
    scaling::NLPScaling{T, nh, ng}
end

function Solver(ocp::OCP{T, nx, nu, nh, ng};
        options::Union{Options{T}, Nothing}=nothing) where {T, nx, nu, nh, ng}
    nstages = ocp.N - 1
    nominal = [TrajectoryElement(T, nx, nu, nh, ng) for _ = 1:nstages]
    current = [TrajectoryElement(T, nx, nu, nh, ng) for _ = 1:nstages]
    terminal_state = SVector{nx, T}(zeros(T, nx))
    nominal_terminal_state = Ref(terminal_state)
    current_terminal_state = Ref(terminal_state)
    update = [UpdateRule(T, nx, nu, nh, ng) for _ = 1:nstages]
    data = solver_data(T)
    options = isnothing(options) ? Options{T}() : options
    scaling = NLPScaling(T, nstages, nh, ng)
	return Solver(ocp, nominal, current, nominal_terminal_state,
        current_terminal_state, update, data, options, scaling)
end

function get_trajectory(solver::Solver)
    x = [nom.x for nom in solver.nominal]
    push!(x, solver.nominal_terminal_state[])
    u = [nom.u for nom in solver.nominal]
	return x, u
end

get_slack_trajectory(solver::Solver) = [nom.s for nom in solver.nominal]

function initialize_interior(value::SVector{n, T}, bounds::BoundConstraints{T, n},
        κ_1::T, κ_2::T) where {T, n}
    return SVector{n, T}(ntuple(Val(n)) do index
        lower = bounds.l[index]
        upper = bounds.u[index]
        has_lower = bounds.maskl[index]
        has_upper = bounds.masku[index]
        if has_lower && has_upper
            lower_interior = lower + min(κ_1 * max(one(T), abs(lower)), κ_2 * (upper - lower))
            upper_interior = upper - min(κ_1 * max(one(T), abs(upper)), κ_2 * (upper - lower))
            return min(max(value[index], lower_interior), upper_interior)
        elseif has_lower
            return max(value[index], κ_1 * max(lower, one(T)) + lower)
        elseif has_upper
            return min(value[index], -κ_1 * max(upper, one(T)) + upper)
        end
        return value[index]
    end)
end

function initialize_trajectory!(solver::Solver{T, nx, nu, nh, ng},
        u::Vector{SVector{nu, T}}, x1::SVector{nx, T}) where {T, nx, nu, nh, ng}
    options = solver.options
    control_limits = solver.ocp.control_limits
    inequalities = solver.ocp.inequality_constraints
    slack_bounds = inequalities.bounds
    parameters = solver.ocp.parameters
    nstages = solver.ocp.N - 1
    length(u) == nstages || throw(DimensionMismatch(
        "expected $(nstages) controls for $(solver.ocp.N) state knots, got $(length(u))"))
    x = x1
    for t = 1:nstages
        ūt = initialize_interior(
            u[t], control_limits, options.κ_1, options.κ_2)
        s̄t = ng > 0 ? evaluate(
            inequalities.g, x, ūt, parameters.inequality_constraints) : @SVector zeros(T, ng)
        s = initialize_interior(
            s̄t, slack_bounds, options.κ_1, options.κ_2)
        ϕ = @SVector zeros(T, nh)
        ν = @SVector zeros(T, ng)
        zl = options.ineq_dual_init .* SVector{nu, T}(control_limits.maskl)
        zu = options.ineq_dual_init .* SVector{nu, T}(control_limits.masku)
        zsl = options.ineq_dual_init .* SVector{ng, T}(slack_bounds.maskl)
        zsu = options.ineq_dual_init .* SVector{ng, T}(slack_bounds.masku)

        solver.nominal[t] = TrajectoryElement(x, ūt, s, ϕ, ν, zl, zu, zsl, zsu)
        x = evaluate(solver.ocp.dynamics.f, x, ūt, parameters.dynamics)
    end
    solver.nominal_terminal_state[] = x
    solver.current_terminal_state[] = x
    compute_nlp_scaling!(solver)
    dual_scale = solver.scaling.objective
    if dual_scale != one(T)
        for t = eachindex(solver.nominal)
            element = solver.nominal[t]
            solver.nominal[t] = TrajectoryElement(
                element.x, element.u, element.s, element.ϕ, element.ν,
                dual_scale .* element.zl, dual_scale .* element.zu,
                dual_scale .* element.zsl, dual_scale .* element.zsu)
        end
    end
end

@inline function gradient_scale(max_gradient::T, gradient_norm::T) where T
    return iszero(gradient_norm) ? one(T) : min(one(T), max_gradient / gradient_norm)
end

function compute_nlp_scaling!(solver::Solver{T, nx, nu, nh, ng}) where {T, nx, nu, nh, ng}
    scaling = solver.scaling
    options = solver.options
    max_gradient = options.nlp_scaling_max_gradient
    parameters = solver.ocp.parameters
    max_gradient > zero(T) || throw(ArgumentError("nlp_scaling_max_gradient must be positive"))

    scaling.objective = one(T)
    fill!(scaling.equality, SVector{nh, T}(ones(T, nh)))
    fill!(scaling.inequality, SVector{ng, T}(ones(T, ng)))
    options.nlp_scaling || return nothing

    objective_gradient_norm = zero(T)
    for t = eachindex(solver.nominal)
        element = solver.nominal[t]
        objective_gradient_norm = max(
            objective_gradient_norm,
            norm(evaluate(solver.ocp.stage_objective.lx, element.x, element.u,
                parameters.stage_objective), Inf),
            norm(evaluate(solver.ocp.stage_objective.lu, element.x, element.u,
                parameters.stage_objective), Inf))

        if nh > 0
            hx = evaluate(solver.ocp.equality_constraints.hx,
                element.x, element.u, parameters.equality_constraints)
            hu = evaluate(solver.ocp.equality_constraints.hu,
                element.x, element.u, parameters.equality_constraints)
            scaling.equality[t] = SVector{nh, T}(ntuple(Val(nh)) do index
                gradient_norm = max(norm(hx[index, :], Inf), norm(hu[index, :], Inf))
                gradient_scale(max_gradient, gradient_norm)
            end)
        end

        if ng > 0
            gx = evaluate(solver.ocp.inequality_constraints.gx,
                element.x, element.u, parameters.inequality_constraints)
            gu = evaluate(solver.ocp.inequality_constraints.gu,
                element.x, element.u, parameters.inequality_constraints)
            scaling.inequality[t] = SVector{ng, T}(ntuple(Val(ng)) do index
                gradient_norm = max(one(T), norm(gx[index, :], Inf), norm(gu[index, :], Inf))
                gradient_scale(max_gradient, gradient_norm)
            end)
        end
    end
    objective_gradient_norm = max(objective_gradient_norm,
        norm(evaluate(solver.ocp.term_objective.lx,
            solver.nominal_terminal_state[], parameters.term_objective), Inf))
    scaling.objective = gradient_scale(max_gradient, objective_gradient_norm)
    return nothing
end

function rescale_duals!(solver::Solver{T}, scale::T) where T
    scale == one(T) && return nothing
    for t = eachindex(solver.nominal)
        element = solver.nominal[t]
        solver.nominal[t] = TrajectoryElement(
            element.x, element.u, element.s,
            scale .* element.ϕ, scale .* element.ν,
            scale .* element.zl, scale .* element.zu,
            scale .* element.zsl, scale .* element.zsu)
    end
    return nothing
end

function update_nominal_trajectory!(solver::Solver)
    for t = eachindex(solver.nominal)
        solver.nominal[t] = solver.current[t]
    end
    solver.nominal_terminal_state[] = solver.current_terminal_state[]
    return nothing
end

function get_feedback(solver::Solver, t::Int)
    α = solver.update[t].α
    β = solver.update[t].β
    ū = solver.nominal[t].u
    x̄ = solver.nominal[t].x
    f = x -> ū + α + β * (x - x̄)
    return f
end

function get_slack_feedback(solver::Solver, t::Int)
    αs = solver.update[t].αs
    βs = solver.update[t].βs
    s̄ = solver.nominal[t].s
    x̄ = solver.nominal[t].x
    return x -> s̄ + αs + βs * (x - x̄)
end
