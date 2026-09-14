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

struct Solver{T, nx, nu, nh, ng, nux, nhx, ngx, O}
    ocp::O
    nominal::Vector{TrajectoryElement{T, nx, nu, nh, ng}}
    current::Vector{TrajectoryElement{T, nx, nu, nh, ng}}
    update::Vector{UpdateRule{T, nx, nu, nh, ng, nux, nhx, ngx}}
	data::SolverData{T}
    options::Options{T}
end

function Solver(ocp::OCP{T, nx, nu, nh, ng};
        options::Union{Options{T}, Nothing}=nothing) where {T, nx, nu, nh, ng}
    nominal = [TrajectoryElement(T, nx, nu, nh, ng) for _ = 1:ocp.N]
    current = [TrajectoryElement(T, nx, nu, nh, ng) for _ = 1:ocp.N]
    update = [UpdateRule(T, nx, nu, nh, ng) for _ = 1:ocp.N]
    data = solver_data(T)
    options = isnothing(options) ? Options{T}() : options
	return Solver(ocp, nominal, current, update, data, options)
end

function get_trajectory(solver::Solver)
    x = [nom.x for nom in solver.nominal]
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
    x = x1
    for t = 1:solver.ocp.N
        ūt = initialize_interior(
            u[t], control_limits, options.κ_1, options.κ_2)
        s̄t = ng > 0 ? inequalities.g(x, ūt) : @SVector zeros(T, ng)
        s = initialize_interior(
            s̄t, slack_bounds, options.κ_1, options.κ_2)
        ϕ = @SVector zeros(T, nh)
        ν = @SVector zeros(T, ng)
        zl = SVector{nu, T}(control_limits.maskl)
        zu = SVector{nu, T}(control_limits.masku)
        zsl = SVector{ng, T}(slack_bounds.maskl)
        zsu = SVector{ng, T}(slack_bounds.masku)

        solver.nominal[t] = TrajectoryElement(x, ūt, s, ϕ, ν, zl, zu, zsl, zsu)
        
        if t < solver.ocp.N
            x = solver.ocp.dynamics.f(x, ūt)
        end
    end
end

function update_nominal_trajectory!(solver::Solver)
    for t = 1:solver.ocp.N
        solver.nominal[t] = solver.current[t]
    end
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
