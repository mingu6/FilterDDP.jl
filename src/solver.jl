struct TrajectoryElement{T, nx, nu, nc}
    x::SVector{nx, T}
    u::SVector{nu, T}
    ϕ::SVector{nc, T}
    zl::SVector{nu, T}
    zu::SVector{nu, T}
end

function TrajectoryElement(T, nx::Int, nu::Int, nc::Int)
    TrajectoryElement{T, nx, nu, nc}(
        SVector{nx, T}(zeros(T, nx)),
        SVector{nu, T}(zeros(T, nu)),
        SVector{nc, T}(zeros(T, nc)),
        SVector{nu, T}(zeros(T, nu)),
        SVector{nu, T}(zeros(T, nu))
        )
end

struct UpdateRule{T, nx, nu, nc}
    α::SVector{nu, T}
    ψ::SVector{nc, T}
    β::SMatrix{nu, nx, T}
    ω::SMatrix{nc, nx, T}
    χl::SVector{nu, T}
    χu::SVector{nu, T}
    ζl::SMatrix{nu, nx, T}
    ζu::SMatrix{nu, nx, T}
end

struct Solver{T, nx, nu, nc}
    ocp::OCP{T, nx, nu, nc}
    nominal::Vector{TrajectoryElement{T, nx, nu, nc}}
    current::Vector{TrajectoryElement{T, nx, nu, nc}}
    update::Vector{UpdateRule{T, nx, nu, nc}}
	data::SolverData{T}
    options::Options{T}
end

function UpdateRule(T, nx::Int, nu::Int, nc::Int)
    UpdateRule{T, nx, nu, nc}(
        SVector{nu, T}(zeros(T, nu)),
        SVector{nc, T}(zeros(T, nc)),
        SMatrix{nu, nx, T}(zeros(T, nu, nx)),
        SMatrix{nc, nx, T}(zeros(T, nc, nx)),
        SVector{nu, T}(zeros(T, nu)),
        SVector{nu, T}(zeros(T, nu)),
        SMatrix{nu, nx, T}(zeros(T, nu, nx)),
        SMatrix{nu, nx, T}(zeros(T, nu, nx))
    )
end

function Solver(ocp::OCP{T, nx, nu, nc}; options::Union{Options{T}, Nothing}=nothing) where {T, nx, nu, nc}
    nominal = [TrajectoryElement(T, nx, nu, nc) for _ = 1:ocp.N]
    current = [TrajectoryElement(T, nx, nu, nc) for _ = 1:ocp.N]
    update = [UpdateRule(T, nx, nu, nc) for _ = 1:ocp.N]
    data = solver_data(T)
    options = isnothing(options) ? Options{T}() : options
	return Solver(ocp, nominal, current, update, data, options)
end

function get_trajectory(solver::Solver{T, nx, nu, nc}) where {T, nx, nu, nc}
    x = [nom.x for nom in solver.nominal]
    u = [nom.u for nom in solver.nominal]
	return x, u
end

function initialize_trajectory!(solver::Solver{T, nx, nu, nc}, u::Vector{SVector{nu, T}}, x1::SVector{nx, T}) where {T, nx, nu, nc}
    options = solver.options
    κ_1 = options.κ_1
    κ_2 = options.κ_2

    cl = solver.ocp.control_limits
    x = x1
    for t = 1:solver.ocp.N
        ūt = @SVector zeros(T, nu)
        mask_lo = cl.maskl .* .!cl.masku
        ūt = ūt + max.(u[t],  options.κ_1 .* max.(cl.l, 1.0) + cl.l) .* mask_lo

        mask_up = cl.masku .* .!cl.maskl
        ūt = ūt + min.(u[t], -options.κ_1 .* max.(cl.u, 1.0) + cl.u) .* mask_up

        mask_both = cl.masku .* cl.maskl
        u1 = cl.l + min.(κ_1 * max.(1.0, abs.(cl.l)), κ_2 .* (cl.u - cl.l))
        u2 = cl.u - min.(κ_1 * max.(1.0, abs.(cl.u)), κ_2 .* (cl.u - cl.l))
        ūt = ūt + min.(max.(u[t], u1), u2) .* mask_both

        mask_none = .!cl.masku .* .!cl.maskl
        ūt = ūt + u[t] .* mask_none

        ϕ = @SVector zeros(T, nc)
        zl = SVector{nu, T}(ones(T, nu) .* cl.maskl)
        zu = SVector{nu, T}(ones(T, nu) .* cl.masku)

        solver.nominal[t] = TrajectoryElement{T, nx, nu, nc}(x, ūt, ϕ, zl, zu)
        
        if t < solver.ocp.N
            x = solver.ocp.dynamics.f(x, ūt)
        end
    end
end

function update_nominal_trajectory!(solver::Solver{T, nx, nu, nc}) where {T, nx, nu, nc}
    for t = 1:solver.ocp.N
        solver.nominal[t] = solver.current[t]
    end
    return nothing
end

function get_feedback(solver::Solver{T, nx, nu, nc}, t::Int) where {T, nx, nu, nc}
    α = solver.update[t].α
    β = solver.update[t].β
    ū = solver.nominal[t].u
    x̄ = solver.nominal[t].x
    f = x -> ū + α + β * (x - x̄)
    return f
end
