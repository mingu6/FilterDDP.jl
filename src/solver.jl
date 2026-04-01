mutable struct Solver{T, nx, nu, nc}
    ocp::OCP{T, nx, nu, nc}
    ws::FilterDDPWorkspace{T, nx, nu, nc}
	data::SolverData{T}
    options::Options{T}
end

function Solver(ocp::OCP{T, nx, nu, nc}; options::Union{Options{T}, Nothing}=nothing) where {T, nx, nu, nc}
    ws = FilterDDPWorkspace(ocp)
    data = solver_data(T)
    options = isnothing(options) ? Options{T}() : options
	return Solver(ocp, ws, data, options)
end

function get_trajectory(solver::Solver{T, nx, nu, nc}) where {T, nx, nu, nc}
    x = [wse.nominal.x for wse in solver.ws]
    u = [wse.nominal.u for wse in solver.ws]
	return x, u
end

function initialize_trajectory!(solver::Solver{T, nx, nu, nc}, u::Vector{SVector{nu, T}}, x1::SVector{nx, T}) where {T, nx, nu, nc}
    options = solver.options
    κ_1 = options.κ_1
    κ_2 = options.κ_2

    solver.ws[1].nominal.x = x1
    for t = 1:solver.ocp.N
        cl = solver.ocp.control_limits[t]

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

        solver.ws[t].nominal.u = ūt

        # initialise inequality constraints

        solver.ws[t].nominal.ul = ūt - cl.l
        solver.ws[t].nominal.uu = cl.u - ūt
        
        t < solver.ocp.N && dynamics!(solver.ocp.dynamics[t], solver.ws[t], solver.ws[t+1]; mode=:nominal)
    end
end

function get_feedback(solver::Solver{T, nx, nu, nc}, t::Int) where {T, nx, nu, nc}
    α = solver.ws[t].α
    β = solver.ws[t].β
    ū = solver.ws[t].nominal.u
    x̄ = solver.ws[t].nominal.x
    f = x -> ū + α + β * (x - x̄)
    return f
end
