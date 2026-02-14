struct OCP{T}
    N::Int
    nx::Vector{Int}
    nu::Vector{Int}
    nc::Vector{Int}
    objective::Vector{Objective{T}}
    dynamics::Vector{Dynamics{T}};
    constraints::Vector{Constraints{T}}
    control_limits::Vector{ControlLimits{T}}
end

function OCP(objective::Vector{Objective{T}}, dynamics::Vector{Dynamics{T}};
    constraints::Union{Vector{Constraints{T}}, Nothing} = nothing,
    control_limits::Union{Vector{ControlLimits{T}}, Nothing} = nothing) where T
    
    N = length(objective)
    @assert length(dynamics) == N-1
    @assert length(constraints) == N
    @assert length(control_limits) == N
    @assert all(a.nx == b.nx for (a, b) in zip(objective, constraints))
    @assert all(a.nu == b.nu for (a, b) in zip(objective, constraints))
    @assert all(a.nx == b.nx for (a, b) in zip(dynamics, constraints))
    @assert all(a.nu == b.nu for (a, b) in zip(dynamics, constraints))
    @assert all(a.nx1 == b.nx for (a, b) in zip(dynamics, constraints[2:end]))
    @assert all(a.nu == length(b.l) for (a, b) in zip(objective, control_limits))

    nx = [c.nx for c in constraints]
    nu = [c.nu for c in constraints]
    nc = [c.nc for c in constraints]

    if isnothing(constraints)
        constraints = [Constraints(T, c.nx, c.nu) for c in constraints]
    end

    if isnothing(control_limits)
        control_limits = [ControlLimits(T, c.nu) for c in constraints]
    end

    return OCP{T}(N, nx, nu, nc, objective, dynamics, constraints, control_limits)
end

function OCP(N::Int, stage_objective::Objective{T}, term_objective::Objective{T}, dynamics::Dynamics{T};
    constraints::Union{Constraints{T}, Nothing} = nothing,
    control_limits::Union{ControlLimits{T}, Nothing} = nothing) where T

    return OCP([[deepcopy(stage_objective) for _ = 1:N-1]..., deepcopy(term_objective)], [deepcopy(dynamics) for _ = 1:N-1]; 
        constraints=[deepcopy(constraints) for _ = 1:N], control_limits=[deepcopy(control_limits) for _ = 1:N])
end

function evaluate_derivatives!(ocp::OCP{T}, ws::FilterDDPWorkspace{T}; mode=:nominal) where T
    jacobians!(ocp.constraints, ws; mode=mode)
    jacobians!(ocp.dynamics, ws; mode=mode)
    gradients!(ocp.objective, ws; mode=mode)
    hessians!(ocp.constraints, ws; mode=mode)
    hessians!(ocp.objective, ws; mode=mode)
    return nothing
end

function FilterDDPWorkspace(ocp::OCP{T}) where T
    return [FilterDDPWorkspaceElement{T}(
                TrajectoryElement(T, c.nx, c.nu, c.nc),
                TrajectoryElement(T, c.nx, c.nu, c.nc),
                zeros(T, c.nu+c.nc, c.nx+1),
                zeros(T, 2*c.nu, c.nx+1),
                zeros(T, c.nu),
                zeros(T, c.nx, c.nx),
                zeros(T, c.nu, c.nx),
                zeros(T, c.nu, c.nu),
                zeros(T, c.nx),
                zeros(T, c.nx, c.nx),
                zeros(T, c.nx),
                zeros(T, c.nu),
                zeros(T, c.nu),
                zeros(T, c.nc),
                zeros(T, c.nx, c.nx),
                zeros(T, c.nu, c.nx),
                zeros(T, c.nu, c.nu),
                zeros(T, c.nc, c.nx),
                zeros(T, c.nc, c.nu),
                zeros(T, c.nc+c.nu, c.nc+c.nu),
                BunchKaufmanWs(zeros(T, c.nc+c.nu, c.nc+c.nu)),
                Pair(zeros(T, c.nc+c.nu), zeros(T, c.nc+c.nu)))
                for c in ocp.constraints]
end
