mutable struct OCP{T, nx, nu, nc}
    N::Int
    objective::Vector{Objective{T, nx, nu}}
    dynamics::Vector{Dynamics{T, nx, nu}};
    constraints::Vector{EqualityConstraints{T, nx, nu, nc}}
    control_limits::Vector{ControlLimits{T, nu}}
    no_eq_constr::Bool
end

function OCP(objective::Vector{Objective{T, nx, nu}}, dynamics::Vector{Dynamics{T, nx, nu}};
    constraints::Union{Vector{EqualityConstraints{T, nx, nu, nc}}, Nothing} = nothing,
    control_limits::Union{Vector{ControlLimits{T, nu}}, Nothing} = nothing) where {T, nx, nu, nc}
    
    no_eq_constr = false
    N = length(objective)
    if isnothing(constraints)
        constraints = [EqualityConstraints(T, nx, nu) for o in objective]
        no_eq_constr = true
    end
    if isnothing(control_limits)
        control_limits = [ControlLimits(T, nu) for o in objective]
    end

    @assert length(dynamics) == N-1
    @assert length(constraints) == N
    @assert length(control_limits) == N

    return OCP(N, objective, dynamics, constraints, control_limits, no_eq_constr)
end

function OCP(N::Int64, stage_objective::Objective{T, nx, nu}, term_objective::Objective{T, nx, nu}, dynamics::Dynamics{T, nx, nu};
        constraints::Union{EqualityConstraints{T, nx, nu, nc}, Nothing} = nothing,
        control_limits::Union{ControlLimits{T, nu}, Nothing} = nothing) where {T, nx, nu, nc}
    if !isnothing(control_limits)
        control_limits = [deepcopy(control_limits) for _ = 1:N]
    end
    if !isnothing(constraints)
        constraints = [deepcopy(constraints) for _ = 1:N]
    end
    return OCP([[deepcopy(stage_objective) for _ = 1:N-1]..., deepcopy(term_objective)], [deepcopy(dynamics) for _ = 1:N-1]; 
        constraints=constraints, control_limits=control_limits)
end

function evaluate_derivatives!(ocp::OCP{T, nx, nu, nc}, ws::FilterDDPWorkspace{T, nx, nu, nc}; mode=:nominal) where {T, nx, nu, nc}
    jacobians!(ocp.constraints, ws; mode=mode)
    jacobians!(ocp.dynamics, ws; mode=mode)
    gradients!(ocp.objective, ws; mode=mode)
    hessians!(ocp.constraints, ws; mode=mode)
    hessians!(ocp.objective, ws; mode=mode)
    return nothing
end

function FilterDDPWorkspace(ocp::OCP{T, nx, nu, nc}) where {T, nx, nu, nc}
    return [FilterDDPWorkspaceElement{T, nx, nu, nc}(
                TrajectoryElement(T, nx, nu, nc),
                TrajectoryElement(T, nx, nu, nc),
                SVector{nu, T}(zeros(T, nu)),
                SVector{nc, T}(zeros(T, nc)),
                SMatrix{nu, nx, T}(zeros(T, nu, nx)),
                SMatrix{nc, nx, T}(zeros(T, nc, nx)),
                SVector{nu, T}(zeros(T, nu)),
                SVector{nu, T}(zeros(T, nu)),
                SMatrix{nu, nx, T}(zeros(T, nu, nx)),
                SMatrix{nu, nx, T}(zeros(T, nu, nx)),
                SVector{nu, T}(zeros(T, nu)),
                SVector{nx, T}(zeros(T, nx)),
                SMatrix{nx, nx, T}(zeros(T, nx, nx)))
                for c in ocp.constraints]
end
