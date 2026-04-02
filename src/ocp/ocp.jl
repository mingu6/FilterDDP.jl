struct OCP{T, nx, nu, nc}
    N::Int
    stage_objective::Objective{T, nx, nu}
    term_objective::Objective{T, nx, nu}
    dynamics::Dynamics{T, nx, nu}
    constraints::EqualityConstraints{T, nx, nu, nc}
    control_limits::ControlLimits{T, nu}
end

function OCP(N::Int64, stage_objective::Objective{T, nx, nu}, term_objective::Objective{T, nx, nu}, dynamics::Dynamics{T, nx, nu};
        constraints::Union{EqualityConstraints{T, nx, nu, nc}, Nothing} = nothing,
        control_limits::Union{ControlLimits{T, nu}, Nothing} = nothing) where {T, nx, nu, nc}
    if isnothing(control_limits)
        control_limits = ControlLimits(T, nu)
    end
    if isnothing(constraints)
        constraints = EqualityConstraints(T, nx, nu)
    end
    return OCP(N, stage_objective, term_objective, dynamics, constraints, control_limits)
end

