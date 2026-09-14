struct OCP{T, nx, nu, nh, ng, D, E, I, C, OS, OT}
    N::Int64
    stage_objective::OS
    term_objective::OT
    dynamics::D
    equality_constraints::E
    inequality_constraints::I
    control_limits::C
end

function build_ocp(N::Int64, stage_objective::Objective{nx, nu},
        term_objective::Objective{nx, nu}, dynamics::Dynamics{nx, nu},
        equality_constraints::EqualityConstraints{nx, nu, nh},
        inequality_constraints::InequalityConstraints{T, nx, nu, ng},
        control_limits::ControlLimits{T, nu}) where {T<:Real, nx, nu, nh, ng}
    return OCP{T, nx, nu, nh, ng, typeof(dynamics),
        typeof(equality_constraints), typeof(inequality_constraints),
        typeof(control_limits), typeof(stage_objective), typeof(term_objective)}(
        N, stage_objective, term_objective, dynamics, equality_constraints,
        inequality_constraints, control_limits)
end

function build_ocp(N::Int64, stage_objective::Objective{nx, nu},
        term_objective::Objective{nx, nu}, dynamics::Dynamics{nx, nu},
        equality_constraints::EqualityConstraints{nx, nu},
        control_limits::ControlLimits{T, nu}) where {T<:Real, nx, nu}
    inequality_constraints = InequalityConstraints(T, nx, nu)
    return build_ocp(N, stage_objective, term_objective, dynamics,
        equality_constraints, inequality_constraints, control_limits)
end

function build_ocp(N::Int64, stage_objective::Objective{nx, nu},
        term_objective::Objective{nx, nu}, dynamics::Dynamics{nx, nu},
        control_limits::ControlLimits{T, nu}) where {T<:Real, nx, nu}
    equality_constraints = EqualityConstraints(nx, nu)
    return build_ocp(N, stage_objective, term_objective, dynamics,
        equality_constraints, control_limits)
end
