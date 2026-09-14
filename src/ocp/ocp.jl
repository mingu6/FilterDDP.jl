struct OCPParameters{T, npo, npd, npe, npi}
    stage_objective::SVector{npo, T}
    term_objective::SVector{npo, T}
    dynamics::SVector{npd, T}
    equality_constraints::SVector{npe, T}
    inequality_constraints::SVector{npi, T}
end

function OCPParameters(;
        stage_objective::SVector{npo, T},
        term_objective::SVector{npo, T},
        dynamics::SVector{npd, T},
        equality_constraints::SVector{npe, T},
        inequality_constraints::SVector{npi, T}
        ) where {T<:Real, npo, npd, npe, npi}
    return OCPParameters(stage_objective, term_objective, dynamics,
        equality_constraints, inequality_constraints)
end

function OCPParameters(::Type{T}) where {T<:Real}
    empty = SVector{0, T}()
    return OCPParameters(empty, empty, empty, empty, empty)
end

struct OCP{T, nx, nu, nh, ng, npo, npd, npe, npi, D, E, I, C, OS, OT}
    N::Int64
    stage_objective::OS
    term_objective::OT
    dynamics::D
    equality_constraints::E
    inequality_constraints::I
    control_limits::C
    parameters::OCPParameters{T, npo, npd, npe, npi}
end

function build_ocp(N::Int64, stage_objective::Objective{nx, nu, npo},
        term_objective::Objective{nx, nu, npo}, dynamics::Dynamics{nx, nu, npd},
        equality_constraints::EqualityConstraints{nx, nu, nh, npe},
        inequality_constraints::InequalityConstraints{T, nx, nu, ng, npi},
        control_limits::ControlLimits{T, nu};
        parameters::OCPParameters{T, npo, npd, npe, npi}=OCPParameters(T)
        ) where {T<:Real, nx, nu, nh, ng, npo, npd, npe, npi}
    return OCP{T, nx, nu, nh, ng, npo, npd, npe, npi, typeof(dynamics),
        typeof(equality_constraints), typeof(inequality_constraints),
        typeof(control_limits), typeof(stage_objective), typeof(term_objective)}(
        N, stage_objective, term_objective, dynamics, equality_constraints,
        inequality_constraints, control_limits, parameters)
end

function build_ocp(N::Int64, stage_objective::Objective{nx, nu, npo},
    term_objective::Objective{nx, nu, npo}, dynamics::Dynamics{nx, nu, npd},
    inequality_constraints::InequalityConstraints{T, nx, nu, ng, npi},
    control_limits::ControlLimits{T, nu};
    parameters::OCPParameters{T, npo, npd, 0, npi}=OCPParameters(T)
    ) where {T<:Real, nx, nu, ng, npo, npd, npi}
    equality_constraints = EqualityConstraints(nx, nu)
    return build_ocp(N, stage_objective, term_objective, dynamics,
        equality_constraints, inequality_constraints, control_limits;
        parameters=parameters)
end

function build_ocp(N::Int64, stage_objective::Objective{nx, nu, npo},
    term_objective::Objective{nx, nu, npo}, dynamics::Dynamics{nx, nu, npd},
    equality_constraints::EqualityConstraints{nx, nu, nh, npe},
    control_limits::ControlLimits{T, nu};
    parameters::OCPParameters{T, npo, npd, npe, 0}=OCPParameters(T)
    ) where {T<:Real, nx, nu, nh, npo, npd, npe}
    inequality_constraints = InequalityConstraints(T, nx, nu)
    return build_ocp(N, stage_objective, term_objective, dynamics,
        equality_constraints, inequality_constraints, control_limits;
        parameters=parameters)
end

function build_ocp(N::Int64, stage_objective::Objective{nx, nu, npo},
    term_objective::Objective{nx, nu, npo}, dynamics::Dynamics{nx, nu, npd},
    control_limits::ControlLimits{T, nu};
    parameters::OCPParameters{T, npo, npd, 0, 0}=OCPParameters(T)
    ) where {T<:Real, nx, nu, npo, npd}
    equality_constraints = EqualityConstraints(nx, nu)
    return build_ocp(N, stage_objective, term_objective, dynamics,
    equality_constraints, control_limits; parameters=parameters)
end
