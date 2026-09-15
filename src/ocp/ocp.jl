struct OCPParameters{T, nps, npt, npd, npe, npi}
    stage_objective::SVector{nps, T}
    term_objective::SVector{npt, T}
    dynamics::SVector{npd, T}
    equality_constraints::SVector{npe, T}
    inequality_constraints::SVector{npi, T}
end

function OCPParameters(;
        stage_objective::SVector{nps, T},
        term_objective::SVector{npt, T},
        dynamics::SVector{npd, T},
        equality_constraints::SVector{npe, T},
        inequality_constraints::SVector{npi, T}
        ) where {T<:Real, nps, npt, npd, npe, npi}
    return OCPParameters(stage_objective, term_objective, dynamics,
        equality_constraints, inequality_constraints)
end

function OCPParameters(::Type{T}) where {T<:Real}
    empty = SVector{0, T}()
    return OCPParameters(empty, empty, empty, empty, empty)
end

struct OCP{T, nx, nu, nh, ng, nps, npt, npd, npe, npi, D, E, I, C, OS, OT}
    N::Int64
    stage_objective::OS
    term_objective::OT
    dynamics::D
    equality_constraints::E
    inequality_constraints::I
    control_limits::C
    parameters::OCPParameters{T, nps, npt, npd, npe, npi}
end

function build_ocp(N::Int64, stage_objective::AbstractStageObjective{nx, nu, nps},
    term_objective::AbstractTerminalObjective{nx, npt},
    dynamics::AbstractDynamics{nx, nu, npd},
    equality_constraints::AbstractEqualityConstraints{nx, nu, nh, npe},
    inequality_constraints::AbstractInequalityConstraints{T, nx, nu, ng, npi},
        control_limits::ControlLimits{T, nu};
        parameters::OCPParameters{T, nps, npt, npd, npe, npi}=OCPParameters(T)
        ) where {T<:Real, nx, nu, nh, ng, nps, npt, npd, npe, npi}
    N >= 2 || throw(ArgumentError("horizon must contain at least two state knots"))
    return OCP{T, nx, nu, nh, ng, nps, npt, npd, npe, npi, typeof(dynamics),
        typeof(equality_constraints), typeof(inequality_constraints),
        typeof(control_limits), typeof(stage_objective), typeof(term_objective)}(
        N, stage_objective, term_objective, dynamics, equality_constraints,
        inequality_constraints, control_limits, parameters)
end

function build_ocp(N::Int64, stage_objective::AbstractStageObjective{nx, nu, nps},
    term_objective::AbstractTerminalObjective{nx, npt},
    dynamics::AbstractDynamics{nx, nu, npd},
    inequality_constraints::AbstractInequalityConstraints{T, nx, nu, ng, npi},
    control_limits::ControlLimits{T, nu};
    parameters::OCPParameters{T, nps, npt, npd, 0, npi}=OCPParameters(T)
    ) where {T<:Real, nx, nu, ng, nps, npt, npd, npi}
    equality_constraints = EqualityConstraints(nx, nu)
    return build_ocp(N, stage_objective, term_objective, dynamics,
        equality_constraints, inequality_constraints, control_limits;
        parameters=parameters)
end

function build_ocp(N::Int64, stage_objective::AbstractStageObjective{nx, nu, nps},
    term_objective::AbstractTerminalObjective{nx, npt},
    dynamics::AbstractDynamics{nx, nu, npd},
    equality_constraints::AbstractEqualityConstraints{nx, nu, nh, npe},
    control_limits::ControlLimits{T, nu};
    parameters::OCPParameters{T, nps, npt, npd, npe, 0}=OCPParameters(T)
    ) where {T<:Real, nx, nu, nh, nps, npt, npd, npe}
    inequality_constraints = InequalityConstraints(T, nx, nu)
    return build_ocp(N, stage_objective, term_objective, dynamics,
        equality_constraints, inequality_constraints, control_limits;
        parameters=parameters)
end

function build_ocp(N::Int64, stage_objective::AbstractStageObjective{nx, nu, nps},
    term_objective::AbstractTerminalObjective{nx, npt},
    dynamics::AbstractDynamics{nx, nu, npd},
    control_limits::ControlLimits{T, nu};
    parameters::OCPParameters{T, nps, npt, npd, 0, 0}=OCPParameters(T)
    ) where {T<:Real, nx, nu, nps, npt, npd}
    equality_constraints = EqualityConstraints(nx, nu)
    return build_ocp(N, stage_objective, term_objective, dynamics,
    equality_constraints, control_limits; parameters=parameters)
end
