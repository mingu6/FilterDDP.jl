mutable struct OCPParameters{T, nps, npt, npd, npe, npi}
    stage_objective::Vector{SVector{nps, T}}
    term_objective::SVector{npt, T}
    dynamics::Vector{SVector{npd, T}}
    equality_constraints::Vector{SVector{npe, T}}
    inequality_constraints::Vector{SVector{npi, T}}
end

function OCPParameters(::Type{T}, N::Int64, ::Val{nps}, ::Val{npt}, ::Val{npd},
        ::Val{npe}, ::Val{npi}, parameters::NamedTuple) where {T<:Real, nps, npt, npd, npe, npi}
    nstages = N - 1
    ocp_parameters = OCPParameters{T, nps, npt, npd, npe, npi}(
        Vector{SVector{nps, T}}(undef, nstages),
        zero(SVector{npt, T}),
        Vector{SVector{npd, T}}(undef, nstages),
        Vector{SVector{npe, T}}(undef, nstages),
        Vector{SVector{npi, T}}(undef, nstages))
    empty = SVector{0, T}()
    set_parameters!(ocp_parameters; stage_objective=empty, term_objective=empty,
        dynamics=empty, equality_constraints=empty, inequality_constraints=empty,
        parameters...)
    return ocp_parameters
end

function parameter_vector(::Type{SVector{np, T}}, p::AbstractVector{<:Real},
        name::Symbol) where {np, T}
    length(p) == np || throw(DimensionMismatch(
        "$(name) parameters have length $(length(p)), expected $(np)"))
    return SVector{np, T}(p)
end

function set_stage_parameters!(destination::Vector{SVector{np, T}},
        p::AbstractVector{<:Real}, name::Symbol) where {np, T}
    fill!(destination, parameter_vector(SVector{np, T}, p, name))
    return destination
end

function set_stage_parameters!(destination::Vector{SVector{np, T}},
        p::AbstractVector{<:AbstractVector{<:Real}}, name::Symbol) where {np, T}
    length(p) == length(destination) || throw(DimensionMismatch(
        "$(name) parameters have $(length(p)) stages, expected $(length(destination))"))
    for (t, pt) in zip(eachindex(destination), p)
        destination[t] = parameter_vector(SVector{np, T}, pt, name)
    end
    return destination
end

function set_parameters!(parameters::OCPParameters{T, nps, npt};
        stage_objective=nothing, term_objective=nothing, dynamics=nothing,
        equality_constraints=nothing, inequality_constraints=nothing) where {T, nps, npt}
    isnothing(stage_objective) || set_stage_parameters!(
        parameters.stage_objective, stage_objective, :stage_objective)
    isnothing(term_objective) || (parameters.term_objective = parameter_vector(
        SVector{npt, T}, term_objective, :term_objective))
    isnothing(dynamics) || set_stage_parameters!(
        parameters.dynamics, dynamics, :dynamics)
    isnothing(equality_constraints) || set_stage_parameters!(
        parameters.equality_constraints, equality_constraints, :equality_constraints)
    isnothing(inequality_constraints) || set_stage_parameters!(
        parameters.inequality_constraints, inequality_constraints, :inequality_constraints)
    return parameters
end

struct OCP{T, nx, nu, nh, ng, nps, npt, npd, npe, npi, D, E, I, OS, OT}
    N::Int64
    stage_objective::OS
    term_objective::OT
    dynamics::D
    equality_constraints::E
    inequality_constraints::I
    control_limits::Vector{BoundConstraints{T, nu}}
    inequality_bounds::Vector{BoundConstraints{T, ng}}
    parameters::OCPParameters{T, nps, npt, npd, npe, npi}
end

function build_ocp(N::Int64, stage_objective::AbstractStageObjective{nx, nu, nps},
    term_objective::AbstractTerminalObjective{nx, npt},
    dynamics::AbstractDynamics{nx, nu, npd},
    equality_constraints::AbstractEqualityConstraints{nx, nu, nh, npe},
    inequality_constraints::AbstractInequalityConstraints{T, nx, nu, ng, npi},
        control_limits::StageBounds{T, nu};
        parameters::NamedTuple=(;),
        inequality_bounds=inequality_constraints.bounds
        ) where {T<:Real, nx, nu, nh, ng, nps, npt, npd, npe, npi}
    N >= 2 || throw(ArgumentError("horizon must contain at least two state knots"))
    ocp_parameters = OCPParameters(T, N, Val(nps), Val(npt), Val(npd), Val(npe),
        Val(npi), parameters)
    ocp_control_limits = set_stage_bounds!(
        Vector{BoundConstraints{T, nu}}(undef, N - 1), control_limits, :control_limits)
    ocp_inequality_bounds = set_stage_bounds!(
        Vector{BoundConstraints{T, ng}}(undef, N - 1), inequality_bounds, :inequality_bounds)
    return OCP{T, nx, nu, nh, ng, nps, npt, npd, npe, npi, typeof(dynamics),
        typeof(equality_constraints), typeof(inequality_constraints),
        typeof(stage_objective), typeof(term_objective)}(
        N, stage_objective, term_objective, dynamics, equality_constraints,
        inequality_constraints, ocp_control_limits, ocp_inequality_bounds,
        ocp_parameters)
end

function build_ocp(N::Int64, stage_objective::AbstractStageObjective{nx, nu, nps},
    term_objective::AbstractTerminalObjective{nx, npt},
    dynamics::AbstractDynamics{nx, nu, npd},
    inequality_constraints::AbstractInequalityConstraints{T, nx, nu, ng, npi},
    control_limits::StageBounds{T, nu};
    parameters::NamedTuple=(;),
    inequality_bounds=inequality_constraints.bounds
    ) where {T<:Real, nx, nu, ng, nps, npt, npd, npi}
    equality_constraints = EqualityConstraints(nx, nu)
    return build_ocp(N, stage_objective, term_objective, dynamics,
        equality_constraints, inequality_constraints, control_limits;
        parameters=parameters, inequality_bounds=inequality_bounds)
end

function build_ocp(N::Int64, stage_objective::AbstractStageObjective{nx, nu, nps},
    term_objective::AbstractTerminalObjective{nx, npt},
    dynamics::AbstractDynamics{nx, nu, npd},
    equality_constraints::AbstractEqualityConstraints{nx, nu, nh, npe},
    control_limits::StageBounds{T, nu};
    parameters::NamedTuple=(;)
    ) where {T<:Real, nx, nu, nh, nps, npt, npd, npe}
    inequality_constraints = InequalityConstraints(T, nx, nu)
    return build_ocp(N, stage_objective, term_objective, dynamics,
        equality_constraints, inequality_constraints, control_limits;
        parameters=parameters)
end

function build_ocp(N::Int64, stage_objective::AbstractStageObjective{nx, nu, nps},
    term_objective::AbstractTerminalObjective{nx, npt},
    dynamics::AbstractDynamics{nx, nu, npd},
    control_limits::StageBounds{T, nu};
    parameters::NamedTuple=(;)
    ) where {T<:Real, nx, nu, nps, npt, npd}
    equality_constraints = EqualityConstraints(nx, nu)
    return build_ocp(N, stage_objective, term_objective, dynamics,
    equality_constraints, control_limits; parameters=parameters)
end

function set_parameters!(ocp::OCP; kwargs...)
    set_parameters!(ocp.parameters; kwargs...)
    return ocp
end

function set_bounds!(ocp::OCP; control_limits=nothing, inequality_bounds=nothing)
    isnothing(control_limits) || set_stage_bounds!(
        ocp.control_limits, control_limits, :control_limits)
    isnothing(inequality_bounds) || set_stage_bounds!(
        ocp.inequality_bounds, inequality_bounds, :inequality_bounds)
    return ocp
end

bound_count(ocp::OCP) = bound_count(ocp.control_limits) + bound_count(ocp.inequality_bounds)
