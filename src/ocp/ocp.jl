RF = RuntimeGeneratedFunction
RFN = Union{RuntimeGeneratedFunction, Nothing}

struct OCP{T, nx, nu, nc, F1, F2, F3, F4, F5, F6, C1, C2, C3, C4, C5, C6, OS1, OS2, OS3, OS4, OS5, OS6, OT1, OT2, OT3, OT4, OT5, OT6}
    N::Int64
    stage_objective::Objective{nx, nu, OS1, OS2, OS3, OS4, OS5, OS6}
    term_objective::Objective{nx, nu, OT1, OT2, OT3, OT4, OT5, OT6}
    dynamics::Dynamics{nx, nu, F1, F2, F3, F4, F5, F6}
    constraints::EqualityConstraints{nx, nu, nc, C1, C2, C3, C4, C5, C6}
    control_limits::ControlLimits{T, nu}
end

function build_ocp(N::Int64, stage_objective::Objective{nx, nu, OS1, OS2, OS3, OS4, OS5, OS6},
        term_objective::Objective{nx, nu, OT1, OT2, OT3, OT4, OT5, OT6}, dynamics::Dynamics{nx, nu, F1, F2, F3, F4, F5, F6},
        constraints::EqualityConstraints{nx, nu, nc, C1, C2, C3, C4, C5, C6}, control_limits::ControlLimits{T, nu}) where
                        {T<:Real, nx, nu, nc, F1<:RF, F2<:RF, F3<:RF, F4<:RF, F5<:RF, F6<:RF,
                        C1<:RFN, C2<:RFN, C3<:RFN, C4<:RFN, C5<:RFN, C6<:RFN,
                        OS1<:RF, OS2<:RF, OS3<:RF, OS4<:RF, OS5<:RF, OS6<:RF,
                        OT1<:RF, OT2<:RF, OT3<:RF, OT4<:RF, OT5<:RF, OT6<:RF}
    l = @SVector ones(T, nx)
    u = floatmax(T) .* l
    l = -floatmax(T) .* l
    control_limits_ = isnothing(control_limits) ? ControlLimits(l, u) : control_limits
    constraints_ = isnothing(constraints) ? EqualityConstraints(nx, nu) : constraints
    return OCP{T, nx, nu, nc, F1, F2, F3, F4, F5, F6, C1, C2, C3, C4, C5, C6, OS1, OS2, OS3, OS4, OS5, OS6, OT1, OT2, OT3, OT4, OT5, OT6}(
        N, stage_objective, term_objective, dynamics, constraints_, control_limits_)
end
