struct ControlLimits{T}
    l::Vector{T}
    u::Vector{T}
    indl::Vector{Int}
    indu::Vector{Int}
    nl::Int
    nu::Int
end

function ControlLimits(l::Vector{T}, u::Vector{T}) where T
    @assert length(l) == length(u)
    @assert all(u .>= l)
    indl = [i for (i, mask) in enumerate(.!isinf.(l)) if mask]
    indu = [i for (i, mask) in enumerate(.!isinf.(u)) if mask]
    nl = sum([1 for b in l if !isinf(b)])
    nu = sum([1 for b in u if !isinf(b)])
    return ControlLimits{T}(l, u, indl, indu, nl, nu)
end

function ControlLimits(T, nu::Int)
    return ControlLimits(-T(Inf) .* ones(T, nu), T(Inf) .* ones(T, nu))
end

function ControlLimits(nu::Int, l::T, u::T) where T
    return ControlLimits(l .* ones(T, nu), u .* ones(T, nu))
end
