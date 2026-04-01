struct ControlLimits{T, nu}
    l::SVector{nu, T}
    u::SVector{nu, T}
    maskl::SVector{nu, Bool}
    masku::SVector{nu, Bool}
    nl::Int
    nu::Int
end

function ControlLimits(l::SVector{nu, T}, u::SVector{nu, T}) where {T, nu}
    @assert length(l) == length(u)
    @assert all(u .>= l)
    maskl = (l .!= -floatmax(T)) .&& .!isinf.(l)
    masku = (u .!= floatmax(T)) .&& .!isinf.(u)
    nlo = sum(maskl)
    nup = sum(masku)
    return ControlLimits{T, nu}(l, u, maskl, masku, nlo, nup)
end

function ControlLimits(T, nu::Int)
    return ControlLimits(nu, -T(Inf), T(Inf))
end

function ControlLimits(nu::Int, l::T, u::T) where T
    l = isinf(l) ? -floatmax(T) : l
    u = isinf(u) ? floatmax(T) : u
    return ControlLimits(SVector{nu, T}(l .* ones(T, nu)), SVector{nu, T}(u .* ones(T, nu)))
end
