struct ControlLimits{T, nu}
    l::SVector{nu, T}
    u::SVector{nu, T}
    maskl::SVector{nu, Bool}
    masku::SVector{nu, Bool}
    nl::Int64
    nu::Int64
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
