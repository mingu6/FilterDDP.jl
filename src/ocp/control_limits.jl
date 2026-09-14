struct BoundConstraints{T, n}
    l::SVector{n, T}
    u::SVector{n, T}
    maskl::SVector{n, Bool}
    masku::SVector{n, Bool}
    nl::Int64
    nu::Int64
end

function BoundConstraints(l::SVector{n, T}, u::SVector{n, T}) where {T, n}
    @assert length(l) == length(u)
    @assert all(u .>= l)
    maskl = (l .!= -floatmax(T)) .&& .!isinf.(l)
    masku = (u .!= floatmax(T)) .&& .!isinf.(u)
    nlo = sum(maskl)
    nup = sum(masku)
    return BoundConstraints{T, n}(l, u, maskl, masku, nlo, nup)
end

const ControlLimits = BoundConstraints
