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

const StageBounds{T, n} = Union{BoundConstraints{T, n}, AbstractVector{BoundConstraints{T, n}}}

function set_stage_bounds!(destination::Vector{BoundConstraints{T, n}},
        bounds::BoundConstraints{T, n}, name::Symbol) where {T, n}
    fill!(destination, bounds)
    return destination
end

function set_stage_bounds!(destination::Vector{BoundConstraints{T, n}},
        bounds::AbstractVector{BoundConstraints{T, n}}, name::Symbol) where {T, n}
    length(bounds) == length(destination) || throw(DimensionMismatch(
        "$(name) have $(length(bounds)) stages, expected $(length(destination))"))
    copyto!(destination, bounds)
    return destination
end

function set_stage_bounds!(destination::Vector{BoundConstraints{T, n}},
        bounds, name::Symbol) where {T, n}
    throw(ArgumentError("$(name) must be a BoundConstraints{$(T), $(n)} or a vector of them"))
end

function bound_count(bounds::Vector{<:BoundConstraints})
    count = 0
    for stage_bounds in bounds
        count += stage_bounds.nl + stage_bounds.nu
    end
    return count
end
