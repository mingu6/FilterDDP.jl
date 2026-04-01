mutable struct TrajectoryElement{T, nx, nu, nc}
    x::SVector{nx, T}
    u::SVector{nu, T}
    c::SVector{nc, T}
    ul::SVector{nu, T}       # value of lower control limits ineq. constraint <- TODO: remove
    uu::SVector{nu, T}       # value of lower control limits ineq. constraint <- TODO: remove
    ϕ::SVector{nc, T}
    zl::SVector{nu, T}
    zu::SVector{nu, T}
    λ::SVector{nx, T}
end

Trajectories{T, nx, nu, ncnx,} = Vector{TrajectoryElement{T, nx, nu, nc}} where {T, nx, nu, nc}

function TrajectoryElement(T, nx::Int, nu::Int, nc::Int)
    TrajectoryElement{T, nx, nu, nc}(
        SVector{nx, T}(zeros(T, nx)),
        SVector{nu, T}(zeros(T, nu)),
        SVector{nc, T}(zeros(T, nc)),
        SVector{nu, T}(zeros(T, nu)),
        SVector{nu, T}(zeros(T, nu)),
        SVector{nc, T}(zeros(T, nc)),
        SVector{nu, T}(zeros(T, nu)),
        SVector{nu, T}(zeros(T, nu)),
        SVector{nx, T}(zeros(T, nx)))
end

mutable struct FilterDDPWorkspaceElement{T, nx, nu, nc}
    current::TrajectoryElement{T, nx, nu, nc}
    nominal::TrajectoryElement{T, nx, nu, nc}
    α::SVector{nu, T}
    ψ::SVector{nc, T}
    β::SMatrix{nu, nx, T}
    ω::SMatrix{nc, nx, T}
    χl::SVector{nu, T}
    χu::SVector{nu, T}
    ζl::SMatrix{nu, nx, T}
    ζu::SMatrix{nu, nx, T}
    Qû::SVector{nu, T}
    V̂x::SVector{nx, T}
    V̂xx::SMatrix{nx, nx, T}
end

FilterDDPWorkspace{T, nx, nu, nc} = Vector{FilterDDPWorkspaceElement{T, nx, nu, nc}} where {T, nx, nu, nc}

function update_nominal_trajectory!(ws::FilterDDPWorkspace{T, nx, nu, nc}) where {T, nx, nu, nc}
    for wse in ws
        wse.nominal.x = wse.current.x
        wse.nominal.u = wse.current.u
        wse.nominal.c = wse.current.c
        wse.nominal.ul = wse.current.ul
        wse.nominal.uu = wse.current.uu
        wse.nominal.ϕ = wse.current.ϕ
        wse.nominal.zl = wse.current.zl
        wse.nominal.zu = wse.current.zu
        wse.nominal.λ = wse.current.λ
    end
    return nothing
end
