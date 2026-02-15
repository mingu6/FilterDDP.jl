struct TrajectoryElement{T}
    x::Vector{T}
    u::Vector{T}
    c::Vector{T}
    ul::Vector{T}       # value of lower control limits ineq. constraint <- TODO: remove
    uu::Vector{T}       # value of lower control limits ineq. constraint <- TODO: remove
    ϕ::Vector{T}
    zl::Vector{T}
    zu::Vector{T}
    λ::Vector{T}
end

Trajectories{T} = Vector{TrajectoryElement{T}} where T

function TrajectoryElement(T, nx::Int, nu::Int, nc::Int)
    TrajectoryElement{T}(
        zeros(T, nx),
        zeros(T, nu),
        zeros(T, nc),
        zeros(T, nu),
        zeros(T, nu),
        zeros(T, nc),
        zeros(T, nu),
        zeros(T, nu),
        zeros(T, nx))
end

struct FilterDDPWorkspaceElement{T}
    current::TrajectoryElement{T}
    nominal::TrajectoryElement{T}
    eq_update_params::Matrix{T}    # [α β; ψ ω]
    ineq_update_params::Matrix{T}  # [χ ζ]
    Qû::Vector{T}
    V̂xx::Matrix{T}
    # temporary storage for intermediate computations
    x_tmp::Vector{T}
    u_tmp1::Vector{T}
    u_tmp2::Vector{T}
    xx_tmp::Matrix{T}
	ux_tmp::Matrix{T}
	uu_tmp::Matrix{T}
    kkt_mat::Matrix{T}
    kkt_mat_ws::BunchKaufmanWs{T}
    kkt_D_cache::Pair{Vector{T}}
end

FilterDDPWorkspace{T} = Vector{FilterDDPWorkspaceElement{T}} where T

function update_nominal_trajectory!(ws::FilterDDPWorkspace{T}) where T
    for wse in ws
        wse.nominal.x .= wse.current.x
        wse.nominal.u .= wse.current.u
        wse.nominal.c .= wse.current.c
        wse.nominal.ul .= wse.current.ul
        wse.nominal.uu .= wse.current.uu
        wse.nominal.ϕ .= wse.current.ϕ
        wse.nominal.zl .= wse.current.zl
        wse.nominal.zu .= wse.current.zu
        wse.nominal.λ .= wse.current.λ
    end
    return nothing
end
