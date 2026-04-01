abstract type DiffMethod end
struct Symbolic <: DiffMethod end
struct FD <: DiffMethod end

mutable struct Objective{T, nx, nu}
    l                     # no type means compiled julia/casadi function 
    lx
    lu
    lxx
    lux
    luu
    lx_mem::SVector{nx, T}
    lu_mem::SVector{nu, T}
    lxx_mem::SMatrix{nx, nx, T}
    lux_mem::SMatrix{nu, nx, T}
    luu_mem::SMatrix{nu, nu, T}
end

function Objective(T, l::Function, nx::Int64, nu::Int64; method::DiffMethod=Symbolic())
    return _Objective(method, T, l, nx, nu)
end

function _Objective(method::Symbolic, T, l::Function, nx::Int64, nu::Int64)
    x = Symbolics.variables(:x, 1:nx)
    u = Symbolics.variables(:u, 1:nu)

    l_ = Symbolics.simplify(l(x, u))
    lx = Symbolics.simplify(Symbolics.gradient(l_, x))
    lu = Symbolics.simplify(Symbolics.gradient(l_, u))
    lxx = Symbolics.simplify(Symbolics.jacobian(lx, x))
    lux = Symbolics.simplify(Symbolics.jacobian(lu, x))
    luu = Symbolics.simplify(Symbolics.jacobian(lu, u))

    l_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), [l_], x, u; skipzeros=true)[1])
    lx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lx, x, u; skipzeros=true)[1])
    lu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lu, x, u; skipzeros=true)[1])
    lxx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lxx, x, u; skipzeros=true)[1])
    lux_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lux, x, u; skipzeros=true)[1])
    luu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), luu, x, u; skipzeros=true)[1])

    lx_mem = @SVector zeros(T, nx)
    lu_mem = @SVector zeros(T, nu)
    lxx_mem = @SMatrix zeros(T, nx, nx)
    lux_mem = @SMatrix zeros(T, nu, nx)
    luu_mem = @SMatrix zeros(T, nu, nu)
    
    return Objective{T, nx, nu}(l_func, lx_func, lu_func, lxx_func, lux_func, luu_func,
                                lx_mem, lu_mem, lxx_mem, lux_mem, luu_mem)
end

function objective(objective_vec::Vector{Objective{T, nx, nu}}, ws::FilterDDPWorkspace{T, nx, nu, nc}; mode=:nominal) where {T, nx, nu, nc}
    Jt = [T(0.0)]
    J = 0.0
    if mode == :nominal
        for (objective, wse) in zip(objective_vec, ws)
            J += objective.l(wse.nominal.x, wse.nominal.u)[1]
        end
    else
        for (objective, wse) in zip(objective_vec, ws)
            J += objective.l(wse.current.x, wse.current.u)[1]
        end
    end
    return J
end

function gradients!(objective_vec::Vector{Objective{T, nx, nu}}, ws::FilterDDPWorkspace{T, nx, nu, nc}; mode=:nominal) where {T, nx, nu, nc}
    if mode == :nominal
        for (objective, wse) in zip(objective_vec, ws)
            objective.lx_mem = objective.lx(wse.nominal.x, wse.nominal.u)
            objective.lu_mem = objective.lu(wse.nominal.x, wse.nominal.u)
        end
    else
        for (objective, wse) in zip(objective_vec, ws)
            objective.lx_mem = objective.lx(wse.current.x, wse.current.u)
            objective.lu_mem = objective.lu(wse.current.x, wse.current.u)
        end
    end
end

function hessians!(objective_vec::Vector{Objective{T, nx, nu}}, ws::FilterDDPWorkspace{T, nx, nu, nc}; mode=:nominal) where {T, nx, nu, nc}
    if mode == :nominal
        for (objective, wse) in zip(objective_vec, ws)
            objective.lxx_mem = objective.lxx(wse.nominal.x, wse.nominal.u)
            objective.lux_mem = objective.lux(wse.nominal.x, wse.nominal.u)
            objective.luu_mem = objective.luu(wse.nominal.x, wse.nominal.u)
        end
    else
        for (objective, wse) in zip(objective_vec, ws)
            objective.lxx_mem = objective.lxx(wse.current.x, wse.current.u)
            objective.lux_mem = objective.lux(wse.current.x, wse.current.u)
            objective.luu_mem = objective.luu(wse.current.x, wse.current.u)
        end
    end
end
