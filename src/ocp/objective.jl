abstract type DiffMethod end
struct Symbolic <: DiffMethod end
struct FD <: DiffMethod end

struct Objective{T}
    nx::Int
    nu::Int
    l                     # no type means compiled julia/casadi function 
    lx
    lu
    lxx
    lux
    luu
    lx_mem::Vector{T}
    lu_mem::Vector{T}
    lxx_mem::Matrix{T}
    lux_mem::Matrix{T}
    luu_mem::Matrix{T}
end

function Objective(T, l::Function, nx::Int, nu::Int; method::DiffMethod=Symbolic())
    return _Objective(method, T, l, nx, nu)
end

function _Objective(method::Symbolic, T, l::Function, nx::Int, nu::Int)
    x = Symbolics.variables(:x, 1:nx)
    u = Symbolics.variables(:u, 1:nu)

    l_ = Symbolics.simplify(l(x, u))
    lx = Symbolics.simplify(Symbolics.gradient(l_, x))
    lu = Symbolics.simplify(Symbolics.gradient(l_, u))
    lxx = Symbolics.simplify(Symbolics.jacobian(lx, x))
    lux = Symbolics.simplify(Symbolics.jacobian(lu, x))
    luu = Symbolics.simplify(Symbolics.jacobian(lu, u))

    l_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), [l_], x, u; skipzeros=true)[2])
    lx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lx, x, u; skipzeros=true)[2])
    lu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lu, x, u; skipzeros=true)[2])
    lxx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lxx, x, u; skipzeros=true)[2])
    lux_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lux, x, u; skipzeros=true)[2])
    luu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), luu, x, u; skipzeros=true)[2])

    lx_mem = zeros(T, nx)
    lu_mem = zeros(T, nu)
    lxx_mem = zeros(T, nx, nx)
    lux_mem = zeros(T, nu, nx)
    luu_mem = zeros(T, nu, nu)
    
    return Objective{T}(nx, nu, l_func, lx_func, lu_func, lxx_func, lux_func, luu_func,
                        lx_mem, lu_mem, lxx_mem, lux_mem, luu_mem)
end

function objective(objective_vec::Vector{Objective{T}}, ws::FilterDDPWorkspace{T}; mode=:nominal) where T
    Jt = [T(0.0)]
    J = 0.0
    if mode == :nominal
        for (objective, wse) in zip(objective_vec, ws)
            objective.l(Jt, wse.nominal.x, wse.nominal.u)
            J += Jt[1]
        end
    else
        for (objective, wse) in zip(objective_vec, ws)
            objective.l(Jt, wse.current.x, wse.current.u)
            J += Jt[1]
        end
    end
    return J
end

function gradients!(objective_vec::Vector{Objective{T}}, ws::FilterDDPWorkspace{T}; mode=:nominal) where T
    if mode == :nominal
        for (objective, wse) in zip(objective_vec, ws)
            objective.lx(objective.lx_mem, wse.nominal.x, wse.nominal.u)
            objective.lu(objective.lu_mem, wse.nominal.x, wse.nominal.u)
        end
    else
        for (objective, wse) in zip(objective_vec, ws)
            objective.lx(objective.lx_mem, wse.current.x, wse.current.u)
            objective.lu(objective.lu_mem, wse.current.x, wse.current.u)
        end
    end
end

function hessians!(objective_vec::Vector{Objective{T}}, ws::FilterDDPWorkspace{T}; mode=:nominal) where T
    if mode == :nominal
        for (objective, wse) in zip(objective_vec, ws)
            objective.lxx(objective.lxx_mem, wse.nominal.x, wse.nominal.u)
            objective.lux(objective.lux_mem, wse.nominal.x, wse.nominal.u)
            objective.luu(objective.luu_mem, wse.nominal.x, wse.nominal.u)
        end
    else
        for (objective, wse) in zip(objective_vec, ws)
            objective.lxx(objective.lxx_mem, wse.current.x, wse.current.u)
            objective.lux(objective.lux_mem, wse.current.x, wse.current.u)
            objective.luu(objective.luu_mem, wse.current.x, wse.current.u)
        end
    end
end

function reset_mem!(objective_vec::Vector{Objective{T}}) where T 
    for objective in objective_vec
        fill!(objective.lx_mem, 0.0) 
        fill!(objective.lu_mem, 0.0) 
        fill!(objective.lxx_mem, 0.0) 
        fill!(objective.lux_mem, 0.0) 
        fill!(objective.luu_mem, 0.0) 
    end 
end
