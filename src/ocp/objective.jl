abstract type DiffMethod end
struct Symbolic <: DiffMethod end
struct FD <: DiffMethod end

struct Objective{T, nx, nu}
    l                     # no type means compiled julia/casadi function 
    lx
    lu
    lxx
    lux
    luu
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
        Symbolics._build_function(Symbolics.JuliaTarget(), [l_], x, u; skipzeros=false)[1])
    lx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lx, x, u; skipzeros=false)[1])
    lu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lu, x, u; skipzeros=false)[1])
    lxx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lxx, x, u; skipzeros=false)[1])
    lux_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lux, x, u; skipzeros=false)[1])
    luu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), luu, x, u; skipzeros=false)[1])
    
    return Objective{T, nx, nu}(l_func, lx_func, lu_func, lxx_func, lux_func, luu_func)
end
