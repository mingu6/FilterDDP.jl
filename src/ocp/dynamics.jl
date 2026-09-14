RF = RuntimeGeneratedFunction

struct Dynamics{nx, nu, np, F1<:RF, F2<:RF, F3<:RF, F4<:RF, F5<:RF, F6<:RF}
    f::F1
    fx::F2
    fu::F3
    fxx::F4
    fux::F5
    fuu::F6
end

function Dynamics(f::F, nx::Int64, nu::Int64) where F<:Function
    x::Vector{Num} = Symbolics.variables(:x, 1:nx)
    u::Vector{Num} = Symbolics.variables(:u, 1:nu)

    y = Symbolics.simplify(f(x, u))
    λ::Vector{Num} = Symbolics.variables(:λ, 1:nx)  # vector variables for Hessian vector products
    
    fx = Symbolics.simplify(Symbolics.jacobian(y, x))
    fu = Symbolics.simplify(Symbolics.jacobian(y, u))
    f_func = @RuntimeGeneratedFunction(Symbolics.build_function(y, x, u)[1])
    fx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fx, x, u; skipzeros=false)[1])
    fu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fu, x, u; skipzeros=false)[1])
    
    fxx = Symbolics.simplify(Symbolics.hessian(λ' * y, x))
    fux = Symbolics.simplify(Symbolics.jacobian(Symbolics.gradient(λ' * y, u), x))
    fuu = Symbolics.simplify(Symbolics.hessian(λ' * y, u))
    fxx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fxx, x, u, λ; skipzeros=false)[1])
    fux_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fux, x, u, λ; skipzeros=false)[1])
    fuu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fuu, x, u, λ; skipzeros=false)[1])

    return Dynamics{nx, nu, 0, typeof(f_func), typeof(fx_func), typeof(fu_func),
                    typeof(fxx_func), typeof(fux_func), typeof(fuu_func)}(
                        f_func, fx_func, fu_func, fxx_func, fux_func, fuu_func)
end

function Dynamics(f::F, nx::Int64, nu::Int64, np::Int64) where F<:Function
    np > 0 || throw(ArgumentError("parameter dimension must be positive"))
    x::Vector{Num} = Symbolics.variables(:x, 1:nx)
    u::Vector{Num} = Symbolics.variables(:u, 1:nu)
    p::Vector{Num} = Symbolics.variables(:p, 1:np)

    y = Symbolics.simplify(f(x, u, p))
    λ::Vector{Num} = Symbolics.variables(:λ, 1:nx)

    fx = Symbolics.simplify(Symbolics.jacobian(y, x))
    fu = Symbolics.simplify(Symbolics.jacobian(y, u))
    f_func = @RuntimeGeneratedFunction(Symbolics.build_function(y, x, u, p)[1])
    fx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fx, x, u, p; skipzeros=false)[1])
    fu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fu, x, u, p; skipzeros=false)[1])

    fxx = Symbolics.simplify(Symbolics.hessian(λ' * y, x))
    fux = Symbolics.simplify(Symbolics.jacobian(Symbolics.gradient(λ' * y, u), x))
    fuu = Symbolics.simplify(Symbolics.hessian(λ' * y, u))
    fxx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fxx, x, u, p, λ; skipzeros=false)[1])
    fux_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fux, x, u, p, λ; skipzeros=false)[1])
    fuu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fuu, x, u, p, λ; skipzeros=false)[1])

    return Dynamics{nx, nu, np, typeof(f_func), typeof(fx_func), typeof(fu_func),
                    typeof(fxx_func), typeof(fux_func), typeof(fuu_func)}(
                        f_func, fx_func, fu_func, fxx_func, fux_func, fuu_func)
end
