struct Dynamics{T, nx, nu}
    f                     # no type means compiled julia/casadi function 
    fx
    fu
    fxx
    fux
    fuu
end

function Dynamics(T, f::Function, nx::Int64, nu::Int64; method::DiffMethod=Symbolic(), quasi_newton::Bool=false)
    return _Dynamics(method, T, f, nx, nu; quasi_newton=quasi_newton)
end

function _Dynamics(method::Symbolic, T, f::Function, nx::Int64, nu::Int64; quasi_newton::Bool=false)
    x = Symbolics.variables(:x, 1:nx)
    u = Symbolics.variables(:u, 1:nu)

    y = Symbolics.simplify(f(x, u))
    λ = Symbolics.variables(:λ, 1:nx)  # vector variables for Hessian vector products
    
    fx = Symbolics.simplify(Symbolics.jacobian(y, x))
    fu = Symbolics.simplify(Symbolics.jacobian(y, u))
    f_func = @RuntimeGeneratedFunction(Symbolics.build_function(y, x, u)[1])
    fx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fx, x, u; skipzeros=false)[1])
    fu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fu, x, u; skipzeros=false)[1])
    if !quasi_newton
        fxx = Symbolics.simplify(Symbolics.hessian(λ' * y, x))
        fux = Symbolics.simplify(Symbolics.jacobian(Symbolics.gradient(λ' * y, u), x))
        fuu = Symbolics.simplify(Symbolics.hessian(λ' * y, u))
        fxx_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), fxx, x, u, λ; skipzeros=false)[1])
        fux_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), fux, x, u, λ; skipzeros=false)[1])
        fuu_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), fuu, x, u, λ; skipzeros=false)[1])
    else
        fxx_func = nothing
        fux_func = nothing
        fuu_func = nothing
    end

    return Dynamics{T, nx, nu}(f_func, fx_func, fu_func, fxx_func, fux_func, fuu_func)
end

# user-provided dynamics and derivatives
function Dynamics(T, f::Function, fx::Function, fu::Function; fxx::Function=nothing, fux::Function=nothing, fuu::Function=nothing)
    return Dynamics{T}(f, fx, fu, fxx, fux, fuu)
end
