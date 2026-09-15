RF = RuntimeGeneratedFunction

struct SymbolicsDynamics{nx, nu, np, F1<:RF, F2<:RF, F3<:RF, F4<:RF, F5<:RF, F6<:RF} <:
    AbstractDynamics{nx, nu, np}
    f::F1
    fx::F2
    fu::F3
    fxx::F4
    fux::F5
    fuu::F6
end

struct ForwardDiffDynamics{nx, nu, np, F} <: AbstractDynamics{nx, nu, np}
    f::F
end

# Public dynamics constructor.
function Dynamics(f::F, nx::Int64, nu::Int64, np::Int64=0;
        backend::DerivativeBackend=SymbolicsBackend()) where F<:Function
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    return Dynamics(backend, f, nx, nu, Val(np))
end

@inline symbolic_dynamics_value(f, x, u, ::Val{0}) = (collect(f(x, u)), ())

@inline function symbolic_dynamics_value(f, x, u, ::Val{np}) where np
    np > 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    p::Vector{Num} = Symbolics.variables(:p, 1:np)
    return collect(f(x, u, p)), (p,)
end

function build_symbolic_dynamics(::Val{nx}, ::Val{nu}, ::Val{np}, y, x, u,
        inputs...) where {nx, nu, np}
    λ::Vector{Num} = Symbolics.variables(:λ, 1:nx)

    fx = Symbolics.simplify(Symbolics.jacobian(y, x))
    fu = Symbolics.simplify(Symbolics.jacobian(y, u))
    f_func = @RuntimeGeneratedFunction(
        Symbolics.build_function(y, x, u, inputs...)[1])
    fx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fx, x, u, inputs...; skipzeros=false)[1])
    fu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fu, x, u, inputs...; skipzeros=false)[1])

    fxx = Symbolics.simplify(Symbolics.hessian(λ' * y, x))
    fux = Symbolics.simplify(Symbolics.jacobian(Symbolics.gradient(λ' * y, u), x))
    fuu = Symbolics.simplify(Symbolics.hessian(λ' * y, u))
    fxx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fxx, x, u, inputs..., λ; skipzeros=false)[1])
    fux_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fux, x, u, inputs..., λ; skipzeros=false)[1])
    fuu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fuu, x, u, inputs..., λ; skipzeros=false)[1])

    return SymbolicsDynamics{nx, nu, np, typeof(f_func), typeof(fx_func), typeof(fu_func),
                    typeof(fxx_func), typeof(fux_func), typeof(fuu_func)}(
                        f_func, fx_func, fu_func, fxx_func, fux_func, fuu_func)
end

function Dynamics(::SymbolicsBackend, f::F, nx::Int64, nu::Int64,
        parameter_count::Val{np}) where {F<:Function, np}
    x::Vector{Num} = Symbolics.variables(:x, 1:nx)
    u::Vector{Num} = Symbolics.variables(:u, 1:nu)
    value, inputs = symbolic_dynamics_value(f, x, u, parameter_count)
    return build_symbolic_dynamics(Val(nx), Val(nu), parameter_count,
        Symbolics.simplify(value), x, u, inputs...)
end

function Dynamics(::ForwardDiffBackend, f::F, nx::Int64, nu::Int64,
        ::Val{np}) where {F<:Function, np}
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    return ForwardDiffDynamics{nx, nu, np, F}(f)
end

@inline function component_value(dynamics::SymbolicsDynamics, x, u, parameters)
    return evaluate(dynamics.f, x, u, parameters)
end

@inline function component_value(
        dynamics::ForwardDiffDynamics{nx}, x, u, parameters) where nx
    return forwarddiff_vector_value(dynamics.f, Val(nx), x, u, parameters)
end

@inline function component_first_order(
    dynamics::SymbolicsDynamics, x, u, parameters)
    yx = evaluate(dynamics.fx, x, u, parameters)
    yu = evaluate(dynamics.fu, x, u, parameters)
    return (; yx, yu)
end

@inline function component_first_order(
        dynamics::ForwardDiffDynamics{nx, nu}, x::SVector{nx, T},
        u::SVector{nu, T}, parameters) where {T, nx, nu}
    return forwarddiff_vector_first_order(
        dynamics.f, Val(nx), x, u, parameters)
end

@inline function component_derivatives(
    dynamics::SymbolicsDynamics, x, u, parameters, multiplier)
    yx = evaluate(dynamics.fx, x, u, parameters)
    yu = evaluate(dynamics.fu, x, u, parameters)
    yxx = evaluate(dynamics.fxx, x, u, parameters, multiplier)
    yux = evaluate(dynamics.fux, x, u, parameters, multiplier)
    yuu = evaluate(dynamics.fuu, x, u, parameters, multiplier)
    return (; yx, yu, yxx, yux, yuu)
end

@inline function component_derivatives(
        dynamics::ForwardDiffDynamics{nx, nu}, x::SVector{nx, T},
        u::SVector{nu, T}, parameters,
        multiplier::SVector{nx, T}) where {T, nx, nu}
    derivatives = forwarddiff_vector_derivatives(
        dynamics.f, Val(nx), x, u, parameters, multiplier)
    return (; derivatives.yx, derivatives.yu, derivatives.yxx,
        derivatives.yux, derivatives.yuu)
end
