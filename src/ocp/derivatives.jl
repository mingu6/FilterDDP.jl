abstract type DerivativeBackend end

struct SymbolicsBackend <: DerivativeBackend end
struct ForwardDiffBackend <: DerivativeBackend end

abstract type AbstractStageObjective{nx, nu, np} end
abstract type AbstractTerminalObjective{nx, np} end
abstract type AbstractDynamics{nx, nu, np} end
abstract type AbstractEqualityConstraints{nx, nu, nh, np} end
abstract type AbstractInequalityConstraints{T, nx, nu, ng, np} end

struct StateControlFunction{nx, nu, F, P}
    f::F
    parameters::P
end

@inline function (wrapped::StateControlFunction{nx, nu})(z) where {nx, nu}
    x = SVector{nx}(ntuple(index -> z[index], Val(nx)))
    u = SVector{nu}(ntuple(index -> z[nx + index], Val(nu)))
    return evaluate(wrapped.f, x, u, wrapped.parameters)
end

struct StateFunction{nx, F, P}
    f::F
    parameters::P
end

@inline function (wrapped::StateFunction{nx})(x) where nx
    return evaluate(wrapped.f, x, wrapped.parameters) + zero(eltype(x))
end

struct ScalarOutput{F}
    f::F
end

@inline function (wrapped::ScalarOutput)(x)
    return wrapped.f(x) + zero(eltype(x))
end

struct StaticOutput{ny, F}
    f::F
end

@inline function (wrapped::StaticOutput{ny})(x) where ny
    return SVector{ny, eltype(x)}(wrapped.f(x))
end

struct WeightedOutput{F, M}
    f::F
    multiplier::M
end

@inline function (wrapped::WeightedOutput)(x)
    return dot(wrapped.multiplier, wrapped.f(x))
end

@inline function combine_state_control(
        x::SVector{nx, T}, u::SVector{nu, T}) where {nx, nu, T}
    return SVector{nx + nu, T}(x..., u...)
end

@inline state_indices(::Val{nx}) where nx =
    SVector{nx}(ntuple(identity, Val(nx)))

@inline control_indices(::Val{nx}, ::Val{nu}) where {nx, nu} =
    SVector{nu}(ntuple(index -> nx + index, Val(nu)))

struct EmptyConstraint end

@inline (::EmptyConstraint)(x, u) = SVector{0, eltype(x)}()
@inline (::EmptyConstraint)(x, u, parameters) = SVector{0, eltype(x)}()

function infer_output_dimension(f, nx::Int, nu::Int, np::Int)
    x = zero(SVector{nx, Float64})
    u = zero(SVector{nu, Float64})
    parameters = zero(SVector{np, Float64})
    return length(evaluate(f, x, u, parameters))
end

@inline function forwarddiff_vector_value(
        f, ::Val{ny}, x, u, parameters) where ny
    return SVector{ny}(evaluate(f, x, u, parameters))
end

@inline function forwarddiff_vector_first_order(
        f, ::Val{ny}, x::SVector{nx, T}, u::SVector{nu, T},
        parameters) where {T, ny, nx, nu}
    z = combine_state_control(x, u)
    base = StateControlFunction{nx, nu, typeof(f), typeof(parameters)}(
        f, parameters)
    wrapped = StaticOutput{ny, typeof(base)}(base)
    jacobian = ForwardDiff.jacobian(wrapped, z)
    x_indices = state_indices(Val(nx))
    u_indices = control_indices(Val(nx), Val(nu))
    yx = jacobian[:, x_indices]
    yu = jacobian[:, u_indices]
    return (; yx, yu)
end

@inline function forwarddiff_vector_derivatives(
        f, ::Val{ny}, x::SVector{nx, T}, u::SVector{nu, T}, parameters,
        multiplier::SVector{ny, T}) where {T, ny, nx, nu}
    z = combine_state_control(x, u)
    base = StateControlFunction{nx, nu, typeof(f), typeof(parameters)}(
        f, parameters)
    wrapped = StaticOutput{ny, typeof(base)}(base)
    initial_value = zero(SVector{ny, T})
    result = DiffResults.JacobianResult(initial_value, z)
    result = ForwardDiff.jacobian!(result, wrapped, z)
    contracted = WeightedOutput{typeof(wrapped), typeof(multiplier)}(
        wrapped, multiplier)
    hessian = ForwardDiff.hessian(contracted, z)
    jacobian = DiffResults.jacobian(result)
    x_indices = state_indices(Val(nx))
    u_indices = control_indices(Val(nx), Val(nu))
    value = DiffResults.value(result)
    yx = jacobian[:, x_indices]
    yu = jacobian[:, u_indices]
    yxx = hessian[x_indices, x_indices]
    yux = hessian[u_indices, x_indices]
    yuu = hessian[u_indices, u_indices]
    return (; value, yx, yu, yxx, yux, yuu)
end