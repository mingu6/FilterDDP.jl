RFN = Union{RuntimeGeneratedFunction, Nothing}

@inline symbolic_constraint_value(constraint, x, u, ::Val{0}) =
    (collect(constraint(x, u)), ())

@inline function symbolic_constraint_value(constraint, x, u, ::Val{np}) where np
    np > 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    p::Vector{Num} = Symbolics.variables(:p, 1:np)
    return collect(constraint(x, u, p)), (p,)
end

function build_symbolic_constraint_functions(value, x, u,
        multiplier_name::Symbol, inputs...)
    value_x = Symbolics.simplify(Symbolics.jacobian(value, x))
    value_u = Symbolics.simplify(Symbolics.jacobian(value, u))
    count = length(value)

    if count > 0
        value_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value, x, u, inputs...)[1])
        value_x_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_x, x, u, inputs...)[1])
        value_u_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_u, x, u, inputs...)[1])

        multiplier::Vector{Num} = Symbolics.variables(multiplier_name, 1:count)
        value_xx = Symbolics.simplify(Symbolics.hessian(value' * multiplier, x))
        value_ux = Symbolics.simplify(Symbolics.jacobian(value_u' * multiplier, x))
        value_uu = Symbolics.simplify(Symbolics.hessian(value' * multiplier, u))
        value_xx_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_xx, x, u, inputs..., multiplier)[1])
        value_ux_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_ux, x, u, inputs..., multiplier; skipzeros=false)[1])
        value_uu_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_uu, x, u, inputs..., multiplier; skipzeros=false)[1])
    else
        value_func = nothing
        value_x_func = nothing
        value_u_func = nothing
        value_xx_func = nothing
        value_ux_func = nothing
        value_uu_func = nothing
    end

    return value_func, value_x_func, value_u_func,
        value_xx_func, value_ux_func, value_uu_func, count
end

struct SymbolicsEqualityConstraints{nx, nu, nh, np, F1<:RFN, F2<:RFN, F3<:RFN, F4<:RFN, F5<:RFN, F6<:RFN} <:
    AbstractEqualityConstraints{nx, nu, nh, np}
    h::F1
    hx::F2
    hu::F3
    hxx::F4
    hux::F5
    huu::F6
end

struct ForwardDiffEqualityConstraints{nx, nu, nh, np, F} <:
        AbstractEqualityConstraints{nx, nu, nh, np}
    h::F
end

# Public equality constraint constructors.
function EqualityConstraints(h::F, nx::Int64, nu::Int64, np::Int64=0;
        backend::DerivativeBackend=SymbolicsBackend(),
        output_dimension::Union{Nothing, Int}=nothing) where F<:Function
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    return EqualityConstraints(
        backend, h, nx, nu, Val(np); output_dimension=output_dimension)
end

function EqualityConstraints(nx::Int64, nu::Int64, np::Int64=0;
        backend::DerivativeBackend=SymbolicsBackend())
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    return EqualityConstraints(backend, nx, nu, Val(np))
end

function EqualityConstraints(::SymbolicsBackend, h::F, nx::Int64, nu::Int64,
        parameter_count::Val{np};
        output_dimension::Union{Nothing, Int}=nothing) where {F<:Function, np}
    x::Vector{Num} = Symbolics.variables(:x, 1:nx)
    u::Vector{Num} = Symbolics.variables(:u, 1:nu)
    value, inputs = symbolic_constraint_value(h, x, u, parameter_count)
    h_func, hx_func, hu_func, hxx_func, hux_func, huu_func, nh =
        build_symbolic_constraint_functions(
            Symbolics.simplify(value), x, u, :ϕ, inputs...)
    isnothing(output_dimension) || output_dimension == nh ||
        throw(DimensionMismatch("expected $(output_dimension) equalities, got $(nh)"))
    return SymbolicsEqualityConstraints{nx, nu, nh, np, typeof(h_func), typeof(hx_func), typeof(hu_func),
        typeof(hxx_func), typeof(hux_func), typeof(huu_func)}(
        h_func, hx_func, hu_func, hxx_func, hux_func, huu_func)
end

function EqualityConstraints(::ForwardDiffBackend, h::F, nx::Int64, nu::Int64,
        ::Val{np}; output_dimension::Union{Nothing, Int}=nothing) where {F<:Function, np}
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    nh = isnothing(output_dimension) ? infer_output_dimension(h, nx, nu, np) : output_dimension
    nh >= 0 || throw(ArgumentError("output dimension must be nonnegative"))
    return ForwardDiffEqualityConstraints{nx, nu, nh, np, F}(h)
end

function EqualityConstraints(::SymbolicsBackend, nx::Int64, nu::Int64,
        ::Val{np}) where np
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    return SymbolicsEqualityConstraints{nx, nu, 0, np, Nothing, Nothing,
        Nothing, Nothing, Nothing, Nothing}(
        nothing, nothing, nothing, nothing, nothing, nothing)
end

function EqualityConstraints(::ForwardDiffBackend, nx::Int64, nu::Int64,
        ::Val{np}) where np
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    return ForwardDiffEqualityConstraints{nx, nu, 0, np, EmptyConstraint}(
        EmptyConstraint())
end

struct SymbolicsInequalityConstraints{T, nx, nu, ng, np, F1<:RFN, F2<:RFN, F3<:RFN, F4<:RFN, F5<:RFN, F6<:RFN} <:
        AbstractInequalityConstraints{T, nx, nu, ng, np}
    g::F1
    gx::F2
    gu::F3
    gxx::F4
    gux::F5
    guu::F6
    bounds::BoundConstraints{T, ng}
end

struct ForwardDiffInequalityConstraints{T, nx, nu, ng, np, F} <:
    AbstractInequalityConstraints{T, nx, nu, ng, np}
    g::F
    bounds::BoundConstraints{T, ng}
end

# Public inequality constraint constructors.
function InequalityConstraints(g::F, nx::Int64, nu::Int64,
    bounds::BoundConstraints{T, ng}, np::Int64=0;
    backend::DerivativeBackend=SymbolicsBackend()) where {F<:Function, T, ng}
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    return InequalityConstraints(backend, g, nx, nu, bounds, Val(np))
end

function InequalityConstraints(::SymbolicsBackend, g::F, nx::Int64, nu::Int64,
    bounds::BoundConstraints{T, ng}, parameter_count::Val{np}) where {F<:Function, T, ng, np}
    x::Vector{Num} = Symbolics.variables(:x, 1:nx)
    u::Vector{Num} = Symbolics.variables(:u, 1:nu)
    value, inputs = symbolic_constraint_value(g, x, u, parameter_count)
    g_func, gx_func, gu_func, gxx_func, gux_func, guu_func, constraint_count =
        build_symbolic_constraint_functions(
            Symbolics.simplify(value), x, u, :ν, inputs...)
    @assert constraint_count == ng
    return SymbolicsInequalityConstraints{T, nx, nu, ng, np,
        typeof(g_func), typeof(gx_func), typeof(gu_func),
        typeof(gxx_func), typeof(gux_func), typeof(guu_func)}(
        g_func, gx_func, gu_func, gxx_func, gux_func, guu_func, bounds)
end

function InequalityConstraints(::ForwardDiffBackend, g::F, nx::Int64, nu::Int64,
        bounds::BoundConstraints{T, ng}, ::Val{np}) where {F<:Function, T, ng, np}
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    return ForwardDiffInequalityConstraints{T, nx, nu, ng, np, F}(g, bounds)
end

function InequalityConstraints(g::F, nx::Int64, nu::Int64,
        l::SVector{ng, T}, u::SVector{ng, T}, np::Int64=0;
        backend::DerivativeBackend=SymbolicsBackend()) where {F<:Function, T, ng}
    return InequalityConstraints(
        g, nx, nu, BoundConstraints(l, u), np; backend=backend)
end

function InequalityConstraints(::Type{T}, nx::Int64, nu::Int64, np::Int64=0;
        backend::DerivativeBackend=SymbolicsBackend()) where T
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    return InequalityConstraints(backend, T, nx, nu, Val(np))
end

function InequalityConstraints(::SymbolicsBackend, ::Type{T},
        nx::Int64, nu::Int64, ::Val{np}) where {T, np}
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    bounds = BoundConstraints(SVector{0, T}(), SVector{0, T}())
    return SymbolicsInequalityConstraints{T, nx, nu, 0, np, Nothing, Nothing,
        Nothing, Nothing, Nothing, Nothing}(
        nothing, nothing, nothing, nothing, nothing, nothing, bounds)
end

function InequalityConstraints(::ForwardDiffBackend, ::Type{T},
        nx::Int64, nu::Int64, ::Val{np}) where {T, np}
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    bounds = BoundConstraints(SVector{0, T}(), SVector{0, T}())
    return ForwardDiffInequalityConstraints{T, nx, nu, 0, np, EmptyConstraint}(
        EmptyConstraint(), bounds)
end

@inline constraint_function(constraints::SymbolicsEqualityConstraints) = constraints.h
@inline constraint_function(constraints::ForwardDiffEqualityConstraints) = constraints.h
@inline constraint_function(constraints::SymbolicsInequalityConstraints) = constraints.g
@inline constraint_function(constraints::ForwardDiffInequalityConstraints) = constraints.g

@inline function component_value(
        constraints::Union{SymbolicsEqualityConstraints, SymbolicsInequalityConstraints},
        x, u, parameters)
    return evaluate(constraint_function(constraints), x, u, parameters)
end

@inline function component_value(
        constraints::AbstractEqualityConstraints{nx, nu, nh},
        x, u, parameters) where {nx, nu, nh}
    return forwarddiff_vector_value(
        constraint_function(constraints), Val(nh), x, u, parameters)
end

@inline function component_value(
        constraints::AbstractInequalityConstraints{T, nx, nu, ng},
        x, u, parameters) where {T, nx, nu, ng}
    return forwarddiff_vector_value(
        constraint_function(constraints), Val(ng), x, u, parameters)
end

@inline function component_first_order(
    constraints::SymbolicsEqualityConstraints, x, u, parameters)
    yx = evaluate(constraints.hx, x, u, parameters)
    yu = evaluate(constraints.hu, x, u, parameters)
    return (; yx, yu)
end

@inline function component_first_order(
    constraints::SymbolicsInequalityConstraints, x, u, parameters)
    yx = evaluate(constraints.gx, x, u, parameters)
    yu = evaluate(constraints.gu, x, u, parameters)
    return (; yx, yu)
end

@inline function component_first_order(
        constraints::ForwardDiffEqualityConstraints{nx, nu, nh},
        x, u, parameters) where {nx, nu, nh}
    return forwarddiff_vector_first_order(
        constraints.h, Val(nh), x, u, parameters)
end

@inline function component_first_order(
        constraints::ForwardDiffInequalityConstraints{T, nx, nu, ng},
        x, u, parameters) where {T, nx, nu, ng}
    return forwarddiff_vector_first_order(
        constraints.g, Val(ng), x, u, parameters)
end

@inline function component_derivatives(
    constraints::SymbolicsEqualityConstraints, x, u, parameters, multiplier)
    value = evaluate(constraints.h, x, u, parameters)
    yx = evaluate(constraints.hx, x, u, parameters)
    yu = evaluate(constraints.hu, x, u, parameters)
    yxx = evaluate(constraints.hxx, x, u, parameters, multiplier)
    yux = evaluate(constraints.hux, x, u, parameters, multiplier)
    yuu = evaluate(constraints.huu, x, u, parameters, multiplier)
    return (; value, yx, yu, yxx, yux, yuu)
end

@inline function component_derivatives(
    constraints::SymbolicsInequalityConstraints, x, u, parameters, multiplier)
    value = evaluate(constraints.g, x, u, parameters)
    yx = evaluate(constraints.gx, x, u, parameters)
    yu = evaluate(constraints.gu, x, u, parameters)
    yxx = evaluate(constraints.gxx, x, u, parameters, multiplier)
    yux = evaluate(constraints.gux, x, u, parameters, multiplier)
    yuu = evaluate(constraints.guu, x, u, parameters, multiplier)
    return (; value, yx, yu, yxx, yux, yuu)
end

@inline function component_derivatives(
        constraints::ForwardDiffEqualityConstraints{nx, nu, nh},
        x, u, parameters, multiplier) where {nx, nu, nh}
    return forwarddiff_vector_derivatives(
        constraints.h, Val(nh), x, u, parameters, multiplier)
end

@inline function component_derivatives(
        constraints::ForwardDiffInequalityConstraints{T, nx, nu, ng},
        x, u, parameters, multiplier) where {T, nx, nu, ng}
    return forwarddiff_vector_derivatives(
        constraints.g, Val(ng), x, u, parameters, multiplier)
end
