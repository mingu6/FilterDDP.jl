RFN = Union{RuntimeGeneratedFunction, Nothing}

function build_constraint_functions(constraint::F, nx::Int64, nu::Int64,
        multiplier_name::Symbol) where F<:Function
    x::Vector{Num} = Symbolics.variables(:x, 1:nx)
    u::Vector{Num} = Symbolics.variables(:u, 1:nu)

    value = Symbolics.simplify(constraint(x, u))
    value_x = Symbolics.simplify(Symbolics.jacobian(value, x))
    value_u = Symbolics.simplify(Symbolics.jacobian(value, u))
    count = length(value)

    if count > 0
        value_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value, x, u)[1])
        value_x_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_x, x, u)[1])
        value_u_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_u, x, u)[1])

        multiplier::Vector{Num} = Symbolics.variables(multiplier_name, 1:count)
        value_xx = Symbolics.simplify(Symbolics.hessian(value' * multiplier, x))
        value_ux = Symbolics.simplify(Symbolics.jacobian(value_u' * multiplier, x))
        value_uu = Symbolics.simplify(Symbolics.hessian(value' * multiplier, u))
        value_xx_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_xx, x, u, multiplier)[1])
        value_ux_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_ux, x, u, multiplier; skipzeros=false)[1])
        value_uu_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_uu, x, u, multiplier; skipzeros=false)[1])
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

function build_constraint_functions(constraint::F, nx::Int64, nu::Int64,
        np::Int64, multiplier_name::Symbol) where F<:Function
    np > 0 || throw(ArgumentError("parameter dimension must be positive"))
    x::Vector{Num} = Symbolics.variables(:x, 1:nx)
    u::Vector{Num} = Symbolics.variables(:u, 1:nu)
    p::Vector{Num} = Symbolics.variables(:p, 1:np)

    value = Symbolics.simplify(constraint(x, u, p))
    value_x = Symbolics.simplify(Symbolics.jacobian(value, x))
    value_u = Symbolics.simplify(Symbolics.jacobian(value, u))
    count = length(value)

    if count > 0
        value_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value, x, u, p)[1])
        value_x_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_x, x, u, p)[1])
        value_u_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_u, x, u, p)[1])

        multiplier::Vector{Num} = Symbolics.variables(multiplier_name, 1:count)
        value_xx = Symbolics.simplify(Symbolics.hessian(value' * multiplier, x))
        value_ux = Symbolics.simplify(Symbolics.jacobian(value_u' * multiplier, x))
        value_uu = Symbolics.simplify(Symbolics.hessian(value' * multiplier, u))
        value_xx_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_xx, x, u, p, multiplier)[1])
        value_ux_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_ux, x, u, p, multiplier; skipzeros=false)[1])
        value_uu_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), value_uu, x, u, p, multiplier; skipzeros=false)[1])
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

struct EqualityConstraints{nx, nu, nh, np, F1<:RFN, F2<:RFN, F3<:RFN, F4<:RFN, F5<:RFN, F6<:RFN}
    h::F1
    hx::F2
    hu::F3
    hxx::F4
    hux::F5
    huu::F6
end

function EqualityConstraints(h::F, nx::Int64, nu::Int64) where F<:Function
    h_func, hx_func, hu_func, hxx_func, hux_func, huu_func, nh =
        build_constraint_functions(h, nx, nu, :ϕ)
    return EqualityConstraints{nx, nu, nh, 0, typeof(h_func), typeof(hx_func), typeof(hu_func),
        typeof(hxx_func), typeof(hux_func), typeof(huu_func)}(
        h_func, hx_func, hu_func, hxx_func, hux_func, huu_func)
end

function EqualityConstraints(h::F, nx::Int64, nu::Int64, np::Int64) where F<:Function
    h_func, hx_func, hu_func, hxx_func, hux_func, huu_func, nh =
        build_constraint_functions(h, nx, nu, np, :ϕ)
    return EqualityConstraints{nx, nu, nh, np, typeof(h_func), typeof(hx_func), typeof(hu_func),
        typeof(hxx_func), typeof(hux_func), typeof(huu_func)}(
        h_func, hx_func, hu_func, hxx_func, hux_func, huu_func)
end

function EqualityConstraints(nx::Int64, nu::Int64)
    return EqualityConstraints((x, u) -> [], nx, nu)
end

function EqualityConstraints(nx::Int64, nu::Int64, np::Int64)
    return EqualityConstraints((x, u, p) -> [], nx, nu, np)
end

struct InequalityConstraints{T, nx, nu, ng, np, F1<:RFN, F2<:RFN, F3<:RFN, F4<:RFN, F5<:RFN, F6<:RFN}
    g::F1
    gx::F2
    gu::F3
    gxx::F4
    gux::F5
    guu::F6
    bounds::BoundConstraints{T, ng}
end

function InequalityConstraints(g::F, nx::Int64, nu::Int64,
        bounds::BoundConstraints{T, ng}) where {F<:Function, T, ng}
    g_func, gx_func, gu_func, gxx_func, gux_func, guu_func, constraint_count =
        build_constraint_functions(g, nx, nu, :ν)
    @assert constraint_count == ng
    return InequalityConstraints{T, nx, nu, ng, 0,
        typeof(g_func), typeof(gx_func), typeof(gu_func),
        typeof(gxx_func), typeof(gux_func), typeof(guu_func)}(
        g_func, gx_func, gu_func, gxx_func, gux_func, guu_func, bounds)
end

function InequalityConstraints(g::F, nx::Int64, nu::Int64,
        bounds::BoundConstraints{T, ng}, np::Int64) where {F<:Function, T, ng}
    g_func, gx_func, gu_func, gxx_func, gux_func, guu_func, constraint_count =
        build_constraint_functions(g, nx, nu, np, :ν)
    @assert constraint_count == ng
    return InequalityConstraints{T, nx, nu, ng, np,
        typeof(g_func), typeof(gx_func), typeof(gu_func),
        typeof(gxx_func), typeof(gux_func), typeof(guu_func)}(
        g_func, gx_func, gu_func, gxx_func, gux_func, guu_func, bounds)
end

function InequalityConstraints(g::F, nx::Int64, nu::Int64,
        l::SVector{ng, T}, u::SVector{ng, T}) where {F<:Function, T, ng}
    return InequalityConstraints(g, nx, nu, BoundConstraints(l, u))
end

function InequalityConstraints(g::F, nx::Int64, nu::Int64,
        l::SVector{ng, T}, u::SVector{ng, T}, np::Int64) where {F<:Function, T, ng}
    return InequalityConstraints(g, nx, nu, BoundConstraints(l, u), np)
end

function InequalityConstraints(::Type{T}, nx::Int64, nu::Int64) where T
    bounds = BoundConstraints(SVector{0, T}(), SVector{0, T}())
    return InequalityConstraints((x, u) -> [], nx, nu, bounds)
end

function InequalityConstraints(::Type{T}, nx::Int64, nu::Int64, np::Int64) where T
    bounds = BoundConstraints(SVector{0, T}(), SVector{0, T}())
    return InequalityConstraints((x, u, p) -> [], nx, nu, bounds, np)
end
