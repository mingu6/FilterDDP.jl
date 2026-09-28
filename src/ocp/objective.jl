struct SymbolicsStageObjective{nx, nu, np, O1, O2, O3, O4, O5, O6} <:
    AbstractStageObjective{nx, nu, np}
    l::O1
    lx::O2
    lu::O3
    lxx::O4
    lux::O5
    luu::O6
end

struct ForwardDiffStageObjective{nx, nu, np, F} <:
    AbstractStageObjective{nx, nu, np}
    l::F
end

struct SymbolicsTerminalObjective{nx, np, O1, O2, O3} <:
    AbstractTerminalObjective{nx, np}
    l::O1
    lx::O2
    lxx::O3
end

struct ForwardDiffTerminalObjective{nx, np, F} <:
        AbstractTerminalObjective{nx, np}
    l::F
end

# Public objective constructors.
function Objective(l::F, nx::Int64, nu::Int64, np::Int64=0;
        backend::DerivativeBackend=SymbolicsBackend()) where F<:Function
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    return Objective(backend, l, nx, nu, Val(np))
end

function TerminalObjective(l::F, nx::Int64, np::Int64=0;
        backend::DerivativeBackend=SymbolicsBackend()) where F<:Function
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    return TerminalObjective(backend, l, nx, Val(np))
end

@inline symbolic_stage_value(l, x, u, ::Val{0}) = (l(x, u), ())

@inline function symbolic_stage_value(l, x, u, ::Val{np}) where np
    np > 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    p::Vector{Num} = Symbolics.variables(:p, 1:np)
    return l(x, u, p), (p,)
end

function build_symbolic_objective(::Val{nx}, ::Val{nu}, ::Val{np}, l_, x, u,
        inputs...) where {nx, nu, np}
    lx = Symbolics.simplify(Symbolics.gradient(l_, x))
    lu = Symbolics.simplify(Symbolics.gradient(l_, u))
    lxx = Symbolics.simplify(Symbolics.jacobian(lx, x))
    lux = Symbolics.simplify(Symbolics.jacobian(lu, x))
    luu = Symbolics.simplify(Symbolics.jacobian(lu, u))

    l_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), [l_], x, u, inputs...; skipzeros=false, cse=true)[1])
    lx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lx, x, u, inputs...; skipzeros=false, cse=true)[1])
    lu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lu, x, u, inputs...; skipzeros=false, cse=true)[1])
    lxx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lxx, x, u, inputs...; skipzeros=false, cse=true)[1])
    lux_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lux, x, u, inputs...; skipzeros=false, cse=true)[1])
    luu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), luu, x, u, inputs...; skipzeros=false, cse=true)[1])

    return SymbolicsStageObjective{nx, nu, np, typeof(l_func), typeof(lx_func), typeof(lu_func),
                     typeof(lxx_func), typeof(lux_func), typeof(luu_func)}(
                        l_func, lx_func, lu_func, lxx_func, lux_func, luu_func)
end

function Objective(::SymbolicsBackend, l::F, nx::Int64, nu::Int64,
        parameter_count::Val{np}) where {F<:Function, np}
    x::Vector{Num} = Symbolics.variables(:x, 1:nx)
    u::Vector{Num} = Symbolics.variables(:u, 1:nu)
    value, inputs = symbolic_stage_value(l, x, u, parameter_count)
    return build_symbolic_objective(Val(nx), Val(nu), parameter_count,
        Symbolics.simplify(value), x, u, inputs...)
end

function Objective(::ForwardDiffBackend, l::F, nx::Int64, nu::Int64,
        ::Val{np}) where {F<:Function, np}
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    return ForwardDiffStageObjective{nx, nu, np, F}(l)
end

@inline symbolic_terminal_value(l, x, ::Val{0}) = (l(x), ())

@inline function symbolic_terminal_value(l, x, ::Val{np}) where np
    np > 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    p::Vector{Num} = Symbolics.variables(:p, 1:np)
    return l(x, p), (p,)
end

function build_symbolic_terminal_objective(::Val{nx}, ::Val{np}, l_, x,
        inputs...) where {nx, np}
    lx = Symbolics.simplify(Symbolics.gradient(l_, x))
    lxx = Symbolics.simplify(Symbolics.jacobian(lx, x))

    l_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), [l_], x, inputs...; skipzeros=false, cse=true)[1])
    lx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lx, x, inputs...; skipzeros=false, cse=true)[1])
    lxx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lxx, x, inputs...; skipzeros=false, cse=true)[1])

    return SymbolicsTerminalObjective{nx, np, typeof(l_func), typeof(lx_func), typeof(lxx_func)}(
        l_func, lx_func, lxx_func)
end

function TerminalObjective(::SymbolicsBackend, l::F, nx::Int64,
        parameter_count::Val{np}) where {F<:Function, np}
    x::Vector{Num} = Symbolics.variables(:x, 1:nx)
    value, inputs = symbolic_terminal_value(l, x, parameter_count)
    return build_symbolic_terminal_objective(Val(nx), parameter_count,
        Symbolics.simplify(value), x, inputs...)
end

function TerminalObjective(::ForwardDiffBackend, l::F, nx::Int64,
    ::Val{np}) where {F<:Function, np}
    np >= 0 || throw(ArgumentError("parameter dimension must be nonnegative"))
    return ForwardDiffTerminalObjective{nx, np, F}(l)
end

@inline function component_value(objective::SymbolicsStageObjective, x, u, parameters)
    return evaluate(objective.l, x, u, parameters)[1]
end

@inline function component_value(objective::ForwardDiffStageObjective,
        x, u, parameters)
    return evaluate(objective.l, x, u, parameters)
end

@inline function component_first_order(
    objective::SymbolicsStageObjective{nx, nu}, x::SVector{nx, T},
        u::SVector{nu, T}, parameters) where {T, nx, nu}
    lx = evaluate(objective.lx, x, u, parameters)
    lu = evaluate(objective.lu, x, u, parameters)
    return (; lx, lu)
end

@inline function component_first_order(
    objective::ForwardDiffStageObjective{nx, nu}, x::SVector{nx, T},
        u::SVector{nu, T}, parameters) where {T, nx, nu}
    z = combine_state_control(x, u)
    base = StateControlFunction{nx, nu, typeof(objective.l), typeof(parameters)}(
        objective.l, parameters)
    wrapped = ScalarOutput{typeof(base)}(base)
    gradient = ForwardDiff.gradient(wrapped, z)
    x_indices = state_indices(Val(nx))
    u_indices = control_indices(Val(nx), Val(nu))
    lx = gradient[x_indices]
    lu = gradient[u_indices]
    return (; lx, lu)
end

@inline function component_derivatives(
    objective::SymbolicsStageObjective{nx, nu}, x::SVector{nx, T},
        u::SVector{nu, T}, parameters) where {T, nx, nu}
    value = evaluate(objective.l, x, u, parameters)[1]
    lx = evaluate(objective.lx, x, u, parameters)
    lu = evaluate(objective.lu, x, u, parameters)
    lxx = evaluate(objective.lxx, x, u, parameters)
    lux = evaluate(objective.lux, x, u, parameters)
    luu = evaluate(objective.luu, x, u, parameters)
    return (; value, lx, lu, lxx, lux, luu)
end

@inline function component_derivatives(
    objective::ForwardDiffStageObjective{nx, nu}, x::SVector{nx, T},
        u::SVector{nu, T}, parameters) where {T, nx, nu}
    z = combine_state_control(x, u)
    base = StateControlFunction{nx, nu, typeof(objective.l), typeof(parameters)}(
        objective.l, parameters)
    wrapped = ScalarOutput{typeof(base)}(base)
    result = DiffResults.HessianResult(z)
    result = ForwardDiff.hessian!(result, wrapped, z)
    gradient = DiffResults.gradient(result)
    hessian = DiffResults.hessian(result)
    x_indices = state_indices(Val(nx))
    u_indices = control_indices(Val(nx), Val(nu))
    value = DiffResults.value(result)
    lx = gradient[x_indices]
    lu = gradient[u_indices]
    lxx = hessian[x_indices, x_indices]
    lux = hessian[u_indices, x_indices]
    luu = hessian[u_indices, u_indices]
    return (; value, lx, lu, lxx, lux, luu)
end

@inline function component_value(
    objective::SymbolicsTerminalObjective, x, parameters)
    return evaluate(objective.l, x, parameters)[1]
end

@inline function component_value(
        objective::ForwardDiffTerminalObjective, x, parameters)
    return evaluate(objective.l, x, parameters)
end

@inline function component_first_order(
    objective::SymbolicsTerminalObjective, x, parameters)
    return evaluate(objective.lx, x, parameters)
end

@inline function component_first_order(
        objective::ForwardDiffTerminalObjective{nx},
        x::SVector{nx}, parameters) where nx
    wrapped = StateFunction{nx, typeof(objective.l), typeof(parameters)}(
        objective.l, parameters)
    return ForwardDiff.gradient(wrapped, x)
end

@inline function component_derivatives(
    objective::SymbolicsTerminalObjective{nx}, x::SVector{nx, T},
        parameters) where {T, nx}
    value = evaluate(objective.l, x, parameters)[1]
    lx = evaluate(objective.lx, x, parameters)
    lxx = evaluate(objective.lxx, x, parameters)
    return (; value, lx, lxx)
end

@inline function component_derivatives(
        objective::ForwardDiffTerminalObjective{nx}, x::SVector{nx, T},
        parameters) where {T, nx}
    wrapped = StateFunction{nx, typeof(objective.l), typeof(parameters)}(
        objective.l, parameters)
    result = DiffResults.HessianResult(x)
    result = ForwardDiff.hessian!(result, wrapped, x)
    value = DiffResults.value(result)
    lx = DiffResults.gradient(result)
    lxx = DiffResults.hessian(result)
    return (; value, lx, lxx)
end
