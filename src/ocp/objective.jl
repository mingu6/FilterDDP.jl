struct Objective{nx, nu, np, O1, O2, O3, O4, O5, O6}
    l::O1
    lx::O2
    lu::O3
    lxx::O4
    lux::O5
    luu::O6
end

struct TerminalObjective{nx, np, O1, O2, O3}
    l::O1
    lx::O2
    lxx::O3
end

function Objective(l::F, nx::Int64, nu::Int64) where F<:Function
    x::Vector{Num} = Symbolics.variables(:x, 1:nx)
    u::Vector{Num} = Symbolics.variables(:u, 1:nu)

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
    
    return Objective{nx, nu, 0, typeof(l_func), typeof(lx_func), typeof(lu_func),
                     typeof(lxx_func), typeof(lux_func), typeof(luu_func)}(
                        l_func, lx_func, lu_func, lxx_func, lux_func, luu_func)
end

function Objective(l::F, nx::Int64, nu::Int64, np::Int64) where F<:Function
    np > 0 || throw(ArgumentError("parameter dimension must be positive"))
    x::Vector{Num} = Symbolics.variables(:x, 1:nx)
    u::Vector{Num} = Symbolics.variables(:u, 1:nu)
    p::Vector{Num} = Symbolics.variables(:p, 1:np)

    l_ = Symbolics.simplify(l(x, u, p))
    lx = Symbolics.simplify(Symbolics.gradient(l_, x))
    lu = Symbolics.simplify(Symbolics.gradient(l_, u))
    lxx = Symbolics.simplify(Symbolics.jacobian(lx, x))
    lux = Symbolics.simplify(Symbolics.jacobian(lu, x))
    luu = Symbolics.simplify(Symbolics.jacobian(lu, u))

    l_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), [l_], x, u, p; skipzeros=false)[1])
    lx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lx, x, u, p; skipzeros=false)[1])
    lu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lu, x, u, p; skipzeros=false)[1])
    lxx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lxx, x, u, p; skipzeros=false)[1])
    lux_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lux, x, u, p; skipzeros=false)[1])
    luu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), luu, x, u, p; skipzeros=false)[1])

    return Objective{nx, nu, np, typeof(l_func), typeof(lx_func), typeof(lu_func),
                     typeof(lxx_func), typeof(lux_func), typeof(luu_func)}(
                        l_func, lx_func, lu_func, lxx_func, lux_func, luu_func)
end

function TerminalObjective(l::F, nx::Int64) where F<:Function
    x::Vector{Num} = Symbolics.variables(:x, 1:nx)

    l_ = Symbolics.simplify(l(x))
    lx = Symbolics.simplify(Symbolics.gradient(l_, x))
    lxx = Symbolics.simplify(Symbolics.jacobian(lx, x))

    l_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), [l_], x; skipzeros=false)[1])
    lx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lx, x; skipzeros=false)[1])
    lxx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lxx, x; skipzeros=false)[1])

    return TerminalObjective{nx, 0, typeof(l_func), typeof(lx_func), typeof(lxx_func)}(
        l_func, lx_func, lxx_func)
end

function TerminalObjective(l::F, nx::Int64, np::Int64) where F<:Function
    np > 0 || throw(ArgumentError("parameter dimension must be positive"))
    x::Vector{Num} = Symbolics.variables(:x, 1:nx)
    p::Vector{Num} = Symbolics.variables(:p, 1:np)

    l_ = Symbolics.simplify(l(x, p))
    lx = Symbolics.simplify(Symbolics.gradient(l_, x))
    lxx = Symbolics.simplify(Symbolics.jacobian(lx, x))

    l_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), [l_], x, p; skipzeros=false)[1])
    lx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lx, x, p; skipzeros=false)[1])
    lxx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), lxx, x, p; skipzeros=false)[1])

    return TerminalObjective{nx, np, typeof(l_func), typeof(lx_func), typeof(lxx_func)}(
        l_func, lx_func, lxx_func)
end
