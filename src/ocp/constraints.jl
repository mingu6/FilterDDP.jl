struct EqualityConstraints{T, nx, nu, nc}
    c
    cx
    cu
    cxx
    cux
    cuu
end

function EqualityConstraints(T, c::Function, nx::Int64, nu::Int64; method::DiffMethod=Symbolic(), quasi_newton::Bool=false)
    return _EqualityConstraints(method, T, c, nx, nu; quasi_newton=quasi_newton)
end

function _EqualityConstraints(method::Symbolic, T, c::Function, nx::Int64, nu::Int64; quasi_newton::Bool=false)
    x = Symbolics.variables(:x, 1:nx)
    u = Symbolics.variables(:u, 1:nu)

    c_ = Symbolics.simplify(c(x, u))
    cx = Symbolics.simplify(Symbolics.jacobian(c_, x))
    cu = Symbolics.simplify(Symbolics.jacobian(c_, u))
    nc = length(c_)

    c_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), c_, x, u)[1])
    cx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), cx, x, u)[1])
    cu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), cu, x, u)[1])
    
    ϕ = Symbolics.variables(:ϕ, 1:nc)  # adjoint for second-order tensor contraction
    if !quasi_newton && nc > 0
        cxx = Symbolics.simplify(Symbolics.hessian(c_' * ϕ, x))
        cux = Symbolics.simplify(Symbolics.jacobian(cu' * ϕ, x))
        cuu = Symbolics.simplify(Symbolics.hessian(c_' * ϕ, u))
        cxx_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), cxx, x, u, ϕ)[1])
        cux_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), cux, x, u, ϕ; skipzeros=false)[1])
        cuu_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), cuu, x, u, ϕ; skipzeros=false)[1])
    else
        cxx_func = nothing
        cux_func = nothing
        cuu_func = nothing
    end

    return EqualityConstraints{T, nx, nu, nc}(c_func, cx_func, cu_func, cxx_func, cux_func, cuu_func)
end

function EqualityConstraints(T, nx::Int, nu::Int)
    return EqualityConstraints{T, nx, nu, 0}(
        (mem, x, u) -> nothing, (mem, x, u) -> nothing, (mem, x, u) -> nothing,
        (mem, x, u, λ) -> nothing, (mem, x, u, λ) -> nothing, (mem, x, u, λ) -> nothing)
end

# user-provided EqualityConstraints and derivatives
function EqualityConstraints(T, nx::Int, nu::Int, nc::Int, c::Function, cx::Function, cu::Function;
    cxx::Function=nothing, cux::Function=nothing, cuu::Function=nothing)
    return EqualityConstraints{T, nx, nu, nc}(c, cx, cu, cxx, cux, cuu)
end
