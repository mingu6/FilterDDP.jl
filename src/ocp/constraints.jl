RFN = Union{RuntimeGeneratedFunction, Nothing}

struct EqualityConstraints{nx, nu, nc, F1<:RFN, F2<:RFN, F3<:RFN, F4<:RFN, F5<:RFN, F6<:RFN}
    c::F1
    cx::F2
    cu::F3
    cxx::F4
    cux::F5
    cuu::F6
end

function EqualityConstraints(c::F, nx::Int64, nu::Int64) where F<:Function
    x::Vector{Num} = Symbolics.variables(:x, 1:nx)
    u::Vector{Num} = Symbolics.variables(:u, 1:nu)

    c_ = Symbolics.simplify(c(x, u))
    cx = Symbolics.simplify(Symbolics.jacobian(c_, x))
    cu = Symbolics.simplify(Symbolics.jacobian(c_, u))
    nc = length(c_)

    if nc > 0
        c_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), c_, x, u)[1])
        cx_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), cx, x, u)[1])
        cu_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), cu, x, u)[1])
        
        ϕ::Vector{Num} = Symbolics.variables(:ϕ, 1:nc)  # adjoint for second-order tensor contraction

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
        c_func = nothing
        cx_func = nothing
        cu_func = nothing
        cxx_func = nothing
        cux_func = nothing
        cuu_func = nothing
    end

    return EqualityConstraints{nx, nu, nc, typeof(c_func), typeof(cx_func), typeof(cu_func),
                    typeof(cxx_func), typeof(cux_func), typeof(cuu_func)}(
                        c_func, cx_func, cu_func, cxx_func, cux_func, cuu_func)
end

function EqualityConstraints(nx::Int64, nu::Int64)
    return EqualityConstraints((x, u) -> [], nx, nu)
end
