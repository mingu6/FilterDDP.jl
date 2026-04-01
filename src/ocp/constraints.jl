mutable struct EqualityConstraints{T, nx, nu, nc}
    c
    cx
    cu
    cxx
    cux
    cuu
    c_mem::SVector{nc, T}
    cx_mem::SMatrix{nc, nx, T}
    cu_mem::SMatrix{nc, nu, T}
    cxx_mem::SMatrix{nx, nx, T}
    cux_mem::SMatrix{nu, nx, T}
    cuu_mem::SMatrix{nu, nu, T}
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
            Symbolics._build_function(Symbolics.JuliaTarget(), cux, x, u, ϕ; skipzeros=true)[1])
        cuu_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), cuu, x, u, ϕ; skipzeros=true)[1])
    else
        cxx_func = nothing
        cux_func = nothing
        cuu_func = nothing
    end

    c_mem = @SVector zeros(T, nc)
    cx_mem = @SMatrix zeros(T, nc, nx)
    cu_mem = @SMatrix zeros(T, nc, nu)
    cxx_mem = @SMatrix zeros(T, nx, nx)                     # no type means compiled julia/casadi function 
    cux_mem = @SMatrix zeros(T, nu, nx)
    cuu_mem = @SMatrix zeros(T, nu, nu)

    return EqualityConstraints{T, nx, nu, nc}(c_func, cx_func, cu_func, cxx_func, cux_func, cuu_func,
                        c_mem, cx_mem, cu_mem, cxx_mem, cux_mem, cuu_mem)
end

function EqualityConstraints(T, nx::Int, nu::Int)
    return EqualityConstraints{T, nx, nu, 0}(
        (mem, x, u) -> nothing, (mem, x, u) -> nothing, (mem, x, u) -> nothing,
        (mem, x, u, λ) -> nothing, (mem, x, u, λ) -> nothing, (mem, x, u, λ) -> nothing,
        SVector{0, T}(zeros(T, 0)), SVector{0, T}(zeros(T, 0, nx)), SMatrix{0, nu, T}(zeros(T, 0, nu)),
        SMatrix{nx, nx, T}(zeros(T, nx, nx)), SMatrix{nu, nx, T}(zeros(T, nu, nx)), SMatrix{nu, nu, T}(zeros(T, nu, nu)))
end

# user-provided EqualityConstraints and derivatives
function EqualityConstraints(T, nx::Int, nu::Int, nc::Int, c::Function, cx::Function, cu::Function;
    cxx::Function=nothing, cux::Function=nothing, cuu::Function=nothing)
    return EqualityConstraints{T, nx, nu, nc}(c, cx, cu, cxx, cux, cuu,
                        SVector{nc, T}(zeros(T, nc)), SMatrix{nc, nx, T}(zeros(T, nc, nx)), SMatrix{nc, nu, T}(zeros(T, nc, nu)),
                        SMatrix{nx, nx, T}(zeros(T, nx, nx)), SMatrix{nu, nx, T}(zeros(T, nu, nx)), SMatrix{nu, nu, T}(zeros(T, nu, nu)))
end

function constraints!(constraints_vec::Vector{EqualityConstraints{T, nx, nu, nc}}, ws::FilterDDPWorkspace{T, nx, nu, nc};
            mode=:nominal) where {T, nx, nu, nc}
    if mode == :nominal && nc > 0
        for (constraints, wse) in zip(constraints_vec, ws)
            wse.nominal.c = constraints.c(wse.nominal.x, wse.nominal.u)
        end
    elseif nc > 0
        for (constraints, wse) in zip(constraints_vec, ws)
            wse.current.c = constraints.c(wse.current.x, wse.current.u)
        end
    end
end

function jacobians!(constraints_vec::Vector{EqualityConstraints{T, nx, nu, nc}}, ws::FilterDDPWorkspace{T, nx, nu, nc};
            mode=:nominal) where {T, nx, nu, nc}
    for (t, (constraints, wse)) in enumerate(zip(constraints_vec, ws))
        if mode == :nominal && nc > 0
            constraints.cx_mem = constraints.cx(wse.nominal.x, wse.nominal.u)
            constraints.cu_mem = constraints.cu(wse.nominal.x, wse.nominal.u)
        elseif nc > 0
            constraints.cx_mem = constraints.cx(wse.current.x, wse.current.u)
            constraints.cu_mem = constraints.cu(wse.current.x, wse.current.u)
        end
    end
end

function hessians!(constraints_vec::Vector{EqualityConstraints{T, nx, nu, nc}}, ws::FilterDDPWorkspace{T, nx, nu, nc};
            mode=:nominal) where {T, nx, nu, nc}
    if mode == :nominal && nc > 0 
        for (constraints, wse) in zip(constraints_vec, ws)
            if !isnothing(constraints.cuu)  # do nothing if Gauss-Newton
                constraints.cxx_mem = constraints.cxx(wse.nominal.x, wse.nominal.u, wse.nominal.ϕ)
                constraints.cux_mem = constraints.cux(wse.nominal.x, wse.nominal.u, wse.nominal.ϕ)
                constraints.cuu_mem = constraints.cuu(wse.nominal.x, wse.nominal.u, wse.nominal.ϕ)
            end
        end
    elseif nc > 0
        for (constraints, wse) in zip(constraints_vec, ws)
            if !isnothing(constraints.cuu)  # do nothing if Gauss-Newton
                constraints.cxx_mem = constraints.cxx(wse.current.x, wse.current.u, wse.current.ϕ)
                constraints.cux_mem = constraints.cux(wse.current.x, wse.current.u, wse.current.ϕ)
                constraints.cuu_mem = constraints.cuu(wse.current.x, wse.current.u, wse.current.ϕ)
            end
        end
    end
end
