struct Constraints{T}
    nx::Int
    nu::Int
    nc::Int
    c                     # no type means compiled julia/casadi function 
    cx
    cu
    cxx
    cux
    cuu
    c_mem::Vector{T}
    cx_mem::Matrix{T}
    cu_mem::Matrix{T}
    cxx_mem::Matrix{T}
    cux_mem::Matrix{T}
    cuu_mem::Matrix{T}
end

function Constraints(T, c::Function, nx::Int, nu::Int; method::DiffMethod=Symbolic(), quasi_newton::Bool=false)
    return _Constraints(method, T, c, nx, nu; quasi_newton=quasi_newton)
end

function _Constraints(method::Symbolic, T, c::Function, nx::Int, nu::Int; quasi_newton::Bool=false)
    x = Symbolics.variables(:x, 1:nx)
    u = Symbolics.variables(:u, 1:nu)

    c_ = Symbolics.simplify(c(x, u))
    cx = Symbolics.simplify(Symbolics.jacobian(c_, x))
    cu = Symbolics.simplify(Symbolics.jacobian(c_, u))
    nc = length(c_)

    c_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), c_, x, u; skipzeros=true)[2])
    cx_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), cx, x, u; skipzeros=true)[2])
    cu_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), cu, x, u; skipzeros=true)[2])
    
    ϕ = Symbolics.variables(:ϕ, 1:nc)  # adjoint for second-order tensor contraction
    if !quasi_newton && nc > 0
        cxx = Symbolics.simplify(Symbolics.hessian(c_' * ϕ, x))
        cux = Symbolics.simplify(Symbolics.jacobian(cu' * ϕ, x))
        cuu = Symbolics.simplify(Symbolics.hessian(c_' * ϕ, u))
        cxx_func = eval(
            Symbolics._build_function(Symbolics.JuliaTarget(), cxx, x, u, ϕ; skipzeros=true)[2])
        cux_func = eval(
            Symbolics._build_function(Symbolics.JuliaTarget(), cux, x, u, ϕ; skipzeros=true)[2])
        cuu_func = eval(
            Symbolics._build_function(Symbolics.JuliaTarget(), cuu, x, u, ϕ; skipzeros=true)[2])
    else
        cxx_func = nothing
        cux_func = nothing
        cuu_func = nothing
    end

    c_mem = zeros(T, nc)
    cx_mem = zeros(T, nc, nx)
    cu_mem = zeros(T, nc, nu)
    cxx_mem = zeros(T, nx, nx)
    cux_mem = zeros(T, nu, nx)
    cuu_mem = zeros(T, nu, nu)

    return Constraints{T}(nx, nu, nc, c_func, cx_func, cu_func, cxx_func, cux_func, cuu_func,
                        c_mem, cx_mem, cu_mem, cxx_mem, cux_mem, cuu_mem)
end

function Constraints(T, nx::Int, nu::Int)
    return Constraints{T}(nx, nu, 0,
        (mem, x, u) -> nothing, (mem, x, u) -> nothing, (mem, x, u) -> nothing,
        (mem, x, u, λ) -> nothing, (mem, x, u, λ) -> nothing, (mem, x, u, λ) -> nothing,
        zeros(T, 0), zeros(T, 0, nx), zeros(T, 0, nu), zeros(T, nx, nx), zeros(T, nu, nx), zeros(T, nu, nu))
end

# user-provided constraints and derivatives
function Constraints(T, nx::Int, nu::Int, nc::Int, c::Function, cx::Function, cu::Function;
    cxx::Function=nothing, cux::Function=nothing, cuu::Function=nothing)
    return Constraints{T}(nx, nu, nc, c, cx, cu, cxx, cux, cuu,
                        zeros(T, nc), zeros(T, nc, nx), zeros(T, nc, nu),
                        zeros(T, nx, nx), zeros(T, nu, nx), zeros(T, nu, nu))
end

function constraints!(constraints_vec::Vector{Constraints{T}}, ws::FilterDDPWorkspace{T}; mode=:nominal) where T
    if mode == :nominal
        for (constraints, wse) in zip(constraints_vec, ws)
            if constraints.nc > 0
                constraints.c(wse.nominal.c, wse.nominal.x, wse.nominal.u)
            end
        end
    else
        for (constraints, wse) in zip(constraints_vec, ws)
            if constraints.nc > 0
                constraints.c(wse.current.c, wse.current.x, wse.current.u)
            end
        end
    end
end

function jacobians!(constraints_vec::Vector{Constraints{T}}, ws::FilterDDPWorkspace{T}; mode=:nominal) where T
    for (t, (constraints, wse)) in enumerate(zip(constraints_vec, ws))
        if constraints.nc > 0
            if mode == :nominal
                constraints.cx(constraints.cx_mem, wse.nominal.x, wse.nominal.u)
                constraints.cu(constraints.cu_mem, wse.nominal.x, wse.nominal.u)
            else
                constraints.cx(constraints.cx_mem, wse.current.x, wse.current.u)
                constraints.cu(constraints.cu_mem, wse.current.x, wse.current.u)
            end
        end
    end
end

function hessians!(constraints_vec::Vector{Constraints{T}}, ws::FilterDDPWorkspace{T}; mode=:nominal) where T
    if mode == :nominal
        for (constraints, wse) in zip(constraints_vec, ws)
            if !isnothing(constraints.cuu)  # do nothing if Gauss-Newton
                constraints.cxx(constraints.cxx_mem, wse.nominal.x, wse.nominal.u, wse.nominal.ϕ)
                constraints.cux(constraints.cux_mem, wse.nominal.x, wse.nominal.u, wse.nominal.ϕ)
                constraints.cuu(constraints.cuu_mem, wse.nominal.x, wse.nominal.u, wse.nominal.ϕ)
            end
        end
    else
        for (constraints, wse) in zip(constraints_vec, ws)
            if !isnothing(constraints.cuu)  # do nothing if Gauss-Newton
                constraints.cxx(constraints.cxx_mem, wse.current.x, wse.current.u, wse.current.ϕ)
                constraints.cux(constraints.cux_mem, wse.current.x, wse.current.u, wse.current.ϕ)
                constraints.cuu(constraints.cuu_mem, wse.current.x, wse.current.u, wse.current.ϕ)
            end
        end
    end
end

function reset_mem!(constraints_vec::Vector{Constraints{T}}) where T
    for constraints in constraints_vec
        fill!(constraints.c_mem, 0.0)
        fill!(constraints.cx_mem, 0.0)
        fill!(constraints.cu_mem, 0.0)
        fill!(constraints.cxx_mem, 0.0)
        fill!(constraints.cux_mem, 0.0)
        fill!(constraints.cuu_mem, 0.0)
    end
end
