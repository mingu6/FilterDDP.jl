mutable struct Dynamics{T, nx, nu}
    f                     # no type means compiled julia/casadi function 
    fx
    fu
    fxx
    fux
    fuu
    f_mem::SVector{nx, T}
    fx_mem::SMatrix{nx, nx, T}
    fu_mem::SMatrix{nx, nu, T}
    fxx_mem::SMatrix{nx, nx, T}
    fux_mem::SMatrix{nu, nx, T}
    fuu_mem::SMatrix{nu, nu, T}
end

function Dynamics(T, f::Function, nx::Int64, nu::Int64; method::DiffMethod=Symbolic(), quasi_newton::Bool=false)
    return _Dynamics(method, T, f, nx, nu; quasi_newton=quasi_newton)
end

function _Dynamics(method::Symbolic, T, f::Function, nx::Int64, nu::Int64; quasi_newton::Bool=false)
    x = Symbolics.variables(:x, 1:nx)
    u = Symbolics.variables(:u, 1:nu)

    y = Symbolics.simplify(f(x, u))
    λ = Symbolics.variables(:λ, 1:nx)  # vector variables for Hessian vector products
    
    fx = Symbolics.simplify(Symbolics.jacobian(y, x))
    fu = Symbolics.simplify(Symbolics.jacobian(y, u))
    f_func = @RuntimeGeneratedFunction(Symbolics.build_function(y, x, u)[1])
    fx_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fx, x, u; skipzeros=true)[1])
    fu_func = @RuntimeGeneratedFunction(
        Symbolics._build_function(Symbolics.JuliaTarget(), fu, x, u; skipzeros=true)[1])
    if !quasi_newton
        fxx = Symbolics.simplify(Symbolics.hessian(λ' * y, x))
        fux = Symbolics.simplify(Symbolics.jacobian(Symbolics.gradient(λ' * y, u), x))
        fuu = Symbolics.simplify(Symbolics.hessian(λ' * y, u))
        fxx_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), fxx, x, u, λ; skipzeros=true)[1])
        fux_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), fux, x, u, λ; skipzeros=true)[1])
        fuu_func = @RuntimeGeneratedFunction(
            Symbolics._build_function(Symbolics.JuliaTarget(), fuu, x, u, λ; skipzeros=true)[1])
    else
        fxx_func = nothing
        fux_func = nothing
        fuu_func = nothing
    end

    f_mem = @SVector zeros(T, nx)
    fx_mem = @SMatrix zeros(T, nx, nx)
    fu_mem = @SMatrix zeros(T, nx, nu)
    fxx_mem = @SMatrix zeros(T, nx, nx)
    fux_mem = @SMatrix zeros(T, nu, nx)
    fuu_mem = @SMatrix zeros(T, nu, nu)

    return Dynamics{T, nx, nu}(f_func, fx_func, fu_func, fxx_func, fux_func, fuu_func,
                    f_mem, fx_mem, fu_mem, fxx_mem, fux_mem, fuu_mem)
end

# user-provided dynamics and derivatives
function Dynamics(T, f::Function, fx::Function, fu::Function; fxx::Function=nothing, fux::Function=nothing, fuu::Function=nothing)
    return Dynamics{T}(f, fx, fu, fxx, fux, fuu,
                        SVector{T, nx}(zeros(T, nx)), SMatrix{T, nx, nx}(zeros(T, nx, nx)), SMatrix{T, nx, nu}(zeros(T, nx, nu)),
                        SMatrix{T, nx, nx}(zeros(T, nx, nx)), SMatrix{T, nu, nx}(zeros(T, nu, nx)), SMatrix{T, nu, nu}(zeros(T, nu, nu)))
end

function dynamics!(dynamics::Dynamics{T, nx, nu}, wse::FilterDDPWorkspaceElement{T, nx, nu, nc},
            wse1::FilterDDPWorkspaceElement{T, nx, nu, nc}; mode=:nominal) where {T, nx, nu, nc}
    if mode == :nominal
        wse1.nominal.x = dynamics.f(wse.nominal.x, wse.nominal.u)
    else
        wse1.current.x = dynamics.f(wse.current.x, wse.current.u)
    end
    return nothing
end

function jacobians!(dynamics_vec::Vector{Dynamics{T, nx, nu}}, ws::FilterDDPWorkspace{T, nx, nu, nc}; mode=:nominal) where {T, nx, nu, nc}
    if mode == :nominal
        for (dynamics, wse) in zip(dynamics_vec, ws)
            dynamics.fx_mem = dynamics.fx(wse.nominal.x, wse.nominal.u)
            dynamics.fu_mem = dynamics.fu(wse.nominal.x, wse.nominal.u)
        end
    else
        for (dynamics, wse) in zip(dynamics_vec, ws)
            dynamics.fx_mem = dynamics.fx(wse.current.x, wse.current.u)
            dynamics.fu_mem = dynamics.fu(wse.current.x, wse.current.u)
        end
    end
end

function hessians!(dynamics::Dynamics{T, nx, nu}, wse::FilterDDPWorkspaceElement{T, nx, nu, nc}, adj::SVector{nx, T};
            mode=:nominal) where {T, nx, nu, nc}
    if !isnothing(dynamics.fuu)  # do nothing if Gauss-Newton
        if mode == :nominal
            dynamics.fxx_mem = dynamics.fxx(wse.nominal.x, wse.nominal.u, adj)
            dynamics.fux_mem = dynamics.fux(wse.nominal.x, wse.nominal.u, adj)
            dynamics.fuu_mem = dynamics.fuu(wse.nominal.x, wse.nominal.u, adj)
        else
            dynamics.fxx_mem = dynamics.fxx(wse.current.x, wse.current.u, adj)
            dynamics.fux_mem = dynamics.fux(wse.current.x, wse.current.u, adj)
            dynamics.fuu_mem = dynamics.fuu(wse.current.x, wse.current.u, adj)
        end
    end
end

