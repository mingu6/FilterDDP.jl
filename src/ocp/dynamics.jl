struct Dynamics{T}
    nx::Int
    nu::Int
    nx1::Int              # number of states at next timestep
    f                     # no type means compiled julia/casadi function 
    fx
    fu
    fxx
    fux
    fuu
    f_mem::Vector{T}
    fx_mem::Matrix{T}
    fu_mem::Matrix{T}
    fxx_mem::Matrix{T}
    fux_mem::Matrix{T}
    fuu_mem::Matrix{T}
end

function Dynamics(T, f::Function, nx::Int, nu::Int; method::DiffMethod=Symbolic(), quasi_newton::Bool=false)
    return _Dynamics(method, T, f, nx, nu; quasi_newton=quasi_newton)
end

function _Dynamics(method::Symbolic, T, f::Function, nx::Int, nu::Int; quasi_newton::Bool=false)
    x = Symbolics.variables(:x, 1:nx)
    u = Symbolics.variables(:u, 1:nu)

    y = Symbolics.simplify(f(x, u))
    nx1 = length(y)
    λ = Symbolics.variables(:λ, 1:nx1)  # vector variables for Hessian vector products
    
    fx = Symbolics.simplify(Symbolics.jacobian(y, x))
    fu = Symbolics.simplify(Symbolics.jacobian(y, u))
    f_func = eval(Symbolics.build_function(y, x, u)[2])
    fx_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), fx, x, u; skipzeros=true)[2])
    fu_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), fu, x, u; skipzeros=true)[2])
    if !quasi_newton
        fxx = Symbolics.simplify(Symbolics.hessian(λ' * y, x))
        fux = Symbolics.simplify(Symbolics.jacobian(Symbolics.gradient(λ' * y, u), x))
        fuu = Symbolics.simplify(Symbolics.hessian(λ' * y, u))
        fxx_func = eval(
            Symbolics._build_function(Symbolics.JuliaTarget(), fxx, x, u, λ; skipzeros=true)[2])
        fux_func = eval(
            Symbolics._build_function(Symbolics.JuliaTarget(), fux, x, u, λ; skipzeros=true)[2])
        fuu_func = eval(
            Symbolics._build_function(Symbolics.JuliaTarget(), fuu, x, u, λ; skipzeros=true)[2])
    else
        fxx_func = nothing
        fux_func = nothing
        fuu_func = nothing
    end

    f_mem = zeros(T, nx1)
    fx_mem = zeros(T, nx1, nx)
    fu_mem = zeros(T, nx1, nu)
    fxx_mem = zeros(T, nx, nx)
    fux_mem = zeros(T, nu, nx)
    fuu_mem = zeros(T, nu, nu)

    return Dynamics{T}(nx, nu, nx1, f_func, fx_func, fu_func, fxx_func, fux_func, fuu_func,
                    f_mem, fx_mem, fu_mem, fxx_mem, fux_mem, fuu_mem)
end

# user-provided dynamics and derivatives
function Dynamics(T, nx::Int, nu::Int, nx1::Int, f::Function, fx::Function, fu::Function;
    fxx::Function=nothing, fux::Function=nothing, fuu::Function=nothing)
    return Dynamics{T}(nx, nu, nx1, f, fx, fu, fxx, fux, fuu,
                        zeros(T, nx1), zeros(T, nx1, nx), zeros(T, nx1, nu),
                        zeros(T, nx, nx), zeros(T, nu, nx), zeros(T, nu, nu))
end

function dynamics!(dynamics::Dynamics{T}, wse::FilterDDPWorkspaceElement{T},
            wse1::FilterDDPWorkspaceElement{T}; mode=:nominal) where T
    if mode == :nominal
        dynamics.f(wse1.nominal.x, wse.nominal.x, wse.nominal.u)
    else
        dynamics.f(wse1.current.x, wse.current.x, wse.current.u)
    end
    return nothing
end

function jacobians!(dynamics_vec::Vector{Dynamics{T}}, ws::FilterDDPWorkspace{T}; mode=:nominal) where T
    if mode == :nominal
        for (dynamics, wse) in zip(dynamics_vec, ws)
            dynamics.fx(dynamics.fx_mem, wse.nominal.x, wse.nominal.u)
            dynamics.fu(dynamics.fu_mem, wse.nominal.x, wse.nominal.u)
        end
    else
        for (dynamics, wse) in zip(dynamics_vec, ws)
            dynamics.fx(dynamics.fx_mem, wse.current.x, wse.current.u)
            dynamics.fu(dynamics.fu_mem, wse.current.x, wse.current.u)
        end
    end
end

function hessians!(dynamics::Dynamics{T}, wse::FilterDDPWorkspaceElement{T}, adj::Vector{T}; mode=:nominal) where T
    if !isnothing(dynamics.fuu)  # do nothing if Gauss-Newton
        if mode == :nominal
            dynamics.fxx(dynamics.fxx_mem, wse.nominal.x, wse.nominal.u, adj)
            dynamics.fux(dynamics.fux_mem, wse.nominal.x, wse.nominal.u, adj)
            dynamics.fuu(dynamics.fuu_mem, wse.nominal.x, wse.nominal.u, adj)
        else
            dynamics.fxx(dynamics.fxx_mem, wse.current.x, wse.current.u, adj)
            dynamics.fux(dynamics.fux_mem, wse.current.x, wse.current.u, adj)
            dynamics.fuu(dynamics.fuu_mem, wse.current.x, wse.current.u, adj)
        end
    end
end

function reset_mem!(dynamics_vec::Vector{Dynamics{T}}) where T
    for dynamics in dynamics_vec 
        fill!(dynamics.f_mem, 0.0)
        fill!(dynamics.fx_mem, 0.0)
        fill!(dynamics.fu_mem, 0.0)
        fill!(dynamics.fxx_mem, 0.0)
        fill!(dynamics.fux_mem, 0.0)
        fill!(dynamics.fuu_mem, 0.0)
    end 
end 
