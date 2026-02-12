struct Dynamics
    evaluate
    jacobian_state
    jacobian_control
    vfxx
    vfux
    vfuu
    num_next_state::Int
    num_state::Int
    num_control::Int
end

Model = Vector{Dynamics}

function Dynamics(f::Function, num_state::Int, num_control::Int; quasi_newton::Bool=false)
    x = Symbolics.variables(:x, 1:num_state)
    u = Symbolics.variables(:u, 1:num_control)

    y = f(x, u)
    num_next_state = length(y)
    vx = Symbolics.variables(:vx, 1:num_next_state)  # vector variables for Hessian vector products
    
    jacobian_state = Symbolics.jacobian(y, x)
    jacobian_control = Symbolics.jacobian(y, u)
    evaluate_func = eval(Symbolics.build_function(y, x, u)[2])
    jacobian_state_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), jacobian_state, x, u; skipzeros=true)[2]
        )
    jacobian_control_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), jacobian_control, x, u; skipzeros=true)[2]
        )
    if !quasi_newton
        vfxx = sum([vx[t] .* Symbolics.hessian(y[t], x) for t in 1:num_next_state])
        vfux = sum([vx[t] .* Symbolics.jacobian(Symbolics.gradient(y[t], u), x) for t in 1:num_next_state])
        vfuu = sum([vx[t] .* Symbolics.hessian(y[t], u) for t in 1:num_next_state])
        vfxx_func = eval(
            Symbolics._build_function(Symbolics.JuliaTarget(), vfxx, x, u, vx; skipzeros=true)[2]
            )
        vfux_func = eval(
            Symbolics._build_function(Symbolics.JuliaTarget(), vfux, x, u, vx; skipzeros=true)[2]
            )
        vfuu_func = eval(
            Symbolics._build_function(Symbolics.JuliaTarget(), vfuu, x, u, vx; skipzeros=true)[2]
            )
    else
        vfxx_func = nothing
        vfux_func = nothing
        vfuu_func = nothing
    end
    return Dynamics(evaluate_func, jacobian_state_func, jacobian_control_func,
            vfxx_func, vfux_func, vfuu_func, num_next_state, num_state, num_control)
end

function dynamics!(d::Dynamics, cache::Vector{T}, state::Vector{T}, control::Vector{T}) where T
    d.evaluate(cache, state, control)
    return nothing
end

function jacobian!(jacobian_states::Vector{Matrix{T}}, jacobian_controls::Vector{Matrix{T}},
            dynamics::Vector{Dynamics}, states::Vector{Vector{T}}, controls::Vector{Vector{T}}) where T
    for (t, d) in enumerate(dynamics)
        d.jacobian_state(jacobian_states[t], states[t], controls[t])
        d.jacobian_control(jacobian_controls[t], states[t], controls[t])
    end
end

# user-provided dynamics and gradients
function Dynamics(f::Function, fx::Function, fu::Function, num_next_state::Int, num_state::Int, num_control::Int,
        ;vfxx::Function=nothing, vfux::Function=nothing, vfuu::Function=nothing)
    return Dynamics(f, fx, fu, vfxx, vfux, vfuu, num_next_state, num_state, num_control)
end

function tensor_contraction!(vfxx_cache::Matrix{T}, vfux_cache::Matrix{T}, vfuu_cache::Matrix{T}, dynamics::Dynamics,
        x::Vector{T}, u::Vector{T}, v::Vector{T}) where T
    if !isnothing(dynamics.vfuu)
        dynamics.vfxx(vfxx_cache, x, u, v)
        dynamics.vfux(vfux_cache, x, u, v)
        dynamics.vfuu(vfuu_cache, x, u, v)
    end
end

function tensor_contraction!(vfxx_cache::Vector{Matrix{T}}, vfux_cache::Vector{Matrix{T}}, vfuu_cache::Vector{Matrix{T}},
        model::Model, states::Vector{Vector{T}}, controls::Vector{Vector{T}}, v::Vector{Vector{T}}) where T
    N = length(states)
    for t in 1:N-1
        if !isnothing(model[t].vhuu)
            dynamics.vfxx(vfxx_cache[t], states[t], controls[t], v[t])
            dynamics.vfux(vfux_cache[t], states[t], controls[t], v[t])
            dynamics.vfuu(vfuu_cache, states[t], controls[t], v[t])
        end
    end
end
