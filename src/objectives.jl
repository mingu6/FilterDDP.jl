struct Objective
    evaluate
    gradient_state
    gradient_control
    hessian_state_state
    hessian_control_control
    hessian_control_state
    num_state::Int
    num_control::Int
end

function Objective(f::Function, num_state::Int, num_control::Int)
    x = Symbolics.variables(:x, 1:num_state)
    u = Symbolics.variables(:u, 1:num_control)

    evaluate = f(x, u)
    gradient_state = Symbolics.gradient(evaluate, x)
    gradient_control = Symbolics.gradient(evaluate, u)
    hessian_state_state = Symbolics.jacobian(gradient_state, x)
    hessian_control_control = Symbolics.jacobian(gradient_control, u)
    hessian_control_state = Symbolics.jacobian(gradient_control, x)

    evaluate_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), [evaluate], x, u; skipzeros=true)[2]
        )
    gradient_state_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), gradient_state, x, u; skipzeros=true)[2]
        )
    gradient_control_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), gradient_control, x, u; skipzeros=true)[2]
        )
    hessian_state_state_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), hessian_state_state, x, u; skipzeros=true)[2]
        )
    hessian_control_control_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), hessian_control_control, x, u; skipzeros=true)[2]
        )
    hessian_control_state_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), hessian_control_state, x, u; skipzeros=true)[2]
        )

    return Objective(evaluate_func, gradient_state_func, gradient_control_func,
        hessian_state_state_func, hessian_control_control_func, hessian_control_state_func,
        num_state, num_control)
end

Objectives = Vector{Objective}

function objective(objectives::Objectives, states::Vector{Vector{T}}, controls::Vector{Vector{T}}) where T
    N = length(states)
    J = T(0.0)
    Jp = [T(0.0)]
    for t in 1:N
        objectives[t].evaluate(Jp, states[t], controls[t])
        J += Jp[1]
    end
    return J
end

function objective_gradient!(gradient_states::Vector{Vector{T}}, gradient_controls::Vector{Vector{T}}, objectives::Objectives,
            states::Vector{Vector{T}}, controls::Vector{Vector{T}}) where T
    N = length(objectives)
    for t in 1:N
        objectives[t].gradient_state(gradient_states[t], states[t], controls[t])
        objectives[t].gradient_control(gradient_controls[t], states[t], controls[t])
    end
end

function objective_hessian!(hessian_state_state::Vector{Matrix{T}}, hessian_control_control::Vector{Matrix{T}},
            hessian_control_state::Vector{Matrix{T}}, objectives::Objectives, states::Vector{Vector{T}}, controls::Vector{Vector{T}}) where T
    N = length(objectives)
    tmp = T[]
    for t in 1:N
        objectives[t].hessian_state_state(hessian_state_state[t], states[t], controls[t])
        objectives[t].hessian_control_control(hessian_control_control[t], states[t], tmp)
        objectives[t].hessian_control_state(hessian_control_state[t], states[t], tmp)
    end
end
