"""
    Objectives Data
"""

struct ObjectivesData{T}
    objectives
    gradient_state::Vector{Vector{T}}
    gradient_control::Vector{Vector{T}}
    hessian_state_state::Vector{Matrix{T}}
    hessian_control_control::Vector{Matrix{T}}
    hessian_control_state::Vector{Matrix{T}}
end

function objectives_data(T, constraints::Constraints, objectives::Objectives)
	gradient_state = [zeros(T, c.num_state) for c in constraints]
    gradient_control = [zeros(T, c.num_control) for c in constraints]
    hessian_state_state =  [zeros(T, c.num_state, c.num_state) for c in constraints]
    hessian_control_control = [zeros(T, c.num_control, c.num_control) for c in constraints]
    hessian_control_state = [zeros(T, c.num_control, c.num_state) for c in constraints]
    ObjectivesData{T}(objectives, gradient_state, gradient_control, hessian_state_state, hessian_control_control, hessian_control_state)
end

function reset!(data::ObjectivesData{T}) where T 
    N = length(data.gradient_state)
    for t = 1:N 
        fill!(data.gradient_state[t], 0.0) 
        fill!(data.hessian_state_state[t], 0.0)
        fill!(data.gradient_control[t], 0.0)
        fill!(data.hessian_control_control[t], 0.0)
        fill!(data.hessian_control_state[t], 0.0)
    end 
end