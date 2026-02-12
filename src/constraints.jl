struct Constraint
    evaluate
    jacobian_state
    jacobian_control
    vcxx
    vcux  # DDP tensor contraction
    vcuu  # DDP tensor contraction
    num_constraint::Int
    num_state::Int
    num_control::Int
    indices_compl::Vector{Int}
end

Constraints = Vector{Constraint}

function Constraint(c::Function, num_state::Int, num_control::Int; quasi_newton::Bool=false, indices_compl=nothing)

    x = Symbolics.variables(:x, 1:num_state)
    u = Symbolics.variables(:u, 1:num_control)

    evaluate = c(x, u)
    jacobian_state = Symbolics.jacobian(evaluate, x)
    jacobian_control = Symbolics.jacobian(evaluate, u)

    evaluate_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), evaluate, x, u; skipzeros=true)[2]
        )
    jacobian_state_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), jacobian_state, x, u; skipzeros=true)[2]
        )
    jacobian_control_func = eval(
        Symbolics._build_function(Symbolics.JuliaTarget(), jacobian_control, x, u; skipzeros=true)[2]
        )

    num_constraint = length(evaluate)
    
    v = Symbolics.variables(:v, 1:num_constraint)  # vector variables for Hessian vector products

    if !quasi_newton && num_constraint > 0
        vcxx = Symbolics.jacobian(jacobian_state' * v, x)
        vcux = Symbolics.jacobian(jacobian_control' * v, x)
        vcuu = Symbolics.jacobian(jacobian_control' * v, u)
        vcxx_func = eval(
            Symbolics._build_function(Symbolics.JuliaTarget(), vcxx, x, u, v; skipzeros=true)[2]
            )
        vcux_func = eval(
            Symbolics._build_function(Symbolics.JuliaTarget(), vcux, x, u, v; skipzeros=true)[2]
            )
        vcuu_func = eval(
            Symbolics._build_function(Symbolics.JuliaTarget(), vcuu, x, u, v; skipzeros=true)[2]
            )
    else
        vcxx_func = nothing
        vcux_func = nothing
        vcuu_func = nothing
    end

    indices_compl = isnothing(indices_compl) ? Int64[] : indices_compl 

    return Constraint(evaluate_func, jacobian_state_func, jacobian_control_func, vcxx_func, vcux_func, vcuu_func,
        num_constraint, num_state, num_control, indices_compl)
end

function Constraint(num_state::Int, num_control::Int)
    return Constraint(
        (c, x, u) -> nothing,
        (c, x, u) -> nothing, (c, x, u) -> nothing,
        (c, x, u, v) -> nothing, (c, x, u, v) -> nothing, (c, x, u, v) -> nothing,
        0, num_state, num_control, Int64[])
end

function Constraint(c::Function, cx::Function, cu::Function, num_constraint::Int, num_state::Int, num_control::Int;
    indices_compl=nothing, vcxx::Function=nothing, vcux::Function=nothing, vcuu::Function=nothing)
    return Constraint(c, cx, cu, vcxx, vcux, vcuu,
        num_constraint, num_state, num_control, indices_compl)
end

function jacobian!(jacobian_states::Vector{Matrix{T}}, jacobian_controls::Vector{Matrix{T}}, constraints::Constraints,
            states::Vector{Vector{T}}, controls::Vector{Vector{T}}) where T
    for (t, con) in enumerate(constraints)
        con.num_constraint == 0 && continue
        con.jacobian_state(jacobian_states[t], states[t], controls[t])
        con.jacobian_control(jacobian_controls[t], states[t], controls[t])
    end
end

function tensor_contraction!(vcxx_cache::Matrix{T}, vcux_cache::Matrix{T}, vcuu_cache::Matrix{T}, constraint::Constraint,
            x::Vector{T}, u::Vector{T}, v::Vector{T}) where T
    if !isnothing(constraint.vcuu)
        constraint.vhxx(vcxx_cache, x, u, v)
        constraint.vhux(vcux_cache, x, u, v)
        constraint.vhuu(vcuu_cache, x, u, v)
    end
end

function tensor_contraction!(vcxx_cache::Vector{Matrix{T}}, vcux_cache::Vector{Matrix{T}}, vcuu_cache::Vector{Matrix{T}}, constraints::Constraints,
            states::Vector{Vector{T}}, controls::Vector{Vector{T}}, v::Vector{Vector{T}}) where T
    N = length(states)
    for t in 1:N
        if !isnothing(constraints[t].vcuu)
            constraints[t].vcxx(vcxx_cache[t], states[t], controls[t], v[t])
            constraints[t].vcux(vcux_cache[t], states[t], controls[t], v[t])
            constraints[t].vcuu(vcuu_cache[t], states[t], controls[t], v[t])
        end
    end
end
