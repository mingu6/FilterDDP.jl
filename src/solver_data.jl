"""
    Solver Data
"""

# solver status 0 - OK, 1 - backward pass failed, 2 - failed frac-to-bound, 3 - filter blocked,
#               4 - failed setep acceptance, 5 - f/w failed, 6 - failed SOC, 7 - failed forward pass line search,
#               8 - max iterations reached
mutable struct SolverData{T}
    max_primal_1::T               # maximum allowable 1-norm of constraint violation (IPOPT θ_max)
    min_primal_1::T               # minimum 1-norm of constraint violation (IPOPT θ_min) 
    step_size::T                  # current step size for line search
    status::Int                   # solver status
    j::Int                        # outer iteration counter (i.e., j-th barrier subproblem)
    k::Int                        # overall iteration counter
    l::Int                        # line search iteration counter
    μ::T                          # current subproblem perturbation value
    reg_last::T                   # regularisation in backward pass
    objective::T                  # objective function value of current iterate
    primal_inf::T                 # ∞-norm of constraint violation (primal infeasibility)
    dual_inf::T                   # ∞-norm of gradient of Lagrangian (dual infeasibility)
    cs_inf_0::T                   # ∞-norm of complementary slackness error
    cs_inf_μ::T                   # ∞-norm of perturbed complementary slackness error
    barrier_lagrangian_curr::T    # barrier Lagrangian function for subproblem at current iterate
    primal_1_curr::T              # 1-norm of constraint violation at current iterate (primal infeasibility)
    barrier_lagrangian_next::T    # barrier Lagrangian function for subproblem at next iterate
    primal_1_next::T              # 1-norm of constraint violation at next iterate (primal infeasibility)
    expected_change_L::T          # expected change in Lagrangian for sufficient decrease condition (Armijo)
    update_filter::Bool           # updated filter at current iteration
    switching::Bool               # switching condition satisfied (sufficient decrease on barrier obj. relative to constr. viol.)
    armijo_passed::Bool           # sufficient decrease condition of barrier obj. satisfied for current iterate
    filter::Vector{Vector{T}}     # filter points TODO: move to staticarrays
end

function solver_data(T)
    max_primal_1 = T(1e6)
    min_primal_1 = T(1e-5)
    step_size = T(0.0)
    status = 0
    j = 0
    k = 0
    l = 0
    μ = T(0.0)
    reg_last = T(0.0)
    objective = T(0.0)
    primal_inf = T(0.0)
    dual_inf = T(0.0)
    cs_inf_0 = T(0.0)
    cs_inf_μ = T(0.0)
    barrier_lagrangian_curr = T(0.0)
    primal_1_curr = T(0.0)
    barrier_lagrangian_next = T(0.0)
    primal_1_next = T(0.0)
    expected_change_L = T(0.0)
    update_filter = false
    switching = false
    armijo_passed = false
    filter = [[T(0.0) , T(0.0)]]

    SolverData(max_primal_1, min_primal_1, step_size, status, j, k, l, 
        μ, reg_last, objective, primal_inf, dual_inf, cs_inf_0, cs_inf_μ, 
        barrier_lagrangian_curr, primal_1_curr, barrier_lagrangian_next, primal_1_next, expected_change_L,
        update_filter, switching, armijo_passed, filter)
end

function reset!(data::SolverData{T}) where T 
    data.max_primal_1 = 1e6
    data.min_primal_1 = 0.0
    data.step_size = 0.0
    data.status = 0
    data.j = 0
    data.k = 0
    data.l = 0
    data.μ = 0.0
    data.reg_last = 0.0
    data.objective = 0.0
    data.primal_inf = 0.0
    data.dual_inf = 0.0
    data.cs_inf_0 = 0.0
    data.cs_inf_μ = 0.0
    data.barrier_lagrangian_curr = 0.0
    data.primal_1_curr = 0.0
    data.barrier_lagrangian_next = 0.0
    data.primal_1_next = 0.0
    data.expected_change_L = 0.0
    data.update_filter = false
    data.switching = false
    data.armijo_passed = false
    data.filter = [[0.0 , 0.0]]
end
