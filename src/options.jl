Base.@kwdef mutable struct Options{T}
    quasi_newton::Bool = false
    optimality_tolerance::T = 1.0e-8
    max_iterations::Int = 1000
    reset_cache::Bool = true
    verbose = false
    print_frequency = 10
    
    μ_init::T = 1.0                # multiplier on barrier parameter initialisation
    ineq_dual_init::T = 1.0        # dual variable initialisation value
    κ_1::T = 0.01                  # fraction-to-boundary initialization for controls
    κ_2::T = 0.01                  # fraction-to-boundary initialization for controls
    
    reg_1::T = 1e-4
    reg_min::T = 1e-20
    reg_max::T = 1e40
    κ_̄w_p::T = 100.0
    κ_w_p::T = 8.0
    κ_w_m::T = 1.0 / 3.0
    κ_c::T = 0.25
    δ_c::T = 1e-8

    κ_ϵ::T = 10.0                  # tolerance factor for lowering barrier parameter
    κ_μ::T = 0.2                   # linear decrease factor for barrier parameter
    θ_μ::T = 1.2                   # superlinear decrease factor for barrier parameter
    τ_min::T = 0.99                # lower bound on fraction-to-boundary parameter
    
    s_max::T = 100.0               # scaling threshold for NLP error
    η_L::T = 1e-4                  # relaxation factor in the armijo condition
    s_L::T = 2.3                   # exponent for linear barrier function model in switching rule
    δ::T = 1.0                     # multiplier for constraint violation in the switching rule
    s_θ::T = 1.1                   # exponent for current constraint violation in the switching rule
    γ_α::T = 0.05                  # safety factor for minimum step size (in (0, 1))
    γ_θ::T = 1e-5                  # relaxation factor in the filter margin for constraint violation (in (0, 1))
    γ_L::T = 1e-5                  # relaxation factor in the filter margin for barrier function (in (0, 1))

    κ_Σ::T = 1e10                  # dual variable rescaling threshold param.

    linsolve_tol::T = 1e-12        # threshold for non-singularity in linear solver
end