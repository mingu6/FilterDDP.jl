using Debugger

function backward_pass!(update_rule::UpdateRuleData{T}, problem::ProblemData{T}, data::SolverData{T},
            options::Options{T}; mode=:nominal, verbose::Bool=false) where T
    N = problem.horizon
    reg::T = 0.0

    # Jacobians of system dynamics
    fx = problem.model.jacobian_state
    fu = problem.model.jacobian_control
    # Tensor contraction of system dynamics (DDP)
    vfxx = problem.model.vfxx
    vfux = problem.model.vfux
    vfuu = problem.model.vfuu
    # Jacobian caches
    cx = problem.constraints_data.jacobian_state
    cu = problem.constraints_data.jacobian_control
    # Tensor contraction of constraints (DDP)
    vcxx = problem.constraints_data.vcxx
    vcux = problem.constraints_data.vcux
    vcuu = problem.constraints_data.vcuu
    # Objective gradients
    lx = problem.objective_data.gradient_state
    lu = problem.objective_data.gradient_control
    # Objective hessians
    lxx = problem.objective_data.hessian_state_state
    luu = problem.objective_data.hessian_control_control
    lux = problem.objective_data.hessian_control_state
    # DDP intermediate variables
    Qû = update_rule.Qû
    C = update_rule.C
    Ĥ = update_rule.Ĥ
    B = update_rule.B
    # Update rule parameters
    α = update_rule.parameters.α
    β = update_rule.parameters.β
    ψ = update_rule.parameters.ψ
    ω = update_rule.parameters.ω
    χl = update_rule.parameters.χl
    ζl = update_rule.parameters.ζl
    χu = update_rule.parameters.χu
    ζu = update_rule.parameters.ζu
    # Value function
    Vx = update_rule.value.gradient
    Vxx = update_rule.value.hessian

    u_tmp1 = update_rule.u_tmp1
    u_tmp2 = update_rule.u_tmp2
    
    x, u, c, il, iu = primal_trajectories(problem, mode=mode)
    ϕ, zl, zu, λ = dual_trajectories(problem, mode=mode)

    μ = data.μ
    δ_c = 0.
    reg = 0.0
    
    while reg <= options.reg_max
        data.status = 0
        
        for t = N:-1:1
            num_control = length(u[t])
            num_constr = length(c[t])

            u_tmp1[t] .= inv.(il[t])
            u_tmp2[t] .= inv.(iu[t])

            χl[t] .= u_tmp1[t]
            χl[t] .*= μ
            χu[t] .= u_tmp2[t]
            χu[t] .*= μ

            # Qû = Lu' -μŪ^{-1}e + fu' * V̂x
            Qû[t] .= lu[t]
            mul!(Qû[t], transpose(cu[t]), ϕ[t], 1.0, 1.0)
            t < N && mul!(Qû[t], transpose(fu[t]), Vx[t+1], 1.0, 1.0)
            Qû[t] .-= χl[t]  # barrier gradient
            Qû[t] .+= χu[t]  # barrier gradient
            
            # C = Lxx + fx' * Vxx * fx + V̄x ⋅ fxx
            C[t] .= lxx[t]
            if t < N
                mul!(update_rule.xx_tmp[t], transpose(fx[t]), Vxx[t+1])
                mul!(C[t], update_rule.xx_tmp[t], fx[t], 1.0, 1.0)
            end
    
            # Ĥ = Luu + Σ + fu' * Vxx * fu + V̄x ⋅ fuu
            u_tmp1[t] .*= zl[t]   # Σ^L
            u_tmp2[t] .*= zu[t]   # Σ^U
            fill!(Ĥ[t], 0.0)
            for i = 1:num_control
                Ĥ[t][i, i] = u_tmp1[t][i] + u_tmp2[t][i]
            end
            if t < N
                mul!(update_rule.ux_tmp[t], transpose(fu[t]), Vxx[t+1])
                mul!(Ĥ[t], update_rule.ux_tmp[t], fu[t], 1.0, 1.0)
            end
            Ĥ[t] .+= luu[t]
    
            # B = Lux + fu' * Vxx * fx + V̄x ⋅ fxu
            B[t] .= lux[t]
            t < N && mul!(B[t], update_rule.ux_tmp[t], fx[t], 1.0, 1.0)
            
            # apply second order tensor contraction terms to Q̂uu, Q̂ux, Q̂xx
            if !options.quasi_newton
                if t < N
                    fn_eval_time_ = time()
                    tensor_contraction!(vfxx[t], vfux[t], vfuu[t], problem.model.dynamics[t], x[t], u[t], λ[t+1])
                    data.fn_eval_time += time() - fn_eval_time_
                    C[t] .+= vfxx[t]
                    B[t] .+= vfux[t]
                    Ĥ[t] .+= vfuu[t]
                end

                Ĥ[t] .+= vcuu[t]
                B[t] .+= vcux[t]
                C[t] .+= vcxx[t]
            end
            
            # inertia calculation and correction (regularisation)
            if reg > 0.0
                for i in 1:num_control
                    Ĥ[t][i, i] += reg
                end
            end

            # setup linear system in backward pass
            update_rule.lhs_tl[t] .= Symmetric(Ĥ[t])
            update_rule.lhs_tr[t] .= transpose(cu[t])
            update_rule.lhs_bl[t] .= cu[t] ## only for out of place
            fill!(update_rule.lhs_br[t], 0.0)

            α[t] .= Qû[t]
            α[t] .*= -1.0
            ψ[t] .= c[t]
            ψ[t] .*= -1.0
            β[t] .= B[t]
            β[t] .*= -1.0
            ω[t] .= cx[t]
            ω[t] .*= -1.0

            if δ_c > 0.0
                for i in 1:num_constr
                    update_rule.lhs_br[t][i, i] -= δ_c
                end
            end

            if num_control > 0
                # bk, data.status, reg, δ_c = inertia_correction!(update_rule.kkt_matrix_ws[t], update_rule.lhs[t], update_rule.D_cache[t],
                #             num_control, μ, reg, data.reg_last, options)
                data.status, reg = inertia_correction_nullspace(update_rule.lhs_tl[t], cu[t], update_rule.parameters.eq[t],
                        num_control, num_constr, reg, data.reg_last, options)
            end
            data.status != 0 && break


            # update parameters of update rule for ineq. dual variables, i.e., 

            # χ_L =  μ inv.(u - u^L) - z^L - Σ^L * α 
            # ζ_L =  - Σ^L * β 
            # χ_U =  μ inv.(u^U - u) - z^U + Σ^U .* α 
            # ζ_U =  Σ^U .* β

            # see update above for Q̂u[t] for first part of χ^L[t] χ^U[t]

            ζl[t] .= β[t]
            ζl[t] .*= u_tmp1[t]
            ζl[t] .*= -1.0

            u_tmp1[t] .*= α[t]
            χl[t] .-= zl[t]
            χl[t] .-= u_tmp1[t]

            ζu[t] .= β[t]
            ζu[t] .*= u_tmp2[t]

            u_tmp2[t] .*= α[t]
            χu[t] .-= zu[t]
            χu[t] .+= u_tmp2[t]

            # Update return function approx. for next timestep 
            # Vxx = C + β' * B + ω' cx
            mul!(Vxx[t], transpose(β[t]), B[t])
            mul!(Vxx[t], transpose(ω[t]), cx[t], 1.0, 1.0)
            Vxx[t] .+= C[t]

            # Vx = Lx' + β' * Qû + ω' c + fx' Vx+
            Vx[t] .= lx[t]
            mul!(Vx[t], transpose(cx[t]), ϕ[t], 1.0, 1.0)
            λ[t] .= Vx[t]
            mul!(Vx[t], transpose(β[t]), Qû[t], 1.0, 1.0)
            mul!(Vx[t], transpose(ω[t]), c[t], 1.0, 1.0)
            t < N && mul!(Vx[t], transpose(fx[t]), Vx[t+1], 1.0, 1.0)

            # λ = Lx' + fx' λ+
            t < N && mul!(λ[t], transpose(fx[t]), λ[t+1], 1.0, 1.0)
        end
        data.status == 0 && break
    end
    data.reg_last = reg
    data.status != 0 && (verbose && (@warn "Backward pass failure, unable to find positive definite iteration matrix."))
end
