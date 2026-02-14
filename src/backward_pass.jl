function backward_pass!(ocp::OCP{T}, ws::FilterDDPWorkspace{T}, data::SolverData{T},
            options::Options{T}; verbose::Bool=false) where T
    reg::T = 0.0
    μ = data.μ
    δ_c = 0.
    reg = 0.0
    
    while reg <= options.reg_max
        data.status = 0
        
        for t = ocp.N:-1:1
            nu = ocp.nu[t]
            nc = ocp.nc[t]
            
            χl = @views ws[t].ineq_update_params[1:nu, 1]
            χu = @views ws[t].ineq_update_params[nu+1:end, 1]
            α = @views ws[t].eq_update_params[1:nu, 1]
            ψ = @views ws[t].eq_update_params[nu+1:end, 1]
            β = @views ws[t].eq_update_params[1:nu, 2:end]
            ω = @views ws[t].eq_update_params[nu+1:end, 2:end]
            ζl = @views ws[t].ineq_update_params[1:nu, 2:end]
            ζu = @views ws[t].ineq_update_params[nu+1:end, 2:end]

            ws[t].u_tmp1 .= inv.(ws[t].nominal.ul)
            ws[t].u_tmp2 .= inv.(ws[t].nominal.uu)

            # just loop???
            χl .= ws[t].u_tmp1
            χl .*= μ
            χu .= ws[t].u_tmp2
            χu .*= μ

            # Qû = Lu' -μŪ^{-1}e + fu' * V̂x
            ws[t].Qû .= ocp.objective[t].lu_mem
            mul!(ws[t].Qû, transpose(ocp.constraints[t].cu_mem), ws[t].nominal.ϕ, 1.0, 1.0)
            t < ocp.N && mul!(ws[t].Qû, transpose(ocp.dynamics[t].fu_mem), ws[t+1].V̄x, 1.0, 1.0)

            ws[t].Qû .-= χl  # barrier gradient
            ws[t].Qû .+= χu  # barrier gradient
            
            # C = Lxx + fx' * Vxx * fx + V̄x ⋅ fxx
            ws[t].C .= ocp.objective[t].lxx_mem
            if t < ocp.N
                mul!(ws[t].xx_tmp, transpose(ocp.dynamics[t].fx_mem), ws[t+1].V̂xx)
                mul!(ws[t].C, ws[t].xx_tmp, ocp.dynamics[t].fx_mem, 1.0, 1.0)
            end
    
            # Ĥ = Luu + Σ + fu' * Vxx * fu + V̄x ⋅ fuu
            ws[t].u_tmp1 .*= ws[t].nominal.zl   # Σ^L
            ws[t].u_tmp2 .*= ws[t].nominal.zu   # Σ^U
            fill!(ws[t].Ĥ, 0.0)
            for i = 1:nu
                ws[t].Ĥ[i, i] = ws[t].u_tmp1[i] + ws[t].u_tmp2[i]
            end
            if t < ocp.N
                mul!(ws[t].ux_tmp, transpose(ocp.dynamics[t].fu_mem), ws[t+1].V̂xx)
                mul!(ws[t].Ĥ, ws[t].ux_tmp, ocp.dynamics[t].fu_mem, 1.0, 1.0)
            end
            ws[t].Ĥ .+= ocp.objective[t].luu_mem
    
            # B = Lux + fu' * Vxx * fx + V̄x ⋅ fxu
            ws[t].B .= ocp.objective[t].lux_mem
            t < ocp.N && mul!(ws[t].B, ws[t].ux_tmp, ocp.dynamics[t].fx_mem, 1.0, 1.0)
            
            # apply second order tensor contraction terms to Q̂uu, Q̂ux, Q̂xx
            if !options.quasi_newton
                if t < ocp.N
                    fn_eval_time_ = time()
                    hessians!(ocp.dynamics[t], ws[t], ws[t+1])
                    data.fn_eval_time += time() - fn_eval_time_
                    ws[t].C .+= ocp.dynamics[t].fxx_mem
                    ws[t].B .+= ocp.dynamics[t].fux_mem
                    ws[t].Ĥ .+= ocp.dynamics[t].fuu_mem
                end

                ws[t].Ĥ .+= ocp.constraints[t].cuu_mem
                ws[t].B .+= ocp.constraints[t].cux_mem
                ws[t].C .+= ocp.constraints[t].cxx_mem
            end
            
            # inertia calculation and correction (regularisation)
            if reg > 0.0
                for i in 1:nu
                    ws[t].Ĥ[i, i] += reg
                end
            end

            # setup linear system in backward pass
            @views ws[t].kkt_mat[1:nu, 1:nu] .= ws[t].Ĥ
            @views ws[t].kkt_mat[1:nu, nu+1:end] .= transpose(ocp.constraints[t].cu_mem)
            @views ws[t].kkt_mat[nu+1:end, nu+1:end] .= 0

            α .= ws[t].Qû
            α .*= -1.0
            ψ .= ws[t].nominal.c
            ψ .*= -1.0
            β .= ws[t].B
            β .*= -1.0
            ω .= ocp.constraints[t].cx_mem
            ω .*= -1.0

            if δ_c > 0.0
                for i in 1:nc
                    @views ws[t].kkt_mat[nu+1:end, nu+1:end][i, i] -= δ_c
                end
            end
        
            bk, data.status, reg, δ_c = inertia_correction!(ws[t].kkt_mat_ws, ws[t].kkt_mat,
                                ws[t].kkt_D_cache, nu, μ, reg, data.reg_last, options)
            data.status != 0 && break

            ldiv!(bk, ws[t].eq_update_params)

            # update parameters of update rule for ineq. dual variables, i.e., 

            # χ_L =  μ inv.(u - u^L) - z^L - Σ^L * α 
            # ζ_L =  - Σ^L * β 
            # χ_U =  μ inv.(u^U - u) - z^U + Σ^U .* α 
            # ζ_U =  Σ^U .* β

            # see update above for Q̂u[t] for first part of χ^L[t] χ^U[t]

            ζl .= β
            ζl .*= ws[t].u_tmp1
            ζl .*= -1.0

            ws[t].u_tmp1 .*= α
            χl .-= ws[t].nominal.zl
            χl .-= ws[t].u_tmp1

            ζu .= β
            ζu .*= ws[t].u_tmp2

            ws[t].u_tmp2 .*= α
            χu .-= ws[t].nominal.zu
            χu .+= ws[t].u_tmp2

            # Update return function approx. for next timestep 
            # Vxx = C + β' * B + ω' cx
            mul!(ws[t].V̂xx, transpose(β), ws[t].B)
            mul!(ws[t].V̂xx, transpose(ω), ocp.constraints[t].cx_mem, 1.0, 1.0)
            ws[t].V̂xx .+= ws[t].C

            # Vx = Lx' + β' * Qû + ω' c + fx' Vx+
            ws[t].V̄x .= ocp.objective[t].lx_mem
            mul!(ws[t].V̄x, transpose(ocp.constraints[t].cx_mem), ws[t].nominal.ϕ, 1.0, 1.0)
            ws[t].nominal.λ .= ws[t].V̄x
            mul!(ws[t].V̄x, transpose(β), ws[t].Qû, 1.0, 1.0)
            mul!(ws[t].V̄x, transpose(ω), ws[t].nominal.c, 1.0, 1.0)
            t < ocp.N && mul!(ws[t].V̄x, transpose(ocp.dynamics[t].fx_mem), ws[t+1].V̄x, 1.0, 1.0)

            # λ = Lx' + fx' λ+
            t < ocp.N && mul!(ws[t].nominal.λ, transpose(ocp.dynamics[t].fx_mem), ws[t+1].nominal.λ, 1.0, 1.0)
        end
        data.status == 0 && break
    end
    data.reg_last = reg
    data.status != 0 && (verbose && (@warn "Backward pass failure, unable to find an iteration matrix with correct inertia."))
end
