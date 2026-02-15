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
            Ĥ = @views ws[t].kkt_mat[1:nu, 1:nu]
            B = @views ws[t].eq_update_params[1:nu, 2:end]

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
            t < ocp.N && mul!(ws[t].Qû, transpose(ocp.dynamics[t].fu_mem), ws[t+1].x_tmp, 1.0, 1.0)

            ws[t].Qû .-= χl  # barrier gradient
            ws[t].Qû .+= χu  # barrier gradient
            
            # C = Lxx + fx' * Vxx * fx + V̄x ⋅ fxx
            ws[t].V̂xx .= ocp.objective[t].lxx_mem
            if t < ocp.N
                mul!(ws[t].xx_tmp, transpose(ocp.dynamics[t].fx_mem), ws[t+1].V̂xx)
                mul!(ws[t].V̂xx, ws[t].xx_tmp, ocp.dynamics[t].fx_mem, 1.0, 1.0)
            end
    
            # Ĥ = Luu + Σ + fu' * Vxx * fu + V̄x ⋅ fuu
            ws[t].u_tmp1 .*= ws[t].nominal.zl   # Σ^L
            ws[t].u_tmp2 .*= ws[t].nominal.zu   # Σ^U
            fill!(Ĥ, 0.0)
            for i = 1:nu
                Ĥ[i, i] = ws[t].u_tmp1[i] + ws[t].u_tmp2[i]
            end
            if t < ocp.N
                mul!(ws[t].ux_tmp, transpose(ocp.dynamics[t].fu_mem), ws[t+1].V̂xx)
                mul!(Ĥ, ws[t].ux_tmp, ocp.dynamics[t].fu_mem, 1.0, 1.0)
            end
            Ĥ .+= ocp.objective[t].luu_mem
    
            # B = Lux + fu' * Vxx * fx + V̄x ⋅ fxu
            B .= ocp.objective[t].lux_mem
            t < ocp.N && mul!(B, ws[t].ux_tmp, ocp.dynamics[t].fx_mem, 1.0, 1.0)
            
            # apply second order tensor contraction terms to Q̂uu, Q̂ux, Q̂xx
            if !options.quasi_newton
                if t < ocp.N
                    fn_eval_time_ = time()
                    if ocp.no_eq_constr
                        hessians!(ocp.dynamics[t], ws[t], ws[t+1].x_tmp)
                    else
                        hessians!(ocp.dynamics[t], ws[t], ws[t+1].nominal.λ)
                    end
                    data.fn_eval_time += time() - fn_eval_time_
                    ws[t].V̂xx .+= ocp.dynamics[t].fxx_mem
                    B .+= ocp.dynamics[t].fux_mem
                    Ĥ .+= ocp.dynamics[t].fuu_mem
                end

                Ĥ .+= ocp.constraints[t].cuu_mem
                B .+= ocp.constraints[t].cux_mem
                ws[t].V̂xx .+= ocp.constraints[t].cxx_mem
            end
            
            # inertia calculation and correction (regularisation)
            if reg > 0.0
                for i in 1:nu
                    Ĥ[i, i] += reg
                end
            end

            # setup linear system in backward pass
            # @views ws[t].kkt_mat[1:nu, 1:nu] .= ws[t].Ĥ
            @views ws[t].kkt_mat[1:nu, nu+1:end] .= transpose(ocp.constraints[t].cu_mem)
            @views ws[t].kkt_mat[nu+1:end, nu+1:end] .= 0

            α .= ws[t].Qû
            α .*= -1.0
            ψ .= ws[t].nominal.c
            ψ .*= -1.0
            ws[t].ux_tmp .= B  # save B before overwritten by linear solve  
            β .*= -1.0
            ω .= ocp.constraints[t].cx_mem
            ω .*= -1.0

            if δ_c > 0.0
                for i in 1:nc
                    @views ws[t].kkt_mat[nu+1:end, nu+1:end][i, i] -= δ_c
                end
            end
            
            data.status, info, rank = ocp.no_eq_constr ? factorise!(
                ws[t].kkt_mat_ws, ws[t].kkt_mat, options.linsolve_tol) : factorise!(
                    ws[t].kkt_mat_ws, ws[t].kkt_mat, ws[t].kkt_D_cache, nu, options.linsolve_tol)   
            fct = ocp.no_eq_constr ? LinearAlgebra.CholeskyPivoted(
                    ws[t].kkt_mat, 'U', ws[t].kkt_mat_ws.piv, rank, options.linsolve_tol, info) :
                    LinearAlgebra.BunchKaufman(ws[t].kkt_mat, ws[t].kkt_mat_ws.ipiv, 'U', true, true, info)

            if data.status == 2
                δ_c = options.δ_c * μ^options.κ_c
                break
            elseif data.status == 1 
                reg = update_reg(reg, data.reg_last, options)
                break
            end
            ldiv!(fct, ws[t].eq_update_params)

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
            mul!(ws[t].V̂xx, transpose(β), ws[t].ux_tmp, 1.0, 1.0)
            mul!(ws[t].V̂xx, transpose(ω), ocp.constraints[t].cx_mem, 1.0, 1.0)
            # ws[t].V̂xx .+= ws[t].C

            # Vx = Lx' + β' * Qû + ω' c + fx' Vx+
            ws[t].x_tmp .= ocp.objective[t].lx_mem
            mul!(ws[t].x_tmp, transpose(ocp.constraints[t].cx_mem), ws[t].nominal.ϕ, 1.0, 1.0)
            ws[t].nominal.λ .= ws[t].x_tmp
            mul!(ws[t].x_tmp, transpose(β), ws[t].Qû, 1.0, 1.0)
            mul!(ws[t].x_tmp, transpose(ω), ws[t].nominal.c, 1.0, 1.0)
            t < ocp.N && mul!(ws[t].x_tmp, transpose(ocp.dynamics[t].fx_mem), ws[t+1].x_tmp, 1.0, 1.0)

            # λ = Lx' + fx' λ+
            t < ocp.N && mul!(ws[t].nominal.λ, transpose(ocp.dynamics[t].fx_mem), ws[t+1].nominal.λ, 1.0, 1.0)
        end
        data.status == 0 && break
    end
    data.reg_last = reg
    data.status != 0 && (verbose && (@warn "Backward pass failure, unable to find an iteration matrix with correct inertia."))
end

function factorise!(ws::BunchKaufmanWs{T}, kkt_mat::Matrix{T}, D_cache::Pair{Vector{T}}, nu::Int64, tol::T) where T
    kkt_mat, ipiv, info = LAPACK.sytrf_rook!(ws, 'U', kkt_mat)
    bk = LinearAlgebra.BunchKaufman(kkt_mat, ipiv, 'U', true, true, info)
    info > 0 && return 2, info, 0      # K is singular (constraint Jacobian may be not full row rank), add bottom right diagonal
    np, _, _ = inertia!(bk, D_cache[1], D_cache[2]; atol=tol)
    (np != nu || info < 0) && return 1, info, 0
    return 0, info, 0
end

function factorise!(ws::CholeskyPivotedWs{T}, kkt_mat::Matrix{T}, tol::T) where T
    _, piv, rank, info = LinearAlgebra.LAPACK.pstrf!(ws, 'U', kkt_mat, tol)
    if info == 0
        return 0, info, rank
    else
        return 1, info, rank
    end
end

function update_reg(reg::T, reg_last::T, options::Options{T}) where T
    if iszero(reg) # initial setting of regularisation
        reg = (reg_last == 0.0) ? options.reg_1 : max(options.reg_min, options.κ_w_m * reg_last)
    else
        reg = (reg_last == 0.0) ? options.κ_̄w_p * reg : options.κ_w_p * reg
    end
    return reg
end
