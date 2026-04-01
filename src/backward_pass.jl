function backward_pass!(ocp::OCP{T, nx, nu, nc}, ws::FilterDDPWorkspace{T, nx, nu, nc}, data::SolverData{T},
            options::Options{T}; verbose::Bool=false) where {T, nx, nu, nc}
    reg::T = 0.0
    μ = data.μ
    δ_c = 0.
    reg = 0.0
    
    while reg <= options.reg_max
        data.status = 0
        
        for t = ocp.N:-1:1
            inv_ul = inv.(ws[t].nominal.ul) .* ocp.control_limits[t].maskl
            inv_uu = inv.(ws[t].nominal.uu) .* ocp.control_limits[t].masku

            # Qû = Lu' -μŪ^{-1}e + fu' * V̂x
            ws[t].Qû = ocp.objective[t].lu_mem + ocp.constraints[t].cu_mem' * ws[t].nominal.ϕ
            if t < ocp.N
                ws[t].Qû = ws[t].Qû + ocp.dynamics[t].fu_mem' * ws[t+1].V̂x
            end
            ws[t].Qû = ws[t].Qû + μ .* (inv_uu - inv_ul)   # barrier gradient
            
            # C = Lxx + fx' * Vxx * fx + V̄x ⋅ fxx
            C = ocp.objective[t].lxx_mem
            if t < ocp.N
                C = C + ocp.dynamics[t].fx_mem' * ws[t+1].V̂xx * ocp.dynamics[t].fx_mem
            end
    
            # Ĥ = Luu + Σ + fu' * Vxx * fu + V̄x ⋅ fuu
            Σ_L = inv_ul .* ws[t].nominal.zl
            Σ_U = inv_uu .* ws[t].nominal.zu
            Ĥ = ocp.objective[t].luu_mem + diagm(Σ_L) + diagm(Σ_U)
    
            # B = Lux + fu' * Vxx * fx + V̄x ⋅ fxu
            B = ocp.objective[t].lux_mem
            if t < ocp.N
                ux_tmp = ocp.dynamics[t].fu_mem' * ws[t+1].V̂xx
                Ĥ = Ĥ + ux_tmp * ocp.dynamics[t].fu_mem
                B = B + ux_tmp * ocp.dynamics[t].fx_mem
            end
            
            # apply second order tensor contraction terms to Q̂uu, Q̂ux, Q̂xx
            if !options.quasi_newton
                if t < ocp.N
                    fn_eval_time_ = time()
                    if ocp.no_eq_constr
                        hessians!(ocp.dynamics[t], ws[t], ws[t+1].V̂x)
                    else
                        hessians!(ocp.dynamics[t], ws[t], ws[t+1].nominal.λ)
                    end
                    data.fn_eval_time += time() - fn_eval_time_
                    C = C + ocp.dynamics[t].fxx_mem
                    B = B + ocp.dynamics[t].fux_mem
                    Ĥ = Ĥ + ocp.dynamics[t].fuu_mem
                end

                Ĥ = Ĥ + ocp.constraints[t].cuu_mem
                B = B + ocp.constraints[t].cux_mem
                C = C + ocp.constraints[t].cxx_mem
            end
            
            # inertia correction / regularisation
            Ĥ = Ĥ + diagm(reg * SVector{nu, T}(ones(T, nu)))

            # if data.k == 0 && t == ocp.N - 56
            #     println("t ", t, " ", ws[t].Qû)
            #     println("t ", t, " ", ws[t].nominal.c)
            #     println("t ", t, " ", B)
            #     println("t ", t, " ", Ĥ)
            #     println("t ", t, " ", ws[t+1].V̂xx)
            # end

            data.status = factorize_ns!(Ĥ, ocp.constraints[t].cu_mem', ws[t].Qû, ws[t].nominal.c, B,
                    ocp.constraints[t].cx_mem, ws[t], options)
            
            if data.status == 1
                if iszero(reg) # initial setting of regularisation
                    reg = (data.reg_last == 0.0) ? options.reg_1 : max(options.reg_min, options.κ_w_m * data.reg_last)
                else
                    reg = (data.reg_last == 0.0) ? options.κ_̄w_p * reg : options.κ_w_p * reg
                end
                break
            end

            # if data.k == 0 && t == ocp.N - 56
            #     println("t ", t, " ", ws[t].α)
            #     println("t ", t, " ", ws[t].β)
            #     throw("oh fuck")
            # end
            
            # update parameters of update rule for ineq. dual variables, i.e., 

            # χ_L =  μ inv.(u - u^L) - z^L - Σ^L * α 
            # ζ_L =  - Σ^L * β 
            # χ_U =  μ inv.(u^U - u) - z^U + Σ^U .* α 
            # ζ_U =  Σ^U .* β

            # see update above for Q̂u[t] for first part of χ^L[t] χ^U[t]

            ws[t].ζl =  -ws[t].β .* Σ_L
            ws[t].χl = inv_ul .* μ - ws[t].nominal.zl - Σ_L .* ws[t].α

            ws[t].ζu =  Σ_U .* ws[t].β
            ws[t].χu = inv_uu .* μ - ws[t].nominal.zu + Σ_U .* ws[t].α

            # Update return V derivatives for next timestep Vxx = C + β' * B + ω' cx
            ws[t].V̂xx = C + ws[t].β' * B 
            if nc > 0
                ws[t].V̂xx = ws[t].V̂xx  + ws[t].ω' * ocp.constraints[t].cx_mem
            end

            # Vx = Lx' + β' * Qû + ω' c + fx' Vx+
            ws[t].V̂x = ocp.objective[t].lx_mem + ocp.constraints[t].cx_mem' * ws[t].nominal.ϕ
            ws[t].nominal.λ = ws[t].V̂x
            ws[t].V̂x = ws[t].V̂x + ws[t].β' * ws[t].Qû + ws[t].ω' * ws[t].nominal.c
            if t < ocp.N
                ws[t].V̂x = ws[t].V̂x + ocp.dynamics[t].fx_mem' * ws[t+1].V̂x
            end

            # λ = Lx' + fx' λ+
            if t < ocp.N
                ws[t].nominal.λ = ws[t].nominal.λ + ocp.dynamics[t].fx_mem' * ws[t+1].nominal.λ
            end
        end

        data.status == 0 && break
    end
    data.reg_last = reg
    data.status != 0 && (verbose && (@warn "Backward pass failure, unable to find an iteration matrix with correct inertia."))
end


function factorize_ns!(H::SMatrix{nu, nu, T}, A::SMatrix{nu, nc, T}, Qu::SVector{nu, T}, c::SVector{nc, T},
            B::SMatrix{nu, nx, T}, cx::SMatrix{nc, nx, T}, wse::FilterDDPWorkspaceElement{T, nx, nu, nc},
            options::Options{T}) where {T, nx, nu, nc}
    if nc > 0
        Q, R = qr([A SMatrix{nu, nu-nc, T}(zeros(T, nu, nu-nc))])
        Y = Q[:, SVector{nc, Int64}(1:nc)]
        Z = Q[:, SVector{nu-nc, Int64}(nc+1:nu)]
        
        AY = LowerTriangular(A' * Y)
        fk = lu(AY)
        α_β_y = fk \ [-c -cx]

        M = Symmetric(Z' * H * Z)
        ck = cholesky(M; check=false)
        ck.info != 0 && return 1

        α_β_z = ck \ (Z' * ([-Qu -B] - H * Y * α_β_y))
        α_β = Y * α_β_y + Z * α_β_z
        wse.α = α_β[:, 1]
        wse.β = α_β[:, SVector{nx, Int64}(2:nx+1)]

        ψ_ω = ((Y' * ([-Qu -B] - H * α_β))' / fk)'
        wse.ψ = ψ_ω[:, 1]
        wse.ω = ψ_ω[:, SVector{nx, Int64}(2:nx+1)]
    else
        ck = cholesky(Symmetric(H); check=false)
        ck.info != 0 && return 1

        α_β = ck \ [-Qu -B]
        wse.α = α_β[:, 1]
        wse.β = α_β[:, SVector{nx, Int64}(2:nx+1)]
    end

    return 0
end
