function backward_pass!(solver::Solver{T, nx, nu, nc}, ocp::OCP{T, nx, nu, nc}, traj::Vector{TrajectoryElement{T, nx, nu, nc}},
            data::SolverData{T}, options::Options{T}; verbose::Bool=false) where {T, nx, nu, nc}
    reg::T = 0.0
    μ = data.μ
    δ_c = 0.
    reg = 0.0

    constraints = ocp.constraints
    dynamics = ocp.dynamics
    cl = ocp.control_limits
    ni = (cl.nl + cl.nu) * ocp.N
    
    while reg <= options.reg_max
        data.status = 0
        V̂x = @SVector zeros(T, nx)
        V̂xx = @SMatrix zeros(T, nx, nx)
        λ = @SVector zeros(T, nx)

        data.barrier_lagrangian_curr = T(0.0)
        data.primal_1_curr = T(0.0)
        data.primal_inf = T(0.0)
        data.cs_inf_μ = T(0.0)
        data.cs_inf_0 = T(0.0)
        data.objective = T(0.0)
        data.dual_inf = T(0.0)
        data.expected_change_L = T(0.0)
        ϕ_norm = T(0.0)
        z_norm = T(0.0) 
        
        for t = ocp.N:-1:1
            x, u, ϕ, zl, zu = traj[t].x, traj[t].u, traj[t].ϕ, traj[t].zl, traj[t].zu

            # evaluate derivatives

            objective = t == ocp.N ? ocp.term_objective : ocp.stage_objective
            lx = objective.lx(x, u)
            lu_ = objective.lu(x, u)
            lxx = objective.lxx(x, u)
            lux = objective.lux(x, u)
            luu = objective.luu(x, u)
            
            if nc > 0
                c = constraints.c(x, u)
                cx = constraints.cx(x, u)
                cu = constraints.cu(x, u)
                cxx = constraints.cxx(x, u, ϕ)
                cux = constraints.cux(x, u, ϕ)
                cuu = constraints.cuu(x, u, ϕ)

                # evaluate constraint violation norms

                data.primal_1_curr += norm(c, 1)
                data.primal_inf = max(data.primal_inf, norm(c, Inf))
            end

            fx = dynamics.fx(x, u)
            fu = dynamics.fu(x, u)
            fxx = nc == 0 ? dynamics.fxx(x, u, V̂x) : dynamics.fxx(x, u, λ)
            fux = nc == 0 ? dynamics.fux(x, u, V̂x) : dynamics.fux(x, u, λ)
            fuu = nc == 0 ? dynamics.fuu(x, u, V̂x) : dynamics.fuu(x, u, λ)

            # evaluate barrier Lagrangian

            ul = u - cl.l
            uu = cl.u - u
            data.barrier_lagrangian_curr -= μ* (dot(log.(ul), cl.maskl) + dot(log.(uu), cl.masku))
            data.objective += objective.l(x, u)[1]

            # evaluate complementary slackness errors
            cs_l = ul .* zl
            cs_u = uu .* zu
            data.cs_inf_0 = max(data.cs_inf_0, norm(cs_l .* cl.maskl, Inf))
            data.cs_inf_μ = max(data.cs_inf_μ, norm((cs_l .- μ).* cl.maskl, Inf))
            data.cs_inf_0 = max(data.cs_inf_0, norm(cs_u .* cl.masku, Inf))
            data.cs_inf_μ = max(data.cs_inf_μ, norm((cs_u .- μ) .* cl.masku, Inf))

            inv_ul = inv.(ul) .* cl.maskl
            inv_uu = inv.(uu) .* cl.masku

            # Qû = Lu' -μŪ^{-1}e + fu' * V̂x
            Qû = lu_ + fu' * V̂x + μ .* (inv_uu - inv_ul)
            
            # C = Lxx + fx' * Vxx * fx + V̄x ⋅ fxx
            C = lxx + fx' * V̂xx * fx + fxx
    
            ux_tmp = fu' * V̂xx
            # Ĥ = Luu + Σ + fu' * Vxx * fu + V̄x ⋅ fuu
            Σ_L = inv_ul .* zl
            Σ_U = inv_uu .* zu
            Ĥ = luu + diagm(Σ_L) + diagm(Σ_U) + ux_tmp * fu + fuu
    
            # B = Lux + fu' * Vxx * fx + V̄x ⋅ fux
            B = lux + ux_tmp * fx + fux

            if nc > 0
                data.barrier_lagrangian_curr += dot(c, ϕ)
                Qû = Qû + cu' * ϕ
                C = C + cxx
                Ĥ = Ĥ + cuu
                B = B + cux
            end
            
            # inertia correction / regularisation
            Ĥ = Ĥ + diagm(reg * SVector{nu, T}(ones(T, nu)))

            # factorise KKT system using nullspace method
            if nc > 0
                A = cu'
                Q, R = qr([A SMatrix{nu, nu-nc, T}(zeros(T, nu, nu-nc))])
                Y = Q[:, SVector{nc, Int64}(1:nc)]
                Z = Q[:, SVector{nu-nc, Int64}(nc+1:nu)]
                
                AY = LowerTriangular(A' * Y)
                fk = lu(AY)
                α_β_y = fk \ [-c -cx]

                Ĥ = Symmetric(Ĥ)
                M = Symmetric(Z' * Ĥ * Z)
                ck = cholesky(M; check=false)
                if ck.info != 0
                    data.status = 1
                else
                    α_β_z = ck \ (Z' * ([-Qû -B] - Ĥ * Y * α_β_y))
                    α_β = Y * α_β_y + Z * α_β_z
                    α = α_β[:, 1]
                    β = α_β[:, SVector{nx, Int64}(2:nx+1)]

                    ψ_ω = ((Y' * ([-Qû -B] - Ĥ * α_β))' / fk)'
                    ψ = ψ_ω[:, 1]
                    ω = ψ_ω[:, SVector{nx, Int64}(2:nx+1)]
                end

            else
                ck = cholesky(Symmetric(Ĥ); check=false)
                ck.info != 0
                if ck.info != 0
                    data.status = 1
                else
                    α_β = ck \ [-Qû -B]
                    α = α_β[:, 1]
                    β = α_β[:, SVector{nx, Int64}(2:nx+1)]
                    ψ = @SVector zeros(T, nc)
                    ω = @SMatrix zeros(T, nc, nx)
                end
            end
            
            if data.status == 1
                if iszero(reg) # initial setting of regularisation
                    reg = (data.reg_last == 0.0) ? options.reg_1 : max(options.reg_min, options.κ_w_m * data.reg_last)
                else
                    reg = (data.reg_last == 0.0) ? options.κ_̄w_p * reg : options.κ_w_p * reg
                end
                break
            end

            # update parameters of update rule for ineq. dual variables, i.e., 

            # χ_L =  μ inv.(u - u^L) - z^L - Σ^L * α 
            # ζ_L =  - Σ^L * β 
            # χ_U =  μ inv.(u^U - u) - z^U + Σ^U .* α 
            # ζ_U =  Σ^U .* β

            # see update above for Q̂u[t] for first part of χ^L[t] χ^U[t]

            ζl =  -β .* Σ_L
            χl = inv_ul .* μ - zl - Σ_L .* α

            ζu =  Σ_U .* β
            χu = inv_uu .* μ - zu + Σ_U .* α

            # update the update rule
            solver.update[t] = UpdateRule{T, nx, nu, nc}(α, ψ, β, ω, χl, χu, ζl, ζu)

            # evaluate optimality dual error
            Lu = lu_ - zl + zu + fu' * λ
            if nc > 0
                Lu = Lu + cu' * ϕ
            end
            data.dual_inf = max(data.dual_inf, norm(Lu, Inf))
            z_norm += sum(zl)
            z_norm += sum(zu)
            ϕ_norm += norm(ϕ, 1)

            # Update return V derivatives for next timestep Vxx = C + β' * B + ω' cx
            V̂xx = C + β' * B 
            if nc > 0
                V̂xx = V̂xx  + ω' * cx
            end

            # Vx = Lx' + β' * Qû + ω' c + fx' Vx+
            V̂x = lx + β' * Qû + fx' * V̂x

            # λ = Lx' + fx' λ+
            λ = lx + fx' * λ

            if nc > 0
                V̂x = V̂x + cx' * ϕ + ω' * c
                λ = λ + cx' * ϕ
            end

            # evaluate sufficient decrease condition in forward pass
            data.expected_change_L += dot(Qû, α)
            nc > 0 && (data.expected_change_L += dot(c, ψ))
        end
        scaling_dual = max(options.s_max, (ϕ_norm + z_norm) / max(ni + nc * ocp.N, 1.0))  / options.s_max
        scaling_cs = max(options.s_max, z_norm / max(ni, 1.0))  / options.s_max
        data.dual_inf /= scaling_dual
        data.cs_inf_0 /= scaling_cs
        data.cs_inf_μ /= scaling_cs
        data.barrier_lagrangian_curr += data.objective
        data.status == 0 && break
    end
    data.reg_last = reg
    data.status != 0 && (verbose && (@warn "Backward pass failure, unable to find an iteration matrix with correct inertia."))
end
