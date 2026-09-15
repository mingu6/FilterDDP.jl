@inline function eliminate_slack_blocks(H, G, slack_curvature,
        rhs_u, rhs_s, rhs_g)
    H_reduced = Symmetric(H + G' * slack_curvature * G)
    rhs_reduced = rhs_u + G' * (rhs_s + slack_curvature * rhs_g)
    return H_reduced, rhs_reduced
end

@inline function recover_slack_blocks(step_u, G, slack_curvature,
        rhs_s, rhs_g)
    step_s = G * step_u - rhs_g
    step_multiplier = slack_curvature * step_s - rhs_s
    return step_s, step_multiplier
end

function backward_pass!(solver::Solver{T, nx, nu, nh, ng, nux, nhx, ngx},
            ocp::OCP{T, nx, nu, nh, ng},
            traj::Vector{TrajectoryElement{T, nx, nu, nh, ng}},
            data::SolverData{T}, options::Options{T}, verbose::Bool=false
            ) where {T, nx, nu, nh, ng, nux, nhx, ngx}
    reg::T = 0.0
    μ = data.μ
    δ_c = 0.
    reg = 0.0

    equalities = ocp.equality_constraints
    inequalities = ocp.inequality_constraints
    dynamics = ocp.dynamics
    control_limits = ocp.control_limits
    slack_bounds = inequalities.bounds
    parameters = ocp.parameters
    objective_scale = solver.scaling.objective
    reg_1 = objective_scale * options.reg_1
    reg_min = objective_scale * options.reg_min
    reg_max = objective_scale * options.reg_max
    nstages = ocp.N - 1
    ni = (control_limits.nl + control_limits.nu +
        slack_bounds.nl + slack_bounds.nu) * nstages
    terminal_state = solver.nominal_terminal_state[]
    terminal_parameters = parameters.term_objective
    
    while reg <= reg_max
        data.status = 0
        terminal_derivatives = component_derivatives(
            ocp.term_objective, terminal_state, terminal_parameters)
        V̂x = objective_scale .* terminal_derivatives.lx
        V̂xx = objective_scale .* terminal_derivatives.lxx
        λ = V̂x

        data.barrier_lagrangian_curr = T(0.0)
        data.primal_1_curr = T(0.0)
        data.primal_inf = T(0.0)
        data.primal_inf_raw = T(0.0)
        data.cs_inf_μ = T(0.0)
        data.cs_inf_0 = T(0.0)
        data.objective = terminal_derivatives.value
        data.dual_inf = T(0.0)
        data.expected_change_L = T(0.0)
        ϕ_norm = T(0.0)
        z_norm = T(0.0) 
        
        for t = nstages:-1:1
            element = traj[t]
            x, u, s = element.x, element.u, element.s
            ϕ, ν = element.ϕ, element.ν
            zl, zu = element.zl, element.zu
            zsl, zsu = element.zsl, element.zsu

            # evaluate derivatives

            objective_parameters = parameters.stage_objective
            objective_derivatives = component_derivatives(
                ocp.stage_objective, x, u, objective_parameters)
            lx = objective_scale .* objective_derivatives.lx
            lu_ = objective_scale .* objective_derivatives.lu
            lxx = objective_scale .* objective_derivatives.lxx
            lux = objective_scale .* objective_derivatives.lux
            luu = objective_scale .* objective_derivatives.luu
            data.objective += objective_derivatives.value
            
            if nh > 0
                equality_derivatives = component_derivatives(
                    equalities, x, u, parameters.equality_constraints, ϕ)
                h = equality_derivatives.value
                hx = equality_derivatives.yx
                hu = equality_derivatives.yu
                hxx = equality_derivatives.yxx
                hux = equality_derivatives.yux
                huu = equality_derivatives.yuu
                h_scale = solver.scaling.equality[t]
                h_diagonal = diagm(h_scale)
                h_scaled = h_scale .* h
                
                # evaluate constraint violation norms
                data.primal_1_curr += norm(h_scaled, 1)
                data.primal_inf = max(data.primal_inf, norm(h_scaled, Inf))
                data.primal_inf_raw = max(data.primal_inf_raw, norm(h, Inf))
            end

            if ng > 0
                inequality_derivatives = component_derivatives(
                    inequalities, x, u, parameters.inequality_constraints, ν)
                g = inequality_derivatives.value
                gx = inequality_derivatives.yx
                gu = inequality_derivatives.yu
                gxx = inequality_derivatives.yxx
                gux = inequality_derivatives.yux
                guu = inequality_derivatives.yuu
                q = g - s
                g_scale = solver.scaling.inequality[t]
                q_scaled = g_scale .* q

                data.primal_1_curr += norm(q_scaled, 1)
                data.primal_inf = max(data.primal_inf, norm(q_scaled, Inf))
                data.primal_inf_raw = max(data.primal_inf_raw, norm(q, Inf))
            end

            unconstrained = nh == 0 && ng == 0
            dynamics_multiplier = unconstrained ? V̂x : λ
            dynamics_derivatives = component_derivatives(
                dynamics, x, u, parameters.dynamics, dynamics_multiplier)
            fx = dynamics_derivatives.yx
            fu = dynamics_derivatives.yu
            fxx = dynamics_derivatives.yxx
            fux = dynamics_derivatives.yux
            fuu = dynamics_derivatives.yuu

            # evaluate barrier Lagrangian

            ul = u - control_limits.l
            uu = control_limits.u - u
            data.barrier_lagrangian_curr -= μ *
                (dot(log.(ul), control_limits.maskl) +
                dot(log.(uu), control_limits.masku))

            if ng > 0
                sl = s - slack_bounds.l
                su = slack_bounds.u - s
                data.barrier_lagrangian_curr -= μ *
                    (dot(log.(sl), slack_bounds.maskl) +
                    dot(log.(su), slack_bounds.masku))
            else
                sl = @SVector zeros(T, ng)
                su = @SVector zeros(T, ng)
            end

            # evaluate complementary slackness errors
            cs_l = ul .* zl
            cs_u = uu .* zu
            data.cs_inf_0 = max(data.cs_inf_0,
                norm(cs_l .* control_limits.maskl, Inf))
            data.cs_inf_μ = max(data.cs_inf_μ,
                norm((cs_l .- μ) .* control_limits.maskl, Inf))
            data.cs_inf_0 = max(data.cs_inf_0,
                norm(cs_u .* control_limits.masku, Inf))
            data.cs_inf_μ = max(data.cs_inf_μ,
                norm((cs_u .- μ) .* control_limits.masku, Inf))

            if ng > 0
                cs_sl = sl .* zsl
                cs_su = su .* zsu
                data.cs_inf_0 = max(data.cs_inf_0,
                    norm(cs_sl .* slack_bounds.maskl, Inf))
                data.cs_inf_μ = max(data.cs_inf_μ,
                    norm((cs_sl .- μ) .* slack_bounds.maskl, Inf))
                data.cs_inf_0 = max(data.cs_inf_0,
                    norm(cs_su .* slack_bounds.masku, Inf))
                data.cs_inf_μ = max(data.cs_inf_μ,
                    norm((cs_su .- μ) .* slack_bounds.masku, Inf))
            end

            inv_ul = inv.(ul) .* control_limits.maskl
            inv_uu = inv.(uu) .* control_limits.masku
            if ng > 0
                inv_sl = inv.(sl) .* slack_bounds.maskl
                inv_su = inv.(su) .* slack_bounds.masku
            else
                inv_sl = @SVector zeros(T, ng)
                inv_su = @SVector zeros(T, ng)
            end

            # Qû = Lu' -μŪ^{-1}e + fu' * V̂x
            Qû = lu_ + fu' * V̂x + μ .* (inv_uu - inv_ul)
            Qŝ = ng > 0 ? -ν + μ .* (inv_su - inv_sl) : @SVector zeros(T, ng)
            
            # C = Lxx + fx' * Vxx * fx + V̄x ⋅ fxx
            C = lxx + fx' * V̂xx * fx + fxx
    
            ux_tmp = fu' * V̂xx
            # Ĥ = Luu + Σ + fu' * Vxx * fu + V̄x ⋅ fuu
            Σ_L = inv_ul .* zl
            Σ_U = inv_uu .* zu
            if ng > 0
                Σs_L = inv_sl .* zsl
                Σs_U = inv_su .* zsu
                Σs = Σs_L + Σs_U
            else
                Σs_L = @SVector zeros(T, ng)
                Σs_U = @SVector zeros(T, ng)
                Σs = @SVector zeros(T, ng)
            end
            Ĥ = luu + diagm(Σ_L) + diagm(Σ_U) + ux_tmp * fu + fuu
    
            # B = Lux + fu' * Vxx * fx + V̄x ⋅ fux
            B = lux + ux_tmp * fx + fux

            if nh > 0
                data.barrier_lagrangian_curr += dot(h, ϕ)
                Qû = Qû + hu' * ϕ
                C = C + hxx
                Ĥ = Ĥ + huu
                B = B + hux
            end

            if ng > 0
                data.barrier_lagrangian_curr += dot(q, ν)
                Qû = Qû + gu' * ν
                C = C + gxx
                Ĥ = Ĥ + guu
                B = B + gux
            end
            
            # inertia correction / regularisation
            reg_diag = @SVector ones(T, nu)
            reg_diag = reg_diag .* reg
            Ĥ = Ĥ + diagm(reg_diag)

            rhs_u = [-Qû -B]
            if ng > 0
                rhs_s = hcat(-Qŝ, @SMatrix zeros(T, ng, nx))
                rhs_g = [-q -gx]
                slack_curvature = diagm(Σs)
                Hsolve, rhs_u_solve = eliminate_slack_blocks(
                    Ĥ, gu, slack_curvature, rhs_u, rhs_s, rhs_g)
            else
                Hsolve = Symmetric(Ĥ)
                nh > 0 && (Ĥ = Hsolve)
                rhs_u_solve = rhs_u
            end

            # factorise the equality-only KKT system using the nullspace method
            if nh > 0
                hu_scaled = h_diagonal * hu
                hx_scaled = h_diagonal * hx
                A = hu_scaled'
                z = @SMatrix zeros(T, nu, nu-nh)
                Q, R = qr([A z])
                Y = Q[:, SVector{nh, Int64}(1:nh)]
                Z = Q[:, SVector{nu-nh, Int64}(nh+1:nu)]
                
                AY = A' * Y
                fk = lu(AY)
                α_β_y = fk \ [-h_scaled -hx_scaled]

                M = Symmetric(Z' * Hsolve * Z)
                ck = cholesky(M; check=false)
            else
                ck = cholesky(Hsolve; check=false)
            end
            ck.info != 0 && (data.status = 1)
            
            if data.status == 1
                if iszero(reg) # initial setting of regularisation
                    reg = (data.reg_last == 0.0) ? reg_1 : max(reg_min, options.κ_w_m * data.reg_last)
                else
                    reg = (data.reg_last == 0.0) ? options.κ_̄w_p * reg : options.κ_w_p * reg
                end
                break
            end

            if nh > 0
                α_β_z = ng > 0 ?
                    ck \ (Z' * (rhs_u_solve - Hsolve * Y * α_β_y)) :
                    ck \ (Z' * ([-Qû -B] - Ĥ * Y * α_β_y))
                α_β = Y * α_β_y + Z * α_β_z
                ψh_ωh_scaled = ng > 0 ?
                    ((Y' * (rhs_u_solve - Hsolve * α_β))' / fk)' :
                    ((Y' * ([-Qû -B] - Ĥ * α_β))' / fk)'
                # Store physical coefficients so dot(h, ϕ) and derivative terms stay unchanged.
                ψh_ωh = h_diagonal * ψh_ωh_scaled
            else
                α_β = ng > 0 ? ck \ rhs_u_solve : ck \ [-Qû -B]
                ψh_ωh = @SMatrix zeros(T, nh, nx + 1)
            end

            if ng > 0
                αs_βs, ψg_ωg = recover_slack_blocks(
                    α_β, gu, slack_curvature, rhs_s, rhs_g)
            else
                αs_βs = @SMatrix zeros(T, ng, nx + 1)
                ψg_ωg = @SMatrix zeros(T, ng, nx + 1)
            end

            α = α_β[:, 1]
            β = α_β[:, SVector{nx, Int64}(2:nx+1)]
            αs = αs_βs[:, 1]
            βs = αs_βs[:, SVector{nx, Int64}(2:nx+1)]
            ψh = ψh_ωh[:, 1]
            ωh = ψh_ωh[:, SVector{nx, Int64}(2:nx+1)]
            ψg = ψg_ωg[:, 1]
            ωg = ψg_ωg[:, SVector{nx, Int64}(2:nx+1)]

            # update parameters of the bound-dual update rules
            ζl = -β .* Σ_L
            χl = inv_ul .* μ - zl - Σ_L .* α

            ζu = β .* Σ_U
            χu = inv_uu .* μ - zu + Σ_U .* α

            if ng > 0
                ζsl = -βs .* Σs_L
                χsl = inv_sl .* μ - zsl - Σs_L .* αs
                ζsu = βs .* Σs_U
                χsu = inv_su .* μ - zsu + Σs_U .* αs
            else
                ζsl = @SMatrix zeros(T, ng, nx)
                χsl = @SVector zeros(T, ng)
                ζsu = @SMatrix zeros(T, ng, nx)
                χsu = @SVector zeros(T, ng)
            end

            # update the update rule
            solver.update[t] = UpdateRule(
                α, αs, ψh, ψg, β, βs, ωh, ωg,
                χl, χu, ζl, ζu, χsl, χsu, ζsl, ζsu)

            # evaluate optimality dual error
            Lu = lu_ - zl + zu + fu' * λ
            if nh > 0
                Lu = Lu + hu' * ϕ
            end
            if ng > 0
                Lu = Lu + gu' * ν
                Ls = -ν - zsl + zsu
                data.dual_inf = max(data.dual_inf, norm(Ls, Inf))
            end

            data.dual_inf = max(data.dual_inf, norm(Lu, Inf))
            z_norm += sum(zl)
            z_norm += sum(zu)
            z_norm += sum(zsl)
            z_norm += sum(zsu)
            nh > 0 && (ϕ_norm += norm(ϕ ./ h_scale, 1))
            ng > 0 && (ϕ_norm += norm(ν ./ g_scale, 1))

            # Update return V derivatives for the next timestep.
            V̂xx = C + β' * B
            V̂x = lx + β' * Qû + fx' * V̂x
            λ = lx + fx' * λ
            if nh > 0
                V̂xx = V̂xx + ωh' * hx
                V̂x = V̂x + hx' * ϕ + ωh' * h
                λ = λ + hx' * ϕ
            end
            if ng > 0
                V̂xx = V̂xx + ωg' * gx
                V̂x = V̂x + βs' * Qŝ + gx' * ν + ωg' * q
                λ = λ + gx' * ν
            end

            # evaluate sufficient decrease condition in forward pass
            data.expected_change_L += dot(Qû, α)
            if nh > 0
                data.expected_change_L += dot(h, ψh)
            end
            if ng > 0
                data.expected_change_L += dot(Qŝ, αs) + dot(q, ψg)
            end
        end
        scaling_dual = max(options.s_max,
            (ϕ_norm + z_norm) / max(ni + (nh + ng) * nstages, 1.0)) / options.s_max
        scaling_cs = max(options.s_max, z_norm / max(ni, 1.0))  / options.s_max
        data.dual_inf /= scaling_dual
        data.cs_inf_0 /= scaling_cs
        data.cs_inf_μ /= scaling_cs
        data.barrier_lagrangian_curr += objective_scale * data.objective
        data.status == 0 && break
    end
    data.reg_last = reg
    data.status != 0 && (verbose && (@warn "Backward pass failure, unable to find an iteration matrix with correct inertia."))
end
