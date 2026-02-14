mutable struct Solver{T}
    ocp::OCP{T}
    ws::FilterDDPWorkspace{T}
	data::SolverData{T}
    options::Options{T}
end

function Solver(ocp::OCP{T}; options::Union{Options{T}, Nothing}=nothing) where T
    ws = FilterDDPWorkspace(ocp)
    data = solver_data(T)
    options = isnothing(options) ? Options{T}() : options
	return Solver(ocp, ws, data, options)
end

function get_trajectory(solver::Solver{T}) where T
    x = [wse.nominal.x for wse in solver.ws]
    u = [wse.nominal.u for wse in solver.ws]
	return x, u
end

function initialize_trajectory!(solver::Solver{T}, u::Vector{Vector{T}}, x1::Vector{T}) where T
    options = solver.options

    solver.ws[1].nominal.x .= x1
    for t = 1:solver.ocp.N
        cl = solver.ocp.control_limits[t]
        u_tmp1 = solver.ws[t].u_tmp1
        u_tmp2 = solver.ws[t].u_tmp2
        ūt = solver.ws[t].nominal.u
        ūl, ūu = solver.ws[t].nominal.ul, solver.ws[t].nominal.uu
        nu = solver.ocp.constraints[t].nu
        for i in 1:nu
            if !isinf(cl.l[i]) && isinf(cl.u[i])
                u_tmp1[i] = max(cl.l[i], 1.0)
                u_tmp1[i] *= options.κ_1
                u_tmp1[i] += cl.l[i]
                ūt[i] = max(u[t][i], u_tmp1[i])
            elseif !isinf(cl.u[i]) && isinf(cl.l[i])
                u_tmp1[i] = max(cl.u[i], 1.0)
                u_tmp1[i] *= -options.κ_1
                u_tmp1[i] += cl.u[i]
                replace!(u_tmp1[i], NaN=>Inf)
                ūt[i] = min(u[t][i], u_tmp1)
            elseif !isinf(cl.u[i]) && !isinf(cl.l[i])
                u_tmp1[i] = cl.l[i] + min(options.κ_1 * max(1.0, abs(cl.l[i])),
                                    options.κ_2 * (cl.u[i] - cl.l[i]))
                u_tmp2[i] = cl.u[i] - min(options.κ_1 * max(1.0, abs(cl.u[i])),
                                    options.κ_2 * (cl.u[i] - cl.l[i]))
                ūt[i] = min(max(u[t][i], u_tmp1[i]), u_tmp2[i])
            else
                ūt[i] = u[t][i]
            end
        end

        # initialise inequality constraints
        ūl .= ūt
        ūl .-= cl.l
        ūu .= cl.u
        ūu .-= ūt

        t < solver.ocp.N && dynamics!(solver.ocp.dynamics[t], solver.ws[t], solver.ws[t+1]; mode=:nominal)
    end
end

