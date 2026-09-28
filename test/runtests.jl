using FilterDDP
using LinearAlgebra
using StaticArrays
using Test

function bundle_isapprox(left, right)
    return propertynames(left) == propertynames(right) &&
        all(isapprox(getproperty(left, name), getproperty(right, name);
            rtol=1.0e-12, atol=1.0e-12) for name in propertynames(left))
end

component_derivative_allocations(component, args...) =
    @allocated FilterDDP.component_derivatives(component, args...)

component_first_order_allocations(component, args...) =
    @allocated FilterDDP.component_first_order(component, args...)

component_value_allocations(component, args...) =
    @allocated FilterDDP.component_value(component, args...)

stage_parameter_allocations(solver, parameters) =
    @allocated set_parameters!(solver; stage_objective=parameters)

filter_update_allocations(data, options) =
    @allocated FilterDDP.update_filter!(data, options)

filter_reset_allocations(data) = @allocated FilterDDP.reset!(data)

@testset "filter storage" begin
    data = FilterDDP.solver_data(Float64)
    options = Options{Float64}(γ_θ=0.1, γ_L=0.1)
    filter = data.filter

    FilterDDP.reset!(data)
    @test data.filter === filter
    @test data.filter == SVector{2, Float64}[SVector(1.0e6, -Inf)]

    push!(data.filter, SVector(2.0, 5.0))
    push!(data.filter, SVector(3.0, 4.0))
    push!(data.filter, SVector(0.5, 6.0))
    data.primal_1_curr = 1.0
    data.barrier_lagrangian_curr = 4.0
    FilterDDP.update_filter!(data, options)

    @test length(data.filter) == 3
    @test data.filter[2] == SVector(0.5, 6.0)
    @test data.filter[3] ≈ SVector(0.9, 3.9)

    sizehint!(data.filter, 8)
    FilterDDP.reset!(data)
    data.primal_1_curr = 1.0
    data.barrier_lagrangian_curr = 4.0
    FilterDDP.update_filter!(data, options)
    FilterDDP.reset!(data)
    data.primal_1_curr = 1.0
    data.barrier_lagrangian_curr = 4.0
    @test filter_update_allocations(data, options) == 0
    @test filter_reset_allocations(data) == 0

    float32_data = FilterDDP.solver_data(Float32)
    FilterDDP.reset!(float32_data)
    @test eltype(float32_data.filter) == SVector{2, Float32}
end

@testset "constraint notation" begin
    equalities = EqualityConstraints((x, u) -> [x[1]^2 + u[1]], 1, 1)
    x = @SVector [2.0]
    u = @SVector [3.0]
    multiplier = @SVector [4.0]

    @test equalities.h(x, u) == @SVector [7.0]
    @test equalities.hx(x, u) == @SMatrix [4.0]
    @test equalities.hu(x, u) == @SMatrix [1.0]
    @test equalities.hxx(x, u, multiplier) == @SMatrix [8.0]
    @test equalities.hux(x, u, multiplier) == @SMatrix [0.0]
    @test equalities.huu(x, u, multiplier) == @SMatrix [0.0]
end

@testset "slack block elimination" begin
    H = @SMatrix [4.0 1.0; 1.0 3.0]
    G = @SMatrix [2.0 -1.0]
    A = @SMatrix [1.0; 1.0]
    slack_curvature = @SMatrix [5.0]
    rhs_u = @SMatrix [1.0 -2.0; 0.5 3.0]
    rhs_s = @SMatrix [0.2 -0.3]
    rhs_h = @SMatrix [0.1 0.4]
    rhs_g = @SMatrix [-0.7 0.6]

    H_reduced, rhs_reduced = FilterDDP.eliminate_slack_blocks(
        H, G, slack_curvature, rhs_u, rhs_s, rhs_g)
    reduced_matrix = [Matrix(H_reduced) Matrix(A);
                      Matrix(transpose(A)) zeros(1, 1)]
    reduced_step = reduced_matrix \ [Matrix(rhs_reduced); Matrix(rhs_h)]
    step_u = reduced_step[1:2, :]
    step_h = reduced_step[3:3, :]
    step_s, step_g = FilterDDP.recover_slack_blocks(
        step_u, G, slack_curvature, rhs_s, rhs_g)

    full_matrix = [
        Matrix(H) zeros(2, 1) Matrix(A) Matrix(transpose(G));
        zeros(1, 2) Matrix(slack_curvature) zeros(1, 1) -ones(1, 1);
        Matrix(transpose(A)) zeros(1, 1) zeros(1, 1) zeros(1, 1);
        Matrix(G) -ones(1, 1) zeros(1, 1) zeros(1, 1)
    ]
    full_step = [step_u; step_s; step_h; step_g]
    full_rhs = [Matrix(rhs_u); Matrix(rhs_s); Matrix(rhs_h); Matrix(rhs_g)]

    @test full_matrix * full_step ≈ full_rhs rtol=1.0e-13
    @test full_step ≈ full_matrix \ full_rhs rtol=1.0e-13
end

@testset "first-class inequality constraints" begin
    nx = 1
    nu = 1
    horizon = 3
    stage_objective = Objective((x, u) -> (u[1] - 1)^2, nx, nu)
    terminal_objective = TerminalObjective(x -> 0.0, nx)
    dynamics = Dynamics((x, u) -> x, nx, nu)
    inequalities = InequalityConstraints(
        (x, u) -> [u[1]], nx, nu, @SVector([0.0]), @SVector([Inf]))
    control_limits = ControlLimits(@SVector([-Inf]), @SVector([Inf]))
    ocp = build_ocp(horizon, stage_objective, terminal_objective, dynamics,
        inequalities, control_limits)
    solver = Solver(ocp; options=Options{Float64}(verbose=false))

    @test ocp.equality_constraints isa FilterDDP.SymbolicsEqualityConstraints{1, 1, 0, 0}
    status = solve!(solver, @SVector([0.0]),
        [@SVector([0.5]) for _ in 1:horizon-1])
    _, controls = get_trajectory(solver)
    slacks = get_slack_trajectory(solver)

    @test status == 0
    @test solver.data.primal_inf <= solver.options.optimality_tolerance
    @test solver.data.dual_inf <= solver.options.optimality_tolerance
    @test all(slack[1] > 0.0 for slack in slacks)
    @test all(inequalities.g(@SVector([0.0]), control)[1] ≈ slack[1]
        for (control, slack) in zip(controls, slacks))
    @test controls[1][1] ≈ 1.0 atol=1.0e-7

    boundary_solver = Solver(ocp; options=Options{Float64}(verbose=false))
    initial_controls = [@SVector([0.5]) for _ in 1:horizon-1]
    FilterDDP.initialize_trajectory!(
        boundary_solver, initial_controls, @SVector([0.0]))
    @test all(element.s[1] == 0.5 for element in boundary_solver.nominal)
    update = boundary_solver.update[1]
    boundary_solver.update[1] = FilterDDP.UpdateRule(
        update.α, @SVector([-1.0]), update.ψh, update.ψg,
        update.β, update.βs, update.ωh, update.ωg,
        update.χl, update.χu, update.ζl, update.ζu,
        update.χsl, update.χsu, update.ζsl, update.ζsu)
    FilterDDP.rollout!(boundary_solver, ocp, boundary_solver.data, 0.99, 1.0)
    @test boundary_solver.data.status == 2
end

@testset "control-limits-only OCP" begin
    @test ControlLimits === BoundConstraints

    nx = 1
    nu = 1
    stage_objective = Objective((x, u) -> u[1]^2, nx, nu)
    terminal_objective = TerminalObjective(x -> x[1]^2, nx)
    dynamics = Dynamics((x, u) -> x, nx, nu)
    control_limits = ControlLimits(@SVector([-1.0]), @SVector([1.0]))
    ocp = build_ocp(2, stage_objective, terminal_objective, dynamics,
        control_limits)

    @test ocp.control_limits == [control_limits]
    @test ocp.equality_constraints isa FilterDDP.SymbolicsEqualityConstraints{1, 1, 0}
    @test ocp.inequality_constraints isa FilterDDP.SymbolicsInequalityConstraints{Float64, 1, 1, 0}

    solver = Solver(ocp; options=Options{Float64}(verbose=false))
    status = solve!(solver, @SVector([0.0]),
        [@SVector([0.5])])
    @test status == 0
end

@testset "state-only terminal objective" begin
    nx = 1
    nu = 1
    stage_objective = Objective((x, u) -> 0.5 * u[1]^2, nx, nu)
    terminal_objective = TerminalObjective(x -> x[1]^2, nx)
    dynamics = Dynamics((x, u) -> x + u, nx, nu)
    control_limits = ControlLimits(@SVector([-Inf]), @SVector([Inf]))
    ocp = build_ocp(2, stage_objective, terminal_objective, dynamics,
        control_limits)
    solver = Solver(ocp; options=Options{Float64}(
        verbose=false, optimality_tolerance=1.0e-10))

    @test_throws DimensionMismatch solve!(solver, @SVector([1.0]),
        [@SVector([0.0]), @SVector([0.0])])
    status = solve!(solver, @SVector([1.0]), [@SVector([0.0])])
    states, controls = get_trajectory(solver)

    @test status == 0
    @test length(states) == 2
    @test length(controls) == 1
    @test controls[1][1] ≈ -2.0 / 3.0 atol=1.0e-9
    @test states[end][1] ≈ 1.0 / 3.0 atol=1.0e-9
    @test solver.data.objective ≈ 1.0 / 3.0 atol=1.0e-9

    nonlinear_stage = Objective((x, u) -> 0.5 * u[2]^2, 1, 2)
    nonlinear_terminal = TerminalObjective(x -> 0.5 * x[1]^2, 1)
    nonlinear_dynamics = Dynamics((x, u) -> [x[1] + u[1]^2 + u[2]], 1, 2)
    equalities = EqualityConstraints((x, u) -> [u[2]], 1, 2)
    nonlinear_limits = ControlLimits(
        @SVector([-Inf, -Inf]), @SVector([Inf, Inf]))
    nonlinear_solver = Solver(build_ocp(2, nonlinear_stage, nonlinear_terminal,
        nonlinear_dynamics, equalities, nonlinear_limits))
    FilterDDP.initialize_trajectory!(nonlinear_solver,
        [@SVector([0.5, 0.0])], @SVector([1.0]))
    FilterDDP.backward_pass!(nonlinear_solver, nonlinear_solver.ocp,
        nonlinear_solver.nominal, nonlinear_solver.data, nonlinear_solver.options)

    @test nonlinear_solver.update[1].α[1] ≈ -5.0 / 14.0 atol=1.0e-12
end

@testset "fixed OCP parameters" begin
    nx = 1
    nu = 1
    nps = 2
    npt = 1
    npd = 2
    npe = 1
    npi = 1
    stage_objective = Objective(
        (x, u, p) -> p[1] * x[1]^2 + p[2] * u[1]^2, nx, nu, nps)
    terminal_objective = TerminalObjective(
        (x, p) -> p[1] * x[1]^2, nx, npt)
    dynamics = Dynamics(
        (x, u, p) -> [p[1] * x[1] + p[2] * u[1]], nx, nu, npd)
    equalities = EqualityConstraints(
        (x, u, p) -> [p[1] * x[1] + u[1]], nx, nu, npe)
    inequalities = InequalityConstraints(
        (x, u, p) -> [p[1] * u[1]], nx, nu,
        @SVector([-Inf]), @SVector([1.0]), npi)
    control_limits = ControlLimits(@SVector([-Inf]), @SVector([Inf]))
    parameters_1 = (
        stage_objective=@SVector([2.0, 3.0]),
        term_objective=@SVector([4.0]),
        dynamics=@SVector([6.0, 7.0]),
        equality_constraints=@SVector([8.0]),
        inequality_constraints=@SVector([9.0]))
    parameters_2 = (
        stage_objective=@SVector([3.0, 4.0]),
        term_objective=@SVector([5.0]),
        dynamics=@SVector([7.0, 8.0]),
        equality_constraints=@SVector([9.0]),
        inequality_constraints=@SVector([10.0]))

    ocp_1 = build_ocp(2, stage_objective, terminal_objective, dynamics,
        equalities, inequalities, control_limits; parameters=parameters_1)
    ocp_2 = build_ocp(2, stage_objective, terminal_objective, dynamics,
        equalities, inequalities, control_limits; parameters=parameters_2)
    x = @SVector [2.0]
    u = @SVector [1.0]

    @test ocp_1.parameters.stage_objective == [parameters_1.stage_objective]
    @test ocp_1.parameters.term_objective == parameters_1.term_objective
    @test ocp_1.parameters.dynamics == [parameters_1.dynamics]
    @test ocp_1.parameters.equality_constraints == [parameters_1.equality_constraints]
    @test ocp_1.parameters.inequality_constraints == [parameters_1.inequality_constraints]
    @test typeof(ocp_1) === typeof(ocp_2)
    @test ocp_1.stage_objective === ocp_2.stage_objective
    @test FilterDDP.evaluate(stage_objective.l, x, u,
        parameters_1.stage_objective) == @SVector [11.0]
    @test FilterDDP.evaluate(terminal_objective.l, x,
        parameters_1.term_objective) == @SVector [16.0]
    @test FilterDDP.evaluate(terminal_objective.lx, x,
        parameters_1.term_objective) == @SVector [16.0]
    @test FilterDDP.evaluate(terminal_objective.lxx, x,
        parameters_1.term_objective) == @SMatrix [8.0]
    @test FilterDDP.evaluate(stage_objective.lx, x, u,
        parameters_1.stage_objective) == @SVector [8.0]
    @test FilterDDP.evaluate(stage_objective.lu, x, u,
        parameters_1.stage_objective) == @SVector [6.0]
    @test FilterDDP.evaluate(dynamics.f, x, u,
        parameters_1.dynamics) == @SVector [19.0]
    @test FilterDDP.evaluate(equalities.h, x, u,
        parameters_1.equality_constraints) == @SVector [17.0]
    @test FilterDDP.evaluate(inequalities.g, x, u,
        parameters_1.inequality_constraints) == @SVector [9.0]
    @test FilterDDP.evaluate(stage_objective.lxx, x, u,
        parameters_1.stage_objective) == @SMatrix [4.0]
    @test FilterDDP.evaluate(equalities.hxx, x, u,
        parameters_1.equality_constraints, @SVector([2.0])) == @SMatrix [0.0]

    equality_only_parameters = (
        stage_objective=parameters_1.stage_objective,
        term_objective=parameters_1.term_objective,
        dynamics=parameters_1.dynamics,
        equality_constraints=parameters_1.equality_constraints)
    equality_only_ocp = build_ocp(2, stage_objective, terminal_objective,
        dynamics, equalities, control_limits; parameters=equality_only_parameters)
    @test equality_only_ocp.inequality_constraints isa
        FilterDDP.SymbolicsInequalityConstraints{Float64, 1, 1, 0, 0}

    inequality_only_parameters = (
        stage_objective=parameters_1.stage_objective,
        term_objective=parameters_1.term_objective,
        dynamics=parameters_1.dynamics,
        inequality_constraints=parameters_1.inequality_constraints)
    inequality_only_ocp = build_ocp(2, stage_objective, terminal_objective,
        dynamics, inequalities, control_limits; parameters=inequality_only_parameters)
    @test inequality_only_ocp.equality_constraints isa
        FilterDDP.SymbolicsEqualityConstraints{1, 1, 0, 0}
end

@testset "stage-dependent parameters" begin
    N = 11
    backend = ForwardDiffBackend()
    stage_objective = Objective(
        (x, u, p) -> (x[1] - p[1])^2 + 1.0e-6 * u[1]^2, 1, 1, 1; backend=backend)
    terminal_objective = TerminalObjective(
        (x, p) -> (x[1] - p[1])^2, 1, 1; backend=backend)
    dynamics = Dynamics((x, u, p) -> @SVector([x[1] + p[1] * u[1]]), 1, 1, 1;
        backend=backend)
    control_limits = ControlLimits(@SVector([-Inf]), @SVector([Inf]))
    x1 = @SVector [0.0]
    controls = [@SVector([0.0]) for _ = 1:N-1]
    reference = [@SVector([sin(0.5 * t)]) for t = 1:N]

    function solve_tracking(parameters)
        ocp = build_ocp(N, stage_objective, terminal_objective, dynamics,
            control_limits; parameters=parameters)
        solver = Solver(ocp; options=Options{Float64}(verbose=false))
        solve!(solver, x1, controls)
        return solver
    end

    tracking = solve_tracking((stage_objective=reference[1:N-1],
        term_objective=reference[N], dynamics=@SVector([1.0])))
    x_tracking, _ = get_trajectory(tracking)
    @test tracking.data.status == 0
    @test tracking.ocp.parameters.stage_objective == reference[1:N-1]
    @test maximum(abs(x_tracking[t][1] - reference[t][1]) for t = 2:N) < 1.0e-3

    constant = solve_tracking((stage_objective=@SVector([0.5]),
        term_objective=@SVector([0.5]), dynamics=@SVector([2.0])))
    repeated = solve_tracking((stage_objective=fill(@SVector([0.5]), N - 1),
        term_objective=[0.5], dynamics=[[2.0] for _ = 1:N-1]))
    @test get_trajectory(constant) == get_trajectory(repeated)

    shifted_reference = [@SVector([cos(0.5 * t)]) for t = 1:N]
    shifted = (stage_objective=shifted_reference[1:N-1],
        term_objective=shifted_reference[N])
    @test set_parameters!(tracking; shifted...) === tracking
    solve!(tracking, x1, controls)
    fresh = solve_tracking((shifted..., dynamics=@SVector([1.0])))
    @test get_trajectory(tracking) == get_trajectory(fresh)

    stage_reference = reference[1:N-1]
    stage_parameter_allocations(tracking, stage_reference)
    @test stage_parameter_allocations(tracking, stage_reference) == 0
    ocp = tracking.ocp
    @test_throws DimensionMismatch set_parameters!(ocp;
        stage_objective=reference[1:N-2])
    @test_throws DimensionMismatch set_parameters!(ocp;
        dynamics=@SVector([1.0, 2.0]))
    @test_throws DimensionMismatch build_ocp(N, stage_objective,
        terminal_objective, dynamics, control_limits;
        parameters=(stage_objective=@SVector([0.0]), term_objective=@SVector([0.0])))
    @test_throws MethodError set_parameters!(ocp; objective=@SVector([0.0]))
end

@testset "stage-dependent bounds" begin
    N = 11
    backend = ForwardDiffBackend()
    stage_objective = Objective(
        (x, u) -> (x[1] - 10.0)^2 + 1.0e-6 * u[1]^2, 1, 1; backend=backend)
    terminal_objective = TerminalObjective(
        x -> (x[1] - 10.0)^2, 1; backend=backend)
    dynamics = Dynamics((x, u) -> x + u, 1, 1; backend=backend)
    inequalities = InequalityConstraints((x, u) -> x + u, 1, 1,
        @SVector([-Inf]), @SVector([Inf]); backend=backend)
    x1 = @SVector [0.0]
    controls = [@SVector([0.0]) for _ = 1:N-1]
    options = Options{Float64}(verbose=false, optimality_tolerance=1.0e-9)

    function solve_bounded(control_limits; kwargs...)
        ocp = build_ocp(N, stage_objective, terminal_objective, dynamics,
            inequalities, control_limits; kwargs...)
        solver = Solver(ocp; options=options)
        solve!(solver, x1, controls)
        return solver
    end

    limit = t -> t <= 5 ? 0.1 * t : Inf
    stage_limits = [ControlLimits(@SVector([-limit(t)]), @SVector([limit(t)]))
        for t = 1:N-1]
    limited = solve_bounded(stage_limits)
    x_limited, u_limited = get_trajectory(limited)
    @test limited.data.status == 0
    @test FilterDDP.bound_count(limited.ocp) == 10
    @test maximum(abs(u_limited[t][1] - limit(t)) for t = 1:5) < 1.0e-6
    @test maximum(abs(x_limited[t][1] - 10.0) for t = 7:N) < 1.0e-4

    upper = t -> 0.5 * t
    stage_bounds = [BoundConstraints(@SVector([-Inf]), @SVector([upper(t)]))
        for t = 1:N-1]
    wide_limits = ControlLimits(@SVector([-Inf]), @SVector([Inf]))
    bounded = solve_bounded(wide_limits; inequality_bounds=stage_bounds)
    x_bounded, _ = get_trajectory(bounded)
    @test bounded.data.status == 0
    @test FilterDDP.bound_count(bounded.ocp) == N - 1
    @test maximum(abs(x_bounded[t + 1][1] - upper(t)) for t = 1:N-1) < 1.0e-6

    constant_limits = ControlLimits(@SVector([-0.5]), @SVector([0.5]))
    constant = solve_bounded(constant_limits)
    repeated = solve_bounded(fill(constant_limits, N - 1))
    @test get_trajectory(constant) == get_trajectory(repeated)

    @test set_bounds!(constant; control_limits=stage_limits,
        inequality_bounds=stage_bounds) === constant
    solve!(constant, x1, controls)
    fresh = solve_bounded(stage_limits; inequality_bounds=stage_bounds)
    @test get_trajectory(constant) == get_trajectory(fresh)

    stage_bound_allocations(solver, bounds) =
        @allocated set_bounds!(solver; control_limits=bounds)
    stage_bound_allocations(constant, stage_limits)
    @test stage_bound_allocations(constant, stage_limits) == 0

    @test_throws DimensionMismatch set_bounds!(constant;
        control_limits=stage_limits[1:N-2])
    @test_throws ArgumentError set_bounds!(constant;
        inequality_bounds=BoundConstraints(@SVector([-1.0, -1.0]), @SVector([1.0, 1.0])))
    @test_throws DimensionMismatch build_ocp(N, stage_objective,
        terminal_objective, dynamics, inequalities, wide_limits;
        inequality_bounds=stage_bounds[1:2])
    @test_throws MethodError set_bounds!(constant; bounds=stage_bounds)
end

@testset "ForwardDiff derivative backend" begin
    nx = 2
    nu = 2
    stage_function = (x, u, p) ->
        p[1] * x[1]^2 + sin(x[2]) * u[1] + p[2] * u[2]^4
    terminal_function = (x, p) -> p[1] * x[1]^2 + cos(x[2])
    dynamics_function = (x, u, p) -> @SVector [
        p[1] * x[1]^2 + x[2] * u[1] + u[2]^2,
        sin(x[2]) + x[1] * u[2] + u[1]^3]
    equality_function = (x, u, p) -> @SVector [
        p[1] * x[1]^2 + x[2] * u[1],
        x[1] * u[2] + u[1]^3]
    inequality_function = (x, u, p) ->
        @SVector [sin(x[2]) + x[1] * u[1] + p[1] * u[2]^2]

    x = @SVector [1.2, -0.4]
    u = @SVector [0.7, -0.2]
    stage_parameters = @SVector [2.0, 3.0]
    terminal_parameters = @SVector [2.0]
    dynamics_parameters = @SVector [2.0]
    equality_parameters = @SVector [2.0]
    inequality_parameters = @SVector [2.0]
    dynamics_multiplier = @SVector [1.5, -0.8]
    equality_multiplier = @SVector [1.5, -0.8]
    inequality_multiplier = @SVector [0.6]
    forwarddiff = ForwardDiffBackend()

    symbolic_stage = Objective(stage_function, nx, nu, 2)
    forwarddiff_stage = Objective(
        stage_function, nx, nu, 2; backend=forwarddiff)
    symbolic_terminal = TerminalObjective(terminal_function, nx, 1)
    forwarddiff_terminal = TerminalObjective(
        terminal_function, nx, 1; backend=forwarddiff)
    symbolic_dynamics = Dynamics(dynamics_function, nx, nu, 1)
    forwarddiff_dynamics = Dynamics(
        dynamics_function, nx, nu, 1; backend=forwarddiff)
    symbolic_equalities = EqualityConstraints(
        equality_function, nx, nu, 1)
    forwarddiff_equalities = EqualityConstraints(
        equality_function, nx, nu, 1; backend=forwarddiff)
    symbolic_inequalities = InequalityConstraints(
        inequality_function, nx, nu,
        @SVector([-Inf]), @SVector([1.0]), 1)
    forwarddiff_inequalities = InequalityConstraints(
        inequality_function, nx, nu,
        @SVector([-Inf]), @SVector([1.0]), 1; backend=forwarddiff)
    empty_equalities = EqualityConstraints(nx, nu, 0; backend=forwarddiff)
    empty_inequalities = InequalityConstraints(
        Float64, nx, nu, 0; backend=forwarddiff)

    @test FilterDDP.component_value(empty_equalities, x, u,
        SVector{0, Float64}()) == SVector{0, Float64}()
    @test FilterDDP.component_value(empty_inequalities, x, u,
        SVector{0, Float64}()) == SVector{0, Float64}()
    @test_throws ArgumentError EqualityConstraints(
        nx, nu, -1; backend=forwarddiff)
    @test_throws ArgumentError InequalityConstraints(
        Float64, nx, nu, -1; backend=forwarddiff)

    @test bundle_isapprox(
        FilterDDP.component_first_order(
            symbolic_stage, x, u, stage_parameters),
        FilterDDP.component_first_order(
            forwarddiff_stage, x, u, stage_parameters))
    symbolic_stage_derivatives = FilterDDP.component_derivatives(
        symbolic_stage, x, u, stage_parameters)
    forwarddiff_stage_derivatives = @inferred FilterDDP.component_derivatives(
        forwarddiff_stage, x, u, stage_parameters)
    @test bundle_isapprox(
        symbolic_stage_derivatives, forwarddiff_stage_derivatives)

    symbolic_terminal_derivatives = FilterDDP.component_derivatives(
        symbolic_terminal, x, terminal_parameters)
    forwarddiff_terminal_derivatives = @inferred FilterDDP.component_derivatives(
        forwarddiff_terminal, x, terminal_parameters)
    @test bundle_isapprox(
        symbolic_terminal_derivatives, forwarddiff_terminal_derivatives)

    symbolic_dynamics_derivatives = FilterDDP.component_derivatives(
        symbolic_dynamics, x, u, dynamics_parameters, dynamics_multiplier)
    forwarddiff_dynamics_derivatives = @inferred FilterDDP.component_derivatives(
        forwarddiff_dynamics, x, u, dynamics_parameters, dynamics_multiplier)
    @test bundle_isapprox(
        symbolic_dynamics_derivatives, forwarddiff_dynamics_derivatives)
    @test forwarddiff_dynamics_derivatives.yux[1, 2] ≈ dynamics_multiplier[1]
    @test forwarddiff_dynamics_derivatives.yux[2, 1] ≈ dynamics_multiplier[2]

    symbolic_equality_derivatives = FilterDDP.component_derivatives(
        symbolic_equalities, x, u, equality_parameters, equality_multiplier)
    forwarddiff_equality_derivatives = @inferred FilterDDP.component_derivatives(
        forwarddiff_equalities, x, u, equality_parameters, equality_multiplier)
    @test bundle_isapprox(
        symbolic_equality_derivatives, forwarddiff_equality_derivatives)
    @test forwarddiff_equality_derivatives.yux[1, 2] ≈ equality_multiplier[1]
    @test forwarddiff_equality_derivatives.yux[2, 1] ≈ equality_multiplier[2]

    @test bundle_isapprox(
        FilterDDP.component_derivatives(symbolic_inequalities, x, u,
            inequality_parameters, inequality_multiplier),
        FilterDDP.component_derivatives(forwarddiff_inequalities, x, u,
            inequality_parameters, inequality_multiplier))

    @test component_value_allocations(
        forwarddiff_stage, x, u, stage_parameters) == 0
    @test component_first_order_allocations(
        forwarddiff_stage, x, u, stage_parameters) == 0
    @test component_derivative_allocations(
        forwarddiff_stage, x, u, stage_parameters) == 0
    @test component_derivative_allocations(
        forwarddiff_terminal, x, terminal_parameters) == 0
    constant_terminal = TerminalObjective(
        x -> 0.0, nx; backend=forwarddiff)
    empty_parameters_for_terminal = SVector{0, Float64}()
    FilterDDP.component_derivatives(
        constant_terminal, x, empty_parameters_for_terminal)
    @test component_derivative_allocations(
        constant_terminal, x, empty_parameters_for_terminal) == 0
    @test component_value_allocations(
        forwarddiff_dynamics, x, u, dynamics_parameters) == 0
    @test component_first_order_allocations(
        forwarddiff_dynamics, x, u, dynamics_parameters) == 0
    @test component_derivative_allocations(forwarddiff_dynamics,
        x, u, dynamics_parameters, dynamics_multiplier) == 0
    @test component_derivative_allocations(forwarddiff_equalities,
        x, u, equality_parameters, equality_multiplier) == 0
    @test component_derivative_allocations(forwarddiff_inequalities,
        x, u, inequality_parameters, inequality_multiplier) == 0

    large_stage_function = (x, u) ->
        sum(abs2, x) + sum(sin, u) + x[1] * u[1]
    large_dynamics_function = (x, u) -> @SVector [
        x[1] + sum(u),
        x[2]^2 + u[1] * u[2],
        sin(x[3]) + u[3]^2,
        x[4] + x[1] * u[4]]
    large_x = SVector{4}(ntuple(index -> 0.1 * index, 4))
    large_u = SVector{15}(ntuple(index -> 0.01 * index, 15))
    large_multiplier = @SVector [1.0, -0.5, 0.3, 0.8]
    empty_parameters = SVector{0, Float64}()
    large_stage = Objective(
        large_stage_function, 4, 15; backend=forwarddiff)
    large_dynamics = Dynamics(
        large_dynamics_function, 4, 15; backend=forwarddiff)
    FilterDDP.component_derivatives(
        large_stage, large_x, large_u, empty_parameters)
    FilterDDP.component_derivatives(
        large_dynamics, large_x, large_u, empty_parameters, large_multiplier)
    @test component_derivative_allocations(
        large_stage, large_x, large_u, empty_parameters) == 0
    @test component_derivative_allocations(large_dynamics,
        large_x, large_u, empty_parameters, large_multiplier) == 0

    function backend_solver(backend;
            dynamics_function=(x, u) -> SVector(x[1]))
        stage = Objective(
            (x, u) -> (u[1] - 1.0)^2 + u[2]^2, 1, 2;
            backend=backend)
        terminal = TerminalObjective(x -> 0.0, 1; backend=backend)
        dynamics = Dynamics(
            dynamics_function, 1, 2; backend=backend)
        equalities = EqualityConstraints(
            (x, u) -> SVector(u[2]), 1, 2; backend=backend)
        inequalities = InequalityConstraints(
            (x, u) -> SVector(u[1]), 1, 2,
            SVector(0.0), SVector(Inf); backend=backend)
        limits = ControlLimits(
            @SVector([-Inf, -Inf]), @SVector([Inf, Inf]))
        return Solver(build_ocp(
            3, stage, terminal, dynamics, equalities, inequalities, limits);
            options=Options{Float64}(verbose=false))
    end

    symbolic_solver = backend_solver(SymbolicsBackend())
    forwarddiff_solver = backend_solver(forwarddiff)
    initial_controls = [@SVector([0.5, 0.1]) for _ = 1:2]

    nonfinite_solver = backend_solver(forwarddiff)
    FilterDDP.initialize_trajectory!(
        nonfinite_solver, initial_controls, @SVector([0.0]))
    update = nonfinite_solver.update[1]
    nonfinite_solver.update[1] = FilterDDP.UpdateRule(
        @SVector([Inf, 0.0]), update.αs, update.ψh, update.ψg,
        update.β, update.βs, update.ωh, update.ωg,
        update.χl, update.χu, update.ζl, update.ζu,
        update.χsl, update.χsu, update.ζsl, update.ζsu)
    FilterDDP.rollout!(nonfinite_solver, nonfinite_solver.ocp,
        nonfinite_solver.data, 0.99, 1.0)
    @test nonfinite_solver.data.status == 5

    domain_error_solver = backend_solver(forwarddiff;
        dynamics_function=(x, u) -> u[1] > 1.0 ?
            SVector(sqrt(-1.0)) : SVector(x[1]))
    FilterDDP.initialize_trajectory!(
        domain_error_solver, initial_controls, @SVector([0.0]))
    update = domain_error_solver.update[1]
    domain_error_solver.update[1] = FilterDDP.UpdateRule(
        @SVector([1.0, 0.0]), update.αs, update.ψh, update.ψg,
        update.β, update.βs, update.ωh, update.ωg,
        update.χl, update.χu, update.ζl, update.ζu,
        update.χsl, update.χsu, update.ζsl, update.ζsu)
    FilterDDP.rollout!(domain_error_solver, domain_error_solver.ocp,
        domain_error_solver.data, 0.99, 1.0)
    @test domain_error_solver.data.status == 5

    @test solve!(symbolic_solver, @SVector([0.0]), initial_controls) == 0
    @test solve!(forwarddiff_solver, @SVector([0.0]), initial_controls) == 0
    symbolic_states, symbolic_controls = get_trajectory(symbolic_solver)
    forwarddiff_states, forwarddiff_controls = get_trajectory(forwarddiff_solver)
    @test symbolic_solver.data.objective ≈ forwarddiff_solver.data.objective
    @test symbolic_states ≈ forwarddiff_states
    @test symbolic_controls ≈ forwarddiff_controls
end

@testset "automatic NLP scaling" begin
    nx = 1
    nu = 2
    horizon = 2
    stage_objective = Objective(
        (x, u) -> 1000.0 * x[1] + 200.0 * u[1], nx, nu)
    terminal_objective = TerminalObjective(x -> 1000.0 * x[1], nx)
    dynamics = Dynamics((x, u) -> x, nx, nu)
    equalities = EqualityConstraints(
        (x, u) -> [1000.0 * x[1] + 200.0 * u[1]], nx, nu)
    inequalities = InequalityConstraints(
        (x, u) -> [1000.0 * x[1] + 200.0 * u[2]], nx, nu,
        @SVector([0.0]), @SVector([Inf]))
    control_limits = ControlLimits(
        @SVector([-Inf, -Inf]), @SVector([Inf, Inf]))
    ocp = build_ocp(horizon, stage_objective, terminal_objective, dynamics,
        equalities, inequalities, control_limits)

    solver = Solver(ocp; options=Options{Float64}(nlp_scaling=true))
    initial_controls = [@SVector([0.0, 0.0]) for _ in 1:horizon-1]
    FilterDDP.initialize_trajectory!(solver, initial_controls, @SVector([0.0]))
    FilterDDP.compute_nlp_scaling!(solver)

    @test solver.scaling.objective ≈ 0.1
    @test all(element.zsl == @SVector([0.1]) for element in solver.nominal)
    @test all(scale == @SVector([0.1]) for scale in solver.scaling.equality)
    @test all(scale == @SVector([0.1]) for scale in solver.scaling.inequality)
    FilterDDP.rescale_duals!(solver, 2.0)
    @test all(element.zsl == @SVector([0.2]) for element in solver.nominal)

    solver.options.nlp_scaling = false
    FilterDDP.compute_nlp_scaling!(solver)
    @test solver.scaling.objective == 1.0
    @test all(scale == @SVector([1.0]) for scale in solver.scaling.equality)
    @test all(scale == @SVector([1.0]) for scale in solver.scaling.inequality)
end

@testset "equality row scaling preserves primal step" begin
    nx = 1
    nu = 2
    horizon = 2
    stage_objective = Objective((x, u) -> 0.5 * dot(u, u), nx, nu)
    terminal_objective = TerminalObjective(x -> 0.0, nx)
    dynamics = Dynamics((x, u) -> x, nx, nu)
    equalities = EqualityConstraints((x, u) -> [1000.0 * u[1]], nx, nu)
    control_limits = ControlLimits(
        @SVector([-Inf, -Inf]), @SVector([Inf, Inf]))
    ocp = build_ocp(horizon, stage_objective, terminal_objective, dynamics,
        equalities, control_limits)
    initial_controls = [@SVector([0.2, 0.3]) for _ in 1:horizon-1]

    scaled_solver = Solver(ocp;
        options=Options{Float64}(nlp_scaling=true, nlp_scaling_max_gradient=100.0))
    unscaled_solver = Solver(ocp;
        options=Options{Float64}(nlp_scaling=false))
    for solver in (scaled_solver, unscaled_solver)
        FilterDDP.initialize_trajectory!(solver, initial_controls, @SVector([0.0]))
        FilterDDP.compute_nlp_scaling!(solver)
        solver.data.μ = solver.options.μ_init
        FilterDDP.backward_pass!(
            solver, solver.ocp, solver.nominal, solver.data, solver.options)
    end

    @test scaled_solver.scaling.equality[1] == @SVector([0.1])
    @test scaled_solver.data.primal_inf ≈ 20.0
    @test scaled_solver.data.primal_inf_raw ≈ 200.0
    @test unscaled_solver.data.primal_inf ≈ 200.0
    for t = 1:horizon-1
        @test scaled_solver.update[t].α ≈ unscaled_solver.update[t].α rtol=1.0e-12
        @test scaled_solver.update[t].β ≈ unscaled_solver.update[t].β rtol=1.0e-12
        @test scaled_solver.update[t].ψh ≈ unscaled_solver.update[t].ψh rtol=1.0e-12
        @test scaled_solver.update[t].ωh ≈ unscaled_solver.update[t].ωh rtol=1.0e-12
    end
end

@testset "scaled objective solve" begin
    nx = 1
    nu = 1
    horizon = 3
    stage_objective = Objective((x, u) -> 1000.0 * (u[1] - 1.0)^2, nx, nu)
    terminal_objective = TerminalObjective(x -> 0.0, nx)
    dynamics = Dynamics((x, u) -> x, nx, nu)
    inequalities = InequalityConstraints(
        (x, u) -> [u[1]], nx, nu, @SVector([0.0]), @SVector([Inf]))
    control_limits = ControlLimits(@SVector([-Inf]), @SVector([Inf]))
    ocp = build_ocp(horizon, stage_objective, terminal_objective, dynamics,
        EqualityConstraints(nx, nu), inequalities, control_limits)
    initial_controls = [@SVector([0.5]) for _ in 1:horizon-1]

    scaled_solver = Solver(ocp; options=Options{Float64}(nlp_scaling=true))
    unscaled_solver = Solver(ocp; options=Options{Float64}(nlp_scaling=false))
    scaled_status = solve!(scaled_solver, @SVector([0.0]), initial_controls)
    unscaled_status = solve!(unscaled_solver, @SVector([0.0]), initial_controls)

    @test scaled_status == 0
    @test unscaled_status == 0
    @test scaled_solver.scaling.objective ≈ 0.1
    @test all(isapprox(control[1], 1.0; atol=1.0e-7)
        for control in last(get_trajectory(scaled_solver)))
    @test all(isapprox(control[1], 1.0; atol=1.0e-7)
        for control in last(get_trajectory(unscaled_solver)))

    scaled_solver.options.nlp_scaling = false
    repeated_status = solve!(scaled_solver)
    @test repeated_status == 0
    @test scaled_solver.scaling.objective == 1.0
end
