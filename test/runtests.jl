using FilterDDP
using LinearAlgebra
using StaticArrays
using Test

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
    terminal_objective = Objective((x, u) -> (u[1] - 1)^2, nx, nu)
    dynamics = Dynamics((x, u) -> x, nx, nu)
    equalities = EqualityConstraints(nx, nu)
    inequalities = InequalityConstraints(
        (x, u) -> [u[1]], nx, nu, @SVector([0.0]), @SVector([Inf]))
    control_limits = ControlLimits(@SVector([-Inf]), @SVector([Inf]))
    ocp = build_ocp(horizon, stage_objective, terminal_objective, dynamics,
        equalities, inequalities, control_limits)
    solver = Solver(ocp; options=Options{Float64}(verbose=false))

    status = solve!(solver, @SVector([0.0]),
        [@SVector([0.5]) for _ in 1:horizon])
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
    initial_controls = [@SVector([0.5]) for _ in 1:horizon]
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
    terminal_objective = Objective((x, u) -> u[1]^2, nx, nu)
    dynamics = Dynamics((x, u) -> x, nx, nu)
    control_limits = ControlLimits(@SVector([-1.0]), @SVector([1.0]))
    ocp = build_ocp(2, stage_objective, terminal_objective, dynamics,
        control_limits)

    @test ocp.control_limits === control_limits
    @test ocp.equality_constraints isa EqualityConstraints{1, 1, 0}
    @test ocp.inequality_constraints isa InequalityConstraints{Float64, 1, 1, 0}

    solver = Solver(ocp; options=Options{Float64}(verbose=false))
    status = solve!(solver, @SVector([0.0]),
        [@SVector([0.5]) for _ in 1:2])
    @test status == 0
end
