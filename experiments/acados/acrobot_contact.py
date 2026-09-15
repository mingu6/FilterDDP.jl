#!/usr/bin/env python3

from __future__ import annotations

import time
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import ACADOS_INFTY, AcadosModel, AcadosOcp, AcadosOcpSolver


DT = 0.05
N_HORIZON = 100
NX = 4
NU = 7
NP = 11
TARGET_POSITION = np.array([np.pi, 0.0])
EXPERIMENT_DIR = Path(__file__).resolve().parent
PARAMETERS_FILE = EXPERIMENT_DIR.parent / "filterddp" / "params" / "acrobot_contact.txt"
RESULTS_FILE = EXPERIMENT_DIR / "results" / "acrobot_contact.txt"
BUILD_DIR = Path("/tmp/filterddp_acados/acrobot_contact")


def mass_matrix(position: ca.SX, parameters: ca.SX) -> ca.SX:
    inertia1 = parameters[1]
    length1 = parameters[2]
    mass2 = parameters[4]
    inertia2 = parameters[5]
    length_com2 = parameters[7]
    coupling = mass2 * length1 * length_com2 * ca.cos(position[1])
    return ca.vertcat(
        ca.horzcat(inertia1 + inertia2 + mass2 * length1**2 + 2.0 * coupling,
                   inertia2 + coupling),
        ca.horzcat(inertia2 + coupling, inertia2),
    )


def coriolis_minus_gravity(
    position: ca.SX, velocity: ca.SX, parameters: ca.SX
) -> ca.SX:
    mass1 = parameters[0]
    length1 = parameters[2]
    length_com1 = parameters[3]
    mass2 = parameters[4]
    length_com2 = parameters[7]
    gravity = parameters[8]
    coupling = mass2 * length1 * length_com2 * ca.sin(position[1])
    coriolis = ca.vertcat(
        ca.horzcat(-2.0 * coupling * velocity[1], -coupling * velocity[1]),
        ca.horzcat(coupling * velocity[0], 0.0),
    )
    gravity_torque = ca.vertcat(
        -mass1 * gravity * length_com1 * ca.sin(position[0])
        - mass2
        * gravity
        * (length1 * ca.sin(position[0]) + length_com2 * ca.sin(position[0] + position[1])),
        -mass2 * gravity * length_com2 * ca.sin(position[0] + position[1]),
    )
    return coriolis @ velocity - gravity_torque


def contact_constraints(state: ca.SX, control: ca.SX, parameters: ca.SX) -> ca.SX:
    previous_position = state[:2]
    position = state[2:]
    next_position = control[1:3]
    force = control[3:5]
    gap_variable = control[5:7]

    previous_midpoint = 0.5 * (previous_position + position)
    next_midpoint = 0.5 * (position + next_position)
    previous_velocity = (position - previous_position) / DT
    next_velocity = (next_position - position) / DT
    momentum_change = (
        mass_matrix(next_midpoint, parameters) @ next_velocity
        - mass_matrix(previous_midpoint, parameters) @ previous_velocity
    )
    average_force = 0.5 * (
        coriolis_minus_gravity(next_midpoint, next_velocity, parameters)
        + coriolis_minus_gravity(previous_midpoint, previous_velocity, parameters)
    )
    contact_jacobian = ca.DM([[0.0, -1.0], [0.0, 1.0]])
    dynamics_residual = momentum_change + DT * (
        average_force
        - ca.vertcat(0.0, control[0])
        - contact_jacobian.T @ force
        + 0.5 * next_velocity
    )
    gap = ca.vertcat(0.5 * np.pi - next_position[1], next_position[1] + 0.5 * np.pi)
    return ca.vertcat(dynamics_residual, gap_variable - gap, force * gap_variable)


def build_ocp(parameters: np.ndarray) -> AcadosOcp:
    state = ca.SX.sym("x", NX)
    control = ca.SX.sym("u", NU)
    model_parameters = ca.SX.sym("p", NP)

    model = AcadosModel()
    model.name = "filterddp_acrobot_contact"
    model.x = state
    model.u = control
    model.p = model_parameters
    model.disc_dyn_expr = ca.vertcat(state[2:], control[1:3])
    model.con_h_expr = contact_constraints(state, control, model_parameters)
    model.con_h_expr_0 = model.con_h_expr
    model.cost_expr_ext_cost = (
        0.01 * DT * control[0] ** 2
        + 2.0 * (control[3] * control[5] + control[4] * control[6])
    )
    terminal_velocity = (state[2:] - state[:2]) / DT
    terminal_error = state[2:] - TARGET_POSITION
    model.cost_expr_ext_cost_e = (
        200.0 * ca.sumsqr(terminal_velocity)
        + 700.0 * ca.sumsqr(terminal_error)
    )

    ocp = AcadosOcp()
    ocp.model = model
    ocp.parameter_values = parameters
    ocp.solver_options.N_horizon = N_HORIZON
    ocp.solver_options.tf = N_HORIZON * DT
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    lower_nonlinear = np.zeros(6)
    upper_nonlinear = np.array([0.0, 0.0, 0.0, 0.0, ACADOS_INFTY, ACADOS_INFTY])
    ocp.constraints.lh = lower_nonlinear
    ocp.constraints.uh = upper_nonlinear
    ocp.constraints.lh_0 = lower_nonlinear
    ocp.constraints.uh_0 = upper_nonlinear
    ocp.constraints.idxbu = np.array([0, 3, 4, 5, 6])
    ocp.constraints.lbu = np.array([-8.0, 0.0, 0.0, 0.0, 0.0])
    ocp.constraints.ubu = np.array([8.0, ACADOS_INFTY, ACADOS_INFTY,
                                     ACADOS_INFTY, ACADOS_INFTY])
    ocp.constraints.x0 = np.zeros(NX)

    options = ocp.solver_options
    options.integrator_type = "DISCRETE"
    options.cost_discretization = "EULER"
    options.nlp_solver_type = "SQP_WITH_FEASIBLE_QP"
    options.nlp_solver_max_iter = 3000
    options.nlp_solver_tol_stat = 1e-7
    options.nlp_solver_tol_eq = 1e-7
    options.nlp_solver_tol_ineq = 1e-7
    options.nlp_solver_tol_comp = 1e-7
    options.hessian_approx = "EXACT"
    options.regularize_method = "PROJECT"
    options.reg_epsilon = 1e-4
    options.globalization = "FUNNEL_L1PEN_LINESEARCH"
    options.globalization_use_SOC = 1
    options.globalization_alpha_reduction = 0.5
    options.globalization_full_step_dual = True
    options.qpscaling_scale_objective = "OBJECTIVE_GERSHGORIN"
    options.qpscaling_scale_constraints = "INF_NORM"
    options.use_constraint_hessian_in_feas_qp = False
    options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    options.qp_solver_cond_N = 20
    options.qp_solver_iter_max = 400
    options.qp_solver_ric_alg = 0
    options.qp_solver_cond_ric_alg = 0
    options.hpipm_mode = "ROBUST"
    options.print_level = 0
    return ocp


def initialize_solver(solver, parameters: np.ndarray) -> None:
    control_guess = np.array([0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01])
    for stage in range(N_HORIZON):
        solver.set(stage, "x", np.zeros(NX))
        solver.set(stage, "u", control_guess)
        solver.set(stage, "p", parameters)
        solver.cost_set(stage, "scaling", 1.0)
    solver.set(N_HORIZON, "x", np.zeros(NX))
    solver.set(N_HORIZON, "p", parameters)


def solve_problem(solver, parameters: np.ndarray) -> dict[str, float | int]:
    solver.reset()
    initialize_solver(solver, parameters)
    solver.solve()
    solver.reset()
    initialize_solver(solver, parameters)
    wall_start = time.perf_counter()
    status = solver.solve()
    wall_time = time.perf_counter() - wall_start
    residuals = np.asarray(solver.get_residuals(recompute=True))
    return {
        "status": status,
        "iterations": int(solver.get_stats("nlp_iter")),
        "objective": float(solver.get_cost()),
        "primal": float(max(residuals[1], residuals[2])),
        "wall_time_ms": 1e3 * wall_time,
    }


def main() -> None:
    physical_parameters = np.atleast_2d(np.loadtxt(PARAMETERS_FILE))
    fixed_parameters = np.tile(np.array([9.81, 0.0, 0.0]), (len(physical_parameters), 1))
    parameter_sets = np.hstack((physical_parameters, fixed_parameters))

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    ocp = build_ocp(parameter_sets[0])
    ocp.code_gen_options.code_export_directory = str(BUILD_DIR / "c_generated_code")
    ocp.code_gen_options.json_file = "acrobot_contact_ocp.json"
    solver = AcadosOcpSolver(ocp, verbose=False)

    results = []
    for seed, parameters in enumerate(parameter_sets, start=1):
        result = solve_problem(solver, parameters)
        results.append((seed, result))
        print(
            f"seed={seed:3d} status={result['status']} "
            f"iterations={result['iterations']:4d} objective={result['objective']:.8e} "
            f"primal={result['primal']:.3e} time={result['wall_time_ms']:.2f} ms"
        )

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="ascii") as output_file:
        output_file.write("seed iterations status objective primal wall_ms\n")
        for seed, result in results:
            output_file.write(
                f"{seed} {result['iterations']} {str(result['status'] == 0).lower()} "
                f"{result['objective']:.12e} {result['primal']:.12e} "
                f"{result['wall_time_ms']:.6f}\n"
            )

    successes = sum(result["status"] == 0 for _, result in results)
    print(f"Solved {successes}/{len(results)} problems; wrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()