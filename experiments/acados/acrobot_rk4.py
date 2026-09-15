#!/usr/bin/env python3

from __future__ import annotations

import time
from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


DT = 0.05
N_HORIZON = 100
NX = 4
NU = 1
NP = 11
TARGET_STATE = np.array([np.pi, 0.0, 0.0, 0.0])
STATE_WEIGHT = np.diag([1.0, 1.0, 0.1, 0.1])
EXPERIMENT_DIR = Path(__file__).resolve().parent
PARAMETERS_FILE = EXPERIMENT_DIR.parent / "filterddp" / "params" / "acrobot_rk4.txt"
RESULTS_FILE = EXPERIMENT_DIR / "results" / "acrobot_rk4.txt"
BUILD_DIR = Path("/tmp/filterddp_acados/acrobot_rk4")

def continuous_dynamics(x: ca.SX, u: ca.SX, parameters: ca.SX) -> ca.SX:
    m1, I1, l1, lc1, m2, I2, _l2, lc2, gravity, b1, b2 = ca.vertsplit(
        parameters
    )
    q1, q2, velocity1, velocity2 = ca.vertsplit(x)

    cosine2 = ca.cos(q2)
    mass11 = I1 + I2 + m2 * l1**2 + 2.0 * m2 * l1 * lc2 * cosine2
    mass12 = I2 + m2 * l1 * lc2 * cosine2
    determinant = mass11 * I2 - mass12**2
    mass_inverse = ca.vertcat(
        ca.horzcat(I2, -mass12),
        ca.horzcat(-mass12, mass11),
    ) / determinant

    gravity_torque = ca.vertcat(
        -m1 * gravity * lc1 * ca.sin(q1)
        - m2 * gravity * (l1 * ca.sin(q1) + lc2 * ca.sin(q1 + q2)),
        -m2 * gravity * lc2 * ca.sin(q1 + q2),
    )
    coriolis = ca.vertcat(
        ca.horzcat(
            -2.0 * m2 * l1 * lc2 * ca.sin(q2) * velocity2,
            -m2 * l1 * lc2 * ca.sin(q2) * velocity2,
        ),
        ca.horzcat(m2 * l1 * lc2 * ca.sin(q2) * velocity1, 0.0),
    )
    velocity = ca.vertcat(velocity1, velocity2)
    acceleration = mass_inverse @ (
        -coriolis @ velocity
        + gravity_torque
        + ca.vertcat(0.0, u[0])
        - ca.vertcat(b1, b2) * velocity
    )
    return ca.vertcat(velocity, acceleration)


def rk4_dynamics(x: ca.SX, u: ca.SX, parameters: ca.SX) -> ca.SX:
    k1 = continuous_dynamics(x, u, parameters)
    k2 = continuous_dynamics(x + 0.5 * DT * k1, u, parameters)
    k3 = continuous_dynamics(x + 0.5 * DT * k2, u, parameters)
    k4 = continuous_dynamics(x + DT * k3, u, parameters)
    return x + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def build_ocp(parameters: np.ndarray):
    state = ca.SX.sym("x", NX)
    control = ca.SX.sym("u", NU)
    model_parameters = ca.SX.sym("p", NP)
    model = AcadosModel()
    model.name = "filterddp_acrobot_rk4"
    model.x = state
    model.u = control
    model.p = model_parameters
    model.disc_dyn_expr = rk4_dynamics(state, control, model_parameters)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.parameter_values = parameters
    ocp.solver_options.N_horizon = N_HORIZON
    ocp.solver_options.tf = N_HORIZON * DT

    ny = NX + NU
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx = np.vstack((np.eye(NX), np.zeros((NU, NX))))
    ocp.cost.Vu = np.vstack((np.zeros((NX, NU)), np.eye(NU)))
    ocp.cost.W = np.diag([2.0, 2.0, 0.2, 0.2, 4.0])
    ocp.cost.yref = np.concatenate((TARGET_STATE, np.zeros(NU)))
    ocp.cost.Vx_e = np.eye(NX)
    ocp.cost.W_e = 2000.0 * STATE_WEIGHT
    ocp.cost.yref_e = TARGET_STATE

    ocp.constraints.x0 = np.zeros(NX)

    options = ocp.solver_options
    options.integrator_type = "DISCRETE"
    options.cost_discretization = "EULER"
    options.nlp_solver_type = "SQP"
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
    options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    options.qp_solver_cond_N = 20
    options.qp_solver_iter_max = 100
    options.qp_solver_ric_alg = 0
    options.qp_solver_cond_ric_alg = 0
    options.hpipm_mode = "ROBUST"
    options.print_level = 0

    return ocp


def initialize_solver(solver, parameters: np.ndarray) -> None:
    for stage in range(N_HORIZON):
        solver.set(stage, "x", np.zeros(NX))
        solver.set(stage, "u", np.zeros(NU))
        solver.set(stage, "p", parameters)
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


def run_experiment(solver, parameter_sets: np.ndarray) -> None:
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


def main() -> None:
    physical_parameters = np.atleast_2d(np.loadtxt(PARAMETERS_FILE))
    fixed_parameters = np.tile(np.array([9.81, 0.0, 0.0]), (len(physical_parameters), 1))
    parameter_sets = np.hstack((physical_parameters, fixed_parameters))

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    ocp = build_ocp(parameter_sets[0])
    ocp.code_gen_options.code_export_directory = str(BUILD_DIR / "c_generated_code")
    ocp.code_gen_options.json_file = "acrobot_rk4_ocp.json"
    solver = AcadosOcpSolver(ocp, verbose=False)

    run_experiment(solver, parameter_sets)


if __name__ == "__main__":
    main()