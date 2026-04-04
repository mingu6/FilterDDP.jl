import numpy as np

import aligator
from aligator import constraints, dynamics, manifolds
import casadi as cs

dt = 0.04
force_lim = 0.3
vel_lim = 3.0
r_push = 0.01

nx = 4
nu = 7
N = 101

tol = 1e-5
mu_init = 0.05

verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(tol, mu_init, verbose=verbose)
solver.max_iters = 3000
solver.reg_min = 1e-5

res = []

with open("../filterddp/params/manip_pushing.txt", 'r') as file:
    for exper_ind, line in enumerate(file):
        params = [float(num_str) for num_str in line.split()]
        zx, zy, c, mu_fric = params[:4]
        obstacle = params[4:]
        r_total = max(zx, zy) + r_push

        class Constraint(aligator.StageFunction):
            def __init__(self) -> None:
                super().__init__(nx, nu, 2)
                fN = cs.SX.sym('fN', 1)
                ft = cs.SX.sym('ft', 1)
                phidot = cs.SX.sym('phidot', 2)
                sc = cs.SX.sym('sc', 2)
                so = cs.SX.sym('so', 1)
                u = cs.vertcat(fN, ft, phidot, sc, so)

                f = cs.vertcat(
                    (mu_fric * fN - ft) * phidot[0] - sc[0],
                    (mu_fric * fN + ft) * phidot[1] - sc[1]
                )
                Ju = cs.jacobian(f, u)
                self.f = cs.Function('f', [u], [f])
                self.Ju = cs.Function('Ju', [u], [Ju])

            def evaluate(self, x, u, data):
                data.value[:] = self.f(u).toarray()[:, 0]

            def computeJacobians(self, x, u, data):
                data.Jx[:, :] = 0.0
                data.Ju[:] = self.Ju(u).toarray()


        class NonNeg(aligator.StageFunction):
            def __init__(self) -> None:
                super().__init__(nx, nu, 6)
                x = cs.SX.sym('x', nx)

                fN = cs.SX.sym('fN', 1)
                ft = cs.SX.sym('ft', 1)
                phidot = cs.SX.sym('phidot', 2)
                sc = cs.SX.sym('sc', 2)
                so = cs.SX.sym('so', 1)
                u = cs.vertcat(fN, ft, phidot, sc, so)

                obs = (obstacle[2] + r_total) ** 2 - cs.sumsqr(x[:2] - obstacle[:2]) - so
                constr = cs.vertcat(
                    obs,
                    -(mu_fric * fN - ft),
                    -(mu_fric * fN + ft),
                    -sc,
                    -so
                )
                self.constr = cs.Function('constr', [x, u], [constr])
                Jx = cs.jacobian(constr, x)
                Ju = cs.jacobian(constr, u)
                self.Ju = cs.Function('Ju', [x, u], [Ju])
                self.Jx = cs.Function('Jx', [x, u], [Jx])

            def evaluate(self, x, u, data):
                data.value[:] = self.constr(x, u).toarray()[:, 0]

            def computeJacobians(self, x, u, data):
                data.Jx[:, :] = self.Jx(x, u).toarray()
                data.Ju[:, :] = self.Ju(x, u).toarray()


        class Limit(aligator.StageFunction):
            def __init__(self) -> None:
                super().__init__(nx, nu, 5)

            def evaluate(self, x, u, data):
                data.value[:4] = u[:4]
                data.value[4] = x[3]

            def computeJacobians(self, x, u, data):
                data.Ju[:, :] = 0.0
                data.Jx[:, :] = 0.0
                data.Jx[4, 3] = 1.0
                for i in range(4):
                    data.Ju[i, i] = 1.0


        class Dynamics(dynamics.ODEAbstract):
            def __init__(self) -> None:
                space = manifolds.VectorSpace(nx)
                super().__init__(space, nu)
                self.L = np.array([1.0, 1.0, c ** -2.0])
                x = cs.SX.sym('x', nx)
                fN = cs.SX.sym('fN', 1)
                ft = cs.SX.sym('ft', 1)
                phidot = cs.SX.sym('phidot', 2)
                sc = cs.SX.sym('sc', 2)
                so = cs.SX.sym('so', 1)
                u = cs.vertcat(fN, ft, phidot, sc, so)

                R = cs.SX(3, 3)
                R[0, 0] = cs.cos(x[2])
                R[0, 1] = -cs.sin(x[2])
                R[1, 0] = cs.sin(x[2])
                R[1, 1] = cs.cos(x[2])
                R[2, 2] = 1.0

                Jc = cs.SX(2, 3)
                Jc[0, 0] = 1.0
                Jc[0, 2] = 0.5 * zx * cs.tan(x[3])
                Jc[1, 1] = 1.0
                Jc[1, 2] = -0.5 * zx

                xdot = cs.vertcat(R @ (self.L * (Jc.T @ u[:2])), phidot[0] - phidot[1])
                self.xdot = cs.Function('xdot', [x, u], [xdot])

                Jx = cs.jacobian(xdot, x)
                Ju = cs.jacobian(xdot, u)
                self.Jx = cs.Function('Jx', [x, u], [Jx])
                self.Ju = cs.Function('Ju', [x, u], [Ju])

            def forward(self, x, u, data):
                data.xdot[:] = self.xdot(x, u).toarray()[:, 0]

            def dForward(self, x, u, data):
                data.Ju[:, :] = self.Ju(x, u).toarray()
                data.Jx[:, :] = self.Jx(x, u).toarray()


        class Cost(aligator.CostAbstract):
            def __init__(self):
                space = manifolds.VectorSpace(nx)
                super().__init__(space, nu)
                fN = cs.SX.sym('fN', 1)
                ft = cs.SX.sym('ft', 1)
                phidot = cs.SX.sym('phidot', 2)
                sc = cs.SX.sym('sc', 2)
                so = cs.SX.sym('so', 1)
                u = cs.vertcat(fN, ft, phidot, sc, so)
                J = 1e-2 * cs.sumsqr(u[:2]) + 2.0 * (cs.sum(sc) + so)
                self.cost = cs.Function('J', [u], [J])
                self.Ju = cs.Function('Ju', [u], [cs.jacobian(J, u)])
                self.Juu = cs.Function('Juu', [u], [cs.hessian(J, u)[0]])

            def evaluate(self, x, u, data):
                data.value = self.cost(u).toarray()[0][0]

            def computeGradients(self, x, u, data):
                data.Lu[:] = self.Ju(u).toarray()

            def computeHessians(self, x, u, data):
                data.Lxx[:, :] = 0.0
                data.Lxu[:, :] = 0.0
                data.Luu[:, :] = self.Juu(u).toarray()


        x0 = np.zeros(nx)
        x_tar = np.array([0.3, 0.4, 1.5 * np.pi, 0.0])
        wterm = np.array([20.0, 20.0, 20.0, 20.0])
        space = manifolds.VectorSpace(nx)

        term_cost = aligator.QuadraticStateCost(space, nu, x_tar, 2 * np.diag(wterm))
        problem = aligator.TrajOptProblem(x0, nu, space, term_cost=term_cost)
        dynmodelc = Dynamics()
        dynmodel = dynamics.IntegratorEuler(dynmodelc, dt)

        # bound constraint

        rcost = Cost()
        constr = Constraint()
        nonneg = NonNeg()
        limit = Limit()

        for i in range(N-1):
            stage = aligator.StageModel(rcost, dynmodel)
            stage.addConstraint(constr, constraints.EqualityConstraintSet())
            stage.addConstraint(nonneg, constraints.NegativeOrthant())
            stage.addConstraint(limit, constraints.BoxConstraint(
                np.array([0.0, -force_lim, 0.0, 0.0, -0.9]),
                np.array([force_lim, force_lim, vel_lim, vel_lim, 0.9]))
                )
            problem.addStage(stage)

        us_init = [0.01 * np.ones(nu)] * (N-1)
        xs_init = aligator.rollout(dynmodel, x0, us_init)

        solver.setup(problem)
        solver.run(problem, xs_init, us_init)

        results = solver.results

        converged = 'true' if results.conv else 'false'
        res.append([str(exper_ind+1), str(results.num_iters), converged,
                    str(results.traj_cost), str(results.primal_infeas)])

iters = 0
cost = 0.0
viol = 0.0
succ = 0
for r in res:
    iters += int(r[1])
    cost += float(r[3])
    viol += float(r[4])
    if r[2] == 'true':
        succ += 1

print("Average number of iterations: ", iters / len(res))
print("Average cost: ", cost / len(res))
print("Average violation: ", viol / len(res))
print("Successes: ", succ)

with open("results/manip_pushing.txt", 'w') as file:
    for line in res:
        file.write(f"{' '.join(line)}\n")
