# FilterDDP.jl - Constrained Differential Dynamic Programming

This repository contains a Julia package for solving constrained optimal control problems (OCPs) with our proposed FilterDDP algorithm. The associated research paper **(accepted to ICRA 2026!)** is located at https://arxiv.org/abs/2504.08278.

FilterDDP is a Differential Dynamic Programming (DDP) algorithm for solving a general class of optimal control problems with nonlinear equality constraints in addition to the discrete-time dynamics. A line-search filter approach is taken to handle these equality constraints. 

Furthermore, an interior point extension analogous to [Pavlov et al.](https://ieeexplore.ieee.org/document/9332234) is applied for handling inequality constraints. The local quadratic convergence of FilterDDP is formally established in the research paper. The global convergence proof is coming soon! 

FilterDDP can solve OCPs of the form
```math
\begin{array}{rl}
    \underset{\mathbf{x}, \mathbf{u}}{\text{minimize}}  & J(\mathbf{x}, \mathbf{u}) = \sum_{t=1}^{N} \ell^t(x_t, u_t) \\
    \text{subject to} & x_1 = \hat{x}_1, \\
    & x_{t+1} = f^t(x_t, u_t) \quad \text{for } t \in\{1,\dots, N-1\}, \\
    & c^t(x_t, u_t) = 0 \quad \text{for } t \in\{1,\dots, N\}, \\
    & b_L \leq u_t \leq b_U \quad \text{for } t \in\{1,\dots, N\},
\end{array}
```
where $\mathbf{x} = (x_1, \dots, x_{N})$ is the state trajectory and $\mathbf{u} = (u_1, \dots, u_{N})$ is the control trajectory. Bounds are specified as $b_L \in [-\infty, \infty)$ and $b_U \in (-\infty, \infty]$.

- All derivatives, including Jacobians and Hessians are automatically generated using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) for user-provided costs/objectives, constraints, and dynamics.

- The size of the state and control vectors may vary across timesteps.

---

# Installation

1. Switch to package mode with `]` and install the FilterDDP package using the following command: `add https://github.com/mingu6/FilterDDP.jl.git`

2. While still in package mode, run `instantiate` to install dependencies.

---

# Quick Start

We illustrate how to use FilterDDP through a simple constrained OCP given by a block push problem. The objective of the task is to push a block to a target state while minimising the total absolute work along the trajectory. More details this OCP can be found within [this tutorial paper](https://epubs.siam.org/doi/10.1137/16m1062569) (Section 8.1).

```julia
using FilterDDP

N = 101         # prediction horizon
nx = 2          # position and velocity
nu = 3          # pushing force, 2x slacks (for the + and - components of abs work)
x1 = [0.0; 0.0]

# ## Define the discrete-time dynamics

f = (x, u) -> x + 0.01 * [x[2], u[1]]
dyn = Dynamics(Float64, f, nx, nu)

# ## Define the stage and terminal objective functions

xN = [1.0, 0.0]
l = (x, u) -> 0.01 * (u[2] + u[3])
lN = (x, u) -> 500.0 * (x - xN)' * (x - xN)
stage_obj = Objective(Float64, l, nx, nu)
term_obj = Objective(Float64, lN, nx, nu)

# ## Define constraint functions

constraints = Constraints(Float64, (x, u) -> [
    u[2] - u[3] - u[1] * x[2]
], nx,  nu)

# ## Define control limits (hard constraints)

cl = ControlLimits([-10.0, 0.0, 0.0], [10.0, Inf, Inf])

# ## Create an OCP object containing the problem definition

ocp = OCP(N, stage_obj, term_obj, dyn; constraints=constraints, control_limits=cl)

# ## Create a solver instance with the OCP and allocate workspace (memory)

options = Options{Float64}(verbose=true, optimality_tolerance=1e-8)
solver = Solver(ocp; options=options)

# ## Initialise trajectories solve OCP

ū = [[0.01; 0.01; 0.01] for k = 1:N]
solve!(solver, x1, ū)
```

---

# Examples

See `experiments/filterddp/` for additional OCPs, including the contact-implicit problems found in the paper.
