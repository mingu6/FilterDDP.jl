# FilterDDP.jl - Constrained Differential Dynamic Programming

This repository contains a Julia package for solving constrained optimal control problems (OCPs) with our proposed FilterDDP algorithm. The associated research paper **(accepted to ICRA 2026!)** is located at https://arxiv.org/abs/2504.08278.

FilterDDP is a Differential Dynamic Programming (DDP) algorithm for solving a general class of optimal control problems with nonlinear equality constraints in addition to the discrete-time dynamics. A line-search filter approach is taken to handle these equality constraints. 

Furthermore, an interior point extension analogous to [Pavlov et al. (2021)](https://ieeexplore.ieee.org/document/9332234) is applied for handling inequality constraints. The local quadratic convergence of FilterDDP is formally established in the research paper. The global convergence proof is coming soon! Note that FilterDDP requires exact Hessians and does not perform well under a quasi-Newton approximation.

## Global convergence proof now (3 June 2026) available [http://arxiv.org/abs/2606.01487](http://arxiv.org/abs/2606.01487)

FilterDDP can solve OCPs of the form
```math
\begin{array}{rl}
    \underset{\mathbf{x}, \mathbf{u}}{\text{minimize}}  & J(\mathbf{x}, \mathbf{u}) = \sum_{t=1}^{N-1} \ell(x_t, u_t) + \ell^N(x_N) \\
    \text{subject to} & x_1 = \hat{x}_1, \\
    & x_{t+1} = f(x_t, u_t) \quad \text{for } t \in\{1,\dots, N-1\}, \\
    & h(x_t, u_t) = 0 \quad \text{for } t \in\{1,\dots, N-1\}, \\
    & b_g^L \leq g(x_t, u_t) \leq b_g^U \quad \text{for } t \in\{1,\dots, N-1\}, \\
    & b_u^L \leq u_t \leq b_u^U \quad \text{for } t \in\{1,\dots, N-1\},
\end{array}
```
where $\mathbf{x} = (x_1, \dots, x_N)$ is the state trajectory and $\mathbf{u} = (u_1, \dots, u_{N-1})$ is the control trajectory. Equality, inequality, and control-bound definitions use `EqualityConstraints`, `InequalityConstraints`, and `ControlLimits`, respectively.
- Derivatives can be computed using either [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) or [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) for user-provided costs/objectives, constraints, and dynamics. Both are allocation free and fast.

- Linear algebra is implemented using statically-typed arrays through [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl). This yields extremely fast solve times for sufficiently small problem sizes (with respect to number of controls, states and constraints).

## Nonlinear Inequality Constraints

Nonlinear inequalities $b_g^L \leq g(x_t,u_t) \leq b_g^U$ are represented by `InequalityConstraints`. FilterDDP in practice adds a slack variable per inequality constraint and transcribes the inequality constraint into a equality constraint and a bound constraint given by

```math
g(x_t,u_t)-s_t=0, \qquad b_g^L \leq s_t \leq b_g^U.
```

The slacks are handled efficiently within the algorithm (in the stage-wise KKT block matrices, the slacks are eliminated). For debugging purposes, it is possible to access slack variables via `get_slack_trajectory` and `get_slack_feedback`.

## Automatic NLP Scaling

FilterDDP provides conservative gradient-based scaling as an opt-in solver option. After the initial trajectory is constructed, the objective and each algebraic constraint row whose gradient infinity norm exceeds `nlp_scaling_max_gradient` are scaled down. The default threshold is `100.0`, following IPOPT's gradient-based scaling rule. Dynamics are not explicitly scaled because they are eliminated by the forward rollout.

```julia
options = Options{Float64}(
    nlp_scaling=true,
    nlp_scaling_max_gradient=100.0)
```

The default is `nlp_scaling=false`, though consider turning it on if you run into numerical issues that may be due to problem scaling. Note that when scaling is enabled, `solver.data.primal_inf` is the scaled infeasibility used by the stopping test; `solver.data.primal_inf_raw` retains the maximum constraint violation in the problem's original units. The reported objective remains in its original units.

## Fixed OCP Parameters

Coefficients that vary between OCP instances but are not decision variables can be passed through statically sized parameter vectors. Parameterized stage objectives, dynamics, and constraints accept `(x, u, p)`, while terminal objectives accept `(x, p)`. The parameter dimension is supplied as the final constructor argument, and derivatives are not taken with respect to the parameters during the OCP solve. For example,

```julia
npo = 1
npd = 1
stage_objective = Objective(
    (x, u, p) -> p[1] * dot(u, u), nx, nu, npo)
terminal_objective = TerminalObjective(
    (x, p) -> p[1] * dot(x - xN, x - xN), nx, npo)
dynamics = Dynamics(
    (x, u, p) -> x + p[1] * u, nx, nu, npd)

empty_parameters = SVector{0, Float64}()
parameters = OCPParameters(
    stage_objective=@SVector([2.0]),
    term_objective=@SVector([200.0]),
    dynamics=@SVector([0.01]),
    equality_constraints=empty_parameters,
    inequality_constraints=empty_parameters)
ocp = build_ocp(N, stage_objective, terminal_objective, dynamics,
    equalities, inequalities, control_limits; parameters=parameters)
```

## Unconstrained (and Bound Constrained) Differential Dynamic Programming

FilterDDP.jl also provides an efficient Julia implmentation of both the unconstrained DDP algorithm by Mayne et al. (1970) and the Interior Point DDP (IPDDP) algorithm by [Pavlov et al. (2021)](https://ieeexplore.ieee.org/document/9332234) specialised to control limits only. The unconstrained DDP algorithm is applied when neither equality nor inequality constraints are added and IPDDP is applied when only control limits are added. In both cases, the dynamics dual variable as defined by (8) and (13) in the FilterDDP paper are replaced with the value function gradient $\bar{V}_x^t$ due to improved numerical performance overall.

A control-limits-only OCP can be constructed directly with `build_ocp(N, stage_objective, terminal_objective, dynamics, control_limits)`.

If the OCP has only one class of nonlinear constraints, omit the empty class:

```julia
equality_only_ocp = build_ocp(N, stage_objective, terminal_objective,
    dynamics, equalities, control_limits)
inequality_only_ocp = build_ocp(N, stage_objective, terminal_objective,
    dynamics, inequalities, control_limits)
```

---

# Benchmarks

FilterDDP can be extremely fast for OCPs compared to general purpose solvers like IPOPT. For example, FilterDDP can **solve a non-prehensile pushing task with nonlinear dynamics and constraints more than 10x faster than IPOPT through JuMP.jl with superior robustness**. The left figure shows wall clock times in milliseconds and the right figure shows number of OCPs (y-axis) solved to an optimality error of $10^{-7}$ by a given iteraiton count (x-axis). IPOPT (B) is a quasi-Newton variant of IPOPT while IPOPT represents the full algorithm with exact Hessians.

<img src=manip_time.png width=300> <img src=manip_succ.png width=300>

The benchmark was run on a Lenovo ThinkPad T14s Gen 5 with an Intel® Core™ Ultra 7 155U, 32GB running Ubuntu 24.04.3 LTS and Julia version 1.13.0-beta3 compiled using `-O3` optimisation level.

---

# Installation

1. Switch to package mode with `]` and install the FilterDDP package using the following command: `add https://github.com/mingu6/FilterDDP.jl.git`

2. While still in package mode, run `instantiate` to install dependencies.

---

# Quick Start

We illustrate how to use FilterDDP through a simple constrained OCP given by a block push problem. The objective of the task is to push a block to a target state while minimising the total absolute work along the trajectory. More details this OCP can be found within [this tutorial paper](https://epubs.siam.org/doi/10.1137/16m1062569) (Section 8.1).

```julia
using StaticArrays
using FilterDDP

N = 101         # prediction horizon
const nx::Int64 = 2          # position and velocity
const nu::Int64 = 2          # pushing force and absolute-work epigraph
x1 = @SVector zeros(2)

# ## Define the discrete-time dynamics

f = (x, u) -> x + 0.01 * @SVector [x[2], u[1]]
dyn = Dynamics(f, nx, nu)

# ## Define the stage and terminal objective functions

xN = @SVector [1.0, 0.0]
l = (x, u) -> 0.01 * (2.0 * u[2] - u[1] * x[2])
lN = x -> 500.0 * (x - xN)' * (x - xN)
stage_obj = Objective(l, nx, nu)
term_obj = TerminalObjective(lN, nx)

# ## Define constraint functions

equalities = EqualityConstraints(nx, nu)
inequalities = InequalityConstraints(
    (x, u) -> @SVector [u[2] - u[1] * x[2]],
    nx, nu, @SVector([0.0]), @SVector([Inf]))

# ## Define control limits (hard constraints)

control_limits = ControlLimits(
    SVector{nu, Float64}([-10.0, 0.0]),
    SVector{nu, Float64}([10.0, Inf]))

# ## Create an OCP object containing the problem definition

ocp = build_ocp(N, stage_obj, term_obj, dyn,
    equalities, inequalities, control_limits)

# ## Create a solver instance with the OCP and allocate workspace (memory)

options = Options{Float64}(verbose=true, optimality_tolerance=1e-8)
solver = Solver(ocp; options=options)

# ## Initialise trajectories solve OCP

ū = [SVector{nu, Float64}([0.01; 0.01]) for k = 1:N-1]
solve!(solver, x1, ū)

# ## Recover solution trajectories post-solve

x_sol, u_sol = get_trajectory(solver)

# ## Recover affine feedback policy of form u = fb(x) = ū + α + β(x - x̄) for MPC at the first timestep

fb = get_feedback(solver, 1)
x = rand(nx)
u = fb(x)
```


---

# Examples

See `experiments/filterddp/` for additional OCPs, including the contact-implicit problems found in the paper.
