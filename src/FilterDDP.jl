module FilterDDP

using LinearAlgebra 
using Symbolics 
using Printf
using Crayons
using FastLapackInterface

include("workspace.jl")
include(joinpath("ocp", "objective.jl"))
include(joinpath("ocp", "dynamics.jl"))
include(joinpath("ocp", "constraints.jl"))
include(joinpath("ocp", "control_limits.jl"))
include(joinpath("ocp", "ocp.jl"))
include("options.jl")
include("solver_data.jl")
include("solver.jl")
include("inertia.jl")
include("backward_pass.jl")
include("forward_pass.jl")
include("print.jl")
include("solve.jl")

export Objective
export Constraints
export Dynamics
export ControlLimits
export OCP
export Solver,
    Options,
    solve!,
    get_trajectory

end # module
