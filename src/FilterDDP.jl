module FilterDDP

using LinearAlgebra 
using Symbolics 
using Printf
using Crayons
using FastLapackInterface

include("objectives.jl")
include("dynamics.jl")
include("constraints.jl")
include("bounds.jl")
include(joinpath("data", "model.jl"))
include(joinpath("data", "objectives.jl"))
include(joinpath("data", "constraints.jl"))
include(joinpath("data", "update_rule.jl"))
include(joinpath("data", "problem.jl"))
include(joinpath("data", "solver.jl"))
include(joinpath("data", "methods.jl"))
include("options.jl")
include("solver.jl")
include("derivatives.jl")
include("inertia_correction.jl")
include("backward_pass.jl")
include("forward_pass.jl")
include("print.jl")
include("solve.jl")

# costs
export Objective

# constraints 
export Constraint

# dynamics 
export Dynamics

# bounds
export Bound

# solver 
export Solver,
    Options,
    solve!,
    get_trajectory

end # module
