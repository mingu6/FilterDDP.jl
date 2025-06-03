using StatsPlots
using Statistics

problemclass = "cartpole_friction"
fs_x = 21
fs_y = 22

include("utils.jl")

names = ["IPDDP2" "IPOPT" "IPOPT (B)" "ProxDDP"]

_, iters_ipd, _, objs_ipd, constrs_ipd, wall_ipd, _ = read_results("ipddp2/results/$problemclass.txt")
_, iters_ipo, _, objs_ipo, constrs_ipo, wall_ipo, _ = read_results("ipopt/results/$problemclass.txt")
_, iters_ipob, _, objs_ipob, constrs_ipob, wall_ipob, _ = read_results("ipopt/results/bfgs_$problemclass.txt")
_, iters_al, _, objs_al, constrs_al, _, _ = read_results("proxddp/results/$problemclass.txt")

# objective function value

boxplot(names, [objs_ipd objs_ipo objs_ipob objs_al], legend=false,
    xtickfontsize=fs_x, ytickfontsize=fs_y, ylims=(0.0, 5.1), size=(650, 500))
savefig("plots/$problemclass/objective3.svg")

# constraint violation value

boxplot(names, [constrs_ipd constrs_ipo constrs_ipob constrs_al], legend=false, yticks=[1e-16, 1e-12, 1e-8, 1e-4, 1],
    xtickfontsize=fs_x, ytickfontsize=fs_y-3, yaxis=:log10, ylims=(1e-16, 1e1), size=(650, 500))
savefig("plots/$problemclass/constr3.svg")

# iteration count

boxplot(names, [iters_ipd iters_ipo iters_ipob iters_al], legend=false,
    xtickfontsize=fs_x, ytickfontsize=fs_y-3, ylims=(0, 1130), size=(650, 500))
savefig("plots/$problemclass/iterations3.svg")

# wall time per iteration

boxplot(names[:, 1:3], [(wall_ipd ./ iters_ipd) (wall_ipo ./ iters_ipo) (wall_ipob ./ iters_ipob)], legend=false,
    xtickfontsize=fs_x, ytickfontsize=fs_y, ylims=(0, 5.9), size=(500, 500))
savefig("plots/$problemclass/time3.svg")

println("Cartpole")
println()
println("Objective (rel IPOPT): ", median(objs_ipd ./ objs_ipo))
println("Iteration Count (rel IPOPT): ", median(iters_ipd ./ iters_ipo))
println("Wall Time per iteration (rel IPOPT): ", median((wall_ipd ./ iters_ipd) ./ (wall_ipo ./ iters_ipo)))
println()
println("Objective (rel IPOPT BFGS): ", median(objs_ipd ./ objs_ipob))
println("Iteration Count (rel IPOPT BFGS): ", median(iters_ipd ./ iters_ipob))
println("Wall Time per iteration (rel IPOPT BFGS): ", median((wall_ipd ./ iters_ipd) ./ (wall_ipob ./ iters_ipob)))
println()
println("Objective (rel ProxDDP): ", median(objs_ipd ./ objs_al))
println("Iteration Count (rel ProxDDP): ", median(iters_ipd ./ iters_al))
println()
