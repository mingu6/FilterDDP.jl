using StatsPlots
using Statistics

problemclass = "concar_quad"
fs_x = 19
fs_y = 22
font_ = "Computer Modern"

include("utils.jl")

names = ["FilterDDP" "IPOPT" "IPOPT (B)" "ProxDDP"]

_, iters_ipd, _, objs_ipd, constrs_ipd, wall_ipd, _ = read_results("FilterDDP/results/$problemclass.txt")
_, iters_ipo, _, objs_ipo, constrs_ipo, wall_ipo, _ = read_results("ipopt/results/$problemclass.txt")
_, iters_ipob, _, objs_ipob, constrs_ipob, wall_ipob, _ = read_results("ipopt/results/bfgs_$problemclass.txt")
_, iters_al, _, objs_al, constrs_al, _, _ = read_results("proxddp/results/$problemclass.txt")

# objective function value

boxplot(names, [objs_ipd objs_ipo objs_ipob objs_al], legend=false,
    fontfamily=font_, xtickfontsize=fs_x, ytickfontsize=fs_y, ylims=(0.0, 7.7), size=(650, 500))
savefig("plots/$problemclass/objective2.pdf")

# constraint violation value

boxplot(names, [constrs_ipd constrs_ipo constrs_ipob constrs_al], legend=false,
    fontfamily=font_, xtickfontsize=fs_x, ytickfontsize=fs_y-3, yaxis=:log10, ylims=(1e-16, 5e-4), size=(650, 500))
savefig("plots/$problemclass/constr2.pdf")

# iteration count

boxplot(names, [iters_ipd iters_ipo iters_ipob iters_al], legend=false,
    fontfamily=font_, xtickfontsize=fs_x, ytickfontsize=fs_y-3, ylims=(0, 750), size=(650, 500))
savefig("plots/$problemclass/iterations2.pdf")

# wall time

boxplot(names[:, 1:3], [(wall_ipd ./ iters_ipd) (wall_ipo ./ iters_ipo) (wall_ipob ./ iters_ipob)], legend=false,
    fontfamily=font_, xtickfontsize=fs_x-3, ytickfontsize=fs_y, ylims=(0, 3.9), size=(500, 500))
savefig("plots/$problemclass/time2.pdf")

println("Car Obstacle Avoidance (Quadratic)")
println()
println("Objective (rel IPOPT): ", median(objs_ipd ./ objs_ipo), " ", median(objs_ipd), " ", median(objs_al))
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
