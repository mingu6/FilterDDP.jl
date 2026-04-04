using StatsPlots
using Statistics

problemclass = "acrobot_contact" # acrobot_contact, cartpole_friction, concar_quad, concar, pushing_1_obs
fs_x = 19
fs_y = 22
font_ = "Computer Modern"

include("utils.jl")

names = ["FilterDDP" "IPOPT" "IPOPT (B)" "ProxDDP"]

_, iters_fddp, status_fddp, objs_fddp, constrs_fddp, wall_fddp, _ = read_results("filterddp/results/$problemclass.txt")
_, iters_ipo, status_ipo, objs_ipo, constrs_ipo, wall_ipo, _ = read_results("ipopt/results/$problemclass.txt")
_, iters_ipob, status_ipob, objs_ipob, constrs_ipob, wall_ipob, _ = read_results("ipopt/results/bfgs_$problemclass.txt")
_, iters_al, status_al, objs_al, constrs_al, _, _ = read_results("proxddp/results/$problemclass.txt")

# objective function value

boxplot(names, [objs_fddp objs_ipo objs_ipob objs_al], legend=false,
    fontfamily=font_, xtickfontsize=fs_x, ytickfontsize=fs_y, size=(650, 500))
savefig("plots/$(problemclass)_objective.pdf")

# constraint violation value

boxplot(names, [constrs_fddp constrs_ipo constrs_ipob constrs_al], legend=false,
    fontfamily=font_, xtickfontsize=fs_x, ytickfontsize=fs_y-3, yaxis=:log10, size=(650, 500))
savefig("plots/$(problemclass)_violation.pdf")

# iteration count

boxplot(names, [iters_fddp iters_ipo iters_ipob iters_al], legend=false,
    fontfamily=font_, xtickfontsize=fs_x, ytickfontsize=fs_y-3, size=(650, 500))
savefig("plots/$(problemclass)_iteration.pdf")

# wall time per iteration (ms)

boxplot(names[:, 1:3], [(wall_fddp ./ iters_fddp) (wall_ipo ./ iters_ipo) (wall_ipob ./ iters_ipob)], legend=false,
    fontfamily=font_, xtickfontsize=fs_x-3, ytickfontsize=fs_y, size=(500, 500))
savefig("plots/$(problemclass)_time_per.pdf")

# cumulative wall time (ms)

boxplot(names[:, 1:2], [wall_fddp wall_ipo], legend=false, fontfamily=font_, xtickfontsize=fs_x-3, ytickfontsize=fs_y, size=(500, 500))
savefig("plots/$(problemclass)_time.pdf")

# number of successful solves by iteration count

function success_by_iteration_count(iters, status; max_iters=3000)
    sucess_by_iters = []
    for x = 1:max_iters
        num_succ = sum((iters .<= x) .* status)
        push!(sucess_by_iters, num_succ)
    end
    return sucess_by_iters
end

success_by_fddp = success_by_iteration_count(iters_fddp, status_fddp)
success_by_ipo = success_by_iteration_count(iters_ipo, status_ipo)
success_by_ipob = success_by_iteration_count(iters_ipob, status_ipob)
success_by_al = success_by_iteration_count(iters_al, status_al)

plot(1:3000, [success_by_fddp success_by_ipo success_by_ipob success_by_al], legend=:right, fontfamily=font_,
    xtickfontsize=fs_y, ytickfontsize=fs_y, legendfontsize=fs_y, linewidth=2, size=(500, 600), label=names)
savefig("plots/$(problemclass)_succ_by_iter.pdf")
