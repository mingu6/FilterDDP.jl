include("../models/acrobot.jl")

# joints and ee points 
function acrobot_positions(q, p)

    θ1 = q[1]
    θ2 = q[2]

    l1 = p.l1
    l2 = p.l2

    p0 = (0, 0)

    p1 = (
        l1 * sin(θ1),
       -l1 * cos(θ1)
    )

    p2 = (
        p1[1] + l2 * sin(θ1 + θ2),
        p1[2] - l2 * cos(θ1 + θ2)
    )

    return p0, p1, p2
end

function animate_acrobot(q_traj, dt, params::DoublePendulum{T};
                         trail = true,
                         fps = 30,
                         filename = "acrobot.gif",
                         results_dir = "results") where T

    print("\nanimating..")

    isdir(results_dir) || mkpath(results_dir)
    filepath = joinpath(results_dir, filename)

    nframes = length(q_traj)
    freq = 1.0 / dt

    frame_step = freq > fps ? round(Int, freq / fps) : 1
    fps        = freq > fps ? fps : freq
    t = 0.0

    L = params.l1 + params.l2

    anim = @animate for k = 1:frame_step:nframes

        q = q_traj[k]

        p0, p1, p2 = acrobot_positions(q, params)

        plot(
            xlims = (-L-0.2, L+0.2),
            ylims = (-L-0.2, L+0.2),
            aspect_ratio = :equal,
            legend = false,
            title = "Acrobot"
        )

        # links
        plot!([p0[1], p1[1]], [p0[2], p1[2]], lw = 4, color = :black)
        plot!([p1[1], p2[1]], [p1[2], p2[2]], lw = 4, color = :black)

        # joints & ee
        scatter!([p0[1], p1[1], p2[1]], [p0[2], p1[2], p2[2]], ms = 6, color = :red)

        # trail of ee
        if trail && k > 1
            pts = [acrobot_positions(q_traj[j], params)[3] for j in 1:k]
            plot!(first.(pts), last.(pts), lw = 1, color = :gray, alpha = 0.4)
        end

        annotate!(L, L,text(@sprintf("t = %.2f s", t), :right, 12, :black))

        t = k * dt
    end

    gif(anim, filepath, fps = fps)

    print("   -> animation finished.")
end
