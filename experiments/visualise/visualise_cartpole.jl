include("../models/cartpole.jl")

# p0 = left cart corner, p1 = pivot point, p2 = ee
function cartpole_positions(q, p::Cartpole{T}; cart_w = 0.6, cart_h = 0.25, cart_y = 0.0) where T
    pos = q[1]
    θ   = q[2]

    l = p.l

    # cart rectangle corners (for plotting)
    left  = pos - cart_w/2
    right = pos + cart_w/2
    bottom = cart_y
    top    = cart_y + cart_h

    # choose pivot at the top center of cart
    pivot = (pos, top)

    # θ=0 downward convention
    tip = (
        pivot[1] + l * sin(θ),
        pivot[2] - l * cos(θ)
    )

    # useful points for drawing cart as a box
    p_bl = (left,  bottom)
    p_br = (right, bottom)
    p_tr = (right, top)
    p_tl = (left,  top)

    return (p_bl, p_br, p_tr, p_tl), pivot, tip
end

function animate_cartpole(q_traj, dt, params::Cartpole{T};
                          trail = true, fps = 30,
                          filename = "cartpole.gif",
                          results_dir = "results") where T

    print("\nanimating..")

    cart_w = 0.6
    cart_h = 0.25
    cart_y = 0.0
    pad = 0.5

    isdir(results_dir) || mkpath(results_dir)
    filepath = joinpath(results_dir, filename)

    nframes = length(q_traj)
    freq = 1.0 / dt

    frame_step = freq > fps ? round(Int, freq / fps) : 1
    fps        = freq > fps ? fps : freq
    t = 0.0

    xs = first.(q_traj)
    xmin = minimum(xs) - (params.l + pad)
    xmax = maximum(xs) + (params.l + pad)

    ymin = cart_y - (params.l + pad)
    ymax = cart_y + cart_h + (params.l + pad)

    anim = @animate for k = 1:frame_step:nframes

        q = q_traj[k]

        cart_pts, pivot, tip = cartpole_positions(
            q, params; cart_w=cart_w, cart_h=cart_h, cart_y=cart_y
        )

        plot(
            xlims = (xmin, xmax),
            ylims = (ymin, ymax),
            aspect_ratio = :equal,
            legend = false,
            title = "Cartpole"
        )

        # ground line
        plot!([xmin, xmax], [cart_y, cart_y], lw=2, color=:black)

        # cart (rectangle)
        xs_cart = [cart_pts[1][1], cart_pts[2][1], cart_pts[3][1], cart_pts[4][1], cart_pts[1][1]]
        ys_cart = [cart_pts[1][2], cart_pts[2][2], cart_pts[3][2], cart_pts[4][2], cart_pts[1][2]]
        plot!(xs_cart, ys_cart, lw=3, color=:black)

        # pole (line)
        plot!([pivot[1], tip[1]], [pivot[2], tip[2]], lw=4, color=:black)

        # joints/masses
        scatter!([pivot[1], tip[1]], [pivot[2], tip[2]], ms=6, color=:red)

        # trail of pole tip
        if trail && k > 1
            tips = [cartpole_positions(q_traj[j], params;
                                       cart_w=cart_w, cart_h=cart_h, cart_y=cart_y)[3]
                    for j in 1:k]
            plot!(first.(tips), last.(tips), lw=1, color=:gray, alpha=0.4)
        end

        annotate!(xmax, ymax, text(@sprintf("t = %.2f s", t), :right, 12, :black))
        
        t = k * dt
    end

    gif(anim, filepath, fps=fps)

    print("   -> animation finished.")
end
