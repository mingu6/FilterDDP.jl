# taken from https://github.com/thowell/optimization_dynamics/blob/main/src/models/cartpole/visuals.jl

include("visualise.jl")

function _create_cartpole!(vis, model;
    i = 0,
    tl = 1.0,
    color = Colors.RGBA(0, 0, 0, tl))

l2 = Cylinder(Point3f0(-model.l * 10.0, 0.0, 0.0),
    Point3f0(model.l * 10.0, 0.0, 0.0),
    convert(Float32, 0.0125))

setobject!(vis["slider_$i"], l2, MeshPhongMaterial(color = Colors.RGBA(0.0, 0.0, 0.0, tl)))

l1 = Cylinder(Point3f0(0.0, 0.0, 0.0),
    Point3f0(0.0, 0.0, model.l),
    convert(Float32, 0.025))

setobject!(vis["arm_$i"], l1,
    MeshPhongMaterial(color = Colors.RGBA(0.0, 0.0, 0.0, tl)))

setobject!(vis["base_$i"], HyperSphere(Point3f0(0.0),
    convert(Float32, 0.1)),
    MeshPhongMaterial(color = color))

setobject!(vis["ee_$i"], HyperSphere(Point3f0(0.0),
    convert(Float32, 0.05)),
    MeshPhongMaterial(color = color))
end

function _set_cartpole!(vis, model, x; i = 0)
    px = x[1] + model.l * sin(x[2])
    pz = -model.l * cos(x[2])
    cable_transform([x[1]; 0;0], [px; 0.0; pz])
    settransform!(vis["arm_$i"], cable_transform([x[1]; 0;0], [px; 0.0; pz]))
    settransform!(vis["base_$i"], Translation([x[1]; 0.0; 0.0]))
    settransform!(vis["ee_$i"], Translation([px; 0.0; pz]))
end

function visualize!(vis, model, q;
    i = 0,
    tl = 1.0,
    Δt = 0.1,
    color = Colors.RGBA(0,0,0,1.0))

default_background!(vis)
_create_cartpole!(vis, model, i = i, color = color, tl = tl)

anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))
nq = length(q)

for t = 1:nq
    MeshCat.atframe(anim,t) do
        # _set_cartpole!(vis, model, q[t], i = i)
        px = q[t][1] + model.l * sin(q[t][2])
        pz = -model.l * cos(q[t][2])
        c2 = Vector{Float64}()
        push!(c2, px)
        push!(c2, 0.0)
        push!(c2, pz)
        c1 = Vector{Float64}()
        push!(c1, q[t][1])
        push!(c1, 0.0)
        push!(c1, 0.0)
        cable_transform(c1, c2)
        settransform!(vis["arm_$i"], cable_transform(c1, c2))
        settransform!(vis["base_$i"], Translation(c1))
        settransform!(vis["ee_$i"], Translation(c2))
    end
end

settransform!(vis["/Cameras/default"],
    compose(Translation(0.0, 0.0, -1.0), LinearMap(RotZ(- pi / 2))))
setvisible!(vis["/Grid"], false)

MeshCat.setanimation!(vis,anim)
end