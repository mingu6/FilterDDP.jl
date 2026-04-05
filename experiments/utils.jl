using Printf
using Statistics

function read_results(fname::String)
    regex_results = r"\s*(\d+)\s+(\d+)\s+(\w+)\s+(\d+.\d+e[+-]\d+)\s+(\d+.\d+e[+-]\d+)\s+(\d+.\d+)"
    regex_no_bm = r"\s*(\d+)\s+(\d+)\s+(\w+)\s+([+-]?\d+.\d+e?[+-]?\d+)?\s+(\d+.\d+e?[+-]?\d+)"
    
    seeds = Int64[]
    iters = Int64[]
    statuss = Bool[]
    objs = Float64[]
    constrs = Float64[]
    walls = Float64[]
    
       open(fname, "r") do io
        lines = readlines(io)
        
        for line in lines
            res = match(regex_results, line)
            if !isnothing(res)
                seed, iter, status, obj, constr, wall = parse_results(res.captures)
                push!(seeds, seed)
                push!(iters, iter)
                push!(statuss, status)
                push!(objs, obj)
                push!(constrs, constr)
                push!(walls, wall)
            end
            !isnothing(res) && continue
            # no benchmarks run for wall time
            res_no_bm = match(regex_no_bm, line)
            if !isnothing(res_no_bm)
                seed, iter, status, obj, constr, wall = parse_results(res_no_bm.captures)
                push!(seeds, seed)
                push!(iters, iter)
                push!(statuss, status)
                push!(objs, obj)
                push!(constrs, constr)
                push!(walls, wall)
            end
        end
    end 
    
    return seeds, iters, statuss, objs, constrs, walls
end

function parse_results(res)
    seed = parse(Int64, res[1])
    iters = parse(Int64, res[2])
    status = parse(Bool, res[3])
    obj = parse(Float64, res[4])
    constr = parse(Float64, res[5])
    if length(res) == 6
        wall = parse(Float64, res[6])
    else
        wall = 0.0
    end
    return seed, iters, status, obj, constr, wall
end
