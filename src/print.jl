function solver_info()
    println(crayon"bold red","   
 _____ _ _ _            ____  ____  ____  
|  ___(_) | |_ ___ _ __|  _ \\|  _ \\|  _ \\ 
| |_  | | | __/ _ \\ '__| | | | | | | |_) |
|  _| | | | ||  __/ |  | |_| | |_| |  __/ 
|_|   |_|_|\\__\\___|_|  |____/|____/|_|                 
    ")
    print(crayon"reset")
end

function iteration_status(
    data::SolverData,
    options::Options
    )

    # header
    if rem(data.k, options.print_frequency) == 0
        @printf "  iter     objective        pr_inf       du_inf       cs_inf     lg(μ)   lg(reg)    alpha     ls   wall_time  solver_time\n"
    end
    

    # iteration information
    @printf(" %5s   %.8e   %.4e   %.4e   %.4e   % 1.2f  %5s   %.4e  %2s    %5.2f    %5.2f\n",
        data.k, data.objective, data.primal_inf, data.dual_inf, data.cs_inf,
        log10(data.μ), data.reg_last == 0.0 ? "    -  " : @sprintf("% 2.4f", log10(data.reg_last)),
        data.step_size, data.l, data.wall_time * 1000, data.solver_time * 1000)
end

function on_exit(data::SolverData)
    println()
    println()
    if data.status == 0
        println("EXIT: Optimal solution found.")
    elseif data.status == 1
        println("EXIT: Failed, unable to find iteration matrix with desired inertia in backward pass.")
    elseif data.status == 7
        println("EXIT: Failed, line-search unable to find acceptable iterate in forward pass.")
    elseif data.status == 8
        println("EXIT: Failed, maximum solver iterations reached.")
    else
        println("DEBUG: This message should not display.")
    end
    println()
end
