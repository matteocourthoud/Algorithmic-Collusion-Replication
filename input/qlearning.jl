"""Q-learning Functions"""

module qlearning

    export simulate_game

    function get_actions(game::Main.init.model, s::Array{Int8,1}, t::Int64)::Array{Int8,1}
        """Get actions"""
        a = Int8.(zeros(game.n));
        pr_explore::Float64 = exp(- t * game.beta);
        e = pr_explore .> rand(1,3);
        for n = 1:game.n
            if e[n]
                a[n] = rand(1:game.k);
            else
                a[n] = argmax(game.Q[n, s[1], s[2], :]);
            end
        end
        return a
    end

    function update_Q(game::Main.init.model, profits::Array{Float32,1}, s::Array{Int8,1}, s1::Array{Int8,1}, a::Array{Int8,1}, stable::Int64)::Tuple{Array{Float32,4},Int64}
        """Update Q function"""
        for n=1:game.n
            old_value = game.Q[n, s[1], s[2], a[n]];
            max_q1 = max(game.Q[n, s1[1], s1[2], :]...);
            new_value = profits[n] + game.delta * max_q1;
            old_argmax = argmax(game.Q[n, s[1], s[2], :])
            game.Q[n, s[1], s[2], a[n]] = (1 - game.alpha) * old_value + game.alpha * new_value;
            # Check stability
            new_argmax = argmax(game.Q[n, s[1], s[2], :])
            same_argmax = Int8(old_argmax == new_argmax);
            stable = (stable + same_argmax) * same_argmax
        end
        return game.Q, stable
    end

    function check_convergence(game::Main.init.model, t::Int64, stable::Int64)::Bool
        """Check if game converged"""
        if rem(t, game.tstable)==0
            print("\nstable = ", stable)
        end
        if stable > game.tstable
            print("\nConverged!")
            return true;
        elseif t==game.tmax
            print("\nERROR! Not Converged!")
            return true;
        else
            return false;
        end
    end

    function export_game(game::Main.init.model)
        """Export game"""
        open("output/games/game1.json", "w") do io
            JSON3.write(io, game)
        end
    end

    # Simulate game
    function simulate_game(game::Main.init.model)::Main.init.model
        s = Int8.([1,1,1]);;
        stable = 0
        # Iterate until convergence
        for t=1:game.tmax
            a = get_actions(game, s, t)
            profits = game.PI[a[1], a[2], :]
            s1 = a
            game.Q, stable = update_Q(game, profits, s, s1, a, stable)
            s = s1;
            if check_convergence(game, t, stable)
                break
            end
        end
        return game
    end

end
