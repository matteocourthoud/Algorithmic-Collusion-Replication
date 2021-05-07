"""Q-learning Functions"""

module qlearning

    export simulate_game

    include("model.jl")

    function get_actions(game, s, t)
        """Get actions"""
        a = Int.(zeros(1, game.n));
        pr_explore = exp(- t * game.beta);
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

    function update_Q(game, profits, s, s1, a)
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
            game.stable = (game.stable + same_argmax) * same_argmax
        end
        return game
    end

    function check_convergence(game, t, tstable, tmax)
        """Check if game converged"""
        if rem(t, tstable)==0
            print("\nstable = ", game.stable)
        end
        if game.stable > tstable
            print("\nConverged!")
            return true;
        elseif t==tmax
            print("\nERROR! Not Converged!")
            return true;
        else
            return false;
        end
    end

    # Simulate game
    function simulate_game(game, tstable, tmax)
        s = game.s0;
        # Iterate until convergence
        for t=1:tmax
            a = get_actions(game, s, t)
            profits = game.PI[a[1], a[2], :]
            s1 = a
            game = update_Q(game, profits, s, s1, a)
            s = s1;
            if check_convergence(game, t, tstable, tmax)
                break
            end
        end
        return game
    end

end
