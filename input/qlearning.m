%% Q-learning Functions


classdef qlearning
    
    % Q-learning functions
    methods (Static)
        
        % Pick strategies by exploration vs exploitation
        function a = pick_strategies(game, s, t)
            a = zeros(1, game.n);
            pr_explore = exp(- t * game.beta);
            e = (pr_explore > rand(1,3));
            for n=1:game.n
                 if e(n)
                    a(n) = randi(game.k);
                else
                    [~, a(n)] = max(game.Q(n, s(1), s(2), :));
                end

            end
        end
                
        % Update Q matrix
        function [game, stable] = update_q(game, s, a, s1, pi, stable)
            for n=1:game.n
                old_value = game.Q(n, s(1), s(2), a(n));
                max_q1 = max(game.Q(n, s1(1), s1(2), :));
                new_value = pi(n) + game.delta * max_q1;
                [~, old_argmax] = max(game.Q(n, s(1), s(2), :));
                game.Q(n, s(1), s(2), a(n)) = (1 - game.alpha) * old_value + game.alpha * new_value;
                % Check stability
                [~, new_argmax] = max(game.Q(n, s(1), s(2), :));
                same_argmax = (old_argmax == new_argmax);
                stable = (stable + same_argmax) * same_argmax;
            end
        end    
        
        % Check if game converged
        function converged = check_convergence(game, t, stable)
            if rem(t, game.tstable)==0
                fprintf("t=%i\n", t)
            end
            if stable > game.tstable
                disp('Converged!')
                converged = true;
            elseif t==game.tmax
                disp('ERROR! Not Converged!')
                converged = true;
            else
                converged = false;
            end
        end
        
        % Simulate game
        function game = simulate_game(game)
            s = game.s0;
            stable = 0;
            % Iterate until convergence
            for t=1:game.tmax
                a = qlearning.pick_strategies(game, s, t);
                pi = game.PI(a(1), a(2), :);
                s1 = a;
                [game, stable] = qlearning.update_q(game, s, a, s1, pi, stable);
                s = s1;
                converged = qlearning.check_convergence(game, t, stable);
                if converged
                    break
                end
            end
        end
        
    end
end