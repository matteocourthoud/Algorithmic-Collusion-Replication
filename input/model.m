%% Model of algorithms and competition


classdef model
    
    % Model properties
    properties
        
        % Primitives
        n = 2                   % Number of firms
        alpha = 0.15            % Learning parameter
        beta = 4e-6             % Learning parameter
        delta = 0.95            % Discount factor
        mu = 0.25               % Product differentiation
        c = 1                   % Marginal cost
        a = 2                   % Value of the product
        a0 = 0                  % Value of the outside option
        k = 15                  % Dimension of the price grid
        tstable = 1e5           % Number of iterations needed for stability
        tmax = 1e7              % Maximum number of iterations
        
        % Derived properties
        sdim                    % Dimension of the state
        s0                      % Initial state
        A                       % Action space
        p_minmax                % Minimum and maximum prices
        PI                      % Profit matrix
        Q                       % Q-function of the firms
        
    end
    
            
    
    % Model functions
    methods (Static)
     
        % Compute Demand
        function d = demand(game, p)
            e = exp((game.a - p) / game.mu);
            d = e / (sum(e) + exp(game.a0 / game.mu));
        end
        
        % Compute first order condition
        function zero = foc(game, p)
            d = game.demand(game, p);
            zero = 1 - (p - game.c) .* (1 - d) / game.mu;
        end

        % Compute first order condition of a monopolist
        function zero = foc_monopoly(game, p)
            d = game.demand(game, p);
            d1 = fliplr(d);
            p1 = fliplr(p);
            zero = 1 - (p - game.c) .* (1 - d) / game.mu + (p1 - game.c) .* d1 / game.mu;
        end
        
        % Computes competitive and monopoly prices
        function p = compute_p_competitive_monopoly(game)
            p = zeros(2, game.n);
            options = optimoptions('fsolve','Display','off');
            options.Algorithm = 'levenberg-marquardt';
            p0 = ones(1, game.n) * 3;
            p(1,:) = fsolve(@(x) game.foc(game, x), p0, options);
            p(2,:) = fsolve(@(x) game.foc_monopoly(game, x), p0, options);
        end
        
        % Get action space of the firms
        function A = init_actions(game)
            a = linspace(min(game.p_minmax(1,:)), max(game.p_minmax(2,:)), game.k-2);
            delta = a(2) - a(1);
            A = linspace(min(a) - delta, max(a) + delta, game.k);
        end
                
        % Get state dimension and initial state
        function [sdim, s0] = init_state(game)
            sdim = [game.k, game.k];
        	s0 = ones(1, length(sdim));
        end
        
        % Compute payoffs
        function pi = compute_profits(game, p)
            d = game.demand(game, p);
            pi = (p - game.c) .* d;
        end
        
        % Initialize Profits (k^n x kp x n)
        function PI = init_PI(game)
            
            % Generate all possible action combinations
            statespace = ones(1, game.n) * game.k;
            c = [];
            for i=statespace
                c = [c, {1:i}];
            end
            all_actions = combvec(c{:})';
            
            % Compute profits
            PI = zeros([statespace, game.n]);
            for i=1:size(all_actions,1)
                a = all_actions(i,:);
                p = game.A(a);
                pi = game.compute_profits(game, p);
                PI(a(1), a(2), :) = pi;
            end
            
        end 
                
        % Initialize Q function (n x #s x k)
        function Q = init_Q(game)
            Q = zeros([game.n, game.sdim, game.k]);
            for n=1:game.n
                pi = reshape(mean(game.PI(:, :, n), 3-n), game.k, 1);
                Q(n,:,:,:) = repmat(pi, 1, game.k, game.k) / (1 - game.delta);
            end
        end
        
        % Init preprocess
        function game = init(game0, varargin)
            
            % Init 
            game = game0;
            
            % Add properties
            for k=2:2:size(varargin,2)
                game.(varargin{k-1}) = varargin{k};
            end
            
            % Set properties
            game.p_minmax = game.compute_p_competitive_monopoly(game);
            game.A = game.init_actions(game);
            [game.sdim, game.s0] = game.init_state(game);
            game.PI = game.init_PI(game);
            game.Q = game.init_Q(game);
        end
        
        
    end
end