"""Model of algorithms and competition"""

module model

    export init_game

    using Parameters, Statistics, Combinatorics, NLsolve

    # Model properties
    @with_kw mutable struct thegame
        """Default Properties"""
        n::Int8 = 2                             # Number of firms
        alpha::Float64 = 0.15                   # Learning parameter
        beta::Float64 = 4e-6                    # Learning parameter
        delta::Float64 = 0.95                   # Discount factor
        mu::Float64 = 0.25                      # Product differentiation
        c::Int8 = 1                             # Marginal cost
        a::Int8 = 2                             # Value of the product
        a0::Int8 = 0                            # Value of the outside option
        k::Int8 = 15                            # Dimension of the price grid
        stable::Int64 = 0                     # Number of stable periods

        """Derived Properties"""
        sdim::Array{Int8,1} = zeros(1)          # Dimension of the state
        s0::Array{Int8,1} = zeros(1)            # Initial state
        A::Array{Float64,1} = zeros(1)          # Action space
        p_minmax::Array{Float64,2} = zeros(1,1) # Minimum and maximum prices
        PI::Array{Float64,3} = zeros(1,1,1)     # Profit matrix
        Q::Array{Float64,4} = zeros(1,1,1,1)    # Q-function of the firms

    end

    function demand(game, p)
        """Compute Demand"""
        e = exp.((game.a .- p) ./ game.mu);
        d = e ./ (sum(e) + exp(game.a0 / game.mu));
        return d
    end

    function foc(game, p)
        """Compute first order condition"""
        d = demand(game, p);
        zero = 1 .- (p .- game.c) .* (1 .- d) ./ game.mu;
        return zero
    end

    function foc_monopoly(game, p)
        """Compute first order condition of a monopolist"""
        d = demand(game, p);
        d1 = reverse(d, dims=1);
        p1 = reverse(p, dims=1);
        zero = 1 .- (p .- game.c) .* (1 .- d) ./ game.mu .+ (p1 .- game.c) .* d1 ./ game.mu;
        return zero
    end

    function compute_p_competitive_monopoly(game)
        """Computes competitive and monopoly prices"""
        p0 = ones(game.n, 1) * 3;
        p_competitive = nlsolve((p -> foc(game, p)), p0).zero;
        p_monopoly = nlsolve((p -> foc_monopoly(game, p)), p0).zero;
        return [p_competitive p_monopoly]
    end

    function init_actions(game)
        """Get action space of the firms"""
        a = range(min(game.p_minmax[:,1]...), max(game.p_minmax[:,2]...), length=game.k-2);
        delta = a[2] - a[1];
        A = range(a[1] - delta, a[end] + delta, length=game.k);
        return A
    end

    function init_state(game)
        """Get state dimension and initial state"""
        sdim = [game.k, game.k];
    	s0 = Int8.([1,1,1]);
        return sdim, s0
    end

    function compute_profits(game, p)
        """Compute payoffs"""
        d = demand(game, p);
        profits = (p .- game.c) .* d;
        return profits
    end

    function init_PI(game)
        """Initialize Profits (k^n x n)"""
        PI = zeros(game.k, game.k, game.n);
        for a1=1:game.k
            for a2=1:game.k
                p = game.A[[a1, a2]];
                PI[a1, a2, :] = compute_profits(game, p);
            end
        end
        return PI
    end

    function init_Q(game)
        """Initialize Q function (n x #s x k)"""
        Q = zeros(game.n, game.k, game.k, game.k);
        for n=1:game.n
            profits = reshape(mean(game.PI[:, :, n], dims=3-n), (1, 1, 1, game.k));
            Q[n, :, :, :] = repeat(profits ./ (1 - game.delta), outer=[1, game.k, game.k, 1]);
        end
        return Q
    end

    function init_game()
        """This function initializes the game"""
        game = thegame();
        game.p_minmax = compute_p_competitive_monopoly(game);
        game.A = init_actions(game);
        game.sdim, game.s0 = init_state(game);
        game.PI = init_PI(game);
        game.Q = init_Q(game);
        return game
    end

end
