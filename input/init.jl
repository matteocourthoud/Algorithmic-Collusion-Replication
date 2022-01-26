"""Model of algorithms and competition"""

module init

    export init_game

    using Parameters, Statistics, Combinatorics, NLsolve, StructTypes

    # Model properties
    @with_kw mutable struct model
        """Default Properties"""
        n::Int8 = 2                             # Number of firms
        alpha::Float32 = 0.15                   # Learning parameter
        beta::Float32 = 4e-6                    # Learning parameter
        delta::Float32 = 0.95                   # Discount factor
        mu::Float32 = 0.25                      # Product differentiation
        c::Int8 = 1                             # Marginal cost
        a::Int8 = 2                             # Value of the product
        a0::Int8 = 0                            # Value of the outside option
        k::Int8 = 15                            # Dimension of the price grid
        tstable::Int32 = 1e5                    # Number of iterations needed for stability
        tmax::Int32 = 1e7                       # Maximum number of iterations

        """Derived Properties"""
        A::Array{Float32,1} = zeros(1)          # Action space
        p_minmax::Array{Float32,2} = zeros(1,1) # Minimum and maximum prices
        PI::Array{Float32,3} = zeros(1,1,1)     # Profit matrix
        Q::Array{Float32,4} = zeros(1,1,1,1)    # Q-function of the firms

    end

    # Save struct type
    StructTypes.StructType(::Type{model}) = StructTypes.Mutable()

    function demand(game::model, p::Array{Float32,1})::Array{Float32,1}
        """Compute Demand"""
        e = exp.((game.a .- p) ./ game.mu);
        d = e ./ (sum(e) + exp(game.a0 / game.mu));
        return d
    end

    function foc(game::model, p::Array{Float32,1})::Array{Float32,1}
        """Compute first order condition"""
        d = demand(game, p);
        zero = 1 .- (p .- game.c) .* (1 .- d) ./ game.mu;
        return zero
    end

    function foc_monopoly(game::model, p::Array{Float32,1})::Array{Float32,1}
        """Compute first order condition of a monopolist"""
        d = demand(game, p);
        d1 = reverse(d, dims=1);
        p1 = reverse(p, dims=1);
        zero = 1 .- (p .- game.c) .* (1 .- d) ./ game.mu .+ (p1 .- game.c) .* d1 ./ game.mu;
        return zero
    end

    function compute_p_competitive_monopoly(game::model)::Array{Float32,2}
        """Computes competitive and monopoly prices"""
        p0 = Float32.(ones(game.n) * 3);
        p_competitive = nlsolve((p -> foc(game, p)), p0).zero;
        p_monopoly = nlsolve((p -> foc_monopoly(game, p)), p0).zero;
        return [p_competitive p_monopoly]
    end

    function init_actions(game::model)::Array{Float32,1}
        """Get action space of the firms"""
        a = range(min(game.p_minmax[:,1]...), max(game.p_minmax[:,2]...), length=game.k-2);
        delta = a[2] - a[1];
        A = collect(a[1]-delta:delta:a[end]+2*delta)
        return A
    end

    function compute_profits(game::model, p::Array{Float32,1})::Array{Float32,1}
        """Compute payoffs"""
        d = demand(game, p);
        profits = (p .- game.c) .* d;
        return profits
    end

    function init_PI(game::model)::Array{Float32,3}
        """Initialize Profits (k^n x n)"""
        PI = zeros(game.k, game.k, game.n);
        for a1=1:game.k
            for a2=1:game.k
                p = [game.A[a1], game.A[a2]];
                PI[a1, a2, :] = compute_profits(game, p);
            end
        end
        return PI
    end

    function init_Q(game::model)::Array{Float32,4}
        """Initialize Q function (n x #s x k)"""
        Q = zeros(game.n, game.k, game.k, game.k);
        for n=1:game.n
            profits = reshape(mean(game.PI[:, :, n], dims=3-n), (1, 1, 1, game.k));
            Q[n, :, :, :] = repeat(profits ./ (1 - game.delta), outer=[1, game.k, game.k, 1]);
        end
        return Q
    end

    function init_game()::model
        """This function initializes the game"""
        game = model();
        game.p_minmax = compute_p_competitive_monopoly(game);
        game.A = init_actions(game);
        game.PI = init_PI(game);
        game.Q = init_Q(game);
        return game
    end

end
