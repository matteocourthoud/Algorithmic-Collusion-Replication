#=
Replication of Artificial Intelligence, Algorithmic Pricing, and Collusion
    by: Calvano, Calzolari, Denicolò (2020)
    at: https://www.aeaweb.org/articles?id=10.1257/aer.20190623
Code
    author: Matteo Courthoud
    date: 07/05/2021
    git: https://github.com/matteocourthoud
    myself: https://matteocourthoud.github.io/
=#

include("input/init.jl")
include("input/qlearning.jl")

# Init algorithm
game = init.init_game();

# Compute equilibrium
game_equilibrium = qlearning.simulate_game(game);
