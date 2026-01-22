"""
@article{egorov2017pomdps,
  author  = {Maxim Egorov and Zachary N. Sunberg and Edward Balaban and Tim A. Wheeler and Jayesh K. Gupta and Mykel J. Kochenderfer},
  title   = {{POMDP}s.jl: A Framework for Sequential Decision Making under Uncertainty},
  journal = {Journal of Machine Learning Research},
  year    = {2017},
  volume  = {18},
  number  = {26},
  pages   = {1-5},
  url     = {http://jmlr.org/papers/v18/16-300.html}
}
"""

using POMDPs
using POMDPTools
using POMDPTools.Policies: PlaybackPolicy, RandomPolicy
using POMDPGifs
using POMDPModels
using NativeSARSOP
using Random
using RockSample
using Cairo
using DataFrames

function actionNum2word(action)

	if action == 1
		return "Sample"
	elseif action == 2
		return "North"
	elseif action == 3
		return "East"
	elseif action == 4
		return "South"
	elseif action == 5
		return "West"
	elseif action >= 6
		return "Sense Rock $(action - 5)"
	else
		return "Invalid Action"
	end

end

function obsNum2word(obs)

	if obs == 1
		return "Good"
	elseif obs == 2
		return "Bad"
	elseif obs == 3
		return "None"
	else
		return "Invalid Observation"
	end

end

# create POMDP

pomdp = RockSamplePOMDP(map_size = (15, 15),
	rocks_positions = [(1,2), (3, 7), (8, 4), (6, 6), (1, 5), (4, 8), (10, 2), (6, 12), (12, 14)],
	sensor_efficiency = 20.0,
	discount_factor = 0.95,
	good_rock_reward = 20.0,
)

# create solver
solver = SARSOPSolver(precision = 1e-3, max_time = 60.0, verbose = false)
policy = solve(solver, pomdp)

# simulate
project_root = dirname(Base.active_project())
output_dir = joinpath(project_root, "outputs")
save_path = joinpath(output_dir, "NoRH_SinglePOMDP.gif")

println("Creating simulation GIF...")
sim = GifSimulator(
	filename=save_path,
	max_steps=100, 
	rng=MersenneTwister(1),
	show_progress=true 
)
saved_gif = simulate(sim, pomdp, policy)
println("GIF saved to: $(saved_gif.filename)")
