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
using DiscreteValueIteration
using Plots
using LinearAlgebra
using Statistics
# using QMDP
using DataFrames
using CSV
using Dates


function global2local(pos, sub_map)

	local_x = pos[1] - sub_map[1] + 1
	local_y = pos[2] - sub_map[2] + 1

	return [local_x, local_y]

end

function local2global(pos, sub_map)

	global_x = pos[1] + sub_map[1] - 1
	global_y = pos[2] + sub_map[2] - 1

	return [global_x, global_y]

end

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

function add_nearest_rock(sub_rocks, rock_pos)

	# println("new rock being added")

	not_in_sub = [x for x in rock_pos if !(Tuple(x) in sub_rocks)]
	minDist = 10000000
	chosen_rock = nothing
	for rock in not_in_sub
		thisDist = abs(pos[1] - rock[1]) + abs(pos[2] - rock[2])
		if thisDist < minDist
			minDist = thisDist
			chosen_rock = rock
			break
		end
	end

	push!(sub_rocks, Tuple(chosen_rock))

	return sub_rocks


end

function make_sub_POMDP(pos, map_size, rock_pos, rock_probs, pomdp, horizon)
    # horizon = 3

    maxx = min(pos[1] + horizon, map_size[1])
    minx = max(pos[1] - horizon, 1)
    maxy = min(pos[2] + horizon, map_size[2])
    miny = max(pos[2] - horizon, 1)

    sub_map = (max(minx, 1), max(miny, 1), min(maxx, map_size[1]), min(maxy, map_size[2]))

	# rocks_reloaded = SVector(SVector(1, 2), SVector(3, 7), SVector(6, 6), SVector(1, 5))


	# rock_dists = [abs(pos[1] - rock[1]) + abs(pos[2] - rock[2]) for rock in rock_pos]


    # Start with rocks already inside the horizon (GLOBAL coords)
    sub_rocks = [(x, y) for (x, y) in rock_pos if sub_map[1] ≤ x ≤ sub_map[3] && sub_map[2] ≤ y ≤ sub_map[4]]

	# sub_rock_indices = [findfirst(x -> Tuple(x) == rock, rock_pos) for rock in sub_rocks]

	# println("subrocks: ", sub_rocks)




    # sub_rocks = map(x -> Tuple(x), sub_rocks) |> collect  # Vector{Tuple{Int,Int}}

    # # Build a helper to read the current global posterior for any rock coordinate
    global_vals_as_tuples = Tuple.(rock_probs.vals)
    get_prob = r -> begin
        gi = findfirst(==(r), global_vals_as_tuples)
        gi === nothing ? 0.0 : rock_probs.probs[gi]
    end


	#make sure each rock has a big enough prob
	rock_thresh = 0.25

	numrocks = length(rock_pos)

	rockpos = rock_probs.vals
	rockprob = rock_probs.probs

	highrocks = Tuple{Int64, Int64}[]
	lowrocks = Tuple{Int64, Int64}[]

	for i in 1:numrocks

		if rockprob[i] > rock_thresh
			push!(highrocks, Tuple(rockpos[i]))

		else
			push!(lowrocks, Tuple(rockpos[i]))

		end

	end

	# println(highrocks)
	# println(lowrocks)

	# println(typeof(highrocks))

	# valuesiosca = intersect(highrocks, sub_rocks)

	# println(typeof(sub_rocks), typeof(highrocks))

	# # print(typeof(sub_rocks))

	# println("rocks not in highrocks: ", valuesiosca)



	if isempty(intersect(highrocks, sub_rocks)) && !isempty(highrocks)

		# println("rock not in high rocks")

		# sub_rocks does not contain any highrocks (above the threshold)

		sub_rocks = add_nearest_rock(sub_rocks, highrocks)

		minsubx = sub_map[1]
		minsuby = sub_map[2]
		maxsubx = sub_map[3]
		maxsuby = sub_map[4]

		for rock in sub_rocks

			maxsubx = max(rock[1], maxsubx)
			maxsuby = max(rock[2], maxsuby)
			minsubx = min(rock[1], minsubx)
			minsuby = min(rock[2], minsuby)


		end

		# sub_map[3] = maxsubx
		# sub_map[4] = maxsuby

		sub_map = [minsubx, minsuby, maxsubx, maxsuby]


	end

	# println("submap: ", sub_map)

	#you always need at least one rock for the RockSample problem
	if isempty(sub_rocks)

		sub_rocks = add_nearest_rock(sub_rocks, rock_pos)

	end



    # Sub-map size
    sub_map_size = (sub_map[3] - sub_map[1] + 1, sub_map[4] - sub_map[2] + 1)

	# println("submapsize: ", sub_map_size)

    # Convert GLOBAL sub_rocks to LOCAL coordinates for the sub-POMDP
    local_sub_rocks = Tuple.(global2local.(sub_rocks, Ref(sub_map)))

	# print(local_sub_rocks)

    # Local init pos
    locpos = global2local(pos, sub_map)

	# print(sub_rocks)

    # Build the sub-POMDP
    sub_pomdp = RockSamplePOMDP(
        map_size = sub_map_size,
        rocks_positions = collect(local_sub_rocks),   # Vector{Tuple{Int,Int}}
        init_pos = locpos,
        sensor_efficiency = pomdp.sensor_efficiency,
        discount_factor = pomdp.discount_factor,
        good_rock_reward = pomdp.good_rock_reward,
        bad_rock_penalty = pomdp.bad_rock_penalty,
        step_penalty = pomdp.step_penalty,
        exit_reward = pomdp.exit_reward,
        sensor_use_penalty = pomdp.sensor_use_penalty
    )

	# println("sub_pomdp rocks: ", sub_pomdp.rocks_positions)

    # # Map each LOCAL rock index in the sub-POMDP back to its GLOBAL index (for posterior updates later)
    numRock = length(sub_pomdp.rocks_positions)
    global_idx_map = Vector{Union{Int, Nothing}}(undef, numRock)
    for i in 1:numRock
        g = sub_rocks[i]  # global tuple aligned with local_sub_rocks[i]
        global_idx_map[i] = findfirst(==(g), global_vals_as_tuples)
    end

    # Build initial belief over the sub-POMDP states using global posteriors
    rockings = zeros(numRock)
    for i in 1:numRock
        gi = global_idx_map[i]
        rockings[i] = (gi === nothing) ? 0.0 : rock_probs.probs[gi]
    end

	# clamp!(rockings, 0.0, 1.0)

    notRockings = ones(numRock) .- rockings

	

    states = ordered_states(sub_pomdp)
    indc = findall(s -> s.pos == locpos, states)
    init_states = states[indc]
    init_probs = zeros(length(init_states))

    j = 1
    for s in init_states
        mask = s.rocks
        init_probs[j] = prod(mask[i] == 0 ? notRockings[i] : rockings[i] for i in 1:numRock)
        j += 1
    end

    total_prob = sum(init_probs)
    if total_prob <= 0
        init_probs .= 1.0 / max(length(init_probs), 1)
    else
        init_probs ./= total_prob
    end

    init_state = SparseCat(init_states, init_probs)
    return sub_pomdp, init_state, rock_probs, sub_map, global_idx_map
end


function get_next_init_state(policy, thisPomdp, rock_probs, sub_map, actionList, actionListNums, init_state, global_idx_map)

	# create an updater for the pollicy
	initpos = thisPomdp.init_pos
	# println("Initial Position in Sub-POMDP: $initpos")
	up = updater(policy)
	# get the initial belief state
	# b0 = initialize_belief(up, initialstate(thisPomdp)) # initialize belief state
	b0 = init_state
	# show(stdout, "text/plain", b0)

	# init_state = initialstate(thisPomdp)
	# println("Initial state: $init_state")

	# init_action = POMDPs.action(policy, b0)

	# println("Initial action: $init_action")


	state = nothing
	action = nothing
	obs = nothing
	rew = nothing

	# simulate the first step after the inital state
	for (s, a, o, r) in stepthrough(thisPomdp, policy, "s,a,o,r", max_steps = 1)
		# println("in state $s")
		# println("took action $(actionNum2word(a))")
		# println("received observation $(obsNum2word(o)) and reward $r")
		# println("----------------------------------------------------")

		state = s
		action = a
		obs = o
		rew = r
	end

	trans = transition(thisPomdp, state, action)

	if action < 6

		push!(actionListNums, action)

		actionWord = actionNum2word(action)

		push!(actionList, actionWord)
	else

		# actionWord = actionNum2word(action)


		rocksensed = thisPomdp.rocks_positions[action - 5]

		observation = obsNum2word(obs)

		actionWord = "Sense Rock at $(local2global(rocksensed, sub_map)) with Observation $observation"

		push!(actionList, actionWord)

		globalRockPos = local2global(rocksensed, sub_map)

		ind = findfirst(==(globalRockPos), rock_probs.vals)

		globalActionNum = ind + 5  # sensing action number in global POMDP

		push!(actionListNums, globalActionNum)


	end



	if POMDPs.isterminal(thisPomdp, trans.val)

		return "TERMINAL"

	end

	# locpos = trans.val.pos

	# display("Transition result: $trans")

	# println("local trans pos: $locpos")

	#initialise the belief after the first action has been taken
	b1 = update(up, b0, action, obs)

	# val = value(policy, b0)
	# val2 = value(policy, b1)
	# println("Value of initial belief: $val")
	# println("Value of next belief: $val2")

	# trim the belief state to only states that are in the same position as the next belief
	S           = eltype(b1.state_list)
	next_states = S[]
	next_probs  = Float64[]

	# display(b1)
	# display(trans.val.pos)

	# Collect states/probs that share the current position
	for (s, p) in zip(b1.state_list, b1.b)
		if s.pos == trans.val.pos
			push!(next_states, s)
			push!(next_probs, p)
		end
	end

	# display(next_states)
	# display(next_probs)

	thisSum = 0

	for r in 1:length(thisPomdp.rocks_positions)  # r is LOCAL rock index
		thisSum = 0.0
		for (i, s) in enumerate(next_states)
			if s.rocks[r] == true
				thisSum += next_probs[i]
			end
		end
		gi = global_idx_map[r]  # global index in rock_probs, or nothing
		if gi === nothing
			@warn "Rock mapping failed; local rock not in global list (check tuple vs SVector types)" local_r=thisPomdp.rocks_positions[r] global_vals=rock_probs.vals
		else
			rock_probs.probs[gi] = thisSum
		end
	end
	next_init_state = SparseCat(next_states, next_probs)
	# display(next_init_state)

	position = next_init_state.vals[1].pos

	# println("Local Position: $position ")

	globpos = local2global(position, sub_map)
	# println("Global Position: $globpos")

	#return the global position
	return globpos

end

# for multiple pomdps
# actionListList = []

# initial large POMDP of the whole space
# pomdp = RockSamplePOMDP(map_size = (15, 15),
# 	rocks_positions = [(1,2), (3, 7), (8, 4), (6, 6), (1, 5), (4, 8), (10, 2), (6, 12), (12, 14)],
# 	sensor_efficiency = 20.0,
# 	discount_factor = 0.95,
# 	good_rock_reward = 20.0,
# )

# pomdp = RockSamplePOMDP(map_size = (15, 15),
# 	rocks_positions = [(6, 14), (12, 12), (2, 10), (4,3), (5,10), (8,9)],
# 	# rocks_positions = [(6,6)], 
# 	init_pos = (1,1),
# 	sensor_efficiency = 20.0,
# 	discount_factor = 0.95,
# 	good_rock_reward = 20.0,
# 	bad_rock_penalty = -5.0
# )
# #initialize rock_probs for belief state
# rock_probs = SparseCat(pomdp.rocks_positions, [0.5 for _ in 1:length(pomdp.rocks_positions)])

# actionList = []
# actionListNums = []
# posList = []

# rawInitState = POMDPs.initialstate(pomdp)

# new_pomdp = RockSamplePOMDP(map_size = (7, 7),
# 	rocks_positions = [(3, 7), (4,3)],
# 	init_pos = (1,1),
# 	sensor_efficiency = 20.0,
# 	discount_factor = 0.95,
# 	good_rock_reward = 20.0,
# 	bad_rock_penalty = -5.0
# )

# # solve full POMDP and create GIF
# solver = SARSOPSolver(precision = 1e-3; max_time = 10.0, verbose = false) #use SARSOP solver
# policy = solve(solver, pomdp) # get policy using SARSOP solver

# println("Creating simulation GIF...")
# sim = GifSimulator(
# 	filename="DroneRockSample.gif",
# 	max_steps=100,  # Reduced steps for testing
# 	rng=MersenneTwister(1),
# 	show_progress=true  # Enable progress display
# )
# saved_gif = simulate(sim, pomdp, policy)
# println("GIF saved to: $(saved_gif.filename)")

# exit()

# many pomdps

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
filename = "experiment_results_$(timestamp).csv"

# Create the file and write the header *once* before the loop
# We use a temporary DataFrame just to set the column names
init_df = DataFrame(
    MapSize = String[], 
    NumRocks = Int[], 
    Duration = Float64[], 
    ResultStatus = String[], # "Success" or "Failed"
	RockPositions = String[],
    ActionList = String[],    # We will save the list as a string
	Reward = Float64[]
)
CSV.write(filename, init_df)

println("Data will be saved to: $filename")

mapsizes = [(15,15)]

numrocks = [15]

MAX_STEPS = 1000


for size in mapsizes

	for numrock in numrocks

		current_seed = rand(1:10^6)
        Random.seed!(current_seed)        

        # Variables to store results for this specific run
        local time_taken = 0.0
        local actions_taken = []
        local status = "Success"

		possible_coords = [(x, y) for x in 1:size[1], y in 1:size[2]]

		random_rock_pos = first(shuffle(possible_coords), numrock)

		println("Running: Size=$size, Rocks=$numrock at positions $random_rock_pos")

		pomdp = RockSamplePOMDP(map_size = size,
			rocks_positions = random_rock_pos,
			# rocks_positions = [(6, 14), (12, 12), (2, 10), (4,3), (5,10), (8,9)],
			# rocks_positions = [(6,6)], 
			init_pos = (1,1),
			sensor_efficiency = 20.0,
			discount_factor = 0.95,
			good_rock_reward = 20.0,
			bad_rock_penalty = -5.0
		)
		#initialize rock_probs for belief state
		rock_probs = SparseCat(pomdp.rocks_positions, [0.5 for _ in 1:length(pomdp.rocks_positions)])

		actionList = []
		actionListNums = []
		posList = []

		rawInitState = POMDPs.initialstate(pomdp)

		try 

			time_taken = @elapsed begin

				for i in 1:MAX_STEPS

					global pos
					if i == 1
						pos = pomdp.init_pos
					end
				
					# display(rock_probs)
				
				
					# println("Iteration: $i")
					# println("Current Position: $pos")

					if numrocks == 3 || numrocks == 5
						horizon = 3
					elseif numrocks == 10 || numrocks == 15
						horizon = 3
					else
						horizon = 3
					end
				
					sub_pomdp, init_state, rock_probs, sub_map, global_idx_map = make_sub_POMDP(pos, pomdp.map_size, pomdp.rocks_positions, rock_probs, pomdp, horizon)
				
					# display(init_state)
				
					# redefine initial state from the original function from RockSample
					POMDPs.initialstate(p::RockSamplePOMDP{K}) where K = init_state
				
					solver = SARSOPSolver(precision = 1e-3; max_time = 1.0, verbose = false) #use SARSOP solver
					# solver = QMDPSolver(max_iterations=20, belres=1e-3, verbose=false) #use QMDP solver
					policy = solve(solver, sub_pomdp) # get policy using SARSOP solver
				
					#get the next initial belief state
					pos = get_next_init_state(policy, sub_pomdp, rock_probs, sub_map, actionList, actionListNums, init_state, global_idx_map)
					push!(posList, pos)
				
					# if pos == "TERMINAL" || i == 20
					# 	println("Reached terminal state. Exiting loop.")
					# 	println("Actions taken: ")
					# 	count = 1;
					# 	for a in actionList
					# 		println("	", a, " -> ", posList[count])
					# 		count += 1
					# 	end
					# 	break
					# end
				
					if pos == "TERMINAL"
						# println("Reached terminal state. Exiting loop.")
						# println("Actions taken: ")
				
						# POMDPs.initialstate(p::RockSamplePOMDP{K}) where K = rawInitState
				
						# # how many steps have positions recorded
						# nsteps = min(length(actionList), length(posList))
				
						# for step_idx in 1:nsteps
						# 	a = actionList[step_idx]
						# 	p = posList[step_idx]  # expected like [x, y]
						# 	# println("    ", a, " -> ", p)
				
						# 	# # Append to actions (long) for multiple pomdps
						# 	# push!(actions_df, (run_label, j, step_idx, a))
				
						# 	# # Append to positions (long) as "[x,y]" string
						# 	# if p isa Vector{Int} && length(p) == 2
						# 	# 	push!(positions_df, (run_label, j, step_idx, "[$(p[1]),$(p[2])]"))
						# 	# end
						# end
						# # set this to true if your list is 0-based
				
				
						break
					elseif i == MAX_STEPS
						# println("Reached maximum steps ($MAX_STEPS). Exiting loop.")
						status = "Timeout: Max Steps Reached ($MAX_STEPS)"
						break
					end
				
					# println("============================================")
				
				end

			end

			actions_taken = actionList

		catch e
			# If the simulation crashes (e.g., MemoryError, Solver Error), we catch it here
			println("!!! CRASH detected for $size / $numrock !!!")
			println("Error: $e")
			status = "Error: $e"
			time_taken = NaN

		end

		# 3. Save to Disk Immediately
        # We convert actionList to a string so it fits in one CSV cell
		row_data = DataFrame(
            MapSize = ["$size"], 
            NumRocks = [numrock],
            Duration = [time_taken],
            ResultStatus = [status],
			RockPositions = [string(random_rock_pos)], # <--- Saving the rocks here
            ActionList = [string(actions_taken)] # This will now contain data
        )

        CSV.write(filename, row_data, append=true)
        println("Saved result for Size=$size, Rocks=$numrock")

	end

end


# for i in 1:5000

# 	global pos
# 	if i == 1
# 		pos = pomdp.init_pos
# 	end

# 	display(rock_probs)


# 	println("Iteration: $i")
# 	println("Current Position: $pos")

# 	sub_pomdp, init_state, rock_probs, sub_map, global_idx_map = make_sub_POMDP(pos, pomdp.map_size, pomdp.rocks_positions, rock_probs, pomdp)

# 	display(init_state)

# 	# redefine initial state from the original function from RockSample
# 	POMDPs.initialstate(p::RockSamplePOMDP{K}) where K = init_state

# 	solver = SARSOPSolver(precision = 1e-3; max_time = 10.0, verbose = false) #use SARSOP solver
# 	# solver = QMDPSolver(max_iterations=20, belres=1e-3, verbose=false) #use QMDP solver
# 	policy = solve(solver, sub_pomdp) # get policy using SARSOP solver

# 	#get the next initial belief state
# 	pos = get_next_init_state(policy, sub_pomdp, rock_probs, sub_map, actionList, actionListNums, init_state, global_idx_map)
# 	push!(posList, pos)

# 	# if pos == "TERMINAL" || i == 20
# 	# 	println("Reached terminal state. Exiting loop.")
# 	# 	println("Actions taken: ")
# 	# 	count = 1;
# 	# 	for a in actionList
# 	# 		println("	", a, " -> ", posList[count])
# 	# 		count += 1
# 	# 	end
# 	# 	break
# 	# end

# 	if pos == "TERMINAL" || i == 5000
# 		println("Reached terminal state. Exiting loop.")
# 		println("Actions taken: ")

# 		POMDPs.initialstate(p::RockSamplePOMDP{K}) where K = rawInitState

# 		# how many steps have positions recorded
# 		nsteps = min(length(actionList), length(posList))

# 		for step_idx in 1:nsteps
# 			a = actionList[step_idx]
# 			p = posList[step_idx]  # expected like [x, y]
# 			println("    ", a, " -> ", p)

# 			# # Append to actions (long) for multiple pomdps
# 			# push!(actions_df, (run_label, j, step_idx, a))

# 			# # Append to positions (long) as "[x,y]" string
# 			# if p isa Vector{Int} && length(p) == 2
# 			# 	push!(positions_df, (run_label, j, step_idx, "[$(p[1]),$(p[2])]"))
# 			# end
# 		end
# 		# set this to true if your list is 0-based


# 		break
# 	end

# 	println("============================================")

# end

# for many pomdps 

# push!(actionListList, actionList)


# end

# Actions: rows = Step, columns = RunLabel, values = Action
# wide_actions = unstack(actions_df, :Step, :RunLabel, :Action)

# # Positions: rows = Step, columns = RunLabel, values = "[x,y]" string
# wide_positions = unstack(positions_df, :Step, :RunLabel, :Position)

# CSV.write("wide_actions_1.csv", wide_actions)
# CSV.write("wide_positions_1.csv", wide_positions)


