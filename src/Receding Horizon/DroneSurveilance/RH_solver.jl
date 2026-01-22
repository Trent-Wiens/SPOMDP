using POMDPs
using POMDPTools
using POMDPModels
using DroneSurveillance
using NativeSARSOP
using Random
using LinearAlgebra
using StaticArrays
using Parameters # required for @with_kw if you are defining the struct locally, otherwise typically in package

# If you need to redefine DSState locally for the script to run without the package source:
# struct DSState
#     quad::Vector{Int64}
#     agent::Vector{Int64}
# end

function actionNum2word(a::Int64)
    if a == 1
        return :north
    elseif a == 2
        return :east
    elseif a == 3
        return :south
    elseif a == 4
        return :west
    elseif a == 5
        return :hover
    else
        error("invalid action number")
    end
end

# --- Helper Functions ---

function global2local(pos, sub_map_bounds)
    # sub_map_bounds is (minx, miny, maxx, maxy)
    # pos is expected to be [x, y]
    local_x = pos[1] - sub_map_bounds[1] + 1
    local_y = pos[2] - sub_map_bounds[2] + 1
    return [local_x, local_y]
end

function local2global(pos, sub_map_bounds)
    global_x = pos[1] + sub_map_bounds[1] - 1
    global_y = pos[2] + sub_map_bounds[2] - 1
    return [global_x, global_y]
end

# --- Receding Horizon Logic ---

function make_sub_POMDP(quad_pos, agent_pos, global_pomdp)
    
    # ... [Keep previous horizon/bounds logic same as before] ...
    
    horizon = 3
    g_size = global_pomdp.size
    
    minx = max(quad_pos[1] - horizon, 1)
    maxx = min(quad_pos[1] + horizon, g_size[1])
    miny = max(quad_pos[2] - horizon, 1)
    maxy = min(quad_pos[2] + horizon, g_size[2])

    sub_bounds = (minx, miny, maxx, maxy)
    sub_size = (maxx - minx + 1, maxy - miny + 1)

    function get_clamped_local(global_p, bounds, local_size)
        lx = global_p[1] - bounds[1] + 1
        ly = global_p[2] - bounds[2] + 1
        cx = clamp(lx, 1, local_size[1])
        cy = clamp(ly, 1, local_size[2])
        return [cx, cy]
    end

    local_region_A = get_clamped_local(global_pomdp.region_A, sub_bounds, sub_size)
    local_region_B = get_clamped_local(global_pomdp.region_B, sub_bounds, sub_size)

    sub_pomdp = DroneSurveillancePOMDP(
        size = sub_size,
        region_A = local_region_A,
        region_B = local_region_B,
        fov = global_pomdp.fov,
        agent_policy = global_pomdp.agent_policy,
        camera = global_pomdp.camera,
        discount_factor = global_pomdp.discount_factor
    )

    local_quad = global2local(quad_pos, sub_bounds)
    local_agent = get_clamped_local(agent_pos, sub_bounds, sub_size)

    # === FIX IS HERE ===
    raw_state = DSState(local_quad, local_agent)
    
    # Wrap the state in a Deterministic distribution
    init_belief = Deterministic(raw_state)

    return sub_pomdp, init_belief, sub_bounds
end

function get_next_action(policy, sub_pomdp, init_belief)
    # policy = AlphaVectorPolicy
    # init_belief = Deterministic{DSState}

    nextAction = action(policy, init_belief)

    thisVal = value(policy, init_belief)

    println("Value: $thisVal")

    return nextAction
end

# --- Main Simulation Loop ---

# 1. Setup Global Problem
global_pomdp = DroneSurveillancePOMDP(
    size = (10,10),
    region_A = [1, 1],
    region_B = [100, 100],
    fov = (3, 3),
    agent_policy = :restricted # Agent moves somewhat predictably
)

# 2. Initialize Positions
quad_pos = [1, 1]
agent_pos = [5, 5] # Start agent somewhere
path_history = [copy(quad_pos)]
rng = MersenneTwister(1)

println("Starting Simulation...")

for t in 1:50
    println("\n--- Step $t ---")
    println("Global Quad: $quad_pos | Global Agent: $agent_pos")

    # A. Construct Local Problem
    sub_pomdp, local_init_state, sub_bounds = make_sub_POMDP(quad_pos, agent_pos, global_pomdp)
    
    # B. Solve Local Problem
    # precision set low for speed in loop
    solver = SARSOPSolver(precision = 1e-1, max_time = 2.0, verbose = false) 
    policy = solve(solver, sub_pomdp)

    # C. Get Action from Local Policy
    local_action = get_next_action(policy, sub_pomdp, local_init_state)
    
    # Actions in DroneSurveillance are usually integers:
    # 1: Left, 2: Right, 3: Up, 4: Down, 5: Hover (Check your specific mapping)
    println("Action Chosen: $(actionNum2word(local_action))")

    # D. Execute on Global Model
    # We need to manually simulate the transition because 'transition' function
    # requires a full DSState and returns a distribution.
    
    global_state = DSState(quad_pos, agent_pos)
    
    # Get next state distribution from the GLOBAL dynamics
    # Note: We pass the integer action directly
    dist = transition(global_pomdp, global_state, local_action)
    
    # Sample next state
    next_global_state = rand(rng, dist)
    
    # Update positions
    global quad_pos = next_global_state.quad
    global agent_pos = next_global_state.agent
    
    push!(path_history, copy(quad_pos))
    
    # Check if agent reached destination (logic internal to DroneSurveillance env usually handles this, 
    # but good to visualize)
    if agent_pos == global_pomdp.region_A
        println("Agent reached Region A!")
    elseif agent_pos == global_pomdp.region_B
        println("Agent reached Region B!")
    end

    if t == 50
        println("Max steps reached.")
    end
end

println("Simulation Complete.")

