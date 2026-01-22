using POMDPs
using POMDPModels
using POMDPTools
using NativeSARSOP
using RockSample
using DataFrames
using CSV
using Dates
using Random
using AdaOPS

# --- HELPER: Convert Observation to String ---
function obs_to_string(o)
    if o == 1
        return "Good"
    elseif o == 2
        return "Bad"
    elseif o == 3
        return "None"
    else
        return "Unknown($o)"
    end
end

# --- HELPER: Action to String (Now includes Observation) ---
function action_to_string(a, o, rocks_positions)
    obs_text = obs_to_string(o)
    
    if a == 1
        return "Sample" # Usually Sample returns Good/Bad reward, observation is often None
    elseif a == 2; return "North"
    elseif a == 3; return "East"
    elseif a == 4; return "South"
    elseif a == 5; return "West"
    elseif a > 5
        # SENSING ACTION
        rock_idx = a - 5
        if rock_idx <= length(rocks_positions)
            # Format: "Sense((5,3)) - Obs: Good"
            return "Sense($(rocks_positions[rock_idx])) - Obs: $obs_text"
        else
            return "Sense(Invalid)"
        end
    else
        return "Unknown"
    end
end

# --- FILE SETUP ---
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
filename = "experiment_results_STANDARD_SARSOP_$(timestamp).csv"

init_df = DataFrame(
    MapSize = String[], 
    NumRocks = Int[], 
    Duration = Float64[], 
    ResultStatus = String[], 
    RockPositions = String[],
    ActionList = String[],
    Seed = Int[]
)
CSV.write(filename, init_df)

println("Data will be saved to: $filename")

# --- CONFIGURATION ---
mapsizes = [(15,15)]
numrocks = [10,10,10,10] 
MAX_STEPS = 1000

# Time allowed for the C++ Solver to think (once the problem is generated)
MAX_SOLVER_TIME = 60.0 

# Time allowed for the WHOLE process
HARD_LIMIT_SECONDS = 500.0

for size in mapsizes
    for numrock in numrocks
        
        # 1. Seeding (Hardcoded as per your request)
        current_seed = rand(1:10^6)
        Random.seed!(current_seed)


        # 2. Setup Variables
        local time_taken = 0.0
        local actions_taken = []
        local status = "Unknown"
        
        # 3. Generate Map (Hardcoded positions as per your request)
        # Note: If you want random again later, switch this back to shuffle(possible_coords)
        possible_coords = [(x, y) for x in 1:size[1], y in 1:size[2]]
        random_rock_pos = first(shuffle(possible_coords), numrock)

        # random_rock_pos = [(4,4), (5,3), (5,5)]

        println("Running Standard SARSOP: Size=$size, Rocks=$numrock (Seed: $current_seed)")

        pomdp = RockSamplePOMDP(
            map_size = size,
            rocks_positions = random_rock_pos,
            init_pos = (1,1),
            sensor_efficiency = 20.0,
            discount_factor = 0.95,
            good_rock_reward = 20.0,
            bad_rock_penalty = -5.0
        )
        

        # 4. The "Hard Limit" Wrapper
        try 
            start_time = time()
            
            heavy_task = @task begin
                # A. SOLVE
                solver = SARSOPSolver(precision=1e-3, max_time=MAX_SOLVER_TIME, verbose=false)
                policy = solve(solver, pomdp)

                # B. SIMULATE
                history = []
                updater = DiscreteUpdater(pomdp)

                for (a, o) in stepthrough(pomdp, policy, updater, "a,o", max_steps=MAX_STEPS)
                    # We pass 'o' to the string converter
                    push!(history, action_to_string(a, o, pomdp.rocks_positions))
                end
                return history

            end

            schedule(heavy_task)

            # Monitor the task
            while !istaskdone(heavy_task)
                if (time() - start_time) > HARD_LIMIT_SECONDS
                    Base.throwto(heavy_task, InterruptException())
                    status = "Timeout: Hard Limit ($HARD_LIMIT_SECONDS s)"
                    break 
                end
                sleep(1.0) 
            end

            time_taken = time() - start_time

            if status == "Unknown" 
                actions_taken = fetch(heavy_task)
                
                if length(actions_taken) < MAX_STEPS
                    status = "Success: Terminal Reached"
                else
                    status = "Timeout: Max Steps Reached"
                end
            end

        catch e
            if isa(e, InterruptException) || status == "Timeout: Hard Limit ($HARD_LIMIT_SECONDS s)"
                 println("   !!! HARD TIMEOUT EXECUTION !!!")
                 status = "Timeout: Hard Limit ($HARD_LIMIT_SECONDS s)"
                 time_taken = HARD_LIMIT_SECONDS
            else
                 println("   !!! CRASH/ERROR !!!")
                 println("   Error: $e")
                 status = "Error: $e"
                 time_taken = NaN
            end
        end

        # 5. Save Results
        row_data = DataFrame(
            MapSize = ["$size"], 
            NumRocks = [numrock],
            Duration = [time_taken],
            ResultStatus = [status],
            RockPositions = [string(random_rock_pos)],
            ActionList = [string(actions_taken)],
            Seed = [current_seed]
        )

        CSV.write(filename, row_data, append=true)
        println("Saved result. Status: $status")
        println("---------------------------------------------------")
    end
end

println("Experiment Complete.")
