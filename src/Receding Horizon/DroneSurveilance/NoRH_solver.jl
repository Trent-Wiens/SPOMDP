using DroneSurveillance
using POMDPs
using POMDPTools

# import a solver from POMDPs.jl e.g. SARSOP
using NativeSARSOP
using Random

# for visualization
using POMDPGifs
import Cairo

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

function observationNum2word(o::Int64, CamType)
    if CamType == QuadCam()
        if o == 1
            return :SW
        elseif o == 2
            return :NW
        elseif o == 3
            return :NE
        elseif o == 4
            return :SE
        elseif o == 5
            return :DET
        elseif o == 6
            return :OUT
        else
            error("invalid observation number for QuadCam")
        end

    elseif CamType == PerfectCam()
        if o == 1
            return :N
        elseif o == 2
            return :E
        elseif o == 3
            return :S
        elseif o == 4
            return :W
        elseif o == 5
            return :DET
        elseif o == 6
            return :NE
        elseif o == 7
            return :SE
        elseif o == 8
            return :SW
        elseif o == 9
            return :NW
        elseif o == 10
            return :OUT
        else
            error("invalid observation number for PerfectCam")
        end
    else
        error("invalid camera type")
    end
end

pomdp = DroneSurveillancePOMDP(
    size = (7, 7),
    camera = PerfectCam(),
) # initialize the problem 

solver = SARSOPSolver(precision=1e-3, max_time=60.0, verbose=false)
policy = solve(solver, pomdp) # solve the problem

project_root = dirname(Base.active_project())
output_dir = joinpath(project_root, "outputs")
save_path = joinpath(output_dir, "NoRH_Surveil.gif")

println("Creating simulation GIF...")
sim = GifSimulator(
	filename=save_path,
	max_steps=50,  # Reduced steps for testing
	rng=MersenneTwister(1),
	show_progress=true  # Enable progress display
)
saved_gif = simulate(sim, pomdp, policy)
println("GIF saved to: $(saved_gif.filename)")

