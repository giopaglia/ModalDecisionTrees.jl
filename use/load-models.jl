
using JLD2

function load_model(model_hash::String, model_savedir::String)
    # println()
    # println()
    # println("Loading model: $(model_hash)...")

    model = load("$(model_savedir)/$(model_hash).jld")

    model = if startswith(model_hash, "tree") || haskey(model, "T")
            model["T"]
        elseif startswith(model_hash, "rf") || haskey(model, "F")
            model["F"]
        else
            error("Unknown model hash type: $(model_hash)")
        end
    model
end
