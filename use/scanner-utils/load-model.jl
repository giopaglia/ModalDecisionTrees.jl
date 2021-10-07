import JLD2

function load_model(model_hash::String, model_savedir::String)
	# println()
	# println()
	# println("Loading model: $(model_hash)...")

	if !isfile("$(model_savedir)/$(model_hash).jld")
		if isfile("$(model_savedir)/rf_$(model_hash).jld")
			model_hash = "rf_" * model_hash
		elseif isfile("$(model_savedir)/tree_$(model_hash).jld")
			model_hash = "tree_" * model_hash
		else
			throw_n_log("File $(model_savedir)/$(model_hash).jld not found!")
		end
	end

	model = load("$(model_savedir)/$(model_hash).jld")

	model = if startswith(model_hash, "tree") || haskey(model, "T")
			model["T"]
		elseif startswith(model_hash, "rf") || haskey(model, "F")
			model["F"]
		else
			throw_n_log("Unknown model hash type: $(model_hash)")
		end
	model
end
