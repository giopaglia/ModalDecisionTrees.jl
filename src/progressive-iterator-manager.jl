
import JSON

function percent(num::Real; digits=2)
	return round.(num.*100, digits=digits)
end

function data_to_string(
		M::Union{DecisionTree.DTree{S, T},DecisionTree.DTNode{S, T}},
		cm::ConfusionMatrix;
		start_s = "(",
		end_s = ")",
		separator = ";",
		alt_separator = ","
	) where {S, T}

	result = start_s
	result *= string(percent(cm.kappa), separator)
	result *= string(percent(cm.sensitivities[1]), separator)
	result *= string(percent(cm.specificities[1]), separator)
	result *= string(percent(cm.PPVs[1]), separator)
	result *= string(percent(cm.overall_accuracy))

	if isa(M, DecisionTree.Forest{S, T})
		result *= separator
		result *= string(percent(M.oob_error))
	end

	result *= end_s

	result
end

function data_to_string(
		Ms::AbstractVector{DecisionTree.Forest{S, T}},
		cms::AbstractVector{ConfusionMatrix};
		start_s = "(",
		end_s = ")",
		separator = ";",
		alt_separator = ","
	) where {S, T}

	result = start_s
	result *= string(percent(mean(map(cm->cm.kappa, cms))), alt_separator)
	result *= string(percent(mean(map(cm->cm.sensitivities[1], cms))), alt_separator)
	result *= string(percent(mean(map(cm->cm.specificities[1], cms))), alt_separator)
	result *= string(percent(mean(map(cm->cm.PPVs[1], cms))), alt_separator)
	result *= string(percent(mean(map(cm->cm.overall_accuracy, cms))), alt_separator)
	result *= string(percent(mean(map(M->M.oob_error, Ms))))
	result *= end_s
	result *= separator
	result *= start_s
	result *= string(var(map(cm->cm.kappa, cms)), alt_separator)
	result *= string(var(map(cm->cm.sensitivities[1], cms)), alt_separator)
	result *= string(var(map(cm->cm.specificities[1], cms)), alt_separator)
	result *= string(var(map(cm->cm.PPVs[1], cms)), alt_separator)
	result *= string(var(map(cm->cm.overall_accuracy, cms)), alt_separator)
	result *= string(var(map(M->M.oob_error, Ms)))
	result *= end_s

	result
end
function init_new_execution_progress_dictionary(file_path::String, exec_ranges::Vector, params_names::Vector{String})
	execution_dictionaries = []

	for combination_tuple in IterTools.product(exec_ranges...)
		d = zip([params_names..., "runs"], [combination_tuple..., []]) |> Dict
		push!(execution_dictionaries, d)
	end

	export_execution_progress_dictionary(file_path, execution_dictionaries)

	execution_dictionaries = import_execution_progress_dictionary(file_path)

	return execution_dictionaries
end

function init_new_execution_progress_dictionary(file_path::String, exec_n_tasks, exec_n_versions, exec_nbands, exec_dataset_kwargs)
	execution_dictionaries = []

	for n_task in exec_n_tasks
		for n_version in exec_n_versions
			for nbands in exec_nbands
				for dataset_kwargs in exec_dataset_kwargs
					push!(execution_dictionaries, Dict(
						"n_task" => n_task,
						"n_version" => n_version,
						"nbands" => nbands,
						"dataset_kwargs" => dataset_kwargs,
						"runs" => []
					))
				end
			end
		end
	end

	export_execution_progress_dictionary(file_path, execution_dictionaries)

	execution_dictionaries = import_execution_progress_dictionary(file_path)

	return execution_dictionaries
end

function export_execution_progress_dictionary(file_path::String, dicts::AbstractVector)
	mkpath(dirname(file_path))
	file = open(file_path, "w+")
	write(file, JSON.json(dicts))
	close(file)
end

function import_execution_progress_dictionary(file_path::String)
	file = open(file_path)
	dicts = JSON.parse(file)
	close(file)
	dicts
end

function append_in_file(file_name::String, text::String)
	mkpath(dirname(file_name))
	file = open(file_name, "a+")
	write(file, text)
	close(file)
end

function is_same_kwargs(dk1::Dict, dk2::NamedTuple{T, N}) where {T, N}
	return dk1 == Dict{String, Any}([String(k) => v for (k,v) in zip(keys(dk2),values(dk2))])
end

function is_same_kwargs(dk1::NamedTuple{T, N}, dk2::Dict) where {T, N}
	return is_same_kwargs(dk2, dk1)
end

function load_or_create_execution_progress_dictionary(file_path::String, args...)
	if isfile(file_path)
		import_execution_progress_dictionary(file_path)
	else
		init_new_execution_progress_dictionary(file_path, args...)
	end
end
