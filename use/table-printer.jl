
import Dates

include("lib.jl")

###############################################################################
############################ OUTPUT HANDLERS ##################################
###############################################################################

id_f = x->x
half_f = x->ceil(Int, x/2)
sqrt_f = x->ceil(Int, sqrt(x))

function print_function(func::Core.Function)::String
	if func === id_f
		"all"
	elseif func === half_f
		"half"
	elseif func === sqrt_f
		"sqrt"
	elseif func === DecisionTree.util.entropy
		"entropy"
	elseif func === DecisionTree.util.gini
		"gini"
	else
		# ""
		string(func)
	end
end

function get_creation_date_string_format(file_name::String)::String
	Dates.format(Dates.unix2datetime(ctime(file_name)), "HH.MM.SS_dd.mm.yyyy")
end

function backup_file_using_creation_date(file_name::String)
	splitted_name = Base.Filesystem.splitext(file_name)
	backup_file_name = splitted_name[1] * "_" * get_creation_date_string_format(file_name) * splitted_name[2]
	mv(file_name, backup_file_name * ".bkp")
end

function string_tree_head(tree_args)::String
	string("T($(print_function(tree_args.loss_function)),$(tree_args.min_samples_leaf),$(tree_args.min_purity_increase),$(tree_args.min_loss_at_leaf))")
end

function string_forest_head(forest_args)::String
	string("RF($(forest_args.n_trees),$(print_function(forest_args.n_subfeatures)),$(print_function(forest_args.n_subrelations)))")
end

function string_head(
		tree_args::AbstractArray,
		forest_args::AbstractArray;
		separator = "\t",
		tree_columns,
		forest_columns,
    columns_before::Union{Integer,Vector{<:AbstractString}} = 1
	)::String
	
	if columns_before isa Integer
		columns_before = fill("", columns_before)
	end

	result = ""
	for str_col in columns_before
		result *= str_col
		result *= string(separator)
	end

	for i in 1:length(tree_args)
		for j in 1:length(tree_columns)
			result *= string_tree_head(tree_args[i])
			result *= string(tree_columns[j])
			result *= string(separator)
		end
	end

	for i in 1:length(forest_args)
		for j in 1:length(forest_columns)
			result *= string_forest_head(forest_args[i])
			result *= string(forest_columns[j])
			result *= string(separator)
		end
	end

	result *= string("\n")

	result
end

function print_head(file_name::String, tree_args::AbstractArray, forest_args::AbstractArray; kwargs...)
	header = string_head(tree_args, forest_args; kwargs...)

	if isfile(file_name)
		file = open(file_name, "r")
		if readlines(file, keep = true)[1] == header
			# If the header is the same nothing to do
			close(file)
			return
		end
		close(file)

		backup_file_using_creation_date(file_name)
	end

	file = open(file_name, "w+")
	write(file, header)
	close(file)
end

function percent(num::Real; digits=2)
	return round.(num.*100, digits=digits)
end

# Print a tree entry in a row
function data_to_string(
		M::DTree,
		cm::ConfusionMatrix,
		time::Dates.Millisecond,
		hash::AbstractString;
		start_s = "(",
		end_s = ")",
		separator = "\t",
		alt_separator = ",",
		best_rule_params = [(t=.8, min_confidence=0.6, min_support=0.1), (t=.65, min_confidence=0.6, min_support=0.1)],
	)

	result = start_s
	result *= string(percent(kappa(cm)),                  alt_separator)
	result *= string(percent(overall_accuracy(cm)),       alt_separator)
	result *= string(percent(safe_macro_sensitivity(cm)), alt_separator)
	result *= string(percent(safe_macro_specificity(cm)), alt_separator)
	result *= string(percent(safe_macro_PPV(cm)),         alt_separator)
	result *= string(percent(safe_macro_NPV(cm)),         alt_separator)
	result *= string(percent(safe_macro_F1(cm)),          alt_separator)
	result *= string(num_nodes(M),                        alt_separator)
	
	m = tree_walk_metrics(M; best_rule_params = best_rule_params)
	for best_rule_p in best_rule_params
		result *= string(m["best_rule_t=$(best_rule_p)"], alt_separator)
	end
	result *= string(human_readable_time_s(time), alt_separator)
	result *= string(hash)
	result *= end_s

	result
end

# Print a forest entry in a row
function data_to_string(
		Ms::AbstractVector{Forest{S}},
		cms::AbstractVector{ConfusionMatrix},
		time::Dates.Millisecond,
		hash::AbstractString;
		start_s = "(",
		end_s = ")",
		separator = "\t",
		alt_separator = ","
	) where {S}

	result = start_s
	result *= string(percent(mean(map(cm->kappa(cm),                  cms))), alt_separator)
	result *= string(percent(mean(map(cm->overall_accuracy(cm),       cms))), alt_separator)
	result *= string(percent(mean(map(cm->safe_macro_sensitivity(cm), cms))), alt_separator)
	result *= string(percent(mean(map(cm->safe_macro_specificity(cm), cms))), alt_separator)
	result *= string(percent(mean(map(cm->safe_macro_PPV(cm),         cms))), alt_separator)
	result *= string(percent(mean(map(cm->safe_macro_NPV(cm),         cms))), alt_separator)
	result *= string(percent(mean(map(cm->safe_macro_F1(cm),          cms))), alt_separator)
	result *= string(percent(mean(map(M->M.oob_error, Ms))),                    alt_separator)
	result *= string(percent(mean(num_nodes.(Ms))))
	result *= end_s
	result *= separator
	result *= start_s
	result *= string(var(map(cm->kappa(cm),                  cms)), alt_separator)
	result *= string(var(map(cm->overall_accuracy(cm),       cms)), alt_separator)
	result *= string(var(map(cm->safe_macro_sensitivity(cm), cms)), alt_separator)
	result *= string(var(map(cm->safe_macro_specificity(cm), cms)), alt_separator)
	result *= string(var(map(cm->safe_macro_PPV(cm),         cms)), alt_separator)
	result *= string(var(map(cm->safe_macro_NPV(cm),         cms)), alt_separator)
	result *= string(var(map(cm->safe_macro_F1(cm),          cms)), alt_separator)
	result *= string(var(map(M->M.oob_error, Ms)),                              alt_separator)
	result *= string(var(num_nodes.(Ms)))
	result *= end_s
	result *= separator
	result *= start_s
	result *= string(human_readable_time_s(time), alt_separator)
	result *= string(hash)
	result *= end_s

	result
end

###############################################################################
###############################################################################

function extract_model(
		file_name::String,
		type::String;
		n_trees::Union{Nothing,Number} = nothing,
		keep_header = true,
		column_separator = "\t",
		exclude_variance = true,
		exclude_parameters = [ "K", "oob_error", "t" ],
		secondary_file_name::Union{Nothing,String} = nothing,
		remove_duplicated_rows = true
	)

	if ! isfile(file_name)
		error("No file with name $(file_name) found.")
	end

	file = open(file_name, "r")
	secondary_table =
		if isnothing(secondary_file_name)
			nothing
		else
			extract_model(
				secondary_file_name, type,
				n_trees = n_trees,
				keep_header = false,
				column_separator = column_separator,
				exclude_variance = exclude_variance,
				exclude_parameters = exclude_parameters,
				secondary_file_name = nothing
			)
		end

	function split_line(line)
		return split(chomp(line), column_separator, keepempty = true)
	end

	function get_proper_columns_indexes(header, type, n_trees)
		function get_tree_number_from_header(header, index)
			return parse(Int, split(replace(header[index], "RF(" => ""), ",")[1])
		end

		function is_variance_column(header, index)::Bool
			return contains(header[index], "σ")
		end

		function is_excluded_column(header, index)::Bool
			for excluded in exclude_parameters
				if strip(split(header[index], ")")[2]) == excluded
					return true
				end
			end
			return false
		end

		local selected_columns = []
		if type == "T"
			for i in 1:length(header)
				if startswith(header[i], "T")
					if is_excluded_column(header, i)
						continue
					end
					push!(selected_columns, i)
				end
			end
		elseif type == "RF"
			for i in 1:length(header)
				if startswith(header[i], "RF") && (isnothing(n_trees) || get_tree_number_from_header(header, i) == n_trees)
					if exclude_variance && is_variance_column(header, i)
						continue
					end
					if is_excluded_column(header, i)
						continue
					end
					push!(selected_columns, i)
				end
			end	
		else
			error("No model known of type $(type); could be \"T\" or \"RF\".")
		end

		selected_columns
	end
	
	header = []
	append!(header, split_line(readline(file)))

	selected_columns = [1]
	append!(selected_columns, get_proper_columns_indexes(header, type, n_trees))

	table::Vector{Vector{Any}} = []
	if keep_header
		push!(table, header[selected_columns])
	end
	for l in readlines(file)
		push!(table, split_line(l)[selected_columns])
	end

	close(file)

	if !isnothing(secondary_table)
		if remove_duplicated_rows
			# TODO: is there a less naive solution to this?
			for st_row in secondary_table
				jump_row = false
				for pt_row in table
					if st_row[1] == pt_row[1]
						jump_row = true
					end
				end
				if !jump_row
					push!(table, st_row)
				end
			end
		else
			append!(table, secondary_table)
		end
	end

	table
end

function string_table_csv(table::Vector{Vector{Any}}; column_separator = "\t")
	result = ""
	for row in table
		for (i, cell) in enumerate(row)
			result *= string(cell)
			if i != length(row)
				result *= column_separator
			end
		end
		result *= "\n"
	end
	result
end

function string_table_latex(table::Vector{Vector{Any}};
		header_size = 1,
		first_column_size = 1,
		v_lin_every_cloumn = 0,
		foot_note = "",
		scale = 1.0
	)
	result = "\\begin{table}[h]\n\\centering\n"

	if scale != 1.0
		result *= "\\resizebox{$(scale)\\linewidth}{!}{"
	else
		result *= "\\resizebox{1\\linewidth}{!}{"
	end

	result *= "\\begin{tabular}{$("c"^first_column_size)|"
	if v_lin_every_cloumn == 0
		result *= "$("l"^(length(table[header_size+1])-first_column_size))"
	else
		for i in (first_column_size+1):length(table[header_size+1])
			result *= "l"
			if (i-first_column_size) % v_lin_every_cloumn == 0 && i != length(table[header_size+1])
				result *= "|"
			end
		end
	end
	result *= "}\n"

	for (i, row) in enumerate(table)
		if i == 1
#			result *= "\\toprule\n"
		end
		for (j, cell) in enumerate(row)
			result *= string(cell)
			if j != length(row)
				result *= " & "
			end
		end
		if i != length(table)
			result *= " \\\\"
		end
		result *= "\n"
		if i == header_size
			result *= "\\hline"
		end
		if i == length(table)
#			result *= "\\bottomrule\n"
		end
	end
	result *= "\\end{tabular}"
	if length(foot_note) > 0
		result *= "\\begin{tablenotes}\n"
		result *= "\\item " * foot_note * "\n"
	  	result *= "\\end{tablenotes}\n"
	end

	# close resizebox
	result *= "}\n"
	result *= "\\end{table}\n"

	result
end
