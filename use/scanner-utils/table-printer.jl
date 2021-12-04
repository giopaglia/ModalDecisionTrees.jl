
import Dates

include("../lib.jl")

###############################################################################
############################ OUTPUT HANDLERS ##################################
###############################################################################


function get_creation_date_string_format(file_name::String)::String
	Dates.format(Dates.unix2datetime(ctime(file_name)), "HH.MM.SS_dd.mm.yyyy")
end

function backup_file_using_creation_date(file_name::String; out_path = "", copy_or_move = :move, file_suffix = ".bkp")
	splitted_name = Base.Filesystem.splitext(file_name)
	backup_file_name = splitted_name[1] * "_" * get_creation_date_string_format(file_name) * splitted_name[2]
	if copy_or_move == :move
		mv(file_name, Filesystem.joinpath(out_path, backup_file_name * file_suffix))
	elseif copy_or_move == :copy
		cp(file_name, Filesystem.joinpath(out_path, backup_file_name * file_suffix), force=true)
	else
		throw_n_log("backup_file_using_creation_date: Unexpected value for copy_or_move $(copy_or_move)")
	end
end

function string_head(
		tree_args::AbstractArray,
		forest_args::AbstractArray;
		separator::String = "\t",
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
		result *= separator
	end

	for i in 1:length(tree_args)
		for j in 1:length(tree_columns)
			result *= "T,$(Tuple(tree_args[i]))"
			result *= string(tree_columns[j])
			result *= separator
		end
	end

	for i in 1:length(forest_args)
		for j in 1:length(forest_columns)
			result *= "RF,$(Tuple(forest_args[i]))"
			result *= string(forest_columns[j])
			result *= separator
		end
	end

	result *= string("\n")

	result
end

function print_result_head(file_name::String, tree_args::AbstractArray, forest_args::AbstractArray; kwargs...)
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
		cm::GenericPerformanceType,
		time::Dates.Millisecond,
		hash::AbstractString,
		columns::AbstractVector;
		train_cm = nothing,
		start_s = "(",
		end_s = ")",
		separator = "\t",
		alt_separator = ",",
		# best_rule_params = [(t=.8, min_confidence=0.6, min_support=0.1), (t=.65, min_confidence=0.6, min_support=0.1)],
	)

	result = start_s
	for (i_col,column) in enumerate(columns)
		result *= string(
			if column == "train_accuracy"
				percent(overall_accuracy(train_cm))                  
			elseif column == "K"
				percent(kappa(cm))                  
			elseif column == "accuracy"
				percent(overall_accuracy(cm)) 
			elseif column == "macro_sensitivity"      
				percent(macro_sensitivity(cm)) 
			elseif column == "safe_macro_sensitivity"      
				percent(safe_macro_sensitivity(cm)) 
			elseif column == "safe_macro_specificity"
				percent(safe_macro_specificity(cm)) 
			elseif column == "safe_macro_PPV"
				percent(safe_macro_PPV(cm))         
			elseif column == "safe_macro_NPV"
				percent(safe_macro_NPV(cm))         
			elseif column == "safe_macro_F1"
				percent(safe_macro_F1(cm))          
			elseif column == "n_nodes"
				num_nodes(M)        
			elseif column == "n_leaves"
				length(M)
			elseif column == "height"
				height(M)
			elseif column == "modal_height"
				modal_height(M)
			elseif column == "train_cor"
				train_cm.cor
			elseif column == "train_MAE"
				train_cm.MAE
			elseif column == "train_RMSE"
				train_cm.RMSE
			elseif column == "cor"
				cm.cor
			elseif column == "MAE"
				cm.MAE
			elseif column == "RMSE"
				cm.RMSE
			else
				"??? $(column) ???"
		end, alt_separator) # (i_col == length(columns) ? "", alt_separator))

		# m = tree_walk_metrics(M; best_rule_params = best_rule_params)
		# for best_rule_p in best_rule_params
		# 	result *= string(m["best_rule_t=$(best_rule_p)"], alt_separator)
		# end
	end
	
	result *= string(human_readable_time_s(time), alt_separator)
	result *= string(hash)
	result *= end_s

	result
end

# Print a forest entry in a row
function data_to_string(
		Ms::AbstractVector{Forest{S}},
		cms::AbstractVector{GenericPerformanceType},
		time::Dates.Millisecond,
		hash::AbstractString,
		columns::AbstractVector;
		train_cm = nothing,
		start_s = "(",
		end_s = ")",
		separator = "\t",
		alt_separator = ",",
	) where {S}

	result = start_s

	for (i_col,column) in enumerate(columns)
		result *= string(
			if column == "train_accuracy"
				percent(mean(map(cm->overall_accuracy(cm),       train_cm)))
			elseif column == "K"
				percent(mean(map(cm->kappa(cm),                  cms)))
			elseif column == "accuracy"
				percent(mean(map(cm->overall_accuracy(cm),       cms)))
			elseif column == "macro_sensitivity"      
				percent(mean(map(cm->macro_sensitivity(cm),      cms)))
			elseif column == "safe_macro_sensitivity"      
				percent(mean(map(cm->safe_macro_sensitivity(cm), cms)))
			elseif column == "safe_macro_specificity"
				percent(mean(map(cm->safe_macro_specificity(cm), cms)))
			elseif column == "safe_macro_PPV"
				percent(mean(map(cm->safe_macro_PPV(cm),         cms)))
			elseif column == "safe_macro_NPV"
				percent(mean(map(cm->safe_macro_NPV(cm),         cms)))
			elseif column == "safe_macro_F1"
				percent(mean(map(cm->safe_macro_F1(cm),          cms)))
			elseif column == "n_trees"
				percent(mean(length.(Ms)))
			elseif column == "n_nodes"
				percent(mean(num_nodes.(Ms)))
			elseif column == "oob_error"
				percent(mean(map(M->M.oob_error, Ms)))
			elseif column == "train_cor"
				mean(map(cm->train_cm.cor,          cms))
			elseif column == "train_MAE"
				mean(map(cm->train_cm.MAE,          cms))
			elseif column == "train_RMSE"
				mean(map(cm->train_cm.RMSE,          cms))
			elseif column == "cor"
				mean(map(cm->cm.cor,          cms))
			elseif column == "MAE"
				mean(map(cm->cm.MAE,          cms))
			elseif column == "RMSE"
				mean(map(cm->cm.RMSE,          cms))
			else
				"??? $(column) ???"
		end, (i_col == length(columns) ? "" : alt_separator))
	end
	result *= end_s
	result *= separator
	result *= start_s
	for (i_col,column) in enumerate(columns)
		result *= string(
			if column == "train_accuracy"
				percent(var(map(cm->overall_accuracy(cm),       train_cm)))
			elseif column == "K"
				percent(var(map(cm->kappa(cm),                  cms)))
			elseif column == "accuracy"
				percent(var(map(cm->overall_accuracy(cm),       cms)))
			elseif column == "macro_sensitivity"      
				percent(var(map(cm->macro_sensitivity(cm),      cms)))
			elseif column == "safe_macro_sensitivity"      
				percent(var(map(cm->safe_macro_sensitivity(cm), cms)))
			elseif column == "safe_macro_specificity"
				percent(var(map(cm->safe_macro_specificity(cm), cms)))
			elseif column == "safe_macro_PPV"
				percent(var(map(cm->safe_macro_PPV(cm),         cms)))
			elseif column == "safe_macro_NPV"
				percent(var(map(cm->safe_macro_NPV(cm),         cms)))
			elseif column == "safe_macro_F1"
				percent(var(map(cm->safe_macro_F1(cm),          cms)))
			elseif column == "n_trees"
				percent(var(length.(Ms)))
			elseif column == "n_nodes"
				percent(var(num_nodes.(Ms)))
			elseif column == "oob_error"
				percent(var(map(M->M.oob_error, Ms)))
			elseif column == "train_cor"
				var(map(cm->train_cm.cor,          cms))
			elseif column == "train_MAE"
				var(map(cm->train_cm.MAE,          cms))
			elseif column == "train_RMSE"
				var(map(cm->train_cm.RMSE,          cms))
			elseif column == "cor"
				var(map(cm->cm.cor,          cms))
			elseif column == "MAE"
				var(map(cm->cm.MAE,          cms))
			elseif column == "RMSE"
				var(map(cm->cm.RMSE,          cms))
			else
				"??? $(column) ???"
		end, (i_col == length(columns) ? "" : alt_separator))
	end
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
		throw_n_log("No file with name $(file_name) found.")
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
			throw_n_log("No model known of type $(type); could be \"T\" or \"RF\".")
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
