################################################################################
################################################################################
################################## Scan script #################################
################################################################################
################################################################################

include("runner.jl")
include("table-printer.jl")
include("progressive-iterator-manager.jl")

main_rng = DecisionTree.mk_rng(1)

train_seed = 1


################################################################################
#################################### FOLDERS ###################################
################################################################################

results_dir = "./results-gandalf"

iteration_progress_json_file_path = results_dir * "/progress.json"
concise_output_file_path = results_dir * "/grouped_in_models.csv"
full_output_file_path = results_dir * "/full_columns.csv"
data_savedir = results_dir * "/gammas"
tree_savedir = results_dir * "/trees"

column_separator = ";"

save_datasets = true
just_produce_datasets_jld = false
saved_datasets_path = results_dir * "/datasets"

################################################################################
##################################### TREES ####################################
################################################################################

# Optimization arguments for single-tree
tree_args = [
#	(
#		loss_function = DecisionTree.util.entropy,
#		min_samples_leaf = 1,
#		min_purity_increase = 0.01,
#		min_loss_at_leaf = 0.6,
#	)
]

for loss_function in [DecisionTree.util.entropy]
	for min_samples_leaf in [4] # [1,2]
		for min_purity_increase in [0.01] # [0.01, 0.001]
			for min_loss_at_leaf in [0.3] # [0.4, 0.6]
				push!(tree_args, 
					(
						loss_function = loss_function,
						min_samples_leaf = min_samples_leaf,
						min_purity_increase = min_purity_increase,
						min_loss_at_leaf = min_loss_at_leaf,
					)
				)
			end
		end
	end
end

println(" $(length(tree_args)) trees")

################################################################################
#################################### FORESTS ###################################
################################################################################

forest_runs = 5
optimize_forest_computation = true


forest_args = []

# for n_trees in [50,100]
# 	for n_subfeatures in [half_f]
# 		for n_subrelations in [id_f]
# 			push!(forest_args, (
# 				n_subfeatures       = n_subfeatures,
# 				n_trees             = n_trees,
# 				partial_sampling    = 1.0,
# 				n_subrelations      = n_subrelations,
# 				# Optimization arguments for trees in a forest (no pruning is performed)
# 				loss_function = DecisionTree.util.entropy,
# 				min_samples_leaf = 1,
# 				min_purity_increase = 0.0,
# 				min_loss_at_leaf = 0.0,
# 			))
# 		end
# 	end
# end


println(" $(length(forest_args)) forests " * (length(forest_args) > 0 ? "(repeated $(forest_runs) times)" : ""))

################################################################################
################################## MODAL ARGS ##################################
################################################################################

modal_args = (
	# initConditions = DecisionTree.startWithRelationGlob,
	initConditions = DecisionTree.startAtCenter,
	# useRelationGlob = true,
	useRelationGlob = false,
)

data_modal_args = (
	ontology = getIntervalOntologyOfDim(Val(1)),
	# ontology = Ontology{ModalLogic.Interval}([ModalLogic.IA_A]),
	# ontology = Ontology{ModalLogic.Interval}([ModalLogic.IA_A, ModalLogic.IA_L, ModalLogic.IA_Li, ModalLogic.IA_D]),
)


################################################################################
##################################### MISC #####################################
################################################################################

# log_level = Logging.Warn
log_level = DecisionTree.DTOverview
# log_level = DecisionTree.DTDebug
# log_level = DecisionTree.DTDetail

timing_mode = :none
# timing_mode = :time
# timing_mode = :btime

round_dataset_to_datatype = false
# round_dataset_to_datatype = UInt16
#round_dataset_to_datatype = Float32
# round_dataset_to_datatype = UInt16

split_threshold = 0.8
# split_threshold = 1.0
# split_threshold = false

use_ontological_form = false

test_flattened = false
test_averaged  = false

legacy_gammas_check = false # true


################################################################################
##################################### SCAN #####################################
################################################################################

exec_dataseed = 1:10

# exec_dataset_name = ["Salinas", "Salinas-A", "PaviaCentre", "IndianPines", "Pavia"]
exec_dataset_name = ["Pavia", "Salinas-A", "PaviaCentre", "IndianPines", "Salinas"]
exec_windowsize_flattened_ontology_test_operators = [(1,false,"o_None","TestOpGeq"),(3,:flattened,"o_None","TestOpGeq"),(3,:averaged,"o_None","TestOpGeq"),(3,false,"o_RCC8","TestOpAll"),(3,false,"o_RCC5","TestOpAll")]
# exec_windowsize_flattened_ontology_test_operators = [(1,false,"o_None","TestOpGeq"),(3,:flattened,"o_None","TestOpGeq"),(3,:averaged,"o_None","TestOpGeq"),(3,false,"o_RCC8","TestOp"),(3,false,"o_RCC5","TestOp")]
# exec_windowsize_flattened_ontology_test_operators = [(3,:averaged,"o_None","TestOpGeq")]
# exec_windowsize_flattened_ontology_test_operators = [(3,false,"o_RCC8","TestOp")]

# https://github.com/JuliaIO/JSON.jl/issues/203
# https://discourse.julialang.org/t/json-type-serialization/9794
# TODO: make test operators types serializable
# exec_test_operators = [ "TestOpAll" ]
# exec_test_operators = [ "TestOp" ]
# exec_test_operators = [ "TestOp_80" ]

test_operators_dict = Dict(
	"TestOpGeq" => [TestOpGeq],
	"TestOp_70" => [TestOpGeq_70, TestOpLeq_70],
	"TestOp_80" => [TestOpGeq_80, TestOpLeq_80],
	"TestOp"    => [TestOpGeq, TestOpLeq],
	"TestOpAll" => [
									TestOpGeq, TestOpLeq,
									TestOpGeq_90, TestOpLeq_90,
									TestOpGeq_80, TestOpLeq_80,
									TestOpGeq_70, TestOpLeq_70,
									TestOpGeq_60, TestOpLeq_60,
									],
)

ontology_dict = Dict(
	"o_RCC8" => getIntervalRCC8OntologyOfDim(Val(2)),
	"o_RCC5" => getIntervalRCC5OntologyOfDim(Val(2)),
	"o_None" => OneWorldOntology,
)


exec_ranges_dict = (
	windowsize_flattened_ontology_test_operators = exec_windowsize_flattened_ontology_test_operators,
	# test_operators                             = exec_test_operators,
	dataset_name                                 = exec_dataset_name,
	dataseed                                     = exec_dataseed,
)

n_samples_per_label = 100

dataset_function = (
	(windowsize,flattened,ontology,test_operators),
	dataset_name,
	dataseed)->SampleLandCoverDataset(
		dataset_name,
		n_samples_per_label,
		windowsize,
		flattened = flattened,
		# n_attributes = 3,
		rng = Random.MersenneTwister(dataseed))

################################################################################
################################### SCAN FILTERS ###############################
################################################################################

dry_run = false

# TODO let iteration_white/blacklist a decision function and not a "in-array" condition?
iteration_whitelist = []

iteration_blacklist = []


################################################################################
################################################################################
################################################################################
################################################################################

mkpath(saved_datasets_path)

if "-f" in ARGS
	if isfile(iteration_progress_json_file_path)
		println("Backing up existing $(iteration_progress_json_file_path)...")
		backup_file_using_creation_date(iteration_progress_json_file_path)
	end
	if isfile(concise_output_file_path)
		println("Backing up existing $(concise_output_file_path)...")
		backup_file_using_creation_date(concise_output_file_path)
	end
	if isfile(full_output_file_path)
		println("Backing up existing $(full_output_file_path)...")
		backup_file_using_creation_date(full_output_file_path)
	end
end

# if the output files does not exists initilize them
print_head(concise_output_file_path, tree_args, forest_args, tree_columns = [""], forest_columns = ["", "σ²", "t"], separator = column_separator)
print_head(full_output_file_path, tree_args, forest_args, separator = column_separator,
	forest_columns = ["K", "sensitivity", "specificity", "precision", "accuracy", "oob_error", "σ² K", "σ² sensitivity", "σ² specificity", "σ² precision", "σ² accuracy", "σ² oob_error", "t"],
)

exec_ranges_names, exec_ranges = collect(string.(keys(exec_ranges_dict))), collect(values(exec_ranges_dict))
history = load_or_create_history(
	iteration_progress_json_file_path, exec_ranges_names, exec_ranges
)

################################################################################
################################################################################
################################################################################
################################################################################
# TODO actually,no need to recreate the dataset when changing, say, testoperators. Make a distinction between dataset params and run params
for params_combination in IterTools.product(exec_ranges...)

	# Unpack params combination
	params_namedtuple = (zip(Symbol.(exec_ranges_names), params_combination) |> Dict |> namedtuple)

	# FILTER ITERATIONS
	if (!is_whitelisted_test(params_namedtuple, iteration_whitelist)) || is_blacklisted_test(params_namedtuple, iteration_blacklist)
		continue
	end

	##############################################################################
	##############################################################################
	##############################################################################

	run_name = join([replace(string(values(value)), ", " => ",") for value in values(params_namedtuple)], ",")

	# Placed here so we can keep track of which iteration is being skipped
	print("Iteration \"$(run_name)\"")

	# CHECK WHETHER THIS ITERATION WAS ALREADY COMPUTED OR NOT
	if iteration_in_history(history, params_namedtuple) && !just_produce_datasets_jld
		println(": skipping")
		continue
	else
		println("...")
	end

	if dry_run
		continue
	end

	##############################################################################
	##############################################################################
	##############################################################################
	
	(windowsize,flattened,ontology,test_operators), dataset_name, dataseed = params_combination
	
	# LOAD DATASET
	dataset_file_name = saved_datasets_path * "/" * run_name

	cur_modal_args = modal_args

	# TODO reduce redundancy with caching function
	dataset = dataset_function(params_combination...)
	dataset_slice = nothing
	# dataset_rng = Random.MersenneTwister(dataseed)
	# dataset, dataset_slice = 
	# 	if save_datasets && isfile(dataset_file_name * ".jld")
	# 		if just_produce_datasets_jld
	# 			continue
	# 		end

	# 		println("Loading dataset $(dataset_file_name * ".jld")...")
			
	# 		dataset = nothing
	# 		n_pos = nothing
	# 		n_neg = nothing
			
	# 		JLD2.@load (dataset_file_name * ".jld") dataset n_pos n_neg
	# 		(X,Y) = dataset
	# 		if isfile(dataset_file_name * "-balanced.jld")
	# 			JLD2.@load (dataset_file_name * "-balanced.jld") balanced_dataset dataset_slice
	# 		else
	# 			n_per_class = min(n_pos, n_neg)
	# 			dataset_slice = Array{Int,2}(undef, 2, n_per_class)
	# 			dataset_slice[1,:] .=          Random.randperm(dataset_rng, n_pos)[1:n_per_class]
	# 			dataset_slice[2,:] .= n_pos .+ Random.randperm(dataset_rng, n_neg)[1:n_per_class]
	# 			dataset_slice = dataset_slice[:]

	# 			balanced_dataset = slice_labeled_dataset((X,Y), dataset_slice)
	# 			typeof(balanced_dataset) |> println
	# 			(X_train, Y_train), (X_test, Y_test) = traintestsplit(balanced_dataset, split_threshold)
	# 			JLD2.@save (dataset_file_name * "-balanced.jld") balanced_dataset dataset_slice
	# 			balanced_train = (X_train, Y_train)
	# 			JLD2.@save (dataset_file_name * "-balanced-train.jld") balanced_train
	# 			balanced_test = (X_test,  Y_test)
	# 			JLD2.@save (dataset_file_name * "-balanced-test.jld")  balanced_test
	# 		end

	# 		dataset = (X,Y)
	#			dataset, dataset_slice
	# 	else
	# 		println("Creating dataset...")
	# 		# TODO wrap dataset creation into a function accepting the rng and other parameters...
	# 		dataset = dataset_function(params_combination)

	# 		n_per_class = min(n_pos, n_neg)

	# 		dataset_slice = Array{Int,2}(undef, 2, n_per_class)
	# 		dataset_slice[1,:] .=          Random.randperm(dataset_rng, n_pos)[1:n_per_class]
	# 		dataset_slice[2,:] .= n_pos .+ Random.randperm(dataset_rng, n_neg)[1:n_per_class]
	# 		dataset_slice = dataset_slice[:]

	# 		if save_datasets
	# 			println("Saving dataset $(dataset_file_name)...")
	# 			(X, Y) = dataset
	# 			JLD2.@save (dataset_file_name * ".jld")                dataset n_pos n_neg
	# 			balanced_dataset = slice_labeled_dataset((X,Y), dataset_slice)
	# 			typeof(balanced_dataset) |> println
	# 			(X_train, Y_train), (X_test, Y_test) = traintestsplit(balanced_dataset, split_threshold)
	# 			JLD2.@save (dataset_file_name * "-balanced.jld") balanced_dataset dataset_slice
	# 			balanced_train = (X_train, Y_train)
	# 			JLD2.@save (dataset_file_name * "-balanced-train.jld") balanced_train
	# 			balanced_test = (X_test,  Y_test)
	# 			JLD2.@save (dataset_file_name * "-balanced-test.jld")  balanced_test
	# 			if just_produce_datasets_jld
	# 				continue
	# 			end
	# 		end
	# 		dataset, dataset_slice
	# 	end
	# println(dataset_slice)

	cur_data_modal_args = merge(data_modal_args, (test_operators = test_operators_dict[test_operators], ontology = ontology_dict[ontology]))
	
	##############################################################################
	##############################################################################
	##############################################################################
	
	# ACTUAL COMPUTATION
	Ts, Fs, Tcms, Fcms, Tts, Fts = exec_run(
				run_name,
				dataset,
				split_threshold             =   split_threshold,
				log_level                   =   log_level,
				round_dataset_to_datatype   =   round_dataset_to_datatype,
				dataset_slice               =   dataset_slice,
				forest_args                 =   forest_args,
				tree_args                   =   tree_args,
				data_modal_args             =   cur_data_modal_args,
				modal_args                  =   cur_modal_args,
				test_flattened              =   test_flattened,
				test_averaged               =   test_averaged,
				legacy_gammas_check         =   legacy_gammas_check,
				use_ontological_form        =   use_ontological_form,
				optimize_forest_computation =   optimize_forest_computation,
				forest_runs                 =   forest_runs,
				data_savedir                =   (data_savedir, run_name),
				tree_savedir                =   tree_savedir,
				train_seed                  =   train_seed,
				timing_mode                 =   timing_mode
			);
	##############################################################################
	##############################################################################
	# PRINT RESULT IN FILES 
	##############################################################################
	##############################################################################

	# PRINT CONCISE
	concise_output_string = string(run_name, column_separator)
	for j in 1:length(tree_args)
		concise_output_string *= string(data_to_string(Ts[j], Tcms[j], Tts[j]; alt_separator=", ", separator = column_separator))
		concise_output_string *= string(column_separator)
	end
	for j in 1:length(forest_args)
		concise_output_string *= string(data_to_string(Fs[j], Fcms[j], Fts[j]; alt_separator=", ", separator = column_separator))
		concise_output_string *= string(column_separator)
	end
	concise_output_string *= string("\n")
	append_in_file(concise_output_file_path, concise_output_string)

	# PRINT FULL
	full_output_string = string(run_name, column_separator)
	for j in 1:length(tree_args)
		full_output_string *= string(data_to_string(Ts[j], Tcms[j], Tts[j]; start_s = "", end_s = "", alt_separator = column_separator))
		full_output_string *= string(column_separator)
	end
	for j in 1:length(forest_args)
		full_output_string *= string(data_to_string(Fs[j], Fcms[j], Fts[j]; start_s = "", end_s = "", alt_separator = column_separator))
		full_output_string *= string(column_separator)
	end
	full_output_string *= string("\n")
	append_in_file(full_output_file_path, full_output_string)

	##############################################################################
	##############################################################################
	# ADD THIS STEP TO THE "HISTORY" OF ALREADY COMPUTED ITERATION
	push_iteration_to_history!(history, params_namedtuple)
	save_history(iteration_progress_json_file_path, history)
	##############################################################################
	##############################################################################
end

println("Done!")
