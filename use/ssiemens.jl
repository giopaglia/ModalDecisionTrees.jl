################################################################################
################################################################################
################################## Scan script #################################
################################################################################
################################################################################

include("scanner.jl")

main_rng = DecisionTree.mk_rng(1)

train_seed = 1

################################################################################
#################################### FOLDERS ###################################
################################################################################

results_dir = "./siemens"

iteration_progress_json_file_path = results_dir * "/progress.json"
data_savedir = results_dir * "/cache"
tree_savedir = results_dir * "/trees"

dry_run = false

# save_datasets = false
save_datasets = true
skip_training = false

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
	for min_samples_leaf in [1] # [1,2]
		for min_purity_increase in [0.01, 0.001]
			for min_loss_at_leaf in [0.4, 0.6]
				push!(tree_args, 
					(
						loss_function       = loss_function,
						min_samples_leaf    = min_samples_leaf,
						min_purity_increase = min_purity_increase,
						min_loss_at_leaf    = min_loss_at_leaf,
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

modal_args = (;
	initConditions = DecisionTree.startWithRelationGlob,
	# initConditions = DecisionTree.startAtCenter,
	# useRelationGlob = true,
	useRelationGlob = false,
)

data_modal_args = (;
	ontology = getIntervalOntologyOfDim(Val(1)),
	# ontology = Ontology{ModalLogic.Interval}([ModalLogic.IA_A]),
	# ontology = Ontology{ModalLogic.Interval}([ModalLogic.IA_A, ModalLogic.IA_L, ModalLogic.IA_Li, ModalLogic.IA_D]),
	test_operators = [TestOpGeq_80, TestOpLeq_80],
)


################################################################################
##################################### MISC #####################################
################################################################################

# log_level = Logging.Warn
log_level = DecisionTree.DTOverview
# log_level = DecisionTree.DTDebug
# log_level = DecisionTree.DTDetail

# timing_mode = :none
timing_mode = :time
# timing_mode = :btime

round_dataset_to_datatype = false
# round_dataset_to_datatype = UInt8
# round_dataset_to_datatype = UInt16
# round_dataset_to_datatype = UInt32
# round_dataset_to_datatype = UInt64
# round_dataset_to_datatype = Float16
# round_dataset_to_datatype = Float32
# round_dataset_to_datatype = Float64

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

exec_from_to = [(1,10)]
# exec_from_to = [(1,120)]
# exec_from_to = [(1,120),(1440-120+1,1440)]


exec_ranges_dict = (
	from_to                                      = exec_from_to,
)

dataset_function = (from,to)->SiemensJuneDataset_not_stratified(from, to)

################################################################################
################################### SCAN FILTERS ###############################
################################################################################

# TODO let iteration_white/blacklist a decision function and not a "in-array" condition?
iteration_whitelist = []

iteration_blacklist = []

################################################################################
################################################################################
################################################################################
################################################################################

mkpath(results_dir)

if "-f" in ARGS
	if isfile(iteration_progress_json_file_path)
		println("Backing up existing $(iteration_progress_json_file_path)...")
		backup_file_using_creation_date(iteration_progress_json_file_path)
	end
end

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

	# Check whether this iteration was already computed or not
	if all(iteration_in_history(history, (params_namedtuple, dataseed)) for dataseed in exec_dataseed) && (!save_datasets) # !skip_training
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
	
	(from,to), = params_combination
	dataset_fun_sub_params = (from,to)
	
	# Load Dataset
	dataset, n_label_samples = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function

	cur_modal_args = modal_args
	cur_data_modal_args = data_modal_args

	## Dataset slices
	# obtain dataseeds that are were not done before
	todo_dataseeds = filter((dataseed)->!iteration_in_history(history, (params_namedtuple, dataseed)), exec_dataseed)
	dataset_slices = [(dataseed,balanced_dataset_slice(n_label_samples, dataseed)) for dataseed in todo_dataseeds]

	println("Dataseeds = $(todo_dataseeds)")

	##############################################################################
	##############################################################################
	##############################################################################
	
	exec_scan(
		run_name,
		dataset;
		### Training params
		train_seed                      =   train_seed,
		modal_args                      =   cur_modal_args,
		tree_args                       =   tree_args,
		tree_post_pruning_purity_thresh =   [],
		forest_args                     =   forest_args,
		forest_runs                     =   forest_runs,
		optimize_forest_computation     =   optimize_forest_computation,
		test_flattened                  =   test_flattened,
		test_averaged                   =   test_averaged,
		### Dataset params
		split_threshold                 =   split_threshold,
		data_modal_args                 =   cur_data_modal_args,
		dataset_slices                  =   dataset_slices,
		round_dataset_to_datatype       =   round_dataset_to_datatype,
		use_ontological_form            =   use_ontological_form,
		### Run params
		results_dir                     =   results_dir,
		data_savedir                    =   data_savedir,
		tree_savedir                    =   tree_savedir,
		legacy_gammas_check             =   legacy_gammas_check,
		log_level                       =   log_level,
		timing_mode                     =   timing_mode,
		### Misc
		save_datasets                   =   save_datasets,
		skip_training                   =   skip_training,
		callback                        =   (dataseed)->begin
			# Add this step to the "history" of already computed iteration
			push_iteration_to_history!(history, (params_namedtuple, dataseed))
			save_history(iteration_progress_json_file_path, history)
		end
	);

end

println("Done!")
