################################################################################
################################################################################
################################## Scan script #################################
################################################################################
################################################################################

include("scanner.jl")

data_seed = 1

train_seed = 1

################################################################################
#################################### FOLDERS ###################################
################################################################################

results_dir = "./pixel/journal-v1"

iteration_progress_json_file_path = results_dir * "/progress.json"
data_savedir  = results_dir * "/cache"
model_savedir = results_dir * "/trees"

dry_run = false
# dry_run = :dataset_only
# dry_run = true

# save_datasets = true
save_datasets = false

skip_training = false

perform_consistency_check = false

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
	for min_samples_leaf in [2, 4] # [1,2]
		for min_purity_increase in [0.01] # [0.01, 0.001]
			for min_loss_at_leaf in [0.3, 0.4, 0.5] # [0.4, 0.6]
				push!(tree_args, 
					(
						loss_function       = loss_function,
						min_samples_leaf    = min_samples_leaf,
						min_purity_increase = min_purity_increase,
						min_loss_at_leaf    = min_loss_at_leaf,
						perform_consistency_check = perform_consistency_check,
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
# for n_trees in [51,101]
for n_trees in [51]
	for n_subfeatures in [half_f]
		for n_subrelations in [id_f]
			for partial_sampling in [0.7]
				push!(forest_args, (
					n_subfeatures       = n_subfeatures,
					n_trees             = n_trees,
					partial_sampling    = partial_sampling,
					n_subrelations      = n_subrelations,
					# Optimization arguments for trees in a forest (no pruning is performed)
					loss_function       = DecisionTree.util.entropy,
					# min_samples_leaf    = 1,
					# min_purity_increase = 0.0,
					# min_loss_at_leaf    = 0.0,
					perform_consistency_check = perform_consistency_check,
				))
			end
		end
	end
end


println(" $(length(forest_args)) forests " * (length(forest_args) > 0 ? "(repeated $(forest_runs) times)" : ""))

################################################################################
################################## MODAL ARGS ##################################
################################################################################

modal_args = (;
	# initConditions = DecisionTree.startWithRelationGlob,
	initConditions = DecisionTree.startAtCenter,
	# useRelationGlob = true,
	useRelationGlob = false,
)

data_modal_args = (;
	# ontology = getIntervalOntologyOfDim(Val(1)),
	# ontology = getIntervalOntologyOfDim(Val(2)),
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

#timing_mode = :none
timing_mode = :time
#timing_mode = :btime
#timing_mode = :profile

# round_dataset_to_datatype = false
# round_dataset_to_datatype = UInt8
round_dataset_to_datatype = UInt16
# round_dataset_to_datatype = UInt32
# round_dataset_to_datatype = UInt64
# round_dataset_to_datatype = Float16
# round_dataset_to_datatype = Float32
# round_dataset_to_datatype = Float64

split_threshold = 0.8
# split_threshold = 1.0
# split_threshold = false

# use_training_form = :dimensional
# use_training_form = :fmd
# use_training_form = :stump
use_training_form = :stump_with_memoization

test_flattened = false
test_averaged  = false

legacy_gammas_check = false
# legacy_gammas_check = true


################################################################################
##################################### SCAN #####################################
################################################################################

exec_dataseed = 1:10

exec_dataset_name = ["Pavia", "Salinas-A", "PaviaCentre", "IndianPines", "Salinas"]

exec_windowsize_flattened_ontology_test_operators = [
	"single-pixel, propositional"       ,
	"3x3, flattened propositional"      ,
	"3x3, RCC8"                         ,
	"3x3, RCC5"                         ,
	#
	"3x3 avg, propositional"            ,
	"5x5 avg, flattened propositional"  ,
	"5x5 avg, RCC8"                     ,
	"5x5 avg, RCC5"                     ,
]

windowsize_flattened_ontology_test_operators_dict = Dict(
	"single-pixel, propositional"      => ((1,7),false,     false,     "-",   "TestOpGeq"),
	"3x3, flattened propositional"     => ((3,7),false,     :flattened,"-",   "TestOpGeq"),
	"3x3, RCC8"                        => ((3,7),false,     false,     "RCC8","TestOpAll"),
	"3x3, RCC5"                        => ((3,7),false,     false,     "RCC5","TestOpAll"),
	#
	"3x3 avg, propositional"           => ((3,7),false,     :averaged, "-",   "TestOpGeq"),
	"5x5 avg, flattened propositional" => ((5,7),("avg", 3),:flattened,"-",   "TestOpGeq"),
	"5x5 avg, RCC8"                    => ((5,7),("avg", 3),false,     "RCC8","TestOpAll"),
	"5x5 avg, RCC5"                    => ((5,7),("avg", 3),false,     "RCC5","TestOpAll"),
)

# exec_windowsize_flattened_ontology_test_operators = [(5,false,"RCC8","TestOp"),(7,false,"RCC8","TestOp"),(9,false,"RCC8","TestOp")]

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
	"TestOp"    => [TestOpGeq,    TestOpLeq],
	"TestOpAll" => [
									TestOpGeq,    TestOpLeq,
									TestOpGeq_90, TestOpLeq_90,
									TestOpGeq_80, TestOpLeq_80,
									TestOpGeq_70, TestOpLeq_70,
									TestOpGeq_60, TestOpLeq_60,
								],
)

ontology_dict = Dict(
	"-"     => ModalLogic.OneWorldOntology,
	"RCC8"  => getIntervalRCC8OntologyOfDim(Val(2)),
	"RCC5"  => getIntervalRCC5OntologyOfDim(Val(2)),
	"IA"    => getIntervalOntologyOfDim(Val(1)),
	"IA7"   => Ontology{ModalLogic.Interval}(ModalLogic.IA7Relations),
	"IA3"   => Ontology{ModalLogic.Interval}(ModalLogic.IA3Relations),
	"IA2D"  => getIntervalOntologyOfDim(Val(2)),
	# "o_ALLiDxA" => Ontology{ModalLogic.Interval2D}([ModalLogic.IA_AA, ModalLogic.IA_LA, ModalLogic.IA_LiA, ModalLogic.IA_DA]),
)


# exec_n_samples_per_label = 2:2
exec_n_samples_per_label = 100:100
exec_n_attributes = -1:-1

exec_ranges = (;
	windowsize_flattened_ontology_test_operators = exec_windowsize_flattened_ontology_test_operators,
	n_samples_per_label                          = exec_n_samples_per_label,
	n_attributes                                 = exec_n_attributes,
	dataset_name                                 = exec_dataset_name,
)

dataset_function = (
	(windowsize,apply_filter,flattened,test_operators),
	n_samples_per_label,n_attributes,
		dataset_name)->LandCoverDataset(
					dataset_name,
					windowsize[1],
					n_samples_per_label;
					pad_window_size = windowsize[2],
					flattened       = flattened,
					apply_filter    = apply_filter,
					n_attributes    = n_attributes,
					seed            = data_seed)

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

if "-d" in ARGS
	dry_run = true
end

# Copy scan script into the results folder
backup_file_using_creation_date(PROGRAM_FILE; copy_or_move = :copy, out_path = results_dir)

exec_ranges_names, exec_ranges_iterators = collect(string.(keys(exec_ranges))), collect(values(exec_ranges))
history = load_or_create_history(
	iteration_progress_json_file_path, exec_ranges_names, exec_ranges_iterators
)

# Log to console AND to .out file, & send Telegram message with Errors
using Logging, LoggingExtras
using Telegram, Telegram.API
using ConfigEnv

i_log_filename,log_filename = 0,""
while i_log_filename == 0 || isfile(log_filename)
	global i_log_filename,log_filename
	i_log_filename += 1
	log_filename = 
		results_dir * "/" *
		(dry_run == :dataset_only ? "datasets-" : "") *
		"$(i_log_filename).out"
end
logfile_io = open(log_filename, "w+")
dotenv()

tg = TelegramClient()
tg_logger = TelegramLogger(tg; async = false)

new_logger = TeeLogger(
	current_logger(),
	SimpleLogger(logfile_io, log_level),
	MinLevelLogger(tg_logger, Logging.Error), # Want to ignore Telegram? Comment out this
)
global_logger(new_logger)

################################################################################
################################################################################
################################################################################
################################################################################
# TODO actually,no need to recreate the dataset when changing, say, testoperators. Make a distinction between dataset params and run params
for params_combination in IterTools.product(exec_ranges_iterators...)

	flush(logfile_io);

	# Unpack params combination
	# params_namedtuple = (zip(Symbol.(exec_ranges_names), params_combination) |> Dict |> namedtuple)
	params_namedtuple = (;zip(Symbol.(exec_ranges_names), params_combination)...)
	
	# FILTER ITERATIONS
	if (!is_whitelisted_test(params_namedtuple, iteration_whitelist)) || is_blacklisted_test(params_namedtuple, iteration_blacklist)
		continue
	end

	##############################################################################
	##############################################################################
	##############################################################################

	run_name = join([replace(string(values(value)), ", " => ",") for value in params_combination], ",")

	# Placed here so we can keep track of which iteration is being skipped
	print("Iteration \"$(run_name)\"")

	# Check whether this iteration was already computed or not
	if all(iteration_in_history(history, (params_namedtuple, dataseed)) for dataseed in exec_dataseed) && (!save_datasets) # !skip_training
		println(": skipping")
		continue
	else
		println("...")
	end

	if dry_run == true
		continue
	end

	##############################################################################
	##############################################################################
	##############################################################################
	
	windowsize_flattened_ontology_test_operators,
	n_samples_per_label,n_attributes,
	dataset_name = params_combination
	
	windowsize_flattened_ontology_test_operators = windowsize_flattened_ontology_test_operators_dict[windowsize_flattened_ontology_test_operators]
	(windowsize,apply_filter,flattened,ontology,test_operators) = windowsize_flattened_ontology_test_operators

	test_operators = test_operators_dict[test_operators]
	ontology       = ontology_dict[ontology]

	dataset_fun_sub_params = (windowsize,apply_filter,flattened,test_operators), n_samples_per_label,n_attributes, dataset_name
	
	# Load Dataset
	dataset, n_label_samples = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function

	if dry_run == :dataset_only
		continue
	end

	cur_modal_args = merge(modal_args,
		(;)
	)
	cur_data_modal_args = merge(data_modal_args,
		(
		test_operators = test_operators,
		ontology       = ontology,
		)
	)

	## Dataset slices
	# obtain dataseeds that are were not done before
	todo_dataseeds = filter((dataseed)->!iteration_in_history(history, (params_namedtuple, dataseed)), exec_dataseed)
	dataset_slices = [(dataseed,balanced_dataset_slice(n_label_samples, dataseed)) for dataseed in todo_dataseeds]

	println("Dataseeds = $(todo_dataseeds)")

	##############################################################################
	##############################################################################
	##############################################################################
	
	exec_scan(
		params_namedtuple,
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
		use_training_form               =   use_training_form,
		### Run params
		results_dir                     =   results_dir,
		data_savedir                    =   data_savedir,
		model_savedir                   =   model_savedir,
		legacy_gammas_check             =   legacy_gammas_check,
		#logger                          =   logger,
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
 
@error "Done!"

close(logfile_io);

exit(0)
