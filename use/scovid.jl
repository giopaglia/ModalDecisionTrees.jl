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

results_dir = "./covid"

iteration_progress_json_file_path = results_dir * "/progress.json"
data_savedir = results_dir * "/cache"
tree_savedir = results_dir * "/trees"

dry_run = false
# dry_run = true

# save_datasets = true
save_datasets = false

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
	for min_samples_leaf in [2] # [1,2]
		for min_purity_increase in [0.01] # [0.01, 0.001]
			for min_loss_at_leaf in [0.2, 0.6] # [0.4, 0.6]
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

for n_trees in [1] # [50,100] TODO
	for n_subfeatures in [half_f]
		for n_subrelations in [id_f]
			push!(forest_args, (
				n_subfeatures       = n_subfeatures,
				n_trees             = n_trees,
				partial_sampling    = 1.0,
				n_subrelations      = n_subrelations,
				# Optimization arguments for trees in a forest (no pruning is performed)
				loss_function       = DecisionTree.util.entropy,
				min_samples_leaf    = 1,
				min_purity_increase = 0.0,
				min_loss_at_leaf    = 0.0,
			))
		end
	end
end


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

# use_training_form = :dimensional
# use_training_form = :fmd
# use_training_form = :stump
# use_training_form = :stump_with_memoization

test_flattened = false
test_averaged  = false

legacy_gammas_check = false
# legacy_gammas_check = true


################################################################################
##################################### SCAN #####################################
################################################################################

exec_dataseed = 1:5 # TODO 10

exec_use_training_form = [:stump, :stump_with_memoization] # TODO

exec_n_tasks = 1:1
exec_n_versions = 1:1 # 1:3 # 1:3 # TODO
exec_nbands = [2] # [20,40,60] TODO

# max_points = 30

exec_dataset_kwargs =   [( # TODO
							max_points = 10,
							ma_size = 45,
							ma_step = 30,
						),(
							max_points = 20,
							ma_size = 45,
							ma_step = 30,
						# ),(
						# 	max_points = 30,
						# 	ma_size = 45,
						# 	ma_step = 30,
						# ),(
						# 	max_points = 30,
						# 	ma_size = 75,
						# 	ma_step = 50,
						# ),(
						# 	max_points = 50,
						# 	ma_size = 45,
						# 	ma_step = 30,
						)
						]

audio_kwargs_partial_mfcc = (
	wintime = 0.025, # in ms          # 0.020-0.040
	steptime = 0.010, # in ms         # 0.010-0.015
	fbtype = :mel,                    # [:mel, :htkmel, :fcmel]
	window_f = DSP.hamming, # [DSP.hamming, (nwin)->DSP.tukey(nwin, 0.25)]
	pre_emphasis = 0.97,              # any, 0 (no pre_emphasis)
	nbands = 40,                      # any, (also try 20)
	sumpower = false,                 # [false, true]
	dither = false,                   # [false, true]
	# bwidth = 1.0,                   # 
	# minfreq = 0.0,
	# maxfreq = (sr)->(sr/2),
	# usecmp = false,
)

audio_kwargs_full_mfcc = (
	wintime=0.025,
	steptime=0.01,
	numcep=13,
	lifterexp=-22,
	sumpower=false,
	preemph=0.97,
	dither=false,
	minfreq=0.0,
	# maxfreq=sr/2,
	nbands=20,
	bwidth=1.0,
	dcttype=3,
	fbtype=:htkmel,
	usecmp=false,
	modelorder=0
)

exec_use_full_mfcc = [false]


wav_preprocessors = Dict(
	"NG" => noise_gate!,
	"Normalize" => normalize!,
)

exec_preprocess_wavs = [
	# ["Normalize"],
	[],
#	["NG", "Normalize"]
]

# https://github.com/JuliaIO/JSON.jl/issues/203
# https://discourse.julialang.org/t/json-type-serialization/9794
# TODO: make test operators types serializable
# exec_test_operators = [ "TestOp" ]
exec_test_operators = [ "TestOp_80" ]

test_operators_dict = Dict(
	"TestOp_70" => [TestOpGeq_70, TestOpLeq_70],
	"TestOp_80" => [TestOpGeq_80, TestOpLeq_80],
	"TestOp"    => [TestOpGeq,    TestOpLeq],
)


exec_ranges = (;
	use_training_form = exec_use_training_form,
	n_task            = exec_n_tasks,
	n_version         = exec_n_versions,
	nbands            = exec_nbands,
	dataset_kwargs    = exec_dataset_kwargs,
	use_full_mfcc     = exec_use_full_mfcc,
	preprocess_wavs   = exec_preprocess_wavs,
	test_operators    = exec_test_operators,
)

dataset_function = (
	(n_task,
		n_version,
		cur_audio_kwargs,
		dataset_kwargs,
		cur_preprocess_wavs,
		use_full_mfcc,)->
	KDDDataset_not_stratified(
		(n_task,n_version),
		cur_audio_kwargs;
		dataset_kwargs...,
		preprocess_wavs = cur_preprocess_wavs,
		use_full_mfcc = use_full_mfcc
	)
)

################################################################################
################################### SCAN FILTERS ###############################
################################################################################

# TODO let iteration_white/blacklist a decision function and not a "in-array" condition?
iteration_whitelist = [
	# TASK 1
	# (
	# 	n_version = 1,
	# 	nbands = 40,
	# 	dataset_kwargs = (max_points = 30, ma_size = 75, ma_step = 50),
	# ),
	# (
	# 	n_version = 1,
	# 	nbands = 60,
	# 	dataset_kwargs = (max_points = 30, ma_size = 75, ma_step = 50),
	# ),
	# # TASK 2
	# (
	# 	n_version = 2,
	# 	nbands = 20,
	# 	dataset_kwargs = (max_points = 30, ma_size = 45, ma_step = 30),
	# ),
	# (
	# 	n_version = 2,
	# 	nbands = 40,
	# 	dataset_kwargs = (max_points = 30, ma_size = 45, ma_step = 30),
	# )
]

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

exec_ranges_names, exec_ranges_iterators = collect(string.(keys(exec_ranges))), collect(values(exec_ranges))
history = load_or_create_history(
	iteration_progress_json_file_path, exec_ranges_names, exec_ranges_iterators
)

################################################################################
################################################################################
################################################################################
################################################################################
# TODO actually,no need to recreate the dataset when changing, say, testoperators. Make a distinction between dataset params and run params
for params_combination in IterTools.product(exec_ranges_iterators...)

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

	if dry_run
		continue
	end

	##############################################################################
	##############################################################################
	##############################################################################
	
	use_training_form, n_task, n_version, nbands, dataset_kwargs, use_full_mfcc, preprocess_wavs, test_operators = params_combination
	
	test_operators = test_operators_dict[test_operators]

	cur_audio_kwargs = merge(
		if use_full_mfcc
			audio_kwargs_full_mfcc
		else
			audio_kwargs_partial_mfcc
		end
		, (nbands=nbands,))

	cur_preprocess_wavs = [ wav_preprocessors[k] for k in preprocess_wavs ]

	cur_modal_args = modal_args
	
	cur_data_modal_args = merge(data_modal_args,
		(
			test_operators = test_operators,
		)
	)

	dataset_fun_sub_params = (
			n_task,
			n_version,
			cur_audio_kwargs,
			dataset_kwargs,
			cur_preprocess_wavs,
			use_full_mfcc,)	

	# Load Dataset
	dataset, n_label_samples = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function

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

exit(0)
