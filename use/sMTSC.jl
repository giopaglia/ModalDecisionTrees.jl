################################################################################
################################################################################
################################## Scan script #################################
################################################################################
################################################################################

include("scanner.jl")

# using MAT
using NPZ

main_rng = DecisionTree.mk_rng(1)

train_seed = 1

################################################################################
#################################### FOLDERS ###################################
################################################################################

results_dir = "./MTSC-more_dataseeds-v5-better"

iteration_progress_json_file_path = results_dir * "/progress.json"
data_savedir  = results_dir * "/data_cache"
model_savedir = results_dir * "/models_cache"

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
#		max_purity_at_leaf = 0.6,
#	)
]

for loss_function in [DecisionTree.util.entropy]
	for min_samples_leaf in [2,4] # [1,2]
		for min_purity_increase in [0.01] # [0.01, 0.001]
			for max_purity_at_leaf in [0.2, 0.6] # [0.4, 0.6]
				push!(tree_args, 
					(
						loss_function             = loss_function,
						min_samples_leaf          = min_samples_leaf,
						min_purity_increase       = min_purity_increase,
						max_purity_at_leaf        = max_purity_at_leaf,
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

for n_trees in [100]
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
				max_purity_at_leaf  = Inf,
				perform_consistency_check = perform_consistency_check,
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
	# ontology = ModalLogic.OneWorldOntology,
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

# n_samples_per_class = 0.8

# split_threshold = 0.8
split_threshold = 1.0
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

# exec_dataseed = [nothing]
# exec_dataseed = [nothing, (1:9)...]
exec_dataseed = [(1:30)...]

# exec_use_training_form = [:dimensional]
exec_use_training_form = [:stump_with_memoization]

# https://github.com/JuliaIO/JSON.jl/issues/203
# https://discourse.julialang.org/t/json-type-serialization/9794
# TODO: make test operators types serializable
# exec_test_operators = [ "TestOp" ]
exec_test_operators = [ "TestOp_80" ]

test_operators_dict = Dict(
	"TestOp_70" => [TestOpGeq_70, TestOpLeq_70],
	"TestOp_80" => [TestOpGeq_80, TestOpLeq_80],
	"TestOp_90" => [TestOpGeq_90, TestOpLeq_90],
	"TestOp"    => [TestOpGeq,    TestOpLeq],
)

exec_dataset_name = [
	"FingerMovements",
	"Libras",
	"LSST",
	"NATOPS",
	"RacketSports",
	
	#"AtrialFibrillation",
	#"Cricket",
	#"EthanolConcentration",
	#"HandMovementDirection",
	#"Heartbeat",
	#"MotorImagery",
	#"SelfRegulationSCP1",
	#"SelfRegulationSCP2",
	#"StandWalkJump",
]

# exec_flatten_ontology = [(false,"interval2D")] # ,(true,"one_world")]
# exec_use_catch22_flatten_ontology = [(false,false,"interval")]
exec_use_catch22_flatten_ontology = [(false,false,"interval"),(false,true,"one_world"),(true,true,"one_world")]

ontology_dict = Dict(
	"one_world"   => ModalLogic.OneWorldOntology,
	"interval"    => getIntervalOntologyOfDim(Val(1)),
	"interval2D"  => getIntervalOntologyOfDim(Val(2)),
)


exec_n_chunks = [missing]
# exec_n_chunks = [60]

# exec_use_catch22 = [false]

exec_ranges = (;
	use_training_form                = exec_use_training_form              ,
	test_operators                   = exec_test_operators                 ,
	dataset_name                     = exec_dataset_name                   ,
	use_catch22_flatten_ontology     = exec_use_catch22_flatten_ontology   ,
	# use_catch22                      = exec_use_catch22                    ,
	n_chunks                         = exec_n_chunks                       ,
)


dataset_function = 
	(dataset_name, n_chunks, flatten, use_catch22)->
	(
		# Multivariate_arffDataset(dataset_name; n_chunks = n_chunks, join_train_n_test = false, flatten = flatten, use_catch22 = use_catch22, mode = mode)
		Multivariate_arffDataset(dataset_name;
			n_chunks = n_chunks,
			join_train_n_test = false,
			flatten = flatten,
			use_catch22 = use_catch22,
			mode = false
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

	if dry_run == true
		continue
	end

	##############################################################################
	##############################################################################
	##############################################################################
	
	use_training_form,
	test_operators,
	dataset_name,
	(use_catch22,flatten,ontology),
	n_chunks = params_combination
	
	test_operators = test_operators_dict[test_operators]
	ontology = ontology_dict[ontology]

	cur_modal_args = modal_args
	
	cur_data_modal_args = merge(data_modal_args,
		(
			test_operators = test_operators,
			ontology = ontology,
		)
	)

	dataset_fun_sub_params = (
		dataset_name, n_chunks, flatten, use_catch22
	)

	# Load Dataset
	# dataset_function(dataset_fun_sub_params...)
	((dataset_train, class_counts_train), (dataset_test, class_counts_test)) = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function

	dataset, class_counts = concat_labeled_datasets(dataset_train, dataset_test, (class_counts_train, class_counts_test)), (class_counts_train .+ class_counts_test)

	# (X,Y) = dataset

	# matwrite("$(run_name).mat", Dict(
	# 	"X" => X,
	# 	"Y" => Y,
	# ); compress = true)

	# class_names = unique(Y)
	# npzwrite("$(run_name)-X.npy", X[1])
	# npzwrite("$(run_name)-Y.npy", map((y)->(findfirst((x)->x==y,class_names)), Y))
	# npzwrite("$(run_name)-class_names.npy", Dict([c => i for (i,c) in enumerate(class_names)]))

	println("train class_distribution: ")
	println(StatsBase.countmap(dataset_train[2]))
	println("test class_distribution: ")
	println(StatsBase.countmap(dataset_test[2]))
	println()
	println("total class_distribution: ")
	println(StatsBase.countmap(dataset[2]))
	println("class_counts: $(class_counts)")

	## Dataset slices
	# obtain dataseeds that are were not done before
	todo_dataseeds = filter((dataseed)->!iteration_in_history(history, (params_namedtuple, dataseed)), exec_dataseed)
	# dataset_slices = [(dataseed, balanced_dataset_slice(class_counts, dataseed; n_samples_per_class = class_counts)) for dataseed in todo_dataseeds]
	dataset_slices = Tuple{Int64,Tuple{Vector{Int64}, Vector{Int64}}}[(dataseed,
		if dataseed == 1
			train_idxs = Integer[]
			c=0
			for i in 1:length(class_counts)
				train_idxs = vcat(train_idxs, c+1:c+class_counts_train[i])
				c += class_counts_train[i]+class_counts_test[i]
			end
			(Vector{Integer}(collect(train_idxs)), Vector{Integer}(collect(setdiff(Set(1:sum(class_counts)), Set(train_idxs)))))
		else
			balanced_dataset_slice(class_counts, dataseed; n_samples_per_class = class_counts_train, also_return_discarted = true)
		end) for dataseed in todo_dataseeds]

	println("Dataseeds = $(todo_dataseeds)")

	for (dataseed,(train_slice,test_slice)) in dataset_slices
		println("train class_distribution: ")
		println(StatsBase.countmap(dataset[2][train_slice]))
		println("test class_distribution: ")
		println(StatsBase.countmap(dataset[2][test_slice]))
		println(train_slice)
		println(test_slice)
		println("...")
		
		# print("ROW: ")
		# print(dataset_name)
		# print("\t")
		# print(dataseed)
		# print("\t")
		# print(train_slice)
		# print("\t")
		# println(test_slice)
		break # Note: Assuming this print is the same for all dataseeds
	end
	println()
	
	if dry_run == :dataset_only
		continue
	end

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
