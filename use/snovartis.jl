################################################################################
################################################################################
################################## Scan script #################################
################################################################################
################################################################################

include("scanner.jl")
using NPZ

main_data_seed = 5

train_seed = 2

################################################################################
#################################### FOLDERS ###################################
################################################################################

results_dir = "./novartis/v4"

iteration_progress_json_file_path = results_dir * "/progress.json"
data_savedir  = results_dir * "/data_cache"
model_savedir = results_dir * "/models_cache"

dry_run = false
# dry_run = :dataset_only
# dry_run = :model_study
# dry_run = true

skip_training = false

# save_datasets = true
save_datasets = false

perform_consistency_check = false

iteration_blacklist = []

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
	for min_samples_leaf in [4] # [1,2]
		# for min_purity_increase in [100.0, 50.0, 10.0, 1.0, 0.0, 0.01, 0.001] # [0.01, 0.001]
		for min_purity_increase in [0.0] # [0.01, 0.001]
			for max_purity_at_leaf in [0.001] # [0.4, 0.6]
				push!(tree_args,
					(
						loss_function       = loss_function,
						min_samples_leaf    = min_samples_leaf,
						min_purity_increase = min_purity_increase,
						max_purity_at_leaf  = max_purity_at_leaf,
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

forest_runs = 1
# optimize_forest_computation = false
optimize_forest_computation = true


forest_args = []

# for n_trees in [50, 100]
for n_trees in [50]
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
					# max_purity_at_leaf  = Inf,
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
	initConditions = DecisionTree.startWithRelationGlob,
	# initConditions = DecisionTree.startAtCenter,
	# allowRelationGlob = true,
	allowRelationGlob = false,
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

# timing_mode = :none
timing_mode = :time
# timing_mode = :btime
#timing_mode = :profile

round_dataset_to_datatype = false
# round_dataset_to_datatype = UInt8
# round_dataset_to_datatype = UInt16
# round_dataset_to_datatype = UInt32
# round_dataset_to_datatype = UInt64
# round_dataset_to_datatype = Float16
# round_dataset_to_datatype = Float32
# round_dataset_to_datatype = Float64

n_cv_folds = 7

# split_threshold = 0.8
split_threshold = 1.0
# split_threshold = false

# use_training_form = :dimensional
# use_training_form = :fmd
# use_training_form = :stump
use_training_form = :stump_with_memoization

test_flattened = false
test_averaged  = false

prefer_nonaug_data = true

################################################################################
##################################### SCAN #####################################
################################################################################

exec_dataseed = 0:5

exec_dataset_name = ["dataset6"]
exec_num_brand = ["brand_1"] # , "brand_2"]
exec_assume_brand_independence = [false] # , true]
exec_use_diff_with_12 = [false] # , true]
exec_limit_instances = [49]

wav_preprocessors = Dict(
	"NoOp"       => identity,
	"NG"         => noise_gate!,
	"Trim"       => trim_wav!,
	"Normalize"  => normalize!,
	"RemSilence" => remove_long_silences!,
)

exec_preprocess_wavs = [
	["Normalize"],
	# ["NoOp"],
	["NG", "Normalize"],
	# ["NG", "Trim", "Normalize"],
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

exec_ontology = [ "IA", ] # "IA7", "IA3",

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


exec_ranges = (; # Order: faster-changing to slower-changing
	dataset_name                 = exec_dataset_name,
	num_brand                    = exec_num_brand,
	assume_brand_independence    = exec_assume_brand_independence,
	use_diff_with_12             = exec_use_diff_with_12,
	limit_instances              = exec_limit_instances,
	#
	test_operators               = exec_test_operators,
	ontology                     = exec_ontology,
)

dataset_function = (
	(dataset_name,
		num_brand,
		assume_brand_independence,
		use_diff_with_12,
		limit_instances,
		)->begin
		X, Y = begin
			function windowing(TD::AbstractArray{Float64, 3}, SD::AbstractArray{Float64, 2}, b_idx::NTuple{2,Int}; window_size::Int=6)
			    TD_x_size, TD_y_size, TD_z_size = size(TD) # (t,A,I)
			    SD_x_size, SD_y_size            = size(SD) # (A,I)

			    b1_idx, b2_idx = b_idx[1], b_idx[2]

			    @assert TD_z_size == SD_y_size "the number of instances must match"

			    windows = TD_x_size - window_size
			    # @show windows

			    nr_instances = TD_z_size * windows
			    # @show nr_instances

			    X_TD = Array{Float64}(undef, window_size, TD_y_size, nr_instances)
			    X_SD = Array{Float64}(undef, SD_x_size, nr_instances)
			    Y_b1 = Array{Float64}(undef, nr_instances)
			    Y_b2 = Array{Float64}(undef, nr_instances)

			    for i in 1:TD_z_size # for all old instances
			        for w in 0:windows-1 # for all windows
			            # data sets
			            X_TD[:,:,(i-1)*windows+1+w] = TD[w+1:w+window_size,:,i]
			            X_SD[:,(i-1)*windows+1+w] = SD[:,i]

			            # b1 and b2
			            Y_b1[(i-1)*windows+1+w] = TD[w+window_size+1,b1_idx,i]
			            Y_b2[(i-1)*windows+1+w] = TD[w+window_size+1,b2_idx,i]
			        end
			    end

			    return X_TD, X_SD, Y_b1, Y_b2
			end

			X_t = npzread("/home/gio/Desktop/SpatialDecisionTree/NovartisDatathon/$(dataset_name)-train-temp.npy")
			X_s = npzread("/home/gio/Desktop/SpatialDecisionTree/NovartisDatathon/$(dataset_name)-train-static.npy")

			i_b12 = 3
			index_b1, index_b2 = 1, 2

			if use_diff_with_12
				# ignore b12
				# X_t[:,i_b12,:]
				# X_t = X_t[:,[[1:i_b12-1]..., [i_b12+1:end]],:]
				X_t = X_t[:,[1, 2, 4:end...],:]
			else
				# restore b1 and b2 from b12-b1 and b12-b2
				X_t[:,index_b1,:] = X_t[:,i_b12,:] - X_t[:,index_b1,:]
				X_t[:,index_b2,:] = X_t[:,i_b12,:] - X_t[:,index_b2,:]
			end

			n_inst = size(X_s, 2)
			# println(n_inst)

			# pwd(pw)

			if assume_brand_independence
				this_brand_idx = (num_brand == "brand_1" ? index_b1 : index_b2)
				X_t = X_t[:,[this_brand_idx, 3:end...],:]
			end

			windows = windowing(X_t, X_s, (index_b1, index_b2); window_size = 6)
			
			X_t, X_s, Y_b1, Y_b2 = windows

			n_windows = size(X_s, 2)

			X_s = reshape(X_s, (1, size(X_s)...))

			perm = Random.shuffle(Random.MersenneTwister(main_data_seed), 1:n_windows)
			
			perm = perm[1:limit_instances]

			X_t  = X_t[:,:,perm]
			X_s  = X_s[:,:,perm]

			Y = begin
				if num_brand == "brand_1"
					Y_b1[perm]
				elseif num_brand == "brand_2"
					Y_b2[perm]
				else
					error("Unknown")
				end
			end

			# [X_t, X_s], Y_b1
			[X_t, X_s], Y
		end
		X, Y
	end
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

################################################################################
################################################################################
################################################################################
################################################################################

models_to_study = Dict([
	(
		"fcmel",8000,false,"stump_with_memoization",("c",3,true,"KDD-norm-partitioned-v1",["NG","Normalize","RemSilence"]),30,(max_points = 50, ma_size = 30, ma_step = 20),false,"TestOp_80","IA"
	) => [
		"tree_d3377114b972e5806a9e0631d02a5b9803c1e81d6cd6633b3dab4d9e22151969"
	],
])

models_to_study = Dict(JSON.json(k) => v for (k,v) in models_to_study)

MakeOntologicalDataset(Xs, test_operators, ontology) = begin
	MultiFrameModalDataset([
		begin
			features = FeatureTypeFun[]

			for i_attr in 1:n_attributes(X)
				for test_operator in test_operators
					if test_operator == TestOpGeq
						push!(features, ModalLogic.AttributeMinimumFeatureType(i_attr))
					elseif test_operator == TestOpLeq
						push!(features, ModalLogic.AttributeMaximumFeatureType(i_attr))
					elseif test_operator isa _TestOpGeqSoft
						push!(features, ModalLogic.AttributeSoftMinimumFeatureType(i_attr, test_operator.alpha))
					elseif test_operator isa _TestOpLeqSoft
						push!(features, ModalLogic.AttributeSoftMaximumFeatureType(i_attr, test_operator.alpha))
					else
						throw_n_log("Unknown test_operator type: $(test_operator), $(typeof(test_operator))")
					end
				end
			end

			featsnops = Vector{<:TestOperatorFun}[
				if any(map(t->isa(feature,t), [AttributeMinimumFeatureType, AttributeSoftMinimumFeatureType]))
					[≥]
				elseif any(map(t->isa(feature,t), [AttributeMaximumFeatureType, AttributeSoftMaximumFeatureType]))
					[≤]
				else
					throw_n_log("Unknown feature type: $(feature), $(typeof(feature))")
					[≥, ≤]
				end for feature in features
			]

			OntologicalDataset(X, ontology, features, featsnops)
		end for X in Xs])
end

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
n_interations = 0
n_interations_done = 0
for params_combination in IterTools.product(exec_ranges_iterators...)

	flush(logfile_io);

	# Unpack params combination
	# params_namedtuple = (zip(Symbol.(exec_ranges_names), params_combination) |> Dict |> namedtuple)
	params_namedtuple = (;zip(Symbol.(exec_ranges_names), params_combination)...)

	# FILTER ITERATIONS
	if (!is_whitelisted_test(params_namedtuple, iteration_whitelist)) || is_blacklisted_test(params_namedtuple, iteration_blacklist)
		continue
	end

	global n_interations += 1

	##############################################################################
	##############################################################################
	##############################################################################

	run_name = join([replace(string(values(value)), ", " => ",") for value in params_combination], ",")

	# Placed here so we can keep track of which iteration is being skipped
	print("Iteration \"$(run_name)\"")

	# Check whether this iteration was already computed or not
	if all(iteration_in_history(history, (params_namedtuple, dataseed)) for dataseed in exec_dataseed) && (!save_datasets)
		println(": skipping")
		continue
	else
		println("...")
	end

	global n_interations_done += 1

	if dry_run == true
		continue
	end

	##############################################################################
	##############################################################################
	##############################################################################

	dataset_name,
	num_brand,
	assume_brand_independence,
	use_diff_with_12,
	limit_instances,
	test_operators,
	ontology = params_combination

	test_operators = test_operators_dict[test_operators]
	ontology       = ontology_dict[ontology]

	cur_modal_args = modal_args

	cur_data_modal_args = merge(data_modal_args,
		(
			test_operators = test_operators,
			ontology       = ontology,
		)
	)

	dataset_fun_sub_params = (
		dataset_name,
		num_brand,
		assume_brand_independence,
		use_diff_with_12,
		limit_instances,
	)

	# if dry_run == :model_study
	# 	# println(JSON.json(params_combination))
	# 	# println(models_to_study)
	# 	# println(keys(models_to_study))
	# 	if JSON.json(params_combination) in keys(models_to_study)

	# 		trees = models_to_study[JSON.json(params_combination)]

	# 		println()
	# 		println()
	# 		println("Study models for $(params_combination): $(trees)")

	# 		if length(trees) == 0
	# 			continue
	# 		end

	# 		println("dataset_fun_sub_params: $(dataset_fun_sub_params)")

	# 		# @assert dataset_fun_sub_params isa String

	# 		# dataset_fun_sub_params = merge(dataset_fun_sub_params, (; mode = :testing))

	# 		datasets = []
	# 		println("TODO")
	# 		# datasets = [
	# 		# 	(mode,if dataset_fun_sub_params isa Tuple
	# 		# 		dataset = dataset_function(dataset_fun_sub_params...; mode = mode)
	# 		# 		# dataset = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function
	# 		# 		(X, Y), (n_pos, n_neg) = dataset
	# 		# 		# elseif dataset_fun_sub_params isa String
	# 		# 		# 	# load_cached_obj("dataset", data_savedir, dataset_fun_sub_params)
	# 		# 		# 	dataset = Serialization.deserialize("$(data_savedir)/dataset_$(dataset_fun_sub_params).jld").train_n_test
	# 		# 		# 	println(typeof(dataset))
	# 		# 		# 	(X, Y), (n_pos, n_neg) = dataset
	# 		# 		# 	(X, Y, nothing), (n_pos, n_neg)

	# 		# 		# TODO should not need these at test time. Instead, extend functions so that one can use a MatricialDataset instead of an OntologicalDataset
	# 		# 		X = MakeOntologicalDataset(X, test_operators, ontology)
	# 		# 		# println(length(Y))
	# 		# 		# println((n_pos, n_neg))

	# 		# 		println(display_structure(X))
	# 		# 		# println(Y)
	# 		# 		dataset = (X, Y), (n_pos, n_neg)
	# 		# 		dataset
	# 		# 	else
	# 		# 		throw_n_log("$(typeof(dataset_fun_sub_params))")
	# 		# 	end) for mode in [:testing, :development]
	# 		# ]

	# 		for model_hash in trees

	# 			println()
	# 			println()
	# 			println("Loading model: $(model_hash)...")

	# 			model = load_model(model_hash, model_savedir)

	# 			println()
	# 			println("Original model (training):")
	# 			if model isa DTree
	# 				print_model(model)
	# 			end

	# 			for (mode,dataset) in datasets
	# 				(X, Y), (n_pos, n_neg) = dataset

	# 				println()

	# 				println()
	# 				println("Regenerated model ($(mode)):")

	# 				if model isa DTree
	# 					regenerated_model = print_apply_model(model, X, Y; print_relative_confidence = true)
	# 					println()
	# 					# print_model(regenerated_model)
	# 				end

	# 				preds = apply_model(model, X);
	# 				cm = confusion_matrix(Y, preds)
	# 				println(cm)

	# 				# readline()
	# 			end
	# 		end
	# 	end
	# end

	# Load Dataset
	dataset = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function

	## Dataset slices
	# obtain dataseeds that are were not done before
	todo_dataseeds = filter((dataseed)->!iteration_in_history(history, (params_namedtuple, dataseed)), exec_dataseed)

	X, Y = dataset
	
	dataset_slices = begin
		n_insts = length(Y)
		@assert (n_insts % n_cv_folds == 0) "$(n_insts) % $(n_cv_folds) != 0"
		n_insts_fold = div(n_insts, n_cv_folds)
		# todo_dataseeds = 1:10
		[(dataseed, begin
				if dataseed == 0
					(Vector{Integer}(collect(1:n_insts)), Vector{Integer}(collect(1:n_insts)))
				else
					test_idxs = 1+(dataseed-1)*n_insts_fold:(dataseed-1)*n_insts_fold+(n_insts_fold)
					(Vector{Integer}(collect(setdiff(Set(1:n_insts), Set(test_idxs)))), Vector{Integer}(collect(test_idxs)))
				end
			end) for dataseed in todo_dataseeds]
	end

	println("Dataseeds = $(todo_dataseeds)")

	if dry_run == :dataset_only
		continue
	end

	##############################################################################
	##############################################################################
	##############################################################################

	if dry_run == false
		exec_scan(
			params_namedtuple,
			dataset;
			is_regression_problem           = true,
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
			# logger                          =   logger,
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
end

println("Done!")
println("# Iterations $(n_interations_done)/$(n_interations)")

# Notify the Telegram Bot
@error "Done!"

close(logfile_io);

exit(0)
