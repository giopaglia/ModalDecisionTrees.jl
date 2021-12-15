################################################################################
################################################################################
################################## Scan script #################################
################################################################################
################################################################################

include("scanner.jl")

using NEVArt

train_seed = 2

################################################################################
#################################### FOLDERS ###################################
################################################################################

results_dir = "./NEVArt/journal-v1"

iteration_progress_json_file_path = results_dir * "/progress.json"
data_savedir  = results_dir * "/data_cache"
model_savedir = results_dir * "/models_cache"

# dry_run = false
dry_run = :dataset_only
# dry_run = :model_study
# dry_run = true

skip_training = false

# save_datasets = true
save_datasets = false

perform_consistency_check = false # true #  = false

iteration_blacklist = []

################################################################################
##################################### TREES ####################################
################################################################################

# Optimization arguments for single-tree
tree_args = [
#	(
#		loss_function = nothing,
#		min_samples_leaf = 1,
#		min_purity_increase = 0.01,
#		max_purity_at_leaf = 0.6,
#	)
]

for loss_function in [nothing]
	for min_samples_leaf in [2,4] # [1,2]
		for min_purity_increase in [0.01] # [0.01, 0.001]
			for max_purity_at_leaf in [0.4, 0.5, 0.6] # [0.4, 0.6]
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

forest_runs = 5
# optimize_forest_computation = false
optimize_forest_computation = true


forest_args = []

for n_trees in [50]
# for n_trees in [50]
	for n_subfeatures in [half_f]
		for n_subrelations in [id_f]
			for partial_sampling in [0.7]
				push!(forest_args, (
					n_subfeatures       = n_subfeatures,
					n_trees             = n_trees,
					partial_sampling    = partial_sampling,
					n_subrelations      = n_subrelations,
					# Optimization arguments for trees in a forest (no pruning is performed)
					loss_function       = nothing,
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

split_threshold = 0.8
# split_threshold = 1.0
# split_threshold = false

# use_training_form = :dimensional
# use_training_form = :fmd
# use_training_form = :stump
# use_training_form = :stump_with_memoization

test_flattened = false
test_averaged  = false

################################################################################
##################################### SCAN #####################################
################################################################################

exec_dataseed = 1:n_cv_folds

# exec_use_training_form = [:dimensional]
exec_use_training_form = [:stump_with_memoization]

EEG_default = (nbands = 60, wintime = 0.05, steptime = 0.025)
ECG_default = (nbands = 30, wintime = 0.025, steptime = 0.0175)

exec_dataset_params = [
	# (ids,signals,lables,static_attrs,signal_transformation,keep_only_bands,force_single_frame)
	("sure-v1",[:EEG],["liked"],String[],Dict{Symbol,NamedTuple}(:EEG => EEG_default),Dict{Symbol,Vector{Int64}}(:EEG => collect(1:25)),false)
	("sure-v1",[:ECG],["liked"],String[],Dict{Symbol,NamedTuple}(:ECG => ECG_default),Dict{Symbol,Vector{Int64}}(:ECG => collect(1:7)),false)
	("sure-v1",[:EEG,:ECG],["liked"],String[],Dict{Symbol,NamedTuple}(:EEG => EEG_default, :ECG => ECG_default),Dict{Symbol,Vector{Int64}}(:EEG => collect(1:25), :ECG => collect(1:7)),false)
	("sure-v1",[:EEG,:ECG],["liked"],String[],Dict{Symbol,NamedTuple}(:EEG => EEG_default, :ECG => ECG_default),Dict{Symbol,Vector{Int64}}(:EEG => collect(1:25), :ECG => collect(1:7)),true)
]

const datasets = Dict{String,Vector{Int64}}(
	"sure-v1" => sure_dataset_ids
)

exec_aggr_points = [5, 10, 15, 20]
exec_length = ["1/4", "2/4", "3/4", "4/4"]

length_dict = Dict{String,Function}(
	"1/4" => x -> max(1, floor(Int64, size(x, 1) * 0.25)),
	"2/4" => x -> max(1, floor(Int64, size(x, 1) * 0.5)),
	"3/4" => x -> max(1, floor(Int64, size(x, 1) * 0.75)),
	"4/4" => x -> size(x, 1)
)
cut_length(X::AbstractArray{T,3} where T, l::Integer) = X[collect(1:l),:,:]
cut_length(X::AbstractArray{T,3} where T, l::AbstractString) = cut_length(X, length_dict[l](X))
cut_length(X::AbstractArray, l) = X

aggr_points(X::AbstractArray, n::Integer) = X
function aggr_points(X::AbstractArray{T,3}, n::Integer) where T
	chunksize = min(ceil(Int64, size(X, 1) / n), 1)

	res = Array{T,3}(undef, (n, size(X, 2), size(X, 3)))

	for i in 0:(n-1)
		for j in 1:size(X, 2)
			for k in 1:size(X, 3)
				left = (i * chunksize) + 1
				right = i == n-1 ? size(X, 1) : (i+1) * chunksize
				println(left, " ", right)
				res[i+1,j,k] = mean(X[collect(left:right),j,k])
			end
		end
	end

	return res
end

# https://github.com/JuliaIO/JSON.jl/issues/203
# https://discourse.julialang.org/t/json-type-serialization/9794
# TODO: make test operators types serializable
# exec_canonical_features = [ "TestOp" ]
exec_canonical_features = [ "TestOp_80" ]

canonical_features_dict = Dict(
	"TestOp_70" => [ModalLogic.TestOpGeq_70, ModalLogic.TestOpLeq_70],
	"TestOp_80" => [ModalLogic.TestOpGeq_80, ModalLogic.TestOpLeq_80],
	"TestOp"    => [ModalLogic.TestOpGeq,    ModalLogic.TestOpLeq],
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
	exec_aggr_points     = exec_aggr_points,
	exec_length          = exec_length,
	exec_dataset_params  = exec_dataset_params,
	use_training_form    = exec_use_training_form,
	canonical_features   = exec_canonical_features,
	ontology             = exec_ontology,
)

function dataset_function(
	dataset_name::AbstractString,
	signals::AbstractVector{Symbol},
	labels::AbstractVector{<:AbstractString},
	static_attrs::AbstractVector{<:AbstractString},
	signal_transformation::Dict{Symbol,<:NamedTuple},
	keep_only_bands::Dict{Symbol,<:AbstractVector{<:Integer}},
	force_single_frame::Bool
)
	return NEVArtDataset(
		"$(data_dir)/NEVArt";
		ids = datasets[dataset_name],
		signals = signals,
		labels = labels,
		static_attrs = static_attrs,
		mode = :painting,
		apply_transfer_function = true,
		normalize_after_transfer_function = true,
		forget_samplerate = false,
	    signal_transformation = signal_transformation,
		keep_only_bands = keep_only_bands,
		return_type = :Matricial, # :MFD, :DataFrame
		force_single_frame = force_single_frame
	)
end

################################################################################
################################### SCAN FILTERS ###############################
################################################################################

# TODO let iteration_white/blacklist a decision function and not a "in-array" condition?
iteration_whitelist = []

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

MakeOntologicalDataset(Xs, canonical_features, ontology) = begin
	MultiFrameModalDataset([
		begin
			features = FeatureTypeFun[]

			for i_attr in 1:n_attributes(X)
				for test_operator in canonical_features
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

	curr_aggr_points,
	curr_length_fraction,
	(dataset_name,signals,lables,static_attrs,signal_transformation,keep_only_bands,force_single_frame),
	use_training_form,
	canonical_features,
	ontology = params_combination

	canonical_features = canonical_features_dict[canonical_features]
	ontology       = ontology_dict[ontology]

	cur_modal_args = modal_args

	cur_data_modal_args = merge(data_modal_args,
		(
			canonical_features = canonical_features,
			ontology       = ontology,
		)
	)

	dataset_fun_sub_params = (
		dataset_name,
		signals,
		lables,
		static_attrs,
		signal_transformation,
		keep_only_bands,
		force_single_frame
	)

	if dry_run == :model_study
		# println(JSON.json(params_combination))
		# println(models_to_study)
		# println(keys(models_to_study))
		if JSON.json(params_combination) in keys(models_to_study)

			trees = models_to_study[JSON.json(params_combination)]

			println()
			println()
			println("Study models for $(params_combination): $(trees)")

			if length(trees) == 0
				continue
			end

			println("dataset_fun_sub_params: $(dataset_fun_sub_params)")

			# @assert dataset_fun_sub_params isa String

			# dataset_fun_sub_params = merge(dataset_fun_sub_params, (; mode = :testing))

			datasets = []
			println("TODO")
			# datasets = [
			# 	(mode,if dataset_fun_sub_params isa Tuple
			# 		dataset = dataset_function(dataset_fun_sub_params...; mode = mode)
			# 		# dataset = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function
			# 		(X, Y), (n_pos, n_neg) = dataset
			# 		# elseif dataset_fun_sub_params isa String
			# 		# 	# load_cached_obj("dataset", data_savedir, dataset_fun_sub_params)
			# 		# 	dataset = Serialization.deserialize("$(data_savedir)/dataset_$(dataset_fun_sub_params).jld").train_n_test
			# 		# 	println(typeof(dataset))
			# 		# 	(X, Y), (n_pos, n_neg) = dataset
			# 		# 	(X, Y, nothing), (n_pos, n_neg)

			# 		# TODO should not need these at test time. Instead, extend functions so that one can use a MatricialDataset instead of an OntologicalDataset
			# 		X = MakeOntologicalDataset(X, canonical_features, ontology)
			# 		# println(length(Y))
			# 		# println((n_pos, n_neg))

			# 		println(display_structure(X))
			# 		# println(Y)
			# 		dataset = (X, Y), (n_pos, n_neg)
			# 		dataset
			# 	else
			# 		throw_n_log("$(typeof(dataset_fun_sub_params))")
			# 	end) for mode in [:testing, :development]
			# ]

			for model_hash in trees

				println()
				println()
				println("Loading model: $(model_hash)...")

				model = load_model(model_hash, model_savedir)

				println()
				println("Original model (training):")
				if model isa DTree
					print_model(model)
				end

				for (mode,dataset) in datasets
					(X, Y), (n_pos, n_neg) = dataset

					println()

					println()
					println("Regenerated model ($(mode)):")

					if model isa DTree
						regenerated_model = print_apply_model(model, X, Y; print_relative_confidence = true)
						println()
						# print_model(regenerated_model)
					end

					preds = apply_model(model, X);
					cm = confusion_matrix(Y, preds)
					println(cm)

					# readline()
				end
			end
		end
	end

	# Load Dataset
	dataset = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function

	######### aggregate points
	# 1) cut length
	dataset = (cut_length(dataset[1], curr_length_fraction), dataset[2])
	# 2) real aggregation
	dataset = (aggr_points(dataset[1], curr_aggr_points), dataset[2])

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

	for ds in dataset_slices
		println(ds)
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
