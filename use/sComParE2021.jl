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

results_dir = "./ComParE2021/september-v3TODO"

iteration_progress_json_file_path =  "$(results_dir)/progress.json"
data_savedir  = "$(results_dir)/cache"
model_savedir = "$(results_dir)/trees"


dry_run = false
#dry_run = :dataset_only
# dry_run = :model_study
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

# for loss_function in [DecisionTree.util.entropy]
for (loss_function, max_purity_at_leaf) in [(DecisionTree.util.entropy, 0.6), (DecisionTree.util.entropy, 0.7), (DecisionTree.util.gini, 0.3), (DecisionTree.util.gini, 0.4)]
	for min_samples_leaf in [2,4] # [1,2]
		for min_purity_increase in [0.01] # [0.01, 0.001]
			# for max_purity_at_leaf in [0.4, 0.5, 0.6] # [0.4, 0.6]
				push!(tree_args, 
					(
						loss_function       = loss_function,
						min_samples_leaf    = min_samples_leaf,
						min_purity_increase = min_purity_increase,
						max_purity_at_leaf  = max_purity_at_leaf,
						perform_consistency_check = perform_consistency_check,
					)
				)
			# end
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

for n_trees in [50,100]
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
					min_samples_leaf    = 1,
					min_purity_increase = 0.0,
					max_purity_at_leaf  = Inf,
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

exec_dataseed = 1:10

# max_sample_rate = 48000
max_sample_rate = 16000
# max_sample_rate = 8000


# exec_use_training_form = [:dimensional]
exec_use_training_form = [:stump_with_memoization]

exec_nbands = [40] # [20,40,60]

exec_dataset_kwargs =   [( # TODO
						#	max_points = 10,
						#	ma_size = 45,
						#	ma_step = 30,
						#),(max_points = 20,
							# ma_size = 45,
							# ma_step = 30,
						# ),(max_points = 20,
						# 	ma_size = 45,
						# 	ma_step = 30,
						# ),(max_points = 30,
						# 	ma_size = 45,
						# 	ma_step = 30,
						# ),(# max_points = 30,
							ma_size = 120,
							ma_step = 100,
						#),(# max_points = 30,
						# 	ma_size = 120,
						# 	ma_step = 80,
						# ),(# max_points = 30,
						#	ma_size = 100,
						#	ma_step = 75,
						#),(# max_points = 30,
						# 	ma_size = 90,
						# 	ma_step = 60,
						# ),(# max_points = 30,
							# ma_size = 75,
							# ma_step = 50,
						# ),(# max_points = 50,
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
	maxfreq = max_sample_rate/2, # Fix this
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
	maxfreq = max_sample_rate/2, # Fix this
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
	["NG", "Normalize"],
	["Normalize"],
	[],
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


exec__2D_or_3D_ontology = [(false,"Interval")] # , (true,"Interval2D")]

ontology_dict = Dict(
	"OneWorld"    => ModalLogic.OneWorldOntology,
	"Interval"    => getIntervalOntologyOfDim(Val(1)),
	"Interval2D"  => getIntervalOntologyOfDim(Val(2)),
)

exec_include_static_data = [true, false] #, true]

exec_use_lowSR = [true, false]

exec_ranges = (;
	_2D_or_3D_ontology   = exec__2D_or_3D_ontology ,
	include_static_data  = exec_include_static_data,
	use_lowSR            = exec_use_lowSR          ,
	dataset_kwargs       = exec_dataset_kwargs     ,
	preprocess_wavs      = exec_preprocess_wavs    ,
	use_full_mfcc        = exec_use_full_mfcc      ,
	use_training_form    = exec_use_training_form  ,
	nbands               = exec_nbands             ,
	test_operators       = exec_test_operators     ,
)


dataset_function =
	(_2D_or_3D,
		include_static_data,
		cur_audio_kwargs,
		use_lowSR,
		dataset_kwargs,
		cur_preprocess_wavs,
		use_full_mfcc; mode = :development)->
	ComParE2021Dataset(;
		subchallenge = "CCS",
		use_lowSR = use_lowSR,
		mode = mode,
		include_static_data = include_static_data,
		treat_as_single_attribute_2D_context = _2D_or_3D,
		#
		audio_kwargs = cur_audio_kwargs,
		dataset_kwargs...,
		preprocess_wavs = cur_preprocess_wavs,
		use_full_mfcc = use_full_mfcc
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


models_to_study = Dict([
	# ("926c7c1e917236847f99e052f4785eae737f210094891863866bb7afe45eb732",

	(
		(false,"Interval"),true,true,(ma_size = 120, ma_step = 100),["Normalize"],false,"stump_with_memoization",40,"TestOp_80"
	) => [
		"rf_749066dbac8f05d768998d03a1b5af86f9350d92fdbbe046f9a3996c7feb237e"
	],
	# (
	# 	(false,"Interval"),true,true,(ma_size = 120, ma_step = 100),["NG","Normalize"],false,"stump_with_memoization",40,"TestOp_80"
	# ) => [
	# "tree_133e11b8405703a7e9bb988fa6e8e117dddbee4d28279c2cc3eea77a50df506c",
	# "tree_ba85456f353973bc747807a5fabc672389c6840bd9faa311bea7288a266072c1",
	# "tree_993d27f4e86aca00a941e2dead6983b6f9344af36c91a3ac6446b624d235bcc2",
	# "rf_6daa56e0812a6c8ed2553ccba3a84730dd36491a18f958824236f68a2b584dc5",
	# "rf_1ffd65156589150a6501f6fb0a44f1a0ab393221228bff02e5dd78db8d3a2db2",
	# ],
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
mkpath(model_savedir)

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
	
	(_2D_or_3D,ontology),
	include_static_data,
	use_lowSR,
	dataset_kwargs,
	preprocess_wavs,
	use_full_mfcc,
	use_training_form,
	nbands,
	test_operators = params_combination
	
	ontology = ontology_dict[ontology]
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
			ontology       = ontology,
			test_operators = test_operators,
		)
	)

	dataset_fun_sub_params = (
		_2D_or_3D,
		include_static_data,
		cur_audio_kwargs,
		use_lowSR,
		dataset_kwargs,
		cur_preprocess_wavs,
		use_full_mfcc,
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

			datasets = [
				(mode,if dataset_fun_sub_params isa Tuple
					dataset = dataset_function(dataset_fun_sub_params...; mode = mode)
					# dataset = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function
					(X, Y), (n_pos, n_neg) = dataset
					# elseif dataset_fun_sub_params isa String
					# 	# load_cached_obj("dataset", data_savedir, dataset_fun_sub_params)
					# 	dataset = Serialization.deserialize("$(data_savedir)/dataset_$(dataset_fun_sub_params).jld").train_n_test
					# 	println(typeof(dataset))
					# 	(X, Y), (n_pos, n_neg) = dataset
					# 	(X, Y, nothing), (n_pos, n_neg)

					# TODO should not need these at test time. Instead, extend functions so that one can use a MatricialDataset instead of an OntologicalDataset
					X = MakeOntologicalDataset(X, test_operators, ontology)
					# println(length(Y))
					# println((n_pos, n_neg))

					println(display_structure(X))
					# println(Y)
					dataset = (X, Y), (n_pos, n_neg)
					dataset
				else
					throw_n_log("$(typeof(dataset_fun_sub_params))")
				end) for mode in [:testing, :development]
			]

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
	# dataset_function(dataset_fun_sub_params...)
	dataset, n_label_samples = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function

	if dry_run == :dataset_only
		continue
	end

	## Dataset slices
	# obtain dataseeds that are were not done before
	todo_dataseeds = filter((dataseed)->!iteration_in_history(history, (params_namedtuple, dataseed)), exec_dataseed)
	dataset_slices = [(dataseed,balanced_dataset_slice(n_label_samples, dataseed)) for dataseed in todo_dataseeds]

	println("Dataseeds = $(todo_dataseeds)")

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

end

println("Done!")

exit(0)
