################################################################################
################################################################################
################################## Scan script #################################
################################################################################
################################################################################

include("scanner.jl")


using Catch22
using DataStructures

include("dataset-analysis.jl")

train_seed = 1


################################################################################
#################################### FOLDERS ###################################
################################################################################

results_dir = "./siemens/TURBOEXPO-regression-v8-analysis-first"

iteration_progress_json_file_path = results_dir * "/progress.json"
data_savedir  = results_dir * "/data_cache"
model_savedir = results_dir * "/models_cache"

dry_run = false
#dry_run = :dataset_only
# dry_run = :model_study
# dry_run = true

skip_training = false
# skip_training = true

#save_datasets = true
save_datasets = false

perform_consistency_check = false # true

iteration_blacklist = []

################################################################################
##################################### TREES ####################################
################################################################################

# Optimization arguments for single-tree
tree_args = [
#	(
#		loss_function = nothing, # DecisionTree.util.entropy
#		min_samples_leaf = 1,
#		min_purity_increase = 0.01,
#		max_purity_at_leaf = 0.6,
#	)
]

for loss_function in [nothing] # DecisionTree.util.variance
	for min_samples_leaf in [4] # [1,2]
		for min_purity_increase in [0.1, 0.02, 0.015, 0.01, 0.0075, 0.005] # , 0.001, 0.0001, 0.00005, 0.00002, 0.00001, 0.0] # [0.01, 0.001]
		# for min_purity_increase in [0.0] # ,0.01, 0.001]
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

forest_runs = 5
optimize_forest_computation = true


forest_args = []

for n_trees in []
# for n_trees in [50, 100]
	for n_subfeatures in [half_f]
		for n_subrelations in [id_f]
			for partial_sampling in [0.7]
				push!(forest_args, (
					n_subfeatures       = n_subfeatures,
					n_trees             = n_trees,
					partial_sampling    = partial_sampling,
					n_subrelations      = n_subrelations,
					# Optimization arguments for trees in a forest (no pruning is performed)
					loss_function       = nothing, # DecisionTree.util.entropy
					# min_samples_leaf    = 1,
					# min_purity_increase = ...,
					# max_purity_at_leaf  = ..,
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
	ontology = getIntervalOntologyOfDim(Val(1)),
	# ontology = Ontology{ModalLogic.Interval}([ModalLogic.IA_A]),
	# ontology = Ontology{ModalLogic.Interval}([ModalLogic.IA_A, ModalLogic.IA_L, ModalLogic.IA_Li, ModalLogic.IA_D]),
	canonical_features	 = [CanonicalFeatureGeq_80, CanonicalFeatureLeq_80],
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

traintest_threshold = 1.0
# train_instances_per_class = 100

split_threshold = 0.0
# split_threshold = 0.8
# split_threshold = 1.0
# split_threshold = false

# use_training_form = :dimensional
# use_training_form = :fmd
# use_training_form = :stump
# use_training_form = :stump_with_memoization

test_flattened = false
test_averaged  = false

prefer_nonaug_data = true

################################################################################
##################################### SCAN #####################################
################################################################################

exec_dataseed = [1:3...,0]

#exec_datadirname = ["Siemens-Data-Features", "Siemens-Data-Measures"]
exec_datadirname = ["Siemens-Data-Measures"]

exec_use_training_form = [:stump_with_memoization]

exec_binning = [
	# nothing,
	# (1.0 => "High-Risk", 3.0 => "Risky", Inf => "Low-Risk"),
	# (4.0 => "High-Risk", 8.0 => "Risky", Inf => "Low-Risk"),
	(4.0 => "High-Risk", Inf => "Low-Risk"),
]

exec_ignore_last_minutes = [10] # , 2*60]
exec_regression_step_in_minutes = [10]
exec_regression_window_in_minutes = [60]

exec_moving_average = [
	# (
	# 	ma_size = 15,
	# 	ma_step = 15,
	# ),
	# (
	# 	ma_size = 12,
	# 	ma_step = 12,
	# ),
	# (
	# 	ma_size = 10,
	# 	ma_step = 10,
	# ),
	# # (
	# # 	ma_size = 6,
	# # 	ma_step = 6,
	# # ),
	# # (
	# # 	ma_size = 5,
	# # 	ma_step = 5,
	# # ),
	# (
	# 	ma_size = 4,
	# 	ma_step = 4,
	# ),
	# (
	# 	ma_size = 3,
	# 	ma_step = 3,
	# ),
	# (
	# 	ma_size = 2,
	# 	ma_step = 2,
	# ),
	(
		ma_size = 1,
		ma_step = 1,
	),
]

# exec_ignore_datasources = [[42894]]
exec_ignore_datasources = [[], [42894]]

exec_ranges = (;
	datadirname                                  = exec_datadirname,
	use_training_form                            = exec_use_training_form,
	binning                                      = exec_binning,
	ignore_last_minutes                          = exec_ignore_last_minutes,
	moving_average                               = exec_moving_average,
	regression_window_in_minutes                 = exec_regression_window_in_minutes,
	regression_step_in_minutes                   = exec_regression_step_in_minutes,
	ignore_datasources                           = exec_ignore_datasources,
)


dataset_function = (datadirname, binning, ignore_last_minutes, moving_average, regression_window_in_minutes, regression_step_in_minutes, ignore_datasources)->begin
	SiemensDataset_regression(datadirname;
		moving_average...,
		binning = binning,
		sortby_datasource = true,
		only_consider_trip_days = true,
		ignore_last_minutes = ignore_last_minutes,
		select_attributes = [1,4,3,23,2,5,22,25,26,(6:21)...],
		regression_window_in_minutes = regression_window_in_minutes,
		regression_step_in_minutes = regression_step_in_minutes,
		ignore_datasources = ignore_datasources,
	)
end

################################################################################
################################### SCAN FILTERS ###############################
################################################################################

iteration_whitelist = []

################################################################################
################################################################################
################################################################################
################################################################################

models_to_study = Dict([
	# (
	# 	"fcmel",8000,false,"stump_with_memoization",("c",3,true,"KDD-norm-partitioned-v1",["NG","Normalize","RemSilence"]),30,(max_points = 50, ma_size = 30, ma_step = 20),false,"TestOp_80","IA"
	# ) => [
	# 	"tree_d3377114b972e5806a9e0631d02a5b9803c1e81d6cd6633b3dab4d9e22151969"
	# ],
])

models_to_study = Dict(JSON.json(k) => v for (k,v) in models_to_study)

# MakeOntologicalDataset(Xs, test_operators, ontology) = begin
# 	MultiFrameModalDataset([
# 		begin
# 			features = FeatureTypeFun[]

# 			for i_attr in 1:n_attributes(X)
# 				for test_operator in test_operators
# 					if test_operator == TestOpGeq
# 						push!(features, ModalLogic.AttributeMinimumFeatureType(i_attr))
# 					elseif test_operator == TestOpLeq
# 						push!(features, ModalLogic.AttributeMaximumFeatureType(i_attr))
# 					elseif test_operator isa _TestOpGeqSoft
# 						push!(features, ModalLogic.AttributeSoftMinimumFeatureType(i_attr, test_operator.alpha))
# 					elseif test_operator isa _TestOpLeqSoft
# 						push!(features, ModalLogic.AttributeSoftMaximumFeatureType(i_attr, test_operator.alpha))
# 					else
# 						throw_n_log("Unknown test_operator type: $(test_operator), $(typeof(test_operator))")
# 					end
# 				end
# 			end

# 			featsnops = Vector{<:TestOperatorFun}[
# 				if any(map(t->isa(feature,t), [AttributeMinimumFeatureType, AttributeSoftMinimumFeatureType]))
# 					[≥]
# 				elseif any(map(t->isa(feature,t), [AttributeMaximumFeatureType, AttributeSoftMaximumFeatureType]))
# 					[≤]
# 				else
# 					throw_n_log("Unknown feature type: $(feature), $(typeof(feature))")
# 					[≥, ≤]
# 				end for feature in features
# 			]

# 			OntologicalDataset(X, ontology, features, featsnops)
# 		end for X in Xs])
# end

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
	
	datadirname, use_training_form, binning, ignore_last_minutes, moving_average, regression_window_in_minutes, regression_step_in_minutes, ignore_datasources = params_combination
	dataset_fun_sub_params = (datadirname, binning, ignore_last_minutes, moving_average, regression_window_in_minutes, regression_step_in_minutes, ignore_datasources)
	
	cur_modal_args = deepcopy(modal_args)
	cur_data_modal_args = deepcopy(data_modal_args)

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
			# 		X = MakeOntologicalDataset(X, test_operators, ontology)
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
	(dataset, datasource_counts), attribute_names = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function
	
	# (dataset, datasource_counts), attribute_names = Serialization.deserialize("./siemens/TURBOEXPO-regression-v6-back-to-classification/data_cache/dataset_974bfd893a83f72736de7c87703bdae6c97250b34ffe1801231e306acb98798b.jld")

	# class_counts = get_class_counts(dataset)

	X, Y = dataset

	##############################################################################
	##############################################################################
	##############################################################################

	# X, Y = randn(19,5,20), rand(0:1, 20)
	p = nothing
	for f in [1]
		# f = [1][1]

		n_desired_attributes = 5
		n_desired_features   = 5

		savefigs = true
		# savefigs = false

		grouped_descriptors = OrderedDict([
			"Basic stats" => [
				:mean_m
				:min_m
				:max_m
			], "Distribution" => [
				:DN_HistogramMode_5
				:DN_HistogramMode_10
			], "Simple temporal statistics" => [
				:SB_BinaryStats_mean_longstretch1
				:DN_OutlierInclude_p_001_mdrmd
				:DN_OutlierInclude_n_001_mdrmd
			], "Linear autocorrelation" => [
				:CO_f1ecac
				:CO_FirstMin_ac
				:SP_Summaries_welch_rect_area_5_1
				:SP_Summaries_welch_rect_centroid
				:FC_LocalSimple_mean3_stderr
			], "Nonlinear autocorrelation" => [
				:CO_trev_1_num
				:CO_HistogramAMI_even_2_5
				:IN_AutoMutualInfoStats_40_gaussian_fmmi
			], "Successive differences" => [
				:MD_hrv_classic_pnn40
				:SB_BinaryStats_diff_longstretch0
				:SB_MotifThree_quantile_hh
				:FC_LocalSimple_mean1_tauresrat
				:CO_Embed2_Dist_tau_d_expfit_meandiff
			], "Fluctuation Analysis" => [
				:SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1
				:SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1
			], "Others" => [
				:SB_TransitionMatrix_3ac_sumdiagcov
				:PD_PeriodicityWang_th0_01
			],
		])
		
		descriptor_abbrs = Dict([
		 	##########################################################################
		 	:mean_m                                        => "M",
		 	:max_m                                         => "MAX",
		 	:min_m                                         => "MIN",
		 	##########################################################################
		 	:DN_HistogramMode_5                            => "Z5",
		 	:DN_HistogramMode_10                           => "Z10",
		 	##########################################################################
		 	:SB_BinaryStats_mean_longstretch1              => "C",
		 	:DN_OutlierInclude_p_001_mdrmd                 => "A",
		 	:DN_OutlierInclude_n_001_mdrmd                 => "B",
		 	##########################################################################
			:CO_f1ecac                                     => "FC",
			:CO_FirstMin_ac                                => "FM",
			:SP_Summaries_welch_rect_area_5_1              => "TP",
			:SP_Summaries_welch_rect_centroid              => "CE",
			:FC_LocalSimple_mean3_stderr                   => "ME",
		 	##########################################################################
			:CO_trev_1_num                                 => "TR",
			:CO_HistogramAMI_even_2_5                      => "AI",
			:IN_AutoMutualInfoStats_40_gaussian_fmmi       => "FMAI",
		 	##########################################################################
			:MD_hrv_classic_pnn40                          => "PD",
			:SB_BinaryStats_diff_longstretch0              => "LP",
			:SB_MotifThree_quantile_hh                     => "EN",
			:FC_LocalSimple_mean1_tauresrat                => "CC",
			:CO_Embed2_Dist_tau_d_expfit_meandiff          => "EF",
		 	##########################################################################
			:SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1        => "FDFA",
			:SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1   => "FLF",
		 	##########################################################################
			:SB_TransitionMatrix_3ac_sumdiagcov            => "TC",
			:PD_PeriodicityWang_th0_01                     => "PM",
		 	##########################################################################
		 ])

		_attribute_abbrs = Dict([
		  "Ambient_air_humidity"                         => "AMB_H",
		  "Ambient_air_temperature"                      => "AMB_T",
		  "Gas_fuel_valve_position"                      => "GAS_P",
		  "Gas_fuel_mass_flow_rate"                      => "GAS_F",
		  "Compr_IGV_position"                           => "IGV_P",
		  "Compressor_outlet_temperature"                => "IGV_T",
		  "Compressor_outlet_pressure"                   => "IGV_T",
		  "Rotational_speed"                             => "SPEED",
		  "Power_output"                                 => "POWER",
		  "Exhaust_temperature_1"                        => "EX_T1",
		  "Exhaust_temperature_2"                        => "EX_T2",
		  "Exhaust_temperature_3"                        => "EX_T3",
		  "Exhaust_temperature_4"                        => "EX_T4",
		  "Exhaust_temperature_5"                        => "EX_T5",
		  "Exhaust_temperature_6"                        => "EX_T6",
		  "Exhaust_temperature_7"                        => "EX_T7",
		  "Exhaust_temperature_8"                        => "EX_T8",
		  "Exhaust_temperature_9"                        => "EX_T9",
		  "Exhaust_temperature_10"                       => "EX_T10",
		  "Exhaust_temperature_11"                       => "EX_T11",
		  "Exhaust_temperature_12"                       => "EX_T12",
		  "Exhaust_temperature_13"                       => "EX_T13",
		  "Exhaust_temperature_14"                       => "EX_T14",
		  "Exhaust_temperature_15"                       => "EX_T15",
		  "Exhaust_temperature_16"                       => "EX_T16",
		 ])

		attribute_abbrs = [_attribute_abbrs[attr_name] for attr_name in attribute_names]

		run_file_prefix = "$(results_dir)/plotdescription-$(run_name)"
		run_file_prefix = replace(run_file_prefix, "\"" => "")
		best_attributes_idxs, best_descriptors = [1,2,8,14,13], [:min_m, :CO_FirstMin_ac, :max_m, :mean_m, :SB_BinaryStats_mean_longstretch1]
		# best_attributes_idxs, best_descriptors = single_frame_blind_feature_selection(
		# 	(X,Y),
		# 	attribute_names,
		# 	grouped_descriptors,
		# 	run_file_prefix,
		# 	n_desired_attributes,
		# 	n_desired_features;
		# 	savefigs = savefigs,
		# 	descriptor_abbrs = descriptor_abbrs,
		# 	attribute_abbrs = attribute_abbrs,
		# 	export_csv = true,
		# 	# join_plots = [],
		# )
		
		# if savefigs
		# 	descriptors = collect(Iterators.flatten([values(grouped_descriptors)...]))
		# 	single_frame_target_aware_analysis((X,Y), attribute_names, descriptors, run_file_prefix; savefigs = savefigs, descriptor_abbrs = descriptor_abbrs, attribute_abbrs = attribute_abbrs, export_csv = true)
		# end

		X = X[:,best_attributes_idxs,:]
		Y = Y[:]

		if savefigs
			single_frame_target_aware_analysis(
				(X,Y),
				attribute_names[best_attributes_idxs],
				best_descriptors,
				run_file_prefix*"-sub";
				savefigs = savefigs,
				descriptor_abbrs = descriptor_abbrs,
				attribute_abbrs = attribute_abbrs[best_attributes_idxs],
				export_csv = true,
			)
		end
		
		# println(typeof(p))
		# println(length(p))

		# catch22_min_channel_length = Dict([
		# 	:DN_HistogramMode_5                            => 3,
		# 	:DN_HistogramMode_10                           => 3,
		# 	:CO_Embed2_Dist_tau_d_expfit_meandiff          => 3,
		# 	:CO_f1ecac                                     => 3,
		# 	:CO_FirstMin_ac                                => 3,
		# 	:CO_HistogramAMI_even_2_5                      => 3,
		# 	:CO_trev_1_num                                 => 3,
		# 	:DN_OutlierInclude_p_001_mdrmd                 => 3,
		# 	:DN_OutlierInclude_n_001_mdrmd                 => 3,
		# 	:FC_LocalSimple_mean1_tauresrat                => 3,
		# 	:FC_LocalSimple_mean3_stderr                   => 5,
		# 	:IN_AutoMutualInfoStats_40_gaussian_fmmi       => 3,
		# 	:MD_hrv_classic_pnn40                          => 3,
		# 	:SB_BinaryStats_diff_longstretch0              => 3,
		# 	:SB_BinaryStats_mean_longstretch1              => 3,
		# 	:SB_MotifThree_quantile_hh                     => 3,
		# 	:SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1   => 3,
		# 	:SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1        => 3,
		# 	:SP_Summaries_welch_rect_area_5_1              => 3,
		# 	:SP_Summaries_welch_rect_centroid              => 3,
		# 	:SB_TransitionMatrix_3ac_sumdiagcov            => 3,
		# 	:PD_PeriodicityWang_th0_01                     => 3,
		# ])

		for f_name in getnames(catch22)
			@eval (function $(Symbol(string(f_name)*"_pos"))(channel)
				val = $(catch22[f_name])(channel)

				if isnan(val)
					# println("WARNING!!! NaN value found! channel = $(channel)")
					-Inf # aggregator_bottom(existential_aggregator(≥), Float64)
				else
					val
				end
			end)
			@eval (function $(Symbol(string(f_name)*"_neg"))(channel)
				val = $(catch22[f_name])(channel)

				if isnan(val)
					# println("WARNING!!! NaN value found! channel = $(channel)")
					Inf # aggregator_bottom(existential_aggregator(≤), Float64)
				else
					val
				end
			end)
		end

		function getCanonicalFeature(f_name)
			if f_name == :min_m
				[CanonicalFeatureGeq_80]
			elseif f_name == :max_m
				[CanonicalFeatureLeq_80]
			elseif f_name == :mean_m
				[StatsBase.mean]
			else
				[(≥, @eval $(Symbol(string(f_name)*"_pos"))),(≤, @eval $(Symbol(string(f_name)*"_neg")))]
			end
		end

		cur_data_modal_args = merge(cur_data_modal_args, (;
			canonical_features = Vector{Union{CanonicalFeature,Function,Tuple{TestOperatorFun,Function}}}(collect(Iterators.flatten(getCanonicalFeature.(best_descriptors))))
		))

	end

	dataset = (X, Y)

	println()
	println("cur_data_modal_args.canonical_features = $(cur_data_modal_args.canonical_features)")
	println("new size(X) = $(size(X))")
	println()
	
	# histogram2d(x, y)

	# using StatsBase, Plots
	# h = fit(Histogram, (x, y), nbins=20)
	# plot(h) # same as histogram2d
	# wireframe(midpoints(h.edges[1]), midpoints(h.edges[2]), h.weights)

	##############################################################################
	##############################################################################
	##############################################################################
	

	# println("class_distribution: ")
	# println(StatsBase.countmap(dataset[2]))
	# println("class_counts: $(class_counts)")

	## Dataset slices
	# obtain dataseeds that are were not done before
	todo_dataseeds = filter((dataseed)->!iteration_in_history(history, (params_namedtuple, dataseed)), exec_dataseed)

	# println(datasource_counts)
	# println(uniq(Y))

	# println(datasource_counts)

	dataset_slices = begin
		n_insts = length(Y)
		# @assert (n_insts % n_cv_folds == 0) "$(n_insts) % $(n_cv_folds) != 0"
		# n_insts_fold = div(n_insts, n_cv_folds)
		
		
		# datasource_counts = [(10,2),(6,8),(13,10),(20,30)]

		# [(datasource_counts[1:(i-1)], datasource_counts[(i+1):end]) for i in 1:length(datasource_counts)]
		base_idxs = begin
			base_idxs = []
			cur_idx = 1
			for i in 1:length(datasource_counts)
				_cur_idxs = []
				for j in 1:length(datasource_counts[i])
					x = cur_idx
					cur_idx += datasource_counts[i][j]
					push!(_cur_idxs, x:(cur_idx-1))
				end
				push!(base_idxs, Tuple(_cur_idxs))
				# [[datasource_counts[1:(i-1)]..., datasource_counts[(i+1):end]...] ]
			end
			@assert length(unique([length(base_idxs[i]) for i in 1:length(base_idxs)])) == 1 "This part of code expects at least all classes to appear for each group. $(datasource_counts) $(base_idxs)"
			base_idxs
		end

		println(datasource_counts)
		println(base_idxs)

		# todo_dataseeds = 1:10
		[(dataseed, begin
				if dataseed == 0

					class_counts, class_grouped_idxs = begin
						class_counts       = []
						class_grouped_idxs = []
						for cl in 1:length(base_idxs[1])
							idxs_groups      = getindex.(base_idxs,cl)
							idxs       = collect(Iterators.flatten(idxs_groups))
							push!(class_counts, length(idxs))
							append!(class_grouped_idxs, idxs)
						end
						Tuple(class_counts), class_grouped_idxs
					end

					perm = balanced_dataset_slice(class_counts, dataseed; n_samples_per_class = floor(Int, minimum(class_counts)*1.0), also_return_discarted = false)

					# (Vector{Integer}(collect(1:n_insts)), Vector{Integer}(collect(1:n_insts))) # Use all instances
					(Vector{Integer}(class_grouped_idxs[perm]), Vector{Integer}(1:n_insts))
				else
					@assert dataseed in 1:length(datasource_counts)
					# @assert datasource_counts[dataseed] .> 0

					# balanced_dataset_slice(class_counts, dataseed; n_samples_per_class = floor(Int, minimum(class_counts)*traintest_threshold), also_return_discarted = true)
					# balanced_dataset_slice(class_counts, dataseed; n_samples_per_class = train_instances_per_class, also_return_discarted = true)

					# train_idxs = ...balanced_dataset_slice
					
					# datasource_counts
					# base_idxs
					train_class_counts, train_class_grouped_idxs, test_idxs = begin
						train_class_counts       = []
						train_class_grouped_idxs = []
						test_idxs = []
						for cl in 1:length(base_idxs[1])
							train_idxs_groups = getindex.(base_idxs,cl)[[1:(dataseed-1)...,(dataseed+1):end...]]
							test_idxs_groups  = getindex.(base_idxs,cl)[[dataseed]]
							_train_idxs       = collect(Iterators.flatten(train_idxs_groups))
							_test_idxs        = collect(Iterators.flatten(test_idxs_groups))
							push!(train_class_counts, length(_train_idxs))
							append!(train_class_grouped_idxs, _train_idxs)
							append!(test_idxs, _test_idxs)
						end
						Tuple(train_class_counts), train_class_grouped_idxs, test_idxs
					end

					# println(train_class_counts)
					# println(train_class_grouped_idxs)

					perm = balanced_dataset_slice(train_class_counts, dataseed; n_samples_per_class = floor(Int, minimum(train_class_counts)*traintest_threshold), also_return_discarted = false)
					
					train_idxs = train_class_grouped_idxs[perm]

					# println(Y[1:4])
					# println(unique(Y[train_idxs]))
					# println(uniq(Y[train_idxs]))
					println(StatsBase.countmap(Y[train_idxs]))
					# readline()

					# train_idxs = train_idxs[1:10]

					@assert all(train_idxs .<= n_insts)
					(Vector{Integer}(collect(train_idxs)), Vector{Integer}(collect(test_idxs)))
					# (Vector{Integer}(collect(train_idxs)), Vector{Integer}(collect(setdiff(Set(1:n_insts), Set(train_idxs)))))

					# a = datasource_counts[1:dataseed-1];
					# idx_base = (length(a) == 0 ? 0 : sum(a))
					# test_idxs = idx_base .+ (1:datasource_counts[dataseed])
					# p = Random.randperm(Random.MersenneTwister(1), round(Int, length(test_idxs)/2))
					# test_idxs = test_idxs[p]
					# # test_idxs = 1+(dataseed-1)*n_insts_fold:(dataseed-1)*n_insts_fold+(n_insts_fold)
					# (Vector{Integer}(collect(test_idxs)), Vector{Integer}(collect(setdiff(Set(1:n_insts), Set(test_idxs)))))

				end
			end) for dataseed in todo_dataseeds]
	end

	println("Dataseeds = $(todo_dataseeds)")

	# for (dataseed,data_slice) in dataset_slices
	# 	println("class_distribution: ")
	# 	println(StatsBase.countmap(dataset[2][data_slice]))
	# 	println("...")
	# 	break # Note: Assuming this print is the same for all dataseeds
	# end
	# println()

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
			is_regression_problem           =   (eltype(Y) != String),
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
