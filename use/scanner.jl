using Pkg
Pkg.activate("..")
using Revise

using DecisionTree
using DecisionTree.ModalLogic

import Random
my_rng() = Random.MersenneTwister(1) # Random.GLOBAL_RNG

using Logging
using ResumableFunctions
using IterTools

using BenchmarkTools
# using ScikitLearnBase
using Statistics
using Test
# using Profile
# using PProf


using DataStructures 

using SHA
using Serialization
using FileIO
import JLD2
import Dates


include("lib.jl")
include("scanner-utils/load-model.jl")

id_f      = identity
half_f(x) = ceil(Int, x/2)
sqrt_f(x) = ceil(Int, sqrt(x))

abstract type Support end

mutable struct ForestEvaluationSupport <: Support
	f::Union{Nothing,Support,AbstractVector{Forest{S}}} where {S}
	f_args::NamedTuple{T, N} where {T, N}
	cm_train::Union{Nothing,AbstractVector{<:GenericPerformanceType}}
	cm::Union{Nothing,AbstractVector{<:GenericPerformanceType}}
	hash::AbstractString
	time::Dates.Millisecond
	enqueued::Bool
	ForestEvaluationSupport(f_args) = new(nothing, f_args, nothing, nothing, "", Dates.Millisecond(0), false)
end

function will_produce_same_forest_with_different_number_of_trees(f1::ForestEvaluationSupport, f2::ForestEvaluationSupport)
	# TODO: find a smart way to handle this (just not needed for now)
	@assert length(f1.f_args) == length(f2.f_args) "Can't compare two forests with different number of arguments."
	for (k, v1) in zip(keys(f1.f_args), values(f1.f_args))
		# do not compare n_trees
		if k == :n_trees
			continue
		end
		if v1 != f2.f_args[k]
			return false
		end
	end
	true
end

function display_cm_as_row(cm::ConfusionMatrix)
	"$(round(overall_accuracy(cm)*100,    digits=2))%\t" *
	"$(join(round.(cm.sensitivities.*100, digits=2), "%\t"))%\t" *
	"$(join(round.(cm.PPVs.*100,          digits=2), "%\t"))%\t" *
	"||\t" *
	# "$(round(cm.mean_accuracy*100, digits=2))%\t" *
	"$(round(cm.kappa*100, digits=2))%\t" *
	# "$(round(DecisionTree.macro_F1(cm)*100, digits=2))%\t" *
	# "$(round.(cm.accuracies.*100, digits=2))%\t" *
	"$(round.(cm.F1s.*100, digits=2))%\t" *
	"|||\t" *
	"$(round(safe_macro_sensitivity(cm)*100,  digits=2))%\t" *
	"$(round(safe_macro_specificity(cm)*100,  digits=2))%\t" *
	"$(round(safe_macro_PPV(cm)*100,          digits=2))%\t" *
	"$(round(safe_macro_NPV(cm)*100,          digits=2))%\t" *
	"$(round(safe_macro_F1(cm)*100,           digits=2))%\t" *
	# "$(round.(cm.sensitivities.*100, digits=2))%\t" *
	# "$(round.(cm.specificities.*100, digits=2))%\t" *
	# "$(round.(cm.PPVs.*100, digits=2))%\t" *
	# "$(round.(cm.NPVs.*100, digits=2))%\t" *
	# "|||\t" *
	# "$(round(DecisionTree.macro_weighted_F1(cm)*100, digits=2))%\t" *
	# # "$(round(DecisionTree.macro_sensitivity(cm)*100, digits=2))%\t" *
	# "$(round(DecisionTree.macro_weighted_sensitivity(cm)*100, digits=2))%\t" *
	# # "$(round(DecisionTree.macro_specificity(cm)*100, digits=2))%\t" *
	# "$(round(DecisionTree.macro_weighted_specificity(cm)*100, digits=2))%\t" *
	# # "$(round(DecisionTree.macro_PPV(cm)*100, digits=2))%\t" *
	# "$(round(DecisionTree.macro_weighted_PPV(cm)*100, digits=2))%\t" *
	# # "$(round(DecisionTree.macro_NPV(cm)*100, digits=2))%\t" *
	# "$(round(DecisionTree.macro_weighted_NPV(cm)*100, digits=2))%\t" *
	""
end


include("caching.jl")
include("scanner-utils/table-printer.jl")
include("scanner-utils/progressive-iterator-manager.jl")
include("datasets.jl")
include("dataset-utils.jl")


# Slice & split the dataset according to dataset_slices & split_threshold
# The instances for which the full StumpFeatModalDataset is computed are either all, or the ones specified for training;
# This depends on whether the dataset is already splitted or not.
@resumable function generate_splits(dataset, split_threshold, round_dataset_to_datatype, save_datasets, run_name, dataset_slices, data_savedir, buildModalDatasets_fun)
	if split_threshold !== false # Dataset is to be splitted
		
		# Unpack dataset
		X, Y = dataset

		# Apply scaling
		if round_dataset_to_datatype != false
			X, Y = round_dataset((X, Y), round_dataset_to_datatype)
		end
		
		# Compute mffmd for the full set of instances
		X_to_train, X_to_test = buildModalDatasets_fun(X, X)
		
		# TODO
		# if save_datasets
		# 	train = (X_to_train, Y)
		# 	JLD2.@save "$(data_savedir)/datasets/$(run_name)-train.jld" train
		# 	test = (X_to_test, Y)
		# 	JLD2.@save "$(data_savedir)/datasets/$(run_name)-test.jld" test
		# end

		for (slice_id,dataset_slice) in dataset_slices

			print("Dataset_slice = ")
			if isnothing(dataset_slice)
				println("nothing")
			elseif isa(dataset_slice, AbstractVector{<:Integer})
				println("($(length(dataset_slice))) -> $(dataset_slice)")
			elseif isa(dataset_slice, NTuple{2,<:AbstractVector{<:Integer}})
				println("($(length.(dataset_slice))) -> $(dataset_slice)")
			else
				throw_n_log("Unknown dataset_slice type: $(typeof(dataset_slice))")
			end

			# Slice instances (for balancing, for example)
			X_train_slice, Y_train_slice, X_test_slice, Y_test_slice = begin
				if isnothing(dataset_slice) || isa(dataset_slice, AbstractVector{<:Integer})
					# Use two different structures for train and test
					X_to_train_slice, X_to_test_slice, Y_slice =
						if isnothing(dataset_slice)
							(
								X_to_train,
								X_to_test,
								Y,
							)
						else
							(
								ModalLogic.slice_dataset(X_to_train, dataset_slice, return_view = true),
								ModalLogic.slice_dataset(X_to_test,  dataset_slice, return_view = true),
								Y[dataset_slice],
							)
					end

					# Split in train/test
					(X_train_slice, Y_train_slice), _ = traintestsplit((X_to_train_slice, Y_slice), split_threshold; return_view = true)
					_, (X_test_slice, Y_test_slice)   = traintestsplit((X_to_test_slice,  Y_slice), split_threshold; return_view = true)
					
					(X_train_slice, Y_train_slice, X_test_slice, Y_test_slice)
				elseif isa(dataset_slice, NTuple{2,<:AbstractVector{<:Integer}})
					println("train slice: $(dataset_slice[1])")
					println("test  slice: $(dataset_slice[2])")
					@assert isone(split_threshold) || iszero(split_threshold) "Can't set a split_threshold (value: $(split_threshold)) when specifying a split dataset_slice (type: $(typeof(dataset_slice)))"
					(
						ModalLogic.slice_dataset(X_to_train, dataset_slice[1], return_view = true),
						Y[dataset_slice[1]],
						ModalLogic.slice_dataset(X_to_test,  dataset_slice[2], return_view = true),
						Y[dataset_slice[2]],
					)
				else
					throw_n_log("Unknown dataset_slice type: $(typeof(dataset_slice))")
				end
			end
			
			# if save_datasets
			# 	JLD2.@save "$(data_savedir)/datasets/$(run_name)-$(slice_id)-dataset_slice.jld" dataset_slice
			# 	sliced = (X_to_train_slice, Y_slice)
			# 	JLD2.@save "$(data_savedir)/datasets/$(run_name)-$(slice_id)-sliced.jld" sliced
			# 	sliced_train = (X_train_slice, Y_train_slice)
			# 	JLD2.@save "$(data_savedir)/datasets/$(run_name)-$(slice_id)-sliced_train.jld" sliced_train
			# 	sliced_test = (X_test_slice, Y_test_slice)
			# 	JLD2.@save "$(data_savedir)/datasets/$(run_name)-$(slice_id)-sliced_test.jld" sliced_test
			# end

			@yield ((X_to_train, X), slice_id, ((X_train_slice, Y_train_slice), (X_test_slice, Y_test_slice)))
		end

	else # Dataset is already splitted

		# Unpack dataset
		(X_train, Y_train), (X_test, Y_test) = dataset

		# Apply scaling
		if round_dataset_to_datatype != false
			(X_train, Y_train), (X_test,  Y_test) = round_dataset(((X_train, Y_train), (X_test,  Y_test)), round_dataset_to_datatype)
		end
		
		# Compute mffmd for the training instances
		X_to_train, X_to_test = buildModalDatasets_fun(X_train, X_test)

		# if save_datasets
		# 	train = (X_to_train, Y_train)
		# 	JLD2.@save "$(data_savedir)/datasets/$(run_name)-train.jld" train
		# 	test = (X_to_test, Y_test)
		# 	JLD2.@save "$(data_savedir)/datasets/$(run_name)-test.jld" test
		# end

		# Slice *training* instances (for balancing, for example)
		for (slice_id,dataset_slice) in dataset_slices

			print("Dataset_slice = ")
			if isnothing(dataset_slice)
				println("nothing")
			elseif isa(dataset_slice, AbstractVector{<:Integer})
				println("($(length(dataset_slice))) -> $(dataset_slice)")
			elseif isa(dataset_slice, NTuple{2,<:AbstractVector{<:Integer}})
				println("($(length.(dataset_slice))) -> $(dataset_slice)")
			else
				throw_n_log("Unknown dataset_slice type: $(typeof(dataset_slice))")
			end

			X_train_slice, Y_train_slice, X_test_slice, Y_test_slice =
				if isnothing(dataset_slice)
					(X_to_train, Y_train, X_to_test, Y_test)
				elseif isa(dataset_slice, AbstractVector{<:Integer})
					(
						ModalLogic.slice_dataset(X_to_train, dataset_slice, return_view = true),
						Y_train[dataset_slice],
						X_to_test,
						Y_test,
					)
				elseif isa(dataset_slice, NTuple{2,<:AbstractVector{<:Integer}})
					@assert isone(split_threshold) || iszero(split_threshold) "Can't set a split_threshold (value: $(split_threshold)) when specifying a split dataset_slice (type: $(typeof(dataset_slice)))"
					throw_n_log("TODO expand code. When isa(dataset_slice, NTuple{2,<:AbstractVector{<:Integer}}) and the dataset is already splitted, must also test on the validation data! Maybe when the dataset is already splitted into ((X_train, Y_train), (X_test, Y_test)), join it and create a dummy dataset_slice")
					(
						ModalLogic.slice_dataset(X_to_train, dataset_slice[1], return_view = true),
						Y_train[dataset_slice[1]],
						ModalLogic.slice_dataset(X_to_train, dataset_slice[2], return_view = true),
						Y_train[dataset_slice[2]],
					)
				else
					throw_n_log("Unknown dataset_slice type: $(typeof(dataset_slice))")
				end

			# if save_datasets
			# 	JLD2.@save "$(data_savedir)/datasets/$(run_name)-$(slice_id)-dataset_slice.jld" dataset_slice
			# 	sliced_train = (X_train_slice, Y_train_slice)
			# 	JLD2.@save "$(data_savedir)/datasets/$(run_name)-$(slice_id)-sliced_train.jld" sliced_train
			# end
			
			@yield ((X_to_train, X_train), slice_id, ((X_train_slice, Y_train_slice), (X_test_slice, Y_test_slice)))
		end
	end
end

# TODO: decouple X_dataset_c from 'timing_mode` and 'data_savedir`
function X_dataset_c(dataset_type_str, data_modal_args, X_all, modal_args, save_datasets, use_form)

	WorldType = world_type(data_modal_args.ontology)

	Xs = MultiFrameModalDataset()

	# Compute stump for each frame
	for (i_frame, X) in enumerate(X_all)
		println("Frame $(i_frame)/$(length(X_all))")

		needToComputeRelationGlob =
			WorldType != OneWorld && (
				(modal_args.useRelationGlob || (modal_args.initConditions == DecisionTree.startWithRelationGlob))
					|| ((modal_args.initConditions isa AbstractVector) && modal_args.initConditions[i_frame] == DecisionTree.startWithRelationGlob)
			)
		########################################################################
		########################################################################
		########################################################################
		
		# Prepare features
		# TODO generalize
		
		features = FeatureTypeFun[]

		for i_attr in 1:n_attributes(X)
			for test_operator in data_modal_args.test_operators
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

		########################################################################
		########################################################################
		########################################################################
		
		X_dataset_sf_c(use_form, X, features, featsnops, needToComputeRelationGlob) = begin

			if use_form == :dimensional
				if timing_mode == :none
					OntologicalDataset(X, data_modal_args.ontology, features, featsnops);
				# elseif timing_mode == :profile
					# @profile OntologicalDataset(X, data_modal_args.ontology, features, featsnops);
				elseif timing_mode == :time
					@time OntologicalDataset(X, data_modal_args.ontology, features, featsnops);
				elseif timing_mode == :btime
					@btime OntologicalDataset($X, $data_modal_args.ontology, $features, $featsnops);
				end
			elseif use_form in [:fmd, :stump, :stump_with_memoization]

				X = OntologicalDataset(X, data_modal_args.ontology, features, featsnops)
			
				if use_form == :fmd
					if timing_mode == :none
						FeatModalDataset(X);
					# elseif timing_mode == :profile
						# @profile FeatModalDataset(X);
					elseif timing_mode == :time
						@time FeatModalDataset(X);
					elseif timing_mode == :btime
						@btime FeatModalDataset($X);
					end
				elseif use_form == :stump
					if timing_mode == :none
						StumpFeatModalDataset(X, computeRelationGlob = needToComputeRelationGlob);
					# elseif timing_mode == :profile
						# @profile StumpFeatModalDataset(X, computeRelationGlob = needToComputeRelationGlob);
					elseif timing_mode == :time
						@time StumpFeatModalDataset(X, computeRelationGlob = needToComputeRelationGlob);
					elseif timing_mode == :btime
						@btime StumpFeatModalDataset($X, computeRelationGlob = $needToComputeRelationGlob);
					end
				elseif use_form == :stump_with_memoization
					if timing_mode == :none
						StumpFeatModalDatasetWithMemoization(X, computeRelationGlob = needToComputeRelationGlob);
					# elseif timing_mode == :profile
						# @profile StumpFeatModalDatasetWithMemoization(X, computeRelationGlob = needToComputeRelationGlob);
					elseif timing_mode == :time
						@time StumpFeatModalDatasetWithMemoization(X, computeRelationGlob = needToComputeRelationGlob);
					elseif timing_mode == :btime
						@btime StumpFeatModalDatasetWithMemoization($X, computeRelationGlob = $needToComputeRelationGlob);
					end
				end
			else
				throw_n_log("Unexpected value for use_form: $(use_form)!")
			end
		end
		
		X_frame = 
			if save_datasets
				# if use_form == :stump_with_memoization # label: use_training_form
				# 	@cachefast_skipsave "$(dataset_type_str)_dataset" data_savedir (use_form, X, features, featsnops, needToComputeRelationGlob) X_dataset_sf_c
				# 	# X_dataset_sf_c(use_form, X, features, featsnops, needToComputeRelationGlob)
				# else
				@cachefast "$(dataset_type_str)_dataset" data_savedir (use_form, X, features, featsnops, needToComputeRelationGlob) X_dataset_sf_c
				# end
			else
				X_dataset_sf_c(use_form, X, features, featsnops, needToComputeRelationGlob)
			end
		
		# println("Ontological form" * display_structure(X))
		# println("Stump form" * display_structure(X_frame))

		push_frame!(Xs, X_frame)
	end
	Xs
end

# This function transforms bare MatricialDatasets into modal datasets in the form of ontological or featmodal dataset
#  The train dataset, unless use_training_form, is transformed in featmodal form, which is optimized for training.
#  The test dataset is kept in ontological form
# function buildModalDatasets(Xs_train_all::Union{MatricialDataset,Vector{<:MatricialDataset}}, X_test::Union{MatricialDataset,Vector{<:MatricialDataset}})	
function buildModalDatasets(X_train, X_test, data_modal_args, modal_args, use_training_form, data_savedir, timing_mode, save_datasets)

	if X_train isa MatricialDataset
		X_train = [X_train]
	end
	if X_test isa MatricialDataset
		X_test = [X_test]
	end
	
	@assert !dataset_has_nonevalues(X_train) "dataset_has_nonevalues(X_train)"
	@assert !dataset_has_nonevalues(X_test)  "dataset_has_nonevalues(X_test)"

	WorldType = world_type(data_modal_args.ontology)

	# The test dataset is kept in its ontological form
	# X_test = MultiFrameModalDataset([OntologicalDataset(X, WorldType) for X in X_test])
	# X_test = MultiFrameModalDataset([OntologicalDataset{eltype(X), ndims(X)-1-1, WorldType}(X, nothing, nothing, nothing) for X in X_test])
	use_test_form = :dimensional
	X_test = X_dataset_c("test", data_modal_args, X_test, modal_args, save_datasets, use_test_form)
	
	# The train dataset is either kept in ontological form, or processed into stump form (which allows for optimized learning)
	X_train =
		if use_training_form in [:dimensional, :fmd, :stump, :stump_with_memoization]
			if save_datasets
				if use_training_form == :stump_with_memoization
					@cachefast_skipsave "training_dataset" data_savedir ("train", data_modal_args, X_train, modal_args, save_datasets, use_training_form) X_dataset_c
				else
					@cachefast          "training_dataset" data_savedir ("train", data_modal_args, X_train, modal_args, save_datasets, use_training_form) X_dataset_c
				end
			else
				X_dataset_c("train", data_modal_args, X_train, modal_args, save_datasets, use_training_form)
			end
		else
			throw_n_log("Unexpected value for use_training_form: $(use_training_form)!")
		end
		
		# println("X_train:")
		# println("  " * display_structure(X_train; indent_str = "  "))

	X_train, X_test
end


function exec_scan(
		params_namedtuple               ::NamedTuple,
		dataset                         ::Tuple;
		is_regression_problem           = false,
		### Training params
		train_seed                      = 1,
		modal_args                      = (),
		tree_args                       = [],
		tree_post_pruning_purity_thresh = [],
		forest_args                     = [],
		forest_runs                     = 1,
		optimize_forest_computation     = false,
		test_flattened                  = false,
		test_averaged                   = false,
		### Dataset params
		split_threshold                 ::Union{Bool,AbstractFloat} = 1.0,
		data_modal_args                 = (),
		dataset_slices                  ::Union{
			AbstractVector{<:Tuple{<:Any,SLICE}},
			AbstractVector{SLICE},
			Nothing} = nothing,
		round_dataset_to_datatype       ::Union{Bool,Type} = false,
		use_training_form               = :stump_with_memoization,
		### Run params
		results_dir                     ::String,
		data_savedir                    ::Union{String,Nothing} = nothing,
		model_savedir                   ::Union{String,Nothing} = nothing,
		legacy_gammas_check             = false,
		log_level                       = nothing, # TODO remove this obsolete parameter
		logger                          = ConsoleLogger(stderr, DecisionTree.DTOverview), # TODO remove this as well?
		timing_mode                     ::Symbol = :time,
		### Misc
		best_rule_params                = [(t=.8, min_confidence=0.6, min_support=0.1), (t=.65, min_confidence=0.6, min_support=0.1)],
		save_datasets                   :: Bool = false,
		skip_training                   :: Bool = false,
		callback                        :: Function = identity,
		dataset_shape_columns           :: Union{AbstractVector,Nothing} = nothing,
	) where {SLICE<:Union{<:AbstractVector{<:Integer},<:NTuple{2,<:AbstractVector{<:Integer}}}}
	
	@assert timing_mode in [:none, :profile, :time, :btime] "Unknown timing_mode!"
	@assert !legacy_gammas_check "legacy_gammas_check parameter is deprecated!" # TODO remove
	
	run_name = join([replace(string(values(value)), ", " => ",") for value in values(params_namedtuple)], ",")
	
	if !isnothing(log_level)
		println("Warning! scanner.log_level parameter is obsolete. Use logger = ConsoleLogger(stderr, $(log_level)) instead!")
		logger = ConsoleLogger(stderr, log_level)
	end

	##############################################################################
	##############################################################################
	# Output files
	##############################################################################
	##############################################################################

	# concise_output_filepath = results_dir * "/grouped_in_models.tsv"
	full_output_filepath    = results_dir * "/full_columns.tsv"

	results_col_sep = "\t"
	
	base_metrics_names = begin
			if is_regression_problem
			[
				"train_cor",
				"train_MAE",
				"train_RMSE",
				"cor",
				"MAE",
				"RMSE",
			]
		else
			[
				"train_accuracy",
				"K",
				"accuracy",
				"macro_sensitivity",
				"safe_macro_sensitivity",
				"safe_macro_specificity",
				"safe_macro_PPV",
				"safe_macro_NPV",
				"safe_macro_F1",
			]
		end
	end

	# TODO restore best_rule_params
	# tree_columns = [base_metrics_names..., "n_nodes", ["best_rule_p $(best_rule_p)" for best_rule_p in best_rule_params]..., "t", "hash"]
	tree_columns = [base_metrics_names..., "n_nodes", "n_leaves", "height", "modal_height",]
	forest_columns = [base_metrics_names..., "oob_error", "n_trees", "n_nodes",]
	
	all_tree_columns = [tree_columns..., "t", "hash"]
	all_forest_columns = [
		forest_columns...,
		["σ² $(n)" for n in forest_columns]...,
		"t", "hash"
	]

	dataset_shape_functions = begin
		if is_regression_problem
			OrderedDict(
				"# insts (TRAIN)"  => (X_train, Y_train, X_test, Y_test)->length(Y_train),
				"# insts (TEST)"   => (X_train, Y_train, X_test, Y_test)->length(Y_test),
			)
		else
			OrderedDict(
				"# insts (TRAIN)"  => (X_train, Y_train, X_test, Y_test)->length(Y_train),
				"# insts (TEST)"   => (X_train, Y_train, X_test, Y_test)->length(Y_test),
				#
				"class_counts (TRAIN)"     => (X_train, Y_train, X_test, Y_test)->begin
					m = StatsBase.countmap(Y_train)
					ks = keys(m) |> collect |> sort
					Tuple([m[k] for k in ks])
				end,
				"class_counts (TEST)"     => (X_train, Y_train, X_test, Y_test)->begin
					m = StatsBase.countmap(Y_test)
					ks = keys(m) |> collect |> sort
					Tuple([m[k] for k in ks])
				end,
				#
				# "n_attributes"     => (X_train, Y_train, X_test, Y_test)->n_attributes(X_train),
				"n_features"       => (X_train, Y_train, X_test, Y_test)->n_features(X_train),
				"channel_size"     => (X_train, Y_train, X_test, Y_test)->channel_size(X_test),
				# "avg_channel_size" => (X_train, Y_train, X_test, Y_test)->?,
			)
		end
	end

	if isnothing(dataset_shape_columns)
		dataset_shape_columns = collect(keys(dataset_shape_functions))
	else
		@assert all([col_name in keys(dataset_shape_functions) for col_name in dataset_shape_columns]) "Unknown column encountered in dataset_shape_columns = $(dataset_shape_columns)"
	end

	# If the output files do not exists initilize them
	# print_result_head(concise_output_filepath, tree_args, forest_args,
	# 	separator = results_col_sep,
	# 	tree_columns = [""],
	# 	forest_columns = ["", "σ²", "t"],
	# 	columns_before = ["Dataseed", "Params-combination"],
	# )
	print_result_head(full_output_filepath, tree_args, forest_args,
		separator = results_col_sep,
		tree_columns = all_tree_columns,
		forest_columns = all_forest_columns,
		columns_before = ["Dataseed", (params_namedtuple |> keys .|> string)..., dataset_shape_columns...],
	)

	##############################################################################
	##############################################################################
	##############################################################################
	##############################################################################

	go_tree(slice_id, X_train, Y_train, X_test, Y_test, tree_args, rng) = begin
		started = Dates.now()
		T =
			if timing_mode == :none
				build_tree(X_train, Y_train; tree_args..., modal_args..., rng = rng)
			# elseif timing_mode == :profile
				# @profile build_tree(X_train, Y_train; tree_args..., modal_args..., rng = rng)
			elseif timing_mode == :time
				@time build_tree(X_train, Y_train; tree_args..., modal_args..., rng = rng)
			elseif timing_mode == :btime
				@btime build_tree($X_train, $Y_train; $tree_args..., $modal_args..., rng = $rng)
			end
		Tt = Dates.now() - started
		println("Train tree:")
		print(T)

		model_save_path = ""
		tree_hash = get_hash_sha256(T)
		if !isnothing(model_savedir)
			model_save_path = model_savedir * "/tree_" * tree_hash * ".jld"
			mkpath(dirname(model_save_path))

			checkpoint_stdout("Saving tree to file $(model_save_path)...")
			JLD2.@save model_save_path T
		end

		# If not full-training
		T_test =
			if split_threshold != 1.0
				println("Test tree:")
				print_apply_tree(T, X_test, Y_test)
			else
				print_apply_tree(T, X_test, Y_test; update_majority = true, do_print = false)
			end
		
		cm_train = confusion_matrix(Y_train, apply_tree(T, X_train))

		println(" test size = $(size(X_test))")
		cm = nothing
		for pruning_purity_threshold in sort(unique([(Float64.(tree_post_pruning_purity_thresh))...,1.0]))
			println(" Purity threshold $pruning_purity_threshold")
			
			T_pruned = prune_tree(T, pruning_purity_threshold)
			preds = apply_tree(T_pruned, X_test);
			cm = confusion_matrix(Y_test, preds)
			# @test overall_accuracy(cm) > 0.99

			cm_str = begin
				if is_regression_problem
					display(cm)
					println()
				else
					display_cm_as_row(cm)
				end
			end
			println("RESULT:\t$(run_name)\t$(slice_id)\t$(tree_args)\t$(modal_args)\t$(pruning_purity_threshold)\t|\t$(cm_str)\t$(model_save_path)")

			println(cm)
			# @show cm

			# println("nodes: ($(num_nodes(T_pruned)), height: $(height(T_pruned)))")
		end
		return (T_test, cm_train, cm, Tt, string(tree_hash));
	end

	go_forest(slice_id, X_train, Y_train, X_test, Y_test, f_args, rng; prebuilt_model::Union{Nothing,AbstractVector{Forest{S}}} = nothing) where {S} = begin
		Fs, Ft = 
			if isnothing(prebuilt_model)
				started = Dates.now()
				[
					if timing_mode == :none
						build_forest(X_train, Y_train; f_args..., modal_args..., rng = rng);
					# elseif timing_mode == :profile
						# @profile build_forest(X_train, Y_train; f_args..., modal_args..., rng = rng);
					elseif timing_mode == :time
						@time build_forest(X_train, Y_train; f_args..., modal_args..., rng = rng);
					elseif timing_mode == :btime
						@btime build_forest($X_train, $Y_train; $f_args..., $modal_args..., rng = $rng);
					end
					for i in 1:forest_runs
				], (Dates.now() - started)
			else
				println("Using slice of a prebuilt forest.")
				# TODO
				# !!! HUGE PROBLEM HERE !!! #
				# BUG: can't compute oob_error of a forest built slicing another forest!!!
				forests::Vector{Forest{S}} = []
				for f in prebuilt_model
					v_forest = @views f.trees[Random.randperm(rng, length(f.trees))[1:f_args.n_trees]]
					v_cms = @views f.cm[Random.randperm(rng, length(f.cm))[1:f_args.n_trees]]
					push!(forests, Forest{S}(v_forest, v_cms, 0.0))
				end
				forests, Dates.Millisecond(0)
			end

		for F in Fs
			print(F)
		end
		
		cms = GenericPerformanceType[]
		cms_train = GenericPerformanceType[]
		hashes = []
		for F in Fs
			println(" test size = $(size(X_test))")
			
			cm_train = confusion_matrix(Y_train, apply_forest(F, X_train))
			push!(cms_train, cm_train)

			preds = apply_forest(F, X_test);
			cm = confusion_matrix(Y_test, preds)
			# @test overall_accuracy(cm) > 0.99

			model_save_path = ""
			hash = get_hash_sha256(F)
			push!(hashes, hash)
			if !isnothing(model_savedir)
				model_save_path = model_savedir * "/rf_" * hash * ".jld"
				mkpath(dirname(model_save_path))

				checkpoint_stdout("Saving random_forest to file $(model_save_path)...")
				JLD2.@save model_save_path F
			end    
  
			cm_str = begin
				if is_regression_problem
					display(cm)
					println()
				else
					display_cm_as_row(cm)
				end
			end

			println("RESULT:\t$(run_name)\t$(slice_id)\t$(f_args)\t$(modal_args)\t$(cm_str)\t$(model_save_path)")

			if !is_regression_problem
				# println("  accuracy: ", round(overall_accuracy(cm)*100, digits=2), "% kappa: ", round(cm.kappa*100, digits=2), "% ")
				for (i,row) in enumerate(eachrow(cm.matrix))
					for val in row
						print(lpad(val,3," "))
					end
					println("  " * "$(round(100*row[i]/sum(row), digits=2))%\t\t" * cm.classes[i])
				end
			end

			println("Forest OOB Error: $(round.(F.oob_error.*100, digits=2))%")

			push!(cms, cm)
		end

		return (Fs, cms_train, cms, Ft, string(Tuple(hashes)));
	end

	go(slice_id, X_train, Y_train, X_test, Y_test) = begin

		Ts   = []
		Fs   = []
		Tcms_train = []
		Fcms_train = []
		Tcms = []
		Fcms = []
		Tts  = []
		Fts  = []
		Thashs  = []
		Fhashs  = []

		for (i_model, this_args) in enumerate(tree_args)
			checkpoint_stdout("Computing tree $(i_model) / $(length(tree_args))...\n$(this_args)")
			this_T, this_Tcm_train, this_Tcm, this_Tt, this_Thash = go_tree(slice_id, X_train, Y_train, X_test, Y_test, this_args, Random.MersenneTwister(train_seed))
			push!(Ts, this_T)
			push!(Tcms_train, this_Tcm_train)
			push!(Tcms, this_Tcm)
			push!(Tts, this_Tt)
			push!(Thashs, this_Thash)
		end

		if optimize_forest_computation
			# initialize support structures
			forest_supports_user_order = Vector{ForestEvaluationSupport}(undef, length(forest_args)) # ordered as user gave them
			for (i_forest, f_args) in enumerate(forest_args)
				forest_supports_user_order[i_forest] = ForestEvaluationSupport(f_args)
			end

			# biggest forest first
			forest_supports_build_order = Vector{ForestEvaluationSupport}() # ordered with biggest forest first
			append!(forest_supports_build_order, forest_supports_user_order)
			sort!(forest_supports_build_order, by = f -> f.f_args.n_trees, rev = true)

			for (i, f) in enumerate(forest_supports_build_order)
				if f.enqueued
					continue
				end

				f.enqueued = true

				for j in i:length(forest_supports_build_order)
					if forest_supports_build_order[j].enqueued
						continue
					end

					if will_produce_same_forest_with_different_number_of_trees(f, forest_supports_build_order[j])
						# reference the forest of the "equivalent" Support structor
						# equivalent means has the same parameters except for the n_trees
						forest_supports_build_order[j].f = forest_supports_build_order[i]
						forest_supports_build_order[j].enqueued = true
					end
				end
			end

			for (i, f) in enumerate(forest_supports_build_order)
				checkpoint_stdout("Computing Random Forest $(i) / $(length(forest_supports_build_order))...")
				model = f

				while isa(model, ForestEvaluationSupport)
					model = model.f
				end
				checkpoint_stdout("$(f.f_args)")

				forest_supports_build_order[i].f,
				forest_supports_build_order[i].cm_train,
				forest_supports_build_order[i].cm,
				forest_supports_build_order[i].time,
				forest_supports_build_order[i].hash = go_forest(slice_id, X_train, Y_train, X_test, Y_test, f.f_args, Random.MersenneTwister(train_seed), prebuilt_model = model)
			end

			# put resulting forests in vector in the order the user gave them
			for (i, f) in enumerate(forest_supports_user_order)
				@assert f.f isa AbstractVector{Forest{S}} where {S} "This is not a Vector of Forests! eltype = $(eltype(f.f))"
				@assert f.cm isa AbstractVector{<:GenericPerformanceType} "This is not a Vector of <:GenericPerformanceType!"
				@assert length(f.f) == forest_runs "There is a support struct with less than $(forest_runs) forests: $(length(f.f))"
				@assert length(f.cm) == forest_runs "There is a support struct with less than $(forest_runs) confusion matrices: $(length(f.cm))"
				@assert f.f_args == forest_args[i] "f_args mismatch! $(f.f_args) == $(f_args[i])"

				push!(Fs, f.f)
				push!(Fcms_train, f.cm_train)
				push!(Fcms, f.cm)
				push!(Fts, f.time)
				push!(Fhashs, f.hash)
			end
		else
			for (i_forest, this_args) in enumerate(forest_args)
				checkpoint_stdout("Computing Random Forest $(i_forest) / $(length(forest_args))...\n$(this_args)")
				this_F, this_Fcm_train, this_Fcm, this_Ft, this_hashes = go_forest(slice_id, X_train, Y_train, X_test, Y_test, this_args, Random.MersenneTwister(train_seed))
				push!(Fs, this_F)
				push!(Fcms_train, this_Fcm_train)
				push!(Fcms, this_Fcm)
				push!(Fts, this_Ft)
				push!(Fhashs, this_hashes)
			end
		end

		##############################################################################
		##############################################################################
		# PRINT RESULT IN FILES 
		##############################################################################
		##############################################################################
		
		function print_result_row(values_for_columns_before, tab_kwargs, output_filepath)
			out_str = string(join(values_for_columns_before, results_col_sep), results_col_sep)
			for j in 1:length(tree_args)
				out_str *= string(data_to_string(Ts[j], Tcms[j], Tts[j], Thashs[j], tree_columns;
					train_cm = Tcms_train[j],
					tab_kwargs...,
					# best_rule_params = best_rule_params,
					))
				out_str *= string(results_col_sep)
			end
			for j in 1:length(forest_args)
				out_str *= string(data_to_string(Fs[j], Fcms[j], Fts[j], Fhashs[j], forest_columns;
					train_cm = Fcms_train[j],
					tab_kwargs...,
					))
				out_str *= string(results_col_sep)
			end
			out_str *= "\n"
			append_in_file(output_filepath, out_str)
		end

		# PRINT CONCISE
		# print_result_row(
		# 	[slice_id, run_name],
		# 	(alt_separator=", ", separator = results_col_sep),
		# 	concise_output_filepath,
		# 	# results_col_sep = results_col_sep, TODO?
		# )

		# PRINT FULL
		# println(typeof(X_train))
		# println(typeof(X_test))
		# println(length(Y_train))
		# println(typeof(Y_train))
		# println(length(Y_test))
		# println(typeof(Y_test))
		# println(n_samples(X_train))
		# println(n_samples(X_test))

		print_result_row(
			[
				slice_id,
				[replace(string(values(value)), ", " => ",") for value in values(params_namedtuple)]...,
				[string(dataset_shape_functions[col_name](X_train, Y_train, X_test, Y_test)) for col_name in dataset_shape_columns]...,
			],
			(start_s = "", end_s = "", alt_separator = results_col_sep),
			full_output_filepath,
			# results_col_sep = results_col_sep, TODO?
		)
		
		callback(slice_id)

		Dict(
			"Ts" => Ts,
			"Fs" => Fs,
			"Tcms_train" => Tcms_train,
			"Fcms_train" => Fcms_train,
			"Tcms" => Tcms,
			"Fcms" => Fcms,
			"Tts" => Tts,
			"Fts" => Fts,
		)
	end
	
	##############################################################################
	##############################################################################
	##############################################################################

	if isa(dataset_slices,AbstractVector{<:AbstractVector{<:Integer}})
		dataset_slices = enumerate(dataset_slices)
	elseif isnothing(dataset_slices)
		dataset_slices = [(0,nothing)]
	end

	##############################################################################
	##############################################################################
	##############################################################################

	println()
	println("executing run '$run_name'...")
	println("dataset type = ", typeof(dataset))
	println()
	println("train_seed   = ", train_seed)
	println("modal_args   = ", modal_args)
	println("tree_args    = ", tree_args)
	# println("tree_post_pruning_purity_thresh    = ", tree_post_pruning_purity_thresh)
	println("forest_args  = ", forest_args)
	println("forest_runs  = ", forest_runs)
	# println("forest_args  = ", length(forest_args), " × some forest_args structure")
	println()
	println("split_threshold   = ", split_threshold)
	# println("best_rule_params  = ", best_rule_params) TODO restore
	println("data_modal_args   = ", data_modal_args)
	println("dataset_slices    = ($(length(dataset_slices)) dataset_slices)")

	# println("round_dataset_to_datatype   = ", round_dataset_to_datatype)
	# println("use_training_form   = ", use_training_form)
	# println("data_savedir   = ", data_savedir)
	# println("model_savedir   = ", model_savedir)
	# println("log_level   = ", log_level)
	# println("timing_mode   = ", timing_mode)

	println()

	old_logger = global_logger(logger);

	rets = []
	
	buildModalDatasets_fun = (X_train, X_test)->buildModalDatasets(X_train, X_test, data_modal_args, modal_args, use_training_form, data_savedir, timing_mode, save_datasets)

	X_full       = nothing
	X_full_input = nothing

	for (X_f_tuple, slice_id, ((X_train,Y_train), (X_test,Y_test))) in generate_splits(dataset, split_threshold, round_dataset_to_datatype, save_datasets, run_name, dataset_slices, data_savedir, buildModalDatasets_fun)

		(X_full, X_full_input) = X_f_tuple
		
		@assert n_samples(X_train) == length(Y_train)
		@assert n_samples(X_test)  == length(Y_test)
		

		println("train dataset:")
		println("  " * display_structure(X_train; indent_str = "  "))

		println("test  dataset:")
		println("  " * display_structure(X_test; indent_str = "  "))

		if !skip_training
			# readline()

			if test_flattened == true
				throw_n_log("TODO handle case test_flattened = true")
			end
			if test_averaged == true
				throw_n_log("TODO handle case test_averaged = true")
			end
			
			# TODO
			# if test_flattened == true
			# 	# Flatten 
			# 	X_train = ...
			# 	X_test = ...
			# 	push!(rets, go(slice_id, X_train, Y_train, X_test, Y_test))
			# end
			# test_averaged ...

			push!(rets, go(slice_id, X_train, Y_train, X_test, Y_test))
			
			# try
			# go(slice_id, X_train, Y_train, X_test, Y_test)
			# catch e
			# 	println("exec_run: An error occurred!")
			# 	println(e)
			# 	return;
			# end
			
		end
	end

	# Finally save the dataset with memoization
	if save_datasets && use_training_form == :stump_with_memoization && !isnothing(X_full)
		@cachefast_overwrite "training_dataset" data_savedir ("train", data_modal_args, X_full_input, modal_args, save_datasets, use_training_form) X_dataset_c X_full
	end

	global_logger(old_logger);

	# Iterators.flatten(first(rets)) |> collect
	# rets = zip(rets...) |> collect
	# rets = map((r)->Iterators.flatten(r) |> collect, rets)	
	# rets

	# rets=[a,b] # debug
	ks = unique(Iterators.flatten(keys.(rets)) |> collect)
	all_rets = Dict()
	for k in ks
		all_rets[k] = [
				if haskey(r, k)
					r[k]
				else
					nothing
				end
			for r in rets]
	end
	all_rets
end
