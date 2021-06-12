using Pkg
Pkg.activate("..")
using Revise

using DecisionTree
using DecisionTree.ModalLogic

import Random
my_rng() = Random.MersenneTwister(1) # Random.GLOBAL_RNG

using Logging
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
import JLD2
import Dates

function get_hash_sha256(var)::String
	io = IOBuffer();
	serialize(io, var)
	result = bytes2hex(sha256(take!(io)))
	close(io)

	result
end

abstract type Support end

mutable struct ForestEvaluationSupport <: Support
	f::Union{Nothing,Support,AbstractVector{DecisionTree.Forest{S, Ta}}} where {S, Ta}
	f_args::NamedTuple{T, N} where {T, N}
	cm::Union{Nothing,AbstractVector{ConfusionMatrix}}
	time::Dates.Millisecond
	enqueued::Bool
	ForestEvaluationSupport(f_args) = new(nothing, f_args, nothing, Dates.Millisecond(0), false)
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

function human_readable_time(ms::Dates.Millisecond)::String
	result = ms.value / 1000
	seconds = round(Int64, result % 60)
	result /= 60
	minutes = round(Int64, result % 60)
	result /= 60
	hours = round(Int64, result % 24)
	return string(string(hours; pad=2), ":", string(minutes; pad=2), ":", string(seconds; pad=2))
end

function checkpoint_stdout(string::String)
	println("● ", Dates.format(Dates.now(), "[ dd/mm/yyyy HH:MM:SS ] "), string)
	flush(stdout)
end

include("datasets.jl")
include("dataset-utils.jl")

#gammas_saving_task = nothing

function testDataset(
		name                            ::String,
		dataset                         ::Tuple,
		split_threshold                 ::Union{Bool,AbstractFloat};
		###
		forest_args                     = [],
		tree_args                       = [],
		optimize_forest_computation     = false,
		tree_post_pruning_purity_thresh = [],
		modal_args                      = (),
		forest_runs                     = 1,
		test_flattened                  = false, # TODO: Also test the same models but propositional (flattened, average+variance+...+curtosis?)
		train_seed                      ::Integer = 1,
		###
		data_modal_args                 = (),
		dataset_slice                   ::Union{AbstractVector,Nothing} = nothing,
		round_dataset_to_datatype       ::Union{Bool,Type} = false,
		precompute_gammas               = true,
		gammas_save_path                ::Union{String,NTuple{2,String},Nothing} = nothing,
		save_tree_path                  ::Union{String,Nothing} = nothing,
		###
		log_level                       = DecisionTree.DTOverview,
		timing_mode                     ::Symbol = :time,
	)
	println("Benchmarking dataset '$name' (train_seed = $(train_seed))...")
	global_logger(ConsoleLogger(stderr, Logging.Warn));

	function buildModalDatasets(modal_args, Xs_train_all::Vector{<:MatricialDataset}, X_test::Vector{<:MatricialDataset})	
		X_test = MultiFrameOntologicalDataset(data_modal_args.ontology, X_test)

		if !precompute_gammas
			error("TODO !precompute_gammas not coded yet")
			Xs_train_all = MultiFrameOntologicalDataset(data_modal_args.ontology, Xs_train_all)
			(modal_args, Xs_train_all, X_test)
		else
			WorldType = world_type(data_modal_args.ontology)

			old_logger = global_logger(ConsoleLogger(stderr, log_level))
			
			Xs_train_all_multiframe_stump_fmd = stumpFeatModalDataset{<:Real,<:AbstractWorld}[]

			for (i_frame, X_train_all) in enumerate(Xs_train_all)
				
				# X_train_all = OntologicalDataset{eltype(X_train_all)}(data_modal_args.ontology,X_train_all)
				X_train_all = OntologicalDataset(data_modal_args.ontology, X_train_all)

				########################################################################
				########################################################################
				########################################################################
				
				# Prepare features
				# TODO generalize
				
				features = FeatureTypeFun[]

				if data_modal_args.test_operators == [TestOpGeq, TestOpLeq]
					for i_attr in 1:n_attributes(X_train_all)
						push!(features, ModalLogic.AttributeMinimumFeatureType(i_attr))
						push!(features, ModalLogic.AttributeMaximumFeatureType(i_attr))
					end
				elseif data_modal_args.test_operators == [TestOpGeq_80, TestOpLeq_80]
					for i_attr in 1:n_attributes(X_train_all)
						push!(features, ModalLogic.AttributeSoftMinimumFeatureType(i_attr, 0.8))
						push!(features, ModalLogic.AttributeSoftMaximumFeatureType(i_attr, 0.8))
					end
				else
					error("TODO fix $(data_modal_args.test_operators)")
				end

				featsnops = Vector{<:TestOperatorFun}[
					if any(map(t->isa(feature,t), [AttributeMinimumFeatureType, AttributeSoftMinimumFeatureType]))
						[≥]
					elseif any(map(t->isa(feature,t), [AttributeMaximumFeatureType, AttributeSoftMaximumFeatureType]))
						[≤]
					else
						error("Unknown feature type: $(feature), $(typeof(feature))")
						[≥, ≤]
					end for feature in features
				]

				############################################################################
				############################################################################
				############################################################################
				
				relationSet = [RelationId, RelationAll, data_modal_args.ontology.relationSet...]
				relationId_id = 1
				relationAll_id = 2

				ontology_relation_ids = map((x)->x+2, 1:length(data_modal_args.ontology.relationSet))
				needToComputeRelationAll = (modal_args.useRelationAll || (modal_args.initConditions == DecisionTree.startWithRelationAll)) || ((modal_args.initConditions isa AbstractVector) && modal_args.initConditions[i_frame] == DecisionTree.startWithRelationAll)

				# Modal relations to compute gammas for
				inUseRelation_ids = if needToComputeRelationAll
					[relationAll_id, ontology_relation_ids...]
				else
					ontology_relation_ids
				end

				# Generate path to gammas jld file

				if isa(gammas_save_path,String) || isnothing(gammas_save_path)
					gammas_save_path = (gammas_save_path, nothing)
				end

				gammas_save_path, dataset_name_str = gammas_save_path

				gammas_jld_path, gammas_hash_index_file, dataset_hash =
					if isnothing(gammas_save_path)
						(nothing, nothing, nothing)
					else
						dataset_hash = get_hash_sha256(X_train_all)
						(
							"$(gammas_save_path)/gammas_$(dataset_hash).jld",
							"$(gammas_save_path)/gammas_hash_index.csv",
							dataset_hash,
						)
					end

				gammas = 
					if !isnothing(gammas_jld_path) && isfile(gammas_jld_path)
						checkpoint_stdout("Loading gammas from file \"$(gammas_jld_path)\"...")

						Serialization.deserialize(gammas_jld_path)
					else
						checkpoint_stdout("Computing gammas for $(dataset_hash)...")
						started = Dates.now()
						gammas = 
							if timing_mode == :none
								DecisionTree.computeGammas(X_train_all,data_modal_args.test_operators,relationSet,relationId_id,inUseRelation_ids);
							elseif timing_mode == :time
								@time DecisionTree.computeGammas(X_train_all,data_modal_args.test_operators,relationSet,relationId_id,inUseRelation_ids);
							elseif timing_mode == :btime
								@btime DecisionTree.computeGammas($X_train_all,$data_modal_args.test_operators,$relationSet,$relationId_id,$inUseRelation_ids);
						end
						gammas_computation_time = (Dates.now() - started)
						checkpoint_stdout("Computed gammas in $(human_readable_time(gammas_computation_time))...")

						if !isnothing(gammas_jld_path)
							checkpoint_stdout("Saving gammas to file \"$(gammas_jld_path)\"...")
							mkpath(dirname(gammas_jld_path))
							Serialization.serialize(gammas_jld_path, gammas)
							# Add record line to the index file of the folder
							if !isnothing(dataset_name_str)
								# Generate path to gammas jld file)
								# TODO fix column_separator here
								append_in_file(gammas_hash_index_file, "$(dataset_hash);$(dataset_name_str)\n")
							end
						end
						gammas
					end
				checkpoint_stdout("├ Type: $(typeof(gammas))")
				checkpoint_stdout("├ Size: $(sizeof(gammas)/1024/1024 |> x->round(x, digits=2)) MBs")
				checkpoint_stdout("└ Dimensions: $(size(gammas))")

				########################################################
				########################################################
				########################################################
				
				# Compute modalDataset (equivalent to gammas)

				fmd =
					if timing_mode == :none
						FeatModalDataset(X_train_all, features, featsnops)
					elseif timing_mode == :time
						@time FeatModalDataset(X_train_all, features, featsnops)
					elseif timing_mode == :btime
						@btime FeatModalDataset($X_train_all, $features, $featsnops)
				end

				println("OntologicalDataset size:\t\t\t$(Base.summarysize(X_train_all) / 1024 / 1024 |> x->round(x, digits=2)) MBs")
				println("FeaturedWorldDataset size:\t\t\t$(Base.summarysize(fmd) / 1024 / 1024 |> x->round(x, digits=2)) MBs")

				# Check consistency between FeaturedWorldDataset and modalDataset

				for i_instance in 1:n_samples(X_train_all)
					instance = ModalLogic.getInstance(X_train_all, i_instance)
					for i_attribute in 1:n_attributes(X_train_all)
						for i_test_operator in 1:2
							for w in ModalLogic.enumAll(WorldType, ModalLogic.inst_channel_size(instance)...)
								
								g = DecisionTree.readGamma(gammas,i_test_operator,w,i_instance,1,i_attribute)
								m = fmd[i_instance, w, (i_test_operator-1)+(i_attribute-1)*2+1]

								if g != m
									println("FeaturedWorldDataset check: g != m\n$(g)\n$(m)\ni_test_operator=$(i_test_operator)\nw=$(w)\ni_instance=$(i_instance)\ni_attribute=$(i_attribute)")
									print("instance: ")
									println(ModalLogic.getInstanceAttribute(instance, i_attribute))
									error("aoe")
								end
							end
						end
					end
				end

				# stump_fmd =
				# 	if timing_mode == :none
				# 		stumpFeatModalDataset(X_train_all, features, featsnops, computeRelationAll = needToComputeRelationAll, timing_mode = timing_mode);
				# 	elseif timing_mode == :time
				# 		@time stumpFeatModalDataset(X_train_all, features, featsnops, computeRelationAll = needToComputeRelationAll, timing_mode = timing_mode);
				# 	elseif timing_mode == :btime
				# 		@btime stumpFeatModalDataset($X_train_all, $features, $featsnops, computeRelationAll = $needToComputeRelationAll, timing_mode = $timing_mode);
				# end

				# global_logger(ConsoleLogger(stderr, DecisionTree.DTDebug))

				stump_fmd =
					if timing_mode == :none
						stumpFeatModalDataset(fmd, computeRelationAll = needToComputeRelationAll);
					elseif timing_mode == :time
						@time stumpFeatModalDataset(fmd, computeRelationAll = needToComputeRelationAll);
					elseif timing_mode == :btime
						@btime stumpFeatModalDataset($fmd, computeRelationAll = $needToComputeRelationAll);
				end

				fmd   = stump_fmd.fmd
				fmd_m = stump_fmd.fmd_m
				fmd_g = stump_fmd.fmd_g

				println(Base.size(X_train_all.domain))
				println(Base.size(fmd.fwd))
				println(Base.size(fmd_m))
				if !isnothing(fmd_g)
					println(Base.size(fmd_g))
				end
				println("Dataset size:\t\t\t$(Base.summarysize(X_train_all) / 1024 / 1024 |> x->round(x, digits=2)) MBs")
				println("modalDataset total size:\t$((Base.summarysize(fmd) + Base.summarysize(fmd_m) + Base.summarysize(fmd_g)) / 1024 / 1024 |> x->round(x, digits=2)) MBs")
				println("├ fmd size:\t\t$(Base.summarysize(fmd) / 1024 / 1024 |> x->round(x, digits=2)) MBs")
				println("├ fmd_m size:\t\t$(Base.summarysize(fmd_m) / 1024 / 1024 |> x->round(x, digits=2)) MBs")
				println("└ fmd_g size:\t\t$(Base.summarysize(fmd_g) / 1024 / 1024 |> x->round(x, digits=2)) MBs")


				# features_n_operators = Tuple{<:FeatureTypeFun,<:TestOperatorFun}[]
				
				# for feature in features
				# 	if typeof(feature) in [AttributeMinimumFeatureType, AttributeSoftMinimumFeatureType]
				# 		push!(features_n_operators, (feature, ≥))
				# 	elseif typeof(feature) in [AttributeMaximumFeatureType, AttributeSoftMaximumFeatureType]
				# 		push!(features_n_operators, (feature, ≤))
				# 	else
				# 		error("Unknown feature type")
				# 		push!(features_n_operators, (feature, ≥))
				# 		push!(features_n_operators, (feature, ≤))
				# 	end
				# end

				# featnaggrs = ModalLogic.prepare_featnaggrs(featsnops)

				firstWorld = WorldType(ModalLogic.firstWorld)

				for i_instance in 1:n_samples(X_train_all)
					instance = ModalLogic.getInstance(X_train_all, i_instance)
					for i_attribute in 1:n_attributes(X_train_all)
						for i_test_operator in 1:2
							
							i_featnaggr = (i_test_operator-1)+(i_attribute-1)*2+1

							g = DecisionTree.readGamma(gammas,i_test_operator,firstWorld,i_instance,2,i_attribute)
							m = fmd_g[i_instance, i_featnaggr]

							if g != m
								println("fmd_g check: g != m\n$(g)\n$(m)\ni_test_operator=$(i_test_operator)\ntest_operator=$(data_modal_args.test_operators[i_test_operator])\ni_featnaggr=$(i_featnaggr)\ni_instance=$(i_instance)\ni_attribute=$(i_attribute)")
								print("instance: ")
								println(ModalLogic.getInstanceAttribute(instance, i_attribute))
								print(instance)
								# error("aoe")
								readline()
							end
						end
					end
				end

				relations = X_train_all.ontology.relationSet

				for i_instance in 1:n_samples(X_train_all)
					instance = ModalLogic.getInstance(X_train_all, i_instance)
					for i_attribute in 1:n_attributes(X_train_all)
						for i_relation in 1:length(relations)
							for i_test_operator in 1:2
								for w in ModalLogic.enumAll(WorldType, ModalLogic.inst_channel_size(instance)...)
									
									i_featnaggr = (i_test_operator-1)+(i_attribute-1)*2+1

									g = DecisionTree.readGamma(gammas,i_test_operator,w,i_instance,2+i_relation,i_attribute)
									m = fmd_m[i_instance, w, i_featnaggr, i_relation]

									if g != m
										println("fmd_m check: g != m\n$(g)\n$(m)\ni_relation=$(i_relation), relation=$(relations[i_relation])\ni_test_operator=$(i_test_operator)\ntest_operator=$(data_modal_args.test_operators[i_test_operator])\nw=$(w)\ni_instance=$(i_instance)\ni_attribute=$(i_attribute)")
										print("channel: ")
										println(ModalLogic.getInstanceAttribute(instance, i_attribute))
										# error("aoe")
										readline()
									end
								end
							end
						end
					end
				end

				push!(Xs_train_all_multiframe_stump_fmd, stump_fmd)
			end

			Xs_train_all_multiframe_stump_fmd = MultiFrameFeatModalDataset(Xs_train_all_multiframe_stump_fmd)
			########################################################
			########################################################
			########################################################
			
			# println("(optimized) modal_args = ", modal_args)
			global_logger(old_logger);
			(modal_args, Xs_train_all_multiframe_stump_fmd, X_test)
		end
	end

	println("forest_args = ", forest_args)
	# println("forest_args = ", length(forest_args), " × some forest_args structure")
	println("tree_args   = ", tree_args)
	println("modal_args  = ", modal_args)
	println(typeof(dataset))

	# Slice & split the dataset according to dataset_slice & split_threshold
	# The instances for which the gammas are computed are either all, or the ones specified for training.	
	# This depends on whether the dataset is already splitted or not.
	modal_args, (X_train, Y_train), (X_test, Y_test) = 
		if split_threshold != false

			# Unpack dataset
			length(dataset) == 2 || error("Wrong dataset length: $(length(dataset))")
			X, Y = dataset

			# Apply scaling
			if round_dataset_to_datatype != false
				error("TODO case round_dataset_to_datatype != false not captured yet")
				X, Y = roundDataset((X, Y), round_dataset_to_datatype)
			end
			
			# Compute mffmd for the full set of instances
			modal_args, X, _ = buildModalDatasets(modal_args, X, X)

			# Slice instances
			X, Y =
				if isnothing(dataset_slice)
					(X, Y,)
				else
					(
						ModalLogic.slice_dataset(X, dataset_slice),
						Y[dataset_slice]
					)
				end
			
			# Split in train/test
			((X_train, Y_train), (X_test, Y_test)) =
				traintestsplit((X, Y), split_threshold)

			modal_args, (X_train, Y_train), (X_test, Y_test)
		else

			# Unpack dataset
			length(dataset) == 2 || error("Wrong dataset length: $(length(dataset))")
			(X_train, Y_train), (X_test, Y_test) = dataset

			# Apply scaling
			if round_dataset_to_datatype != false
				error("TODO case round_dataset_to_datatype != false not captured yet")
				(X_train, Y_train), (X_test,  Y_test) = roundDataset(((X_train, Y_train), (X_test,  Y_test)), round_dataset_to_datatype)
			end
			
			# Compute mffmd for the training instances
			modal_args, X_train, X_test = buildModalDatasets(modal_args, X_train, X_test)

			# Slice training instances
			X_train, Y_train =
				if isnothing(dataset_slice)
					(X_train, Y_train,)
				else
					(
					ModalLogic.slice_dataset(X_train, dataset_slice),
					Y_train[dataset_slice],
					)
				end

			modal_args, (X_train, Y_train), (X_test, Y_test)
		end

	# println(" n_samples = $(size(X_train)[end-1])")
	println(" train size = $(size(X_train))")
	# global_logger(ConsoleLogger(stderr, Logging.Info))
	# global_logger(ConsoleLogger(stderr, log_level))
	# global_logger(ConsoleLogger(stderr, DecisionTree.DTDebug))

	function display_cm_as_row(cm::ConfusionMatrix)
		"|\t" *
		"$(round(cm.overall_accuracy*100, digits=2))%\t" *
		"$(join(round.(cm.sensitivities.*100, digits=2), "%\t"))%\t" *
		"$(join(round.(cm.PPVs.*100, digits=2), "%\t"))%\t" *
		"||\t" *
		# "$(round(cm.mean_accuracy*100, digits=2))%\t" *
		"$(round(cm.kappa*100, digits=2))%\t" *
		# "$(round(DecisionTree.macro_F1(cm)*100, digits=2))%\t" *
		# "$(round.(cm.accuracies.*100, digits=2))%\t" *
		"$(round.(cm.F1s.*100, digits=2))%\t" *
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
		# # "$(round(DecisionTree.mean_PPV(cm)*100, digits=2))%\t" *
		# "$(round(DecisionTree.macro_weighted_PPV(cm)*100, digits=2))%\t" *
		# # "$(round(DecisionTree.mean_NPV(cm)*100, digits=2))%\t" *
		# "$(round(DecisionTree.macro_weighted_NPV(cm)*100, digits=2))%\t" *
		""
	end

	go_tree(tree_args, rng) = begin
		started = Dates.now()
		T =
			if timing_mode == :none
				build_tree(X_train, Y_train; tree_args..., modal_args..., rng = rng)
			elseif timing_mode == :time
				@time build_tree(X_train, Y_train; tree_args..., modal_args..., rng = rng)
			elseif timing_mode == :btime
				@btime build_tree($X_train, $Y_train; $tree_args..., $modal_args..., rng = $rng)
			end
		Tt = Dates.now() - started
		println("Train tree:")
		print(T)

		if !isnothing(save_tree_path)
			tree_hash = get_hash_sha256(T)
			total_save_path = save_tree_path * "/tree_" * tree_hash * ".jld"
			mkpath(dirname(total_save_path))

			checkpoint_stdout("Saving tree to file $(total_save_path)...")
			JLD2.@save total_save_path T
		end

		if X_train != X_test
			println("Test tree:")
			print_apply_tree(T, X_test, Y_test)
		end

		println(" test size = $(size(X_test))")
		cm = nothing
		for pruning_purity_threshold in sort(unique([(Float64.(tree_post_pruning_purity_thresh))...,1.0]))
			println(" Purity threshold $pruning_purity_threshold")
			
			T_pruned = prune_tree(T, pruning_purity_threshold)
			preds = apply_tree(T_pruned, X_test);
			cm = confusion_matrix(Y_test, preds)
			# @test cm.overall_accuracy > 0.99

			println("RESULT:\t$(name)\t$(tree_args)\t$(modal_args)\t$(pruning_purity_threshold)\t$(display_cm_as_row(cm))")
			
			println(cm)
			# @show cm

			# println("nodes: ($(num_nodes(T_pruned)), height: $(height(T_pruned)))")
		end
		return (T, cm, Tt);
	end

	go_forest(f_args, rng; prebuilt_model::Union{Nothing,AbstractVector{DecisionTree.Forest{S, T}}} = nothing) where {S,T} = begin
		Fs, Ft = 
			if isnothing(prebuilt_model)
				started = Dates.now()
				[
					if timing_mode == :none
						build_forest(X_train, Y_train; f_args..., modal_args..., rng = rng);
					elseif timing_mode == :time
						@time build_forest(X_train, Y_train; f_args..., modal_args..., rng = rng);
					elseif timing_mode == :btime
						@btime build_forest($X_train, $Y_train; $f_args..., $modal_args..., rng = $rng);
					end
					for i in 1:forest_runs
				], (Dates.now() - started)
			else
				println("Using slice of a prebuilt forest.")
				# !!! HUGE PROBLEM HERE !!! #
				# BUG: can't compute oob_error of a forest built slicing another forest!!!
				forests::Vector{DecisionTree.Forest{S, T}} = []
				for f in prebuilt_model
					v_forest = @views f.trees[Random.randperm(rng, length(f.trees))[1:f_args.n_trees]]
					v_cms = @views f.cm[Random.randperm(rng, length(f.cm))[1:f_args.n_trees]]
					push!(forests, DecisionTree.Forest{S, T}(v_forest, v_cms, 0.0))
				end
				forests, Dates.Millisecond(0)
			end

		for F in Fs
			print(F)
		end
		
		cms = []
		for F in Fs
			println(" test size = $(size(X_test))")
			
			preds = apply_forest(F, X_test);
			cm = confusion_matrix(Y_test, preds)
			# @test cm.overall_accuracy > 0.99

			println("RESULT:\t$(name)\t$(f_args)\t$(modal_args)\t$(display_cm_as_row(cm))")

			# println("  accuracy: ", round(cm.overall_accuracy*100, digits=2), "% kappa: ", round(cm.kappa*100, digits=2), "% ")
			for (i,row) in enumerate(eachrow(cm.matrix))
				for val in row
					print(lpad(val,3," "))
				end
				println("  " * "$(round(100*row[i]/sum(row), digits=2))%\t\t" * cm.classes[i])
			end

			println("Forest OOB Error: $(round.(F.oob_error.*100, digits=2))%")

			push!(cms, cm)
		end

		return (Fs, cms, Ft);
	end

	go() = begin
		Ts = []
		Fs = []
		Tcms = []
		Fcms = []
		Tts = []
		Fts = []

		old_logger = global_logger(ConsoleLogger(stderr, log_level))

		for (i_model, this_args) in enumerate(tree_args)
			checkpoint_stdout("Computing Tree $(i_model) / $(length(tree_args))...")
			this_T, this_Tcm, this_Tt = go_tree(this_args, Random.MersenneTwister(train_seed))
			push!(Ts, this_T)
			push!(Tcms, this_Tcm)
			push!(Tts, this_Tt)
		end

		# # TODO
		# if test_flattened == true
		# 	T, Tcm = go_flattened_tree()
		# 	# Flatten 
		# 	(X_train,Y_train), (X_test,Y_test) = dataset
		# 	X_train = ...
		# 	X_test = ...
		# 	dataset = (X_train,Y_train), (X_test,Y_test)
		# end

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

				forest_supports_build_order[i].f, forest_supports_build_order[i].cm, forest_supports_build_order[i].time = go_forest(f.f_args, Random.MersenneTwister(train_seed), prebuilt_model = model)
			end

			# put resulting forests in vector in the order the user gave them
			for (i, f) in enumerate(forest_supports_user_order)
				@assert f.f isa AbstractVector{DecisionTree.Forest{S, T}} where {S,T} "This is not a Vector of Forests! eltype = $(eltype(f.f))"
				@assert f.cm isa AbstractVector{ConfusionMatrix} "This is not a Vector of ConfusionMatrix!"
				@assert length(f.f) == forest_runs "There is a support struct with less than $(forest_runs) forests: $(length(f.f))"
				@assert length(f.cm) == forest_runs "There is a support struct with less than $(forest_runs) confusion matrices: $(length(f.cm))"
				@assert f.f_args == forest_args[i] "f_args mismatch! $(f.f_args) == $(f_args[i])"

				push!(Fs, f.f)
				push!(Fcms, f.cm)
				push!(Fts, f.time)
			end
		else
			for (i_forest, f_args) in enumerate(forest_args)
				checkpoint_stdout("Computing Random Forest $(i_forest) / $(length(forest_args))...")
				this_F, this_Fcm, this_Ft = go_forest(f_args, Random.MersenneTwister(train_seed))
				push!(Fs, this_F)
				push!(Fcms, this_Fcm)
				push!(Fts, this_Ft)
			end
		end

		global_logger(old_logger);

		Ts, Fs, Tcms, Fcms, Tts, Fts
	end

	# try
	go()
	# catch e
	# 	println("testDataset: An error occurred!")
	# 	println(e)
	# 	return;
	# end
end
