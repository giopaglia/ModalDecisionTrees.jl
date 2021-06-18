# The code in this file is a small port from scikit-learn's and numpy's
# library which is distributed under the 3-Clause BSD license.
# The rest of DecisionTree.jl is released under the MIT license.

# written by Poom Chiarawongse <eight1911@gmail.com>

module treeclassifier
	
	export fit

	using ..ModalLogic
	using ..DecisionTree
	using DecisionTree.util
	using Logging: @logmsg
	import Random
	import StatsBase

	mutable struct NodeMeta{U}
		region             :: UnitRange{Int}                   # a slice of the samples used to decide the split of the node
		depth              :: Int
		modal_depth        :: Int
		# worlds      :: AbstractVector{WorldSet{W}}         # current set of worlds for each training instance
		purity             :: U                                # purity grade attained at training time
		label              :: Label                            # most likely label
		is_leaf            :: Bool                             # whether this is a leaf node, or a split one
		# split node-only properties
		split_at           :: Int                              # index of samples
		l                  :: NodeMeta{U}                    # left child
		r                  :: NodeMeta{U}                    # right child
		i_frame            :: Integer          # Id of frame
		relation           :: R where R<:AbstractRelation      # modal operator (e.g. RelationId for the propositional case)
		feature            :: FeatureTypeFun                      # feature used for splitting
		test_operator      :: TestOperatorFun                  # test_operator (e.g. <=)
		threshold          :: T where T                                # threshold value
		onlyUseRelationAll :: Vector{Bool}

		function NodeMeta{U}(
				region      :: UnitRange{Int},
				depth       :: Int,
				modal_depth :: Int,
				oura        :: Vector{Bool},
				) where {U}
			node = new{U}()
			node.region = region
			node.depth = depth
			node.modal_depth = modal_depth
			node.purity = U(NaN)
			node.is_leaf = false
			node.onlyUseRelationAll = oura
			node
		end
	end

	struct Tree{S}
		root           :: NodeMeta{Float64}
		list           :: Vector{S}
		labels         :: Vector{Label}
		initConditions :: Vector{<:DecisionTree._initCondition}
	end

	# Find an optimal local split satisfying the given constraints
	#  (e.g. max_depth, min_samples_leaf, etc.)
	# TODO move this function inside the caller function, and get rid of all parameters
	function _split!(
		Xs                  :: MultiFrameFeatModalDataset, # the modal dataset
		Y                   :: AbstractVector{Label},      # the label array
		W                   :: AbstractVector{U},          # the weight vector
		Ss                  :: AbstractVector{<:AbstractVector{WST} where {WorldType,WST<:WorldSet{WorldType}}}, # the vector of current worlds
		####################
		loss_function       :: Function,
		node                :: NodeMeta{<:AbstractFloat}, # the node to split
		max_depth           :: Int,                         # the maximum depth of the resultant tree
		min_samples_leaf    :: Int,                         # the minimum number of samples each leaf needs to have
		min_loss_at_leaf    :: AbstractFloat,               # maximum purity allowed on a leaf
		min_purity_increase :: AbstractFloat,               # minimum purity increase needed for a split
		####################
		n_subrelations      :: Vector{<:Function},
		n_subfeatures       :: Vector{<:Integer},           # number of features to use to split
		useRelationAll      :: Vector{Bool},
		####################
		indX                :: AbstractVector{<:Integer},   # an array of sample indices (we split using samples in indX[node.region])
		####################
		# Arrays for optimization purposes
		####################
		nc                  :: AbstractVector{U},   # nc maintains a dictionary of all labels in the samples
		ncl                 :: AbstractVector{U},   # ncl maintains the counts of labels on the left
		ncr                 :: AbstractVector{U},   # ncr maintains the counts of labels on the right
		consistency_sat_check :: Union{Nothing,AbstractVector{Bool}},
		####################
		Yf                  :: AbstractVector{Label},
		Wf                  :: AbstractVector{U},
		Sfs                 :: AbstractVector{<:AbstractVector{WST} where {WorldType,WST<:WorldSet{WorldType}}},
		####################
		rng                 :: Random.AbstractRNG,
	) where {U}


		# Region of indX to use to perform the split
		region = node.region
		n_instances = length(region)
		r_start = region.start - 1

		# Class counts
		nc[:] .= zero(U)
		@simd for i in region
			@inbounds nc[Y[indX[i]]] += W[indX[i]]
		end
		nt = sum(nc)
		node.purity = loss_function(nc, nt)
		node.label = argmax(nc) # Assign the most likely label before the split

		@logmsg DTDebug "_split!(...) " n_instances region nt

		# Preemptive leaf conditions
		if (
			# If all the instances belong to the same class, make this a leaf
			 (nc[node.label]       == nt)
			# No binary split can honor min_samples_leaf if there aren't as many as
			#  min_samples_leaf*2 instances in the first place
			 || (min_samples_leaf * 2 >  n_instances)
			# If the node is pure enough, avoid splitting # TODO rename purity to loss
			 || (node.purity          <= min_loss_at_leaf)
			# Honor maximum depth constraint
			 || (max_depth            <= node.depth))
			node.is_leaf = true
			@logmsg DTDetail "leaf created: " (min_samples_leaf * 2 >  n_instances) (nc[node.label] == nt) (node.purity  <= min_loss_at_leaf) (max_depth <= node.depth)
			return
		end

		# Gather all values needed for the current set of instances
		# TODO also slice the dataset?
		@simd for i in 1:n_instances
			Yf[i] = Y[indX[i + r_start]]
			Wf[i] = W[indX[i + r_start]]
		end

		for i_frame in 1:n_frames(Xs)
			@simd for i in 1:n_instances
				Sfs[i_frame][i] = Ss[i_frame][indX[i + r_start]]
			end
		end

		# Optimization-tracking variables
		best_frame = -1
		best_purity__nt = typemin(Float64)
		best_relation = RelationNone
		best_feature = FeatureTypeNone
		best_test_operator = nothing
		best_threshold = nothing
		best_consistency = nothing
		
		#####################
		## Find best split ##
		#####################
		## Test all decisions
		# For each frame (modal dataset)
		for (i_frame,
					(X,
					frame_Sf,
					frame_n_subrelations,
					frame_n_subfeatures,
					frame_useRelationAll,
					frame_onlyUseRelationAll)) in enumerate(zip(frames(Xs), Sfs, n_subrelations, n_subfeatures, useRelationAll, node.onlyUseRelationAll))

			@logmsg DTDetail "  Frame $(best_frame)/$(length(frames(Xs)))"

			allow_propositional_decisions, allow_modal_decisions, allow_global_decisions, modal_relations_inds, features_inds = begin
				
				# Derive subset of features to consider
				# Note: using "sample" function instead of "randperm" allows to insert weights for features which may be wanted in the future 
				features_inds = StatsBase.sample(rng, 1:n_features(X), frame_n_subfeatures, replace = false)
				sort!(features_inds)

				# Derive all available relations
				allow_propositional_decisions, allow_modal_decisions, allow_global_decisions = begin
					if frame_onlyUseRelationAll
						false, false, true
					else
						true, true, frame_useRelationAll
					end
				end

				tot_relations = 0
				if allow_modal_decisions
					tot_relations += length(relations(X))
				end
				if allow_global_decisions
					tot_relations += 1
				end
				
				# Derive subset of relations to consider
				n_subrel = Int(frame_n_subrelations(tot_relations))
				modal_relations_inds = StatsBase.sample(rng, 1:tot_relations, n_subrel, replace = false)
				sort!(modal_relations_inds)

				# Check whether the global relation survived
				if allow_global_decisions
					allow_global_decisions = (tot_relations in modal_relations_inds)
					modal_relations_inds = filter!(r->r≠tot_relations, modal_relations_inds)
					tot_relations = length(modal_relations_inds)
				end
				allow_propositional_decisions, allow_modal_decisions, allow_global_decisions, modal_relations_inds, features_inds
			end
			
			# println(modal_relations_inds)
			# println(features_inds)
			# readline()

			########################################################################
			########################################################################
			########################################################################
			
			for ((relation, test_operator, feature, threshold), aggr_thresholds) in generate_feasible_decisions(X, indX[region], frame_Sf, allow_propositional_decisions, allow_modal_decisions, allow_global_decisions, modal_relations_inds, features_inds)
				
				# println((relation, test_operator, feature, threshold))
				# Re-initialize right class counts
				nr = zero(U)
				ncr .= zero(U)
				if isa(consistency_sat_check,Vector)
					consistency_sat_check[1:n_instances] .= 1
				end
				for i_instance in 1:n_instances
					gamma = aggr_thresholds[i_instance]
					satisfied = ModalLogic.evaluate_thresh_decision(test_operator, gamma, threshold)
					@logmsg DTDetail " instance $i_instance/$n_instances: (f=$(gamma)) -> satisfied = $(satisfied)"
					
					# TODO make this satisfied a fuzzy value
					if !satisfied
						nr += Wf[i_instance]
						ncr[Yf[i_instance]] += Wf[i_instance]
					else
						if isa(consistency_sat_check,Vector)
							consistency_sat_check[i_instance] = 0
						end
					end
				end

				# Calculate left class counts
				ncl .= nc .- ncr
				nl = nt - nr
				@logmsg DTDebug "  (n_left,n_right) = ($nl,$nr)"

				# Honor min_samples_leaf
				if nl >= min_samples_leaf && (n_instances - nl) >= min_samples_leaf
					purity__nt = -(nl * loss_function(ncl, nl) + nr * loss_function(ncr, nr))
					if purity__nt > best_purity__nt # && !isapprox(purity__nt, best_purity__nt)
						#################################
						best_frame          = i_frame
						#################################
						best_purity__nt     = purity__nt
						#################################
						best_relation       = relation
						best_feature        = feature
						best_test_operator  = test_operator
						best_threshold      = threshold
						#################################
						# print((relation, test_operator, feature, threshold))
						# println(" NEW BEST $best_frame, $best_purity__nt/nt")
						@logmsg DTDetail "  Found new optimum in frame $(best_frame): " (best_purity__nt/nt) best_relation best_feature best_test_operator best_threshold
						#################################
						best_consistency  = if isa(consistency_sat_check,Vector)
								consistency_sat_check[1:n_instances]
							else
								nr
						end
						#################################
					end
				end
			end # END decisions
		end # END frame

		# @logmsg DTOverview "purity increase" best_purity__nt/nt node.purity (best_purity__nt/nt + node.purity) (best_purity__nt/nt - node.purity)
		# If the best split is good, partition and split accordingly
		@inbounds if (best_purity__nt == typemin(Float64)
									|| (best_purity__nt/nt + node.purity <= min_purity_increase))
			@logmsg DTDebug " Leaf" best_purity__nt min_purity_increase (best_purity__nt/nt) node.purity ((best_purity__nt/nt) + node.purity)
			node.is_leaf = true
			return
		else
			best_purity = best_purity__nt/nt

			# split the samples into two parts:
			#  ones for which the is satisfied and those for whom it's not
			node.purity         = best_purity
			node.i_frame        = best_frame
			node.relation       = best_relation
			node.feature        = best_feature
			node.test_operator  = best_test_operator
			node.threshold      = best_threshold
			
			# Compute new world sets (= take a modal step)

			# println(decision_str)
			decision_str = display_decision(node.i_frame, node.relation, node.feature, node.test_operator, node.threshold)
			
			# TODO instead of using memory, here, just use two opposite indices and perform substitutions. indj = n_instances
			unsatisfied_flags = fill(1, n_instances)
			for i_instance in 1:n_instances
				# TODO perform step with an OntologicalModalDataset

				# instance = ModalLogic.getInstance(X, node.i_frame, indX[i_instance + r_start])
				X = get_frame(Xs, node.i_frame)
				Sf = Sfs[node.i_frame]
				# instance = ModalLogic.getInstance(X, indX[i_instance + r_start])

				# println(instance)
				# println(Sf[i_instance])
				(satisfied,Ss[node.i_frame][indX[i_instance + r_start]]) = ModalLogic.modalStep(X, indX[i_instance + r_start], Sf[i_instance], node.relation, node.feature, node.test_operator, node.threshold)
				@logmsg DTDetail " [$satisfied] Instance $(i_instance)/$(n_instances)" Sf[i_instance] (if satisfied Ss[node.i_frame][indX[i_instance + r_start]] end)
				# println(satisfied)
				# println(Ss[node.i_frame][indX[i_instance + r_start]])
				# readline()

				# I'm using unsatisfied because sorting puts YES instances first, but TODO use the inverse sorting and use satisfied flag instead
				unsatisfied_flags[i_instance] = !satisfied
			end

			@logmsg DTOverview " Branch ($(sum(unsatisfied_flags))+$(n_instances-sum(unsatisfied_flags))=$(n_instances) samples) on frame $(node.i_frame) with decision: $(decision_str), purity $(node.purity)"

			@logmsg DTDetail " unsatisfied_flags" unsatisfied_flags

			if length(unique(unsatisfied_flags)) == 1
				error("An uninformative split was reached. Something's off\nPurity: $(node.purity)\nSplit: $(decision_str)\nUnsatisfied flags: $(unsatisfied_flags)")
			end
			
			# Check consistency
			consistency = if isa(consistency_sat_check,Vector)
					unsatisfied_flags
				else
					sum(unsatisfied_flags)
			end

			if best_consistency != consistency
				errStr = "Something's wrong with the optimization steps.\n"
				errStr *= "Branch ($(sum(unsatisfied_flags))+$(n_instances-sum(unsatisfied_flags))=$(n_instances) samples) on frame $(node.i_frame) with decision: $(decision_str), purity $(best_purity)\n"
				errStr *= "Different separation was expected:\n"
				if isa(consistency_sat_check,Vector)
					errStr *= "Actual: $(consistency) ($(sum(consistency)))\n"
					errStr *= "Expected: $(best_consistency) ($(sum(best_consistency)))\n"
					diff = best_consistency.-consistency
					errStr *= "Difference: $(diff) ($(sum(abs.(diff))))\n"
				else
					errStr *= "Actual: $(consistency)\n"
					errStr *= "Expected: $(best_consistency)\n"
					diff = best_consistency-consistency
					errStr *= "Difference: $(diff)\n"
				end
				
				# for i in 1:n_instances
					# errStr *= "$(ModalLogic.getChannel(Xs, indX[i + r_start], best_feature))\t$(Sf[i])\t$(!(unsatisfied_flags[i]==1))\t$(Ss[node.i_frame][indX[i + r_start]])\n";
				# end

				# throw or error(Base.ErrorException(errStr))
				println("ERROR! " * errStr)
				error("ERROR! " * errStr)
			end

			@logmsg DTDetail "pre-partition" region indX[region] unsatisfied_flags
			node.split_at = util.partition!(indX, unsatisfied_flags, 0, region)
			@logmsg DTDetail "post-partition" indX[region] node.split_at

			# For debug:
			# indX = rand(1:10, 10)
			# unsatisfied_flags = rand([1,0], 10)
			# partition!(indX, unsatisfied_flags, 0, 1:10)
			
			# Sort [Xf, Yf, Wf, Sf and indX] by Xf
			# util.q_bi_sort!(unsatisfied_flags, indX, 1, n_instances, r_start)
			# node.split_at = searchsortedfirst(unsatisfied_flags, true)
		end
		# println("END split!")
		# readline()
	end

	# Split node at a previously-set node.split_at value.
	# The children inherits some of the data
	@inline function fork!(node::NodeMeta{U}) where {U}
		ind = node.split_at
		region = node.region
		depth = node.depth+1
		mdepth = (node.relation == RelationId ? node.modal_depth : node.modal_depth+1)
		@logmsg DTDetail "fork!(...): " node ind region mdepth

		# onlyUseRelationAll changes:
		# on the left node, the frame where the decision was taken
		l_oura = copy(node.onlyUseRelationAll)
		l_oura[node.i_frame] = false
		r_oura = node.onlyUseRelationAll

		# no need to copy because we will copy at the end
		node.l = NodeMeta{U}(region[    1:ind], depth, mdepth, l_oura)
		node.r = NodeMeta{U}(region[ind+1:end], depth, mdepth, r_oura)
	end

	function check_input(
			Xs                      :: MultiFrameFeatModalDataset,
			Y                       :: AbstractVector{S},
			W                       :: AbstractVector{U},
			##########################################################################
			loss_function           :: Function,
			max_depth               :: Int,
			min_samples_leaf        :: Int,
			min_purity_increase     :: AbstractFloat,
			min_loss_at_leaf        :: AbstractFloat,
			##########################################################################
			n_subrelations          :: Vector{<:Function},
			n_subfeatures           :: Vector{<:Integer},
			initConditions          :: Vector{<:DecisionTree._initCondition},
			useRelationAll          :: Vector{Bool},
		) where {S, U}
		n_instances = n_samples(Xs)

		if length(Y) != n_instances
			error("dimension mismatch between dataset and label vector Y: ($(n_instances)) vs $(size(Y))")
		elseif length(W) != n_instances
			error("dimension mismatch between dataset and weights vector W: ($(n_instances)) vs $(size(W))")
		############################################################################
		elseif length(n_subrelations) != n_frames(Xs)
			error("mismatching number of n_subrelations with number of frames: $(length(n_subrelations)) vs $(n_frames(Xs))")
		elseif length(n_subfeatures)  != n_frames(Xs)
			error("mismatching number of n_subfeatures with number of frames: $(length(n_subfeatures)) vs $(n_frames(Xs))")
		elseif length(initConditions) != n_frames(Xs)
			error("mismatching number of initConditions with number of frames: $(length(initConditions)) vs $(n_frames(Xs))")
		elseif length(useRelationAll) != n_frames(Xs)
			error("mismatching number of useRelationAll with number of frames: $(length(useRelationAll)) vs $(n_frames(Xs))")
		############################################################################
		# elseif any(n_relations(Xs) .< n_subrelations)
		# 	error("in at least one frame the total number of relations is less than the number "
		# 		* "of relations required at each split\n"
		# 		* "# relations:    " * string(n_relations(Xs)) * "\n\tvs\n"
		# 		* "# subrelations: " * string(n_subrelations |> collect))
		# elseif length(findall(n_subrelations .< 0)) > 0
		# 	error("total number of relations $(n_subrelations) must be >= zero ")
		elseif any(n_features(Xs) .< n_subfeatures)
			error("in at least one frame the total number of features is less than the number "
				* "of features required at each split\n"
				* "# features:    " * string(n_features(Xs)) * "\n\tvs\n"
				* "# subfeatures: " * string(n_subfeatures |> collect))
		elseif length(findall(n_subfeatures .< 0)) > 0
			error("total number of features $(n_subfeatures) must be >= zero ")
		elseif min_samples_leaf < 1
			error("min_samples_leaf must be a positive integer "
				* "(given $(min_samples_leaf))")
		# if loss_function in [util.entropy]
		# 	min_loss_at_leaf_thresh = 0.75 # min_purity_increase 0.01
		# 	min_purity_increase_thresh = 0.5
		# 	if (min_loss_at_leaf >= min_loss_at_leaf_thresh)
		# 		println("Warning! It is advised to use min_loss_at_leaf<$(min_loss_at_leaf_thresh) with loss $(loss_function)"
		# 			* "(given $(min_loss_at_leaf))")
		# 	elseif (min_purity_increase >= min_purity_increase_thresh)
		# 		println("Warning! It is advised to use min_loss_at_leaf<$(min_purity_increase_thresh) with loss $(loss_function)"
		# 			* "(given $(min_purity_increase))")
		# end
		elseif loss_function in [util.gini, util.zero_one] && (min_loss_at_leaf > 1.0 || min_loss_at_leaf <= 0.0)
			error("min_loss_at_leaf for loss $(loss_function) must be in (0,1]"
				* "(given $(min_loss_at_leaf))")
		elseif max_depth < -1
			error("unexpected value for max_depth: $(max_depth) (expected:"
				* " max_depth >= 0, or max_depth = -1 for infinite depth)")
		end

		# TODO make sure how missing, nothing, NaN & infinite can be handled
		# if nothing in Xs.fwd
		# 	error("Warning! This algorithm doesn't allow nothing values in Xs.fwd")
		# elseif any(isnan.(Xs.fwd)) # TODO make sure that this does its job.
		# 	error("Warning! This algorithm doesn't allow NaN values in Xs.fwd")
		# else
		if nothing in Y
			error("Warning! This algorithm doesn't allow nothing values in Y")
		# elseif any(isnan.(Y))
		# 	error("Warning! This algorithm doesn't allow NaN values in Y")
		elseif nothing in W
			error("Warning! This algorithm doesn't allow nothing values in W")
		elseif any(isnan.(W))
			error("Warning! This algorithm doesn't allow NaN values in W")
		end

	end

	# function optimize_tree_parameters!(
	# 		X               :: OntologicalDataset{T, N},
	# 		initCondition   :: DecisionTree._initCondition,
	# 		useRelationAll  :: Bool,
	# 		test_operators  :: AbstractVector{<:TestOperator}
	# 	) where {T, N}

	# 	# Adimensional ontological datasets:
	# 	#  flatten to adimensional case + strip of all relations from the ontology
	# 	if prod(channel_size(X)) == 1
	# 		if (length(ontology(X).relationSet) > 0)
	# 			warn("The OntologicalDataset provided has degenerate channel_size $(channel_size(X)), and more than 0 relations: $(ontology(X).relationSet).")
	# 		end
	# 		# X = OntologicalDataset{T, 0}(ModalLogic.strip_ontology(ontology(X)), @views ModalLogic.strip_domain(domain(X)))
	# 	end

	# 	ontology_relations = deepcopy(ontology(X).relationSet)

	# 	# Fix test_operators order
	# 	test_operators = unique(test_operators)
	# 	ModalLogic.sort_test_operators!(test_operators)
		
	# 	# Adimensional operators:
	# 	#  in the adimensional case, some pairs of operators (e.g. <= and >)
	# 	#  are complementary, and thus it is redundant to check both at the same node.
	# 	#  We avoid this by only keeping one of the two operators.
	# 	if prod(channel_size(X)) == 1
	# 		# No ontological relation
	# 		ontology_relations = []
	# 		if test_operators ⊆ ModalLogic.all_lowlevel_test_operators
	# 			test_operators = [TestOpGeq]
	# 			# test_operators = filter(e->e ≠ TestOpGeq,test_operators)
	# 		else
	# 			warn("Test operators set includes non-lowlevel test operators. Update this part of the code accordingly.")
	# 		end
	# 	end

	# 	# Softened operators:
	# 	#  when the biggest world only has a few values, softened operators fallback
	# 	#  to being hard operators
	# 	# max_world_wratio = 1/prod(max_channel_size(X))
	# 	# if TestOpGeq in test_operators
	# 	# 	test_operators = filter((e)->(typeof(e) != _TestOpGeqSoft || e.alpha < 1-max_world_wratio), test_operators)
	# 	# end
	# 	# if TestOpLeq in test_operators
	# 	# 	test_operators = filter((e)->(typeof(e) != _TestOpLeqSoft || e.alpha < 1-max_world_wratio), test_operators)
	# 	# end


	# 	# Binary relations (= unary modal operators)
	# 	# Note: the identity relation is the first, and it is the one representing
	# 	#  propositional splits.
		
	# 	if RelationId in ontology_relations
	# 		error("Found RelationId in ontology provided. No need.")
	# 		# ontology_relations = filter(e->e ≠ RelationId, ontology_relations)
	# 	end

	# 	if RelationAll in ontology_relations
	# 		error("Found RelationAll in ontology provided. Use useRelationAll = true instead.")
	# 		# ontology_relations = filter(e->e ≠ RelationAll, ontology_relations)
	# 		# useRelationAll = true
	# 	end

	# 	relationSet = [RelationId, RelationAll, ontology_relations...]
	# 	relationId_id = 1
	# 	relationAll_id = 2
	# 	ontology_relation_ids = map((x)->x+2, 1:length(ontology_relations))

	# 	needToComputeRelationAll = (useRelationAll || (initCondition == startWithRelationAll))

	# 	# Modal relations to compute gammas for
	# 	inUseRelation_ids = if needToComputeRelationAll
	# 		[relationAll_id, ontology_relation_ids...]
	# 	else
	# 		ontology_relation_ids
	# 	end

	# 	# Relations to use at each split
	# 	availableRelation_ids = []

	# 	push!(availableRelation_ids, relationId_id)
	# 	if useRelationAll
	# 		push!(availableRelation_ids, relationAll_id)
	# 	end

	# 	availableRelation_ids = [availableRelation_ids..., ontology_relation_ids...]

	# 	(
	# 		test_operators, relationSet,
	# 		relationId_id, relationAll_id,
	# 		inUseRelation_ids, availableRelation_ids
	# 	)
	# end

	function _fit(
			Xs                      :: MultiFrameFeatModalDataset,
			Y                       :: AbstractVector{Label},
			W                       :: AbstractVector{U},
			##########################################################################
			loss_function           :: Function,
			n_classes               :: Int,
			max_depth               :: Int,
			min_samples_leaf        :: Int, # TODO generalize to min_samples_leaf_relative and min_weight_leaf
			min_purity_increase     :: AbstractFloat,
			min_loss_at_leaf        :: AbstractFloat,
			##########################################################################
			n_subrelations          :: Vector{<:Function},
			n_subfeatures           :: Vector{<:Integer},
			initConditions          :: Vector{<:DecisionTree._initCondition},
			useRelationAll          :: Vector{Bool},
			##########################################################################
			rng = Random.GLOBAL_RNG :: Random.AbstractRNG
		) where {U}

		n_instances = n_samples(Xs)

		# TODO consolidate functions like this
		init_world_sets(Xs::MultiFrameFeatModalDataset, initConditions::AbstractVector{<:DecisionTree._initCondition}) = begin
			Ss = Vector{Vector{WST} where {WorldType,WST<:WorldSet{WorldType}}}(undef, n_frames(Xs))
			for (i_frame,X) in enumerate(ModalLogic.frames(Xs))
				WT = world_type(X)
				Ss[i_frame] = WorldSet{WT}[initws_function(X, i_instance)(initConditions[i_frame]) for i_instance in 1:n_samples(Xs)]
			end
			Ss
		end

		# Initialize world sets for each instance
		Ss = init_world_sets(Xs, initConditions)


		# Memory support for class counts
		nc  = Vector{U}(undef, n_classes)
		ncl = Vector{U}(undef, n_classes)
		ncr = Vector{U}(undef, n_classes)
		consistency_sat_check = Vector{Bool}(undef, n_instances)
		# consistency_sat_check = nothing

		# Memory support for worldsets, labels, weights 
		# Xf = Array{T, N+1}(undef, channel_size(X)..., n_instances)
		Sfs = Vector{Vector{WST} where {WorldType,WST<:WorldSet{WorldType}}}(undef, n_frames(Xs))
		for i_frame in 1:n_frames(Xs)
			WT = world_type(Xs, i_frame)
			Sfs[i_frame] = Vector{WorldSet{WT}}(undef, n_instances)
		end
		Yf = Vector{Label}(undef, n_instances)
		Wf = Vector{U}(undef, n_instances)
		
		# Memory support for the instances distribution throughout the tree
		#  this is an array of indices that will be recursively permuted and partitioned
		indX = collect(1:n_instances)
		
		# TODO
		# DecisionTree.checkGammasConsistency(gammas[i], get_frame(Xs, i), test_operators, relationSet)

		# Let the core algorithm begin!

		# Create root node
		onlyUseRelationAll = [(iC == startWithRelationAll) for iC in initConditions]
		root = NodeMeta{Float64}(1:n_instances, 0, 0, onlyUseRelationAll)
		# Process stack of nodes
		stack = NodeMeta{Float64}[root]
		@inbounds while length(stack) > 0
			# Pop node and process it
			node = pop!(stack)
			_split!(
				Xs,
				Y,
				W,
				Ss,
				########################################################################
				loss_function,
				node,
				max_depth,
				min_samples_leaf,
				min_loss_at_leaf,
				min_purity_increase,
				########################################################################
				n_subrelations,
				n_subfeatures,
				useRelationAll,
				########################################################################
				indX,
				nc,
				ncl,
				ncr,
				consistency_sat_check,
				########################################################################
				Yf,
				Wf,
				Sfs,
				########################################################################
				rng
			)
			# After processing, if needed, perform the split and push the two children for a later processing step
			if !node.is_leaf
				fork!(node)
				# Note: the left (positive) child is not limited to RelationAll, whereas the right child is only if the current node is as well.
				push!(stack, node.l)
				push!(stack, node.r)
			end
		end

		return (root, indX)
	end

	# In the modal case, dataset instances are Kripke models.
	# In this implementation, we don't accept a generic Kripke model in the explicit form of
	#  a graph; instead, an instance is a dimensional domain (e.g. a matrix or a 3D matrix) onto which
	#  worlds and relations are determined by a given Ontology.

	function fit(;
			# TODO Add default values for this function? loss_function = util.entropy
			Xs                      :: MultiFrameFeatModalDataset,
			Y                       :: AbstractVector{S},
			W                       :: Union{AbstractVector{U},Nothing},
			##########################################################################
			# Logic-agnostic training parameters
			loss_function           :: Function,
			max_depth               :: Int,
			min_samples_leaf        :: Int,
			min_purity_increase     :: AbstractFloat,
			min_loss_at_leaf        :: AbstractFloat, # TODO add this to scikit's interface.
			##########################################################################
			# Modal parameters
			n_subrelations          :: Vector{<:Function},
			n_subfeatures           :: Vector{<:Integer},
			initConditions          :: Vector{<:DecisionTree._initCondition},
			useRelationAll          :: Vector{Bool},
			# TODO add consistency_step_data
			##########################################################################
			rng = Random.GLOBAL_RNG :: Random.AbstractRNG
		) where {S, U}

		# TODO ...
		# if isnothing(consistency_step_data)
		# 	gammas = fill(nothing, n_frames(Xs))
		# else
		# 	@assert n_frames(Xs) == length(gammas) "MultiFrameFeatModalDataset and gammas sizes mismatch: n_frames(Xs) = $(n_frames(Xs)); length(gammas) = $(length(gammas))"
		# end

		# Use unary weights if no weight is supplied
		if isnothing(W)
			# TODO optimize w in the case of all-ones: write a subtype of AbstractVector:
			#  AllOnesVector, so that getindex(W, i) = 1 and sum(W) = size(W).
			#  This allows the compiler to optimize constants at compile-time
			W = fill(1, n_samples(Xs))
		end

		# Check validity of the input
		check_input(
			Xs,
			Y,
			W,
			##########################################################################
			loss_function,
			max_depth,
			min_samples_leaf,
			min_purity_increase,
			min_loss_at_leaf,
			##########################################################################
			n_subrelations,
			n_subfeatures,
			initConditions,
			useRelationAll,
		)

		# Transform labels to categorical form
		labels, Y_ = util.assign(Y)
		# print(labels, Y_)

		# Call core learning function
		root, indX = _fit(
				Xs,
				Y_,
				W,
				########################################################################
				loss_function,
				length(labels),
				max_depth,
				min_samples_leaf,
				min_purity_increase,
				min_loss_at_leaf,
				########################################################################
				n_subrelations,
				n_subfeatures,
				initConditions,
				useRelationAll,
				########################################################################
				rng,
		)
		
		# Create tree with labels and categorical leaves
		return Tree{S}(root, labels, indX, initConditions)
	end
end
