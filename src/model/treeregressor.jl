# The code in this file is a small port from scikit-learn's and numpy's
# library which is distributed under the 3-Clause BSD license.
# The rest of DecisionTree.jl is released under the MIT license.

# written by Poom Chiarawongse <eight1911@gmail.com>

module treeregressor
	
	import DecisionTree: fit

	using ..ModalLogic
	using ..DecisionTree
	using DecisionTree.util
	const L = DecisionTree.RegressionLabel
	using Logging: @logmsg
	import Random
	import StatsBase
	using StructuredArrays # , FillArrays # TODO choose one

	mutable struct NodeMeta{U}
		region             :: UnitRange{Int}                   # a slice of the samples used to decide the split of the node
		depth              :: Int
		modal_depth        :: Int
		# worlds      :: AbstractVector{WorldSet{W}}         # current set of worlds for each training instance
		purity             :: U                                # purity grade attained at training time
		label              :: L                            # most likely label
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
		
		onlyallowRelationGlob:: Vector{Bool}

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
			node.onlyallowRelationGlob = oura
			node
		end
	end

	struct Tree{S}
		root           :: NodeMeta{Float64}
		list           :: Vector{S}
		initConditions :: Vector{<:DecisionTree._initCondition}
	end

	# Find an optimal local split satisfying the given constraints
	#  (e.g. max_depth, min_samples_leaf, etc.)
	# TODO move this function inside the caller function, and get rid of all parameters
	Base.@propagate_inbounds function _split!(
		node                  :: NodeMeta{<:AbstractFloat}, # the node to split
		####################
		Xs                    :: MultiFrameModalDataset, # the modal dataset
		Y                     :: AbstractVector{L},      # the label array
		W                     :: AbstractVector{U},          # the weight vector
		Ss                    :: AbstractVector{<:AbstractVector{WST} where {WorldType,WST<:WorldSet{WorldType}}}, # the vector of current worlds
		####################
		loss_function         :: Function,
		max_depth             :: Int,                         # the maximum depth of the resultant tree
		min_samples_leaf      :: Int,                         # the minimum number of samples each leaf needs to have
		max_purity_at_leaf    :: AbstractFloat,               # maximum purity allowed on a leaf
		min_purity_increase   :: AbstractFloat,               # minimum purity increase needed for a split
		####################
		n_subrelations        :: Vector{<:Function},
		n_subfeatures         :: Vector{<:Integer},           # number of features to use to split
		allowRelationGlob     :: Vector{Bool},
		####################
		indX                  :: AbstractVector{<:Integer},   # an array of sample indices (we split using samples in indX[node.region])
		####################
		_perform_consistency_check :: Union{Val{true},Val{false}},
		####################
		writing_lock          :: Threads.Condition,
		####################
		rng                   :: Random.AbstractRNG,
	) where {U}
		
		# Region of indX to use to perform the split
		region = node.region
		n_instances = length(region)
		r_start = region.start - 1

		# Gather all values needed for the current set of instances
		# TODO also slice the dataset?
		@inbounds Yf = Y[indX[region]]
		@inbounds Wf = W[indX[region]]

		# Yf = Vector{L}(undef, n_instances)
		# Wf = Vector{U}(undef, n_instances)
		# @inbounds @simd for i in 1:n_instances
		# 	Yf[i] = Y[indX[i + r_start]]
		# 	Wf[i] = W[indX[i + r_start]]
		# end

		############################################################################
		# Prepare counts
		############################################################################
		sums, (tsum, nt),
		(node.purity, node.label) = begin
			sums = [Wf[i]*Yf[i]       for i in 1:n_instances]
			# ssqs = [Wf[i]*Yf[i]*Yf[i] for i in 1:n_instances]
			
			# tssq = zero(U)
			# tssq = sum(ssqs)
			# tsum = zero(U)
			tsum = sum(sums)
			# nt = zero(U)
			nt = sum(Wf)
			# @inbounds @simd for i in 1:n_instances
			# 	# tssq += Wf[i]*Yf[i]*Yf[i]
			# 	# tsum += Wf[i]*Yf[i]
			# 	nt += Wf[i]
			# end

			# purity = (tsum * label) # TODO use loss function
			# purity = tsum * tsum # TODO use loss function
			# tmean = tsum/nt
			# purity = -((tssq - 2*tmean*tsum + (tmean^2*nt)) / (nt-1)) # TODO use loss function
			purity = -(loss_function(sums, nt))
			label =  tsum / nt # Assign the most likely label before the split
			sums, (tsum, nt), (purity, label)
		end
		############################################################################
		############################################################################
		############################################################################

		@logmsg DTDebug "_split!(...) " n_instances region nt

		############################################################################
		# Preemptive leaf conditions
		############################################################################
		if (
			# No binary split can honor min_samples_leaf if there aren't as many as
			#  min_samples_leaf*2 instances in the first place
			    (min_samples_leaf * 2 >  n_instances)
		  # equivalent to old_purity > -1e-7
			 # || (tsum * node.label    > -1e-7 * nt + tssq)
			# Honor maximum depth constraint
			 || (max_depth            <= node.depth))
			node.is_leaf = true
			@logmsg DTDetail "leaf created: " (min_samples_leaf * 2 >  n_instances) (tsum * node.label    > -1e-7 * nt + tssq) (tsum * node.label) (-1e-7 * nt + tssq) (max_depth <= node.depth)
			return
		end
		############################################################################
		############################################################################
		############################################################################

		# TODO try this solution for rsums and lsums
		# rsums = Vector{U}(undef, n_instances)
		# lsums = Vector{U}(undef, n_instances)
		# @simd for i in 1:n_instances
		# 	rsums[i] = zero(U)
		# 	lsums[i] = zero(U)
		# end

		Sfs = Vector{AbstractVector{WST} where {WorldType,WST<:AbstractVector{WorldType}}}(undef, n_frames(Xs))
		for i_frame in 1:n_frames(Xs)
			Sfs[i_frame] = Vector{Vector{world_type(Xs, i_frame)}}(undef, n_instances)
			@simd for i in 1:n_instances
				Sfs[i_frame][i] = Ss[i_frame][indX[i + r_start]]
			end
		end

		# Optimization-tracking variables
		best_i_frame = -1
		best_purity_times_nt = typemin(Float64)
		best_relation = RelationNone
		best_feature = FeatureTypeNone
		best_test_operator = nothing
		best_threshold = nothing
		if isa(_perform_consistency_check,Val{true})
			consistency_sat_check = Vector{Bool}(undef, n_instances)
		end
		best_consistency = nothing
		
		#####################
		## Find best split ##
		#####################
		## Test all decisions
		# For each frame (modal dataset)
		@inbounds for (i_frame,
					(X,
					frame_Sf,
					frame_n_subrelations,
					frame_n_subfeatures,
					frame_allowRelationGlob,
					frame_onlyallowRelationGlob)) in enumerate(zip(frames(Xs), Sfs, n_subrelations, n_subfeatures, allowRelationGlob, node.onlyallowRelationGlob))

			@logmsg DTDetail "  Frame $(best_i_frame)/$(length(frames(Xs)))"

			allow_propositional_decisions, allow_modal_decisions, allow_global_decisions, modal_relations_inds, features_inds = begin
				
				# Derive subset of features to consider
				# Note: using "sample" function instead of "randperm" allows to insert weights for features which may be wanted in the future 
				features_inds = StatsBase.sample(rng, 1:n_features(X), frame_n_subfeatures, replace = false)
				sort!(features_inds)

				# Derive all available relations
				allow_propositional_decisions, allow_modal_decisions, allow_global_decisions = begin
					if world_type(X) == OneWorld
						true, false, false
					elseif frame_onlyallowRelationGlob
						false, false, true
					else
						true, true, frame_allowRelationGlob
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
			
			@inbounds for ((relation, feature, test_operator, threshold), aggr_thresholds) in generate_feasible_decisions(X, indX[region], frame_Sf, allow_propositional_decisions, allow_modal_decisions, allow_global_decisions, modal_relations_inds, features_inds)
				
				# println(display_decision(i_frame, relation, feature, test_operator, threshold))

				########################################################################
				# Apply decision to all instances
				########################################################################
				(rsums, nr, lsums, nl, rsum, lsum), consistency_sat_check = begin
					# Initialize right counts
					# rssq = zero(U)
					rsum = zero(U)
					nr   = zero(U)
					# TODO experiment with running mean instead, because this may cause a lot of memory inefficiency
					# https://it.wikipedia.org/wiki/Algoritmi_per_il_calcolo_della_varianza
					rsums = Float64[] # Vector{U}(undef, n_instances)
					lsums = Float64[] # Vector{U}(undef, n_instances)

					if isa(_perform_consistency_check,Val{true})
						consistency_sat_check .= 1
					end
					for i_instance in 1:n_instances
						gamma = aggr_thresholds[i_instance]
						satisfied = ModalLogic.evaluate_thresh_decision(test_operator, gamma, threshold)
						@logmsg DTDetail " instance $i_instance/$n_instances: (f=$(gamma)) -> satisfied = $(satisfied)"
						
						# TODO make this satisfied a fuzzy value
						if !satisfied
							push!(rsums, sums[i_instance])
							# rsums[i_instance] = sums[i_instance]
							nr   += Wf[i_instance]
							rsum += sums[i_instance]
							# rssq += ssqs[i_instance]
						else
							push!(lsums, sums[i_instance])
							# lsums[i_instance] = sums[i_instance]
							if isa(_perform_consistency_check,Val{true})
								consistency_sat_check[i_instance] = 0
							end
						end
					end

					# Calculate left counts
					# lsum = tsum - rsum
					# lssq = tssq - rssq
					nl   = nt - nr
					
					(rsums, nr, lsums, nl, rsum, lsum), consistency_sat_check
				end

				@logmsg DTDebug "  (n_left,n_right) = ($nl,$nr)"

				########################################################################
				########################################################################
				########################################################################
				
				# Honor min_samples_leaf
				if nl >= min_samples_leaf && (n_instances - nl) >= min_samples_leaf
					purity_times_nt = begin
						- (nl * loss_function(lsums, nl) + nr * loss_function(rsums, nr))
											# TODO use loss_function instead
											# ORIGINAL: TODO understand how it works
											# purity_times_nt = (rsum * rsum / nr) + (lsum * lsum / nl)
											# Variance with ssqs
											# purity_times_nt = (rmean, lmean = rsum/nr, lsum/nl; - (nr * (rssq - 2*rmean*rsum + (rmean^2*nr)) / (nr-1) + (nl * (lssq - 2*lmean*lsum + (lmean^2*nl)) / (nl-1))))
											# Variance
											# var = (x)->sum((x.-StatsBase.mean(x)).^2) / (length(x)-1)
											# purity_times_nt = - (nr * var(rsums)) + nl * var(lsums))
											# Simil-variance that is easier to compute but it doesn't work with few samples on the leaves
											# var = (x)->sum((x.-StatsBase.mean(x)).^2)
											# purity_times_nt = - (var(rsums) + var(lsums))
											# println("purity_times_nt: $(purity_times_nt)")
					end
					if purity_times_nt > best_purity_times_nt # && !isapprox(purity_times_nt, best_purity_times_nt)
						#################################
						best_i_frame             = i_frame
						#################################
						best_purity_times_nt     = purity_times_nt
						#################################
						best_relation            = relation
						best_feature             = feature
						best_test_operator       = test_operator
						best_threshold           = threshold
						#################################
						# print((relation, test_operator, feature, threshold))
						# println(" NEW BEST $best_i_frame, $best_purity_times_nt/nt")
						@logmsg DTDetail "  Found new optimum in frame $(best_i_frame): " (best_purity_times_nt/nt) best_relation best_feature best_test_operator best_threshold
						#################################
						best_consistency = begin
							if isa(_perform_consistency_check,Val{true})
								consistency_sat_check[1:n_instances]
							else
								nr
							end
						end
						#################################
					end
				end
			end # END decisions
		end # END frame

		# println("best_purity_times_nt = $(best_purity_times_nt)")
		# println("nt =  $(nt)")
		# println("node.purity =  $(node.purity)")
		# println("best_purity_times_nt / nt - node.purity = $(best_purity_times_nt / nt - node.purity)")
		# println("min_purity_increase * nt =  $(min_purity_increase) * $(nt) = $(min_purity_increase * nt)")

		# @logmsg DTOverview "purity_times_nt increase" best_purity_times_nt/nt node.purity (best_purity_times_nt + node.purity) (best_purity_times_nt - node.purity)
		# If the best split is good, partition and split accordingly
		@inbounds if (
			##########################################################################
			##########################################################################
			##########################################################################
			best_purity_times_nt == typemin(Float64)
									# || best_purity_times_nt - tsum * node.label <= min_purity_increase * nt # ORIGINAL
									|| (best_purity_times_nt / nt - node.purity <= min_purity_increase * nt)
								)
			@logmsg DTDebug " Leaf" best_purity_times_nt tsum node.label min_purity_increase nt (best_purity_times_nt / nt - tsum * node.label) (min_purity_increase * nt)
			##########################################################################
			##########################################################################
			##########################################################################
			node.is_leaf = true
			return
		else
			best_purity = best_purity_times_nt/nt
			
			# Compute new world sets (= take a modal step)

			# println(decision_str)
			decision_str = display_decision(best_i_frame, best_relation, best_feature, best_test_operator, best_threshold)

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
				_sat, _ss = ModalLogic.modal_step(X, indX[i_instance + r_start], Sf[i_instance], node.relation, node.feature, node.test_operator, node.threshold)
				Threads.lock(writing_lock)
				(satisfied,Ss[node.i_frame][indX[i_instance + r_start]]) = _sat, _ss
				Threads.unlock(writing_lock)
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
				throw_n_log("An uninformative split was reached. Something's off\nPurity: $(node.purity)\nSplit: $(decision_str)\nUnsatisfied flags: $(unsatisfied_flags)")
			end
			
			# Check consistency
			consistency = if isa(_perform_consistency_check,Val{true})
					unsatisfied_flags
				else
					sum(unsatisfied_flags)
			end

			if best_consistency != consistency
				errStr = "Something's wrong with the optimization steps for relation $(node.relation), feature $(node.feature) and test_operator $(node.test_operator).\n"
				errStr *= "Branch ($(sum(unsatisfied_flags))+$(n_instances-sum(unsatisfied_flags))=$(n_instances) samples) on frame $(node.i_frame) with decision: $(decision_str), purity $(best_purity)\n"
				errStr *= "Different partition was expected:\n"
				if isa(_perform_consistency_check,Val{true})
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

				throw_n_log("ERROR! " * errStr)
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

		# onlyallowRelationGlob changes:
		# on the left node, the frame where the decision was taken
		l_oura = copy(node.onlyallowRelationGlob)
		l_oura[node.i_frame] = false
		r_oura = node.onlyallowRelationGlob

		# no need to copy because we will copy at the end
		node.l = NodeMeta{U}(region[    1:ind], depth, mdepth, l_oura)
		node.r = NodeMeta{U}(region[ind+1:end], depth, mdepth, r_oura)
	end

	function check_input(
			Xs                      :: MultiFrameModalDataset,
			Y                       :: AbstractVector{S},
			W                       :: AbstractVector{U},
			##########################################################################
			loss_function           :: Function,
			max_depth               :: Int,
			min_samples_leaf        :: Int,
			min_purity_increase     :: AbstractFloat,
			max_purity_at_leaf      :: AbstractFloat,
			##########################################################################
			n_subrelations          :: Vector{<:Function},
			n_subfeatures           :: Vector{<:Integer},
			initConditions          :: Vector{<:DecisionTree._initCondition},
			allowRelationGlob       :: Vector{Bool},
		) where {S, U}
		n_instances = n_samples(Xs)

		if length(Y) != n_instances
			throw_n_log("dimension mismatch between dataset and label vector Y: ($(n_instances)) vs $(size(Y))")
		elseif length(W) != n_instances
			throw_n_log("dimension mismatch between dataset and weights vector W: ($(n_instances)) vs $(size(W))")
		############################################################################
		elseif length(n_subrelations) != n_frames(Xs)
			throw_n_log("mismatching number of n_subrelations with number of frames: $(length(n_subrelations)) vs $(n_frames(Xs))")
		elseif length(n_subfeatures)  != n_frames(Xs)
			throw_n_log("mismatching number of n_subfeatures with number of frames: $(length(n_subfeatures)) vs $(n_frames(Xs))")
		elseif length(initConditions) != n_frames(Xs)
			throw_n_log("mismatching number of initConditions with number of frames: $(length(initConditions)) vs $(n_frames(Xs))")
		elseif length(allowRelationGlob) != n_frames(Xs)
			throw_n_log("mismatching number of allowRelationGlob with number of frames: $(length(allowRelationGlob)) vs $(n_frames(Xs))")
		############################################################################
		# elseif any(n_relations(Xs) .< n_subrelations)
		# 	throw_n_log("in at least one frame the total number of relations is less than the number "
		# 		* "of relations required at each split\n"
		# 		* "# relations:    " * string(n_relations(Xs)) * "\n\tvs\n"
		# 		* "# subrelations: " * string(n_subrelations |> collect))
		# elseif length(findall(n_subrelations .< 0)) > 0
		# 	throw_n_log("total number of relations $(n_subrelations) must be >= zero ")
		elseif any(n_features(Xs) .< n_subfeatures)
			throw_n_log("in at least one frame the total number of features is less than the number "
				* "of features required at each split\n"
				* "# features:    " * string(n_features(Xs)) * "\n\tvs\n"
				* "# subfeatures: " * string(n_subfeatures |> collect))
		elseif length(findall(n_subfeatures .< 0)) > 0
			throw_n_log("total number of features $(n_subfeatures) must be >= zero ")
		elseif min_samples_leaf < 1
			throw_n_log("min_samples_leaf must be a positive integer "
				* "(given $(min_samples_leaf))")
		elseif loss_function in [util.gini, util.zero_one] && (max_purity_at_leaf > 1.0 || max_purity_at_leaf <= 0.0)
			throw_n_log("max_purity_at_leaf for loss $(loss_function) must be in (0,1]"
				* "(given $(max_purity_at_leaf))")
		elseif max_depth < -1
			throw_n_log("unexpected value for max_depth: $(max_depth) (expected:"
				* " max_depth >= 0, or max_depth = -1 for infinite depth)")
		end

		# TODO make sure how missing, nothing, NaN & infinite can be handled
		# if nothing in Xs.fwd
		# 	throw_n_log("Warning! This algorithm doesn't allow nothing values in Xs.fwd")
		# elseif any(isnan.(Xs.fwd)) # TODO make sure that this does its job.
		# 	throw_n_log("Warning! This algorithm doesn't allow NaN values in Xs.fwd")
		# else
		if nothing in Y
			throw_n_log("Warning! This algorithm doesn't allow nothing values in Y")
		# elseif any(isnan.(Y))
		# 	throw_n_log("Warning! This algorithm doesn't allow NaN values in Y")
		elseif nothing in W
			throw_n_log("Warning! This algorithm doesn't allow nothing values in W")
		elseif any(isnan.(W))
			throw_n_log("Warning! This algorithm doesn't allow NaN values in W")
		end

	end


	function _fit(
			Xs                      :: MultiFrameModalDataset,
			Y                       :: AbstractVector{L},
			W                       :: AbstractVector{U},
			##########################################################################
			loss_function           :: Function,
			max_depth               :: Int,
			min_samples_leaf        :: Int, # TODO generalize to min_samples_leaf_relative and min_weight_leaf
			min_purity_increase     :: AbstractFloat,
			max_purity_at_leaf      :: AbstractFloat,
			##########################################################################
			n_subrelations          :: Vector{<:Function},
			n_subfeatures           :: Vector{<:Integer},
			initConditions          :: Vector{<:DecisionTree._initCondition},
			allowRelationGlob       :: Vector{Bool},
			##########################################################################
			_perform_consistency_check :: Union{Val{true},Val{false}},
			##########################################################################
			rng = Random.GLOBAL_RNG :: Random.AbstractRNG
		) where {U}

		n_instances = n_samples(Xs)

		# TODO consolidate functions like this
		init_world_sets(Xs::MultiFrameModalDataset, initConditions::AbstractVector{<:DecisionTree._initCondition}) = begin
			Ss = Vector{Vector{WST} where {WorldType,WST<:WorldSet{WorldType}}}(undef, n_frames(Xs))
			for (i_frame,X) in enumerate(ModalLogic.frames(Xs))
				WT = world_type(X)
				Ss[i_frame] = WorldSet{WT}[initws_function(X, i_instance)(initConditions[i_frame]) for i_instance in 1:n_samples(Xs)]
				# Ss[i_frame] = WorldSet{WT}[[ModalLogic.Interval(1,2)] for i_instance in 1:n_samples(Xs)]
			end
			Ss
		end
		
		# Initialize world sets for each instance
		Ss = init_world_sets(Xs, initConditions)

		# Memory support for the instances distribution throughout the tree
		#  this is an array of indices that will be recursively permuted and partitioned
		indX = collect(1:n_instances)
		
		# Let the core algorithm begin!

		# Create root node
		onlyallowRelationGlob = [(iC == startWithRelationGlob) for iC in initConditions]
		root = NodeMeta{Float64}(1:n_instances, 0, 0, onlyallowRelationGlob)
		# Process stack of nodes
		stack = NodeMeta{Float64}[root]
		currently_processed_nodes::Vector{NodeMeta{Float64}} = []
		writing_lock = Threads.Condition()
		@inbounds while length(stack) > 0
			rngs = [DecisionTree.spawn_rng(rng) for _n in 1:length(stack)]
			# Pop nodes and queue them to be processed
			while length(stack) > 0
				push!(currently_processed_nodes, pop!(stack))
			end
			Threads.@threads for (i_node, node) in collect(enumerate(currently_processed_nodes))
				_split!(
					node,
					######################################################################
					Xs,
					Y,
					W,
					Ss,
					######################################################################
					loss_function,
					max_depth,
					min_samples_leaf,
					max_purity_at_leaf,
					min_purity_increase,
					######################################################################
					n_subrelations,
					n_subfeatures,
					allowRelationGlob,
					######################################################################
					indX,
					######################################################################
					_perform_consistency_check,
					######################################################################
					writing_lock,
					######################################################################
					rngs[i_node]
				)
			end
			# After processing, if needed, perform the split and push the two children for a later processing step
			for node in currently_processed_nodes
				if !node.is_leaf
					fork!(node)
					# Note: the left (positive) child is not limited to RelationGlob, whereas the right child is only if the current node is as well.
					push!(stack, node.l)
					push!(stack, node.r)
				end
			end
			empty!(currently_processed_nodes)
		end

		return (root, indX)
	end

	# In the modal case, dataset instances are Kripke models.
	# In this implementation, we don't accept a generic Kripke model in the explicit form of
	#  a graph; instead, an instance is a dimensional domain (e.g. a matrix or a 3D matrix) onto which
	#  worlds and relations are determined by a given Ontology.

	function DecisionTree.fit(
			# TODO Add default values for this function?
			Xs                      :: MultiFrameModalDataset,
			Y                       :: AbstractVector{S},
			# Use unary weights if no weight is supplied
			W                       :: AbstractVector{U} = UniformArray{Int}(1,n_samples(Xs)) # from StructuredArrays
			;
			# W                       :: AbstractVector{U} = fill(1, n_samples(Xs)),
			# W                       :: AbstractVector{U} = Ones{Int}(n_samples(Xs)),      # from FillArrays
			##########################################################################
			# Logic-agnostic training parameters
			loss_function           :: Union{Nothing,Function} = nothing,
			max_depth               :: Int,
			min_samples_leaf        :: Int,
			min_purity_increase     :: AbstractFloat,
			max_purity_at_leaf      :: AbstractFloat, # TODO add this to scikit's interface.
			##########################################################################
			# Modal parameters
			n_subrelations          :: Vector{<:Function},
			n_subfeatures           :: Vector{<:Integer},
			initConditions          :: Vector{<:DecisionTree._initCondition},
			allowRelationGlob       :: Vector{Bool},
			##########################################################################
			perform_consistency_check :: Bool,
			##########################################################################
			rng = Random.GLOBAL_RNG :: Random.AbstractRNG
		) where {S<:Float64, U}

		if isnothing(loss_function)
			loss_function = util.variance
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
			max_purity_at_leaf,
			##########################################################################
			n_subrelations,
			n_subfeatures,
			initConditions,
			allowRelationGlob,
		)

		Y_ = Y
		
		# Call core learning function
		root, indX = _fit(
				Xs,
				Y_,
				W,
				########################################################################
				loss_function,
				max_depth,
				min_samples_leaf,
				min_purity_increase,
				max_purity_at_leaf,
				########################################################################
				n_subrelations,
				n_subfeatures,
				initConditions,
				allowRelationGlob,
				########################################################################
				Val(perform_consistency_check),
				########################################################################
				rng,
		)
		
		# Create tree with labels and categorical leaves
		return Tree{S}(root, indX, initConditions)
	end
end
