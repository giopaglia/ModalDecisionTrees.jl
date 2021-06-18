# Utilities

# include("../util.jl")
using .util: Label
using .ModalLogic

include("tree.jl")

# Conversion: NodeMeta (node + training info) -> DTNode (bare decision tree model)
function _convert(
		node   :: treeclassifier.NodeMeta,
		list   :: AbstractVector{S},
		labels :: AbstractVector{S}) where {U<:Real, S<:String}
	if node.is_leaf
		return DTLeaf(list[node.label], labels[node.region])
	else
		left  = _convert(node.l, list, labels)
		right = _convert(node.r, list, labels)
		return DTInternal(node.i_frame, node.relation, node.feature, node.test_operator, node.threshold, left, right)
	end
end

################################################################################
########################## Matricial Dataset ###################################
################################################################################

# # Build models on (multi-dimensional) arrays
function build_stump(
	bare_dataset  :: MatricialDataset{T,D},
	labels        :: AbstractVector{String},
	weights       :: Union{Nothing,AbstractVector{U}} = nothing;
	ontology      :: Ontology = ModalLogic.getIntervalOntologyOfDim(Val(D-2)),
	kwargs...) where {T, D, U}
	build_stump(OntologicalDataset{T,D-2}(ontology, bare_dataset), labels, weights; kwargs...)
end

function build_tree(
	bare_dataset  :: MatricialDataset{T,D},
	labels        :: AbstractVector{String},
	weights       :: Union{Nothing,AbstractVector{U}} = nothing;
	ontology      :: Ontology = ModalLogic.getIntervalOntologyOfDim(Val(D-2)),
	kwargs...) where {T, D, U}
	build_tree(OntologicalDataset{T,D-2}(ontology, bare_dataset), labels, weights; kwargs...)
end

function build_forest(
	bare_dataset  :: MatricialDataset{T,D};
	labels        :: AbstractVector{String},
	# weights       :: Union{Nothing,AbstractVector{U}} = nothing TODO
	ontology      :: Ontology = ModalLogic.getIntervalOntologyOfDim(Val(D-2)),
	kwargs...) where {T, D, U}
	# build_forest(OntologicalDataset{T,D-2}(ontology,bare_dataset), labels, weights; kwargs...)
	build_forest(OntologicalDataset{T,D-2}(ontology, bare_dataset), labels; kwargs...)
end

################################################################################
########################## Modal Dataset #######################################
################################################################################

# # Build models on (multi-dimensional) arrays
function build_stump(
	ontol_dataset :: OntologicalDataset{T, N, WorldType},
	labels        :: AbstractVector{String},
	weights       :: Union{Nothing,AbstractVector{U}} = nothing;
	kwargs...) where {T, N, U, WorldType}
	build_stump(MultiFrameFeatModalDataset(ontol_dataset), labels, weights; kwargs...)
end

function build_tree(
	ontol_dataset :: OntologicalDataset{T, N, WorldType},
	labels        :: AbstractVector{String},
	weights       :: Union{Nothing,AbstractVector{U}} = nothing;
	kwargs...) where {T, N, U, WorldType}
	build_tree(MultiFrameFeatModalDataset(ontol_dataset), labels, weights; kwargs...)
end

function build_forest(
	ontol_dataset :: OntologicalDataset{T, N, WorldType};
	labels        :: AbstractVector{String},
	# weights       :: Union{Nothing,AbstractVector{U}} = nothing TODO
	kwargs...) where {T, N, U, WorldType}
	# build_forest(MultiFrameFeatModalDataset(ontol_dataset), labels, weights; kwargs...)
	build_forest(MultiFrameFeatModalDataset(ontol_dataset), labels; kwargs...)
end

################################################################################
########################## Actual Build Funcs ##################################
################################################################################

# Build a stump (tree with depth 1)
function build_stump(
		X	                :: MultiFrameFeatModalDataset,
		Y                 :: AbstractVector{String},
		W                 :: Union{Nothing,AbstractVector{U}} = nothing;
		kwargs...) where {N, U}
	@assert !haskey(kwargs, :max_depth) || kwargs.max_depth == 1 "build_stump doesn't allow max_depth != 1"
	build_tree(X, Y, W; max_depth = 1, kwargs...)
end

# Build a tree
function build_tree(
	Xs                  :: MultiFrameFeatModalDataset,
	Y                   :: AbstractVector{S},
	W                   :: Union{Nothing,AbstractVector{U}}   = nothing;
	##############################################################################
	loss_function       :: Function                           = util.entropy,
	max_depth           :: Int                                = -1,
	min_samples_leaf    :: Int                                = 1,
	min_purity_increase :: AbstractFloat                      = 0.0,
	min_loss_at_leaf    :: AbstractFloat                      = -Inf,
	##############################################################################
	n_subrelations      :: Union{Function,Vector{Function}}               = identity,
	n_subfeatures       :: Union{Function,Vector{Function}}               = identity,
	initConditions      :: Union{_initCondition,Vector{_initCondition}}   = startWithRelationAll,
	useRelationAll      :: Union{Bool,Vector{Bool}}                       = true,
	##############################################################################
	rng                 :: Random.AbstractRNG = Random.GLOBAL_RNG) where {S, U}

	if useRelationAll isa Bool
		useRelationAll = fill(useRelationAll, n_frames(Xs))
	end
	if n_subrelations isa Function
		n_subrelations = fill(n_subrelations, n_frames(Xs))
	end
	if n_subfeatures isa Function
		n_subfeatures  = fill(n_subfeatures, n_frames(Xs))
	end
	if initConditions isa _initCondition
		initConditions = fill(initConditions, n_frames(Xs))
	end

	if max_depth == -1
		max_depth = typemax(Int)
	end

	rng = mk_rng(rng)
	t = treeclassifier.fit(
		Xs                  = Xs,
		Y                   = Y,
		W                   = W,
		############################################################################
		loss_function       = loss_function,
		max_depth           = max_depth,
		min_samples_leaf    = min_samples_leaf,
		min_purity_increase = min_purity_increase,
		min_loss_at_leaf    = min_loss_at_leaf,
		############################################################################
		n_subrelations      = n_subrelations,
		n_subfeatures       = [ n_subfeatures[i](n_features(get_frame(Xs, i))) for i in 1:n_frames(Xs) ],
		initConditions      = initConditions,
		useRelationAll      = useRelationAll,
		############################################################################
		rng                 = rng)

	root = _convert(t.root, t.list, Y[t.labels])
	DTree(root, world_types(Xs), initConditions)
end

function build_forest(
	Xs                  :: MultiFrameFeatModalDataset,
	Y                   :: AbstractVector{S}
	;
	# , W                   :: Union{Nothing,AbstractVector{U}} = nothing; TODO these must also be used for the calculation of the oob_error
	##############################################################################
	# Forest logic-agnostic parameters
	n_trees             = 100,
	partial_sampling    = 0.7,      # portion of instances sampled (without replacement) by each tree
	##############################################################################
	# Tree logic-agnostic parameters
	loss_function       :: Function           = util.entropy,
	max_depth           :: Int                = -1,
	min_samples_leaf    :: Int                = 1,
	min_purity_increase :: AbstractFloat      = 0.0,
	min_loss_at_leaf    :: AbstractFloat      = -Inf,
	##############################################################################
	# Modal parameters
	n_subrelations      :: Union{Function,Vector{Function}}               = identity,
	n_subfeatures       :: Union{Function,Vector{Function}}               = x -> ceil(Int, sqrt(x)),
	initConditions      :: Union{_initCondition,Vector{_initCondition}}   = startWithRelationAll,
	useRelationAll      :: Union{Bool,Vector{Bool}}                       = true,
	##############################################################################
	rng                 :: Random.AbstractRNG = Random.GLOBAL_RNG) where {S, U}

	rng = mk_rng(rng)

	if useRelationAll isa Bool
		useRelationAll = fill(useRelationAll, n_frames(Xs))
	end
	if n_subrelations isa Function
		n_subrelations = fill(n_subrelations, n_frames(Xs))
	end
	if n_subfeatures isa Function
		n_subfeatures  = fill(n_subfeatures, n_frames(Xs))
	end
	if initConditions isa _initCondition
		initConditions = fill(initConditions, n_frames(Xs))
	end

	if n_trees < 1
		error("the number of trees must be >= 1")
	end
	
	if !(0.0 < partial_sampling <= 1.0)
		error("partial_sampling must be in the range (0,1]")
	end
	
	# precompute-gammas, since they are shared by all trees
	# TODO remove
	# if isnothing(gammas)
	# 	gammas = fill(nothing, n_frames(Xs))
	# end
	# for i in 1:length(gammas)
	# 	if isnothing(gammas)
	# 		(
	# 			test_operators, relationSet,
	# 			relationId_id, relationAll_id,
	# 			availableModalRelation_ids, allAvailableRelation_ids
	# 		) = treeclassifier.optimize_tree_parameters!(get_frame(Xs, i), initConditions[i], useRelationAll, test_operators)
	# 		gammas[i] = computeGammas(get_frame(Xs, i),WorldType,test_operators,relationSet,relationId_id,availableModalRelation_ids)
	# 	end
	# end

	t_samples = n_samples(X)
	num_samples = floor(Int, partial_sampling * t_samples)

	trees = Vector{DTree{S}}(undef, n_trees)
	cms = Vector{ConfusionMatrix}(undef, n_trees)
	oob_samples = Vector{Vector{Integer}}(undef, n_trees)

	rngs = [spawn_rng(rng) for i in 1:n_trees]
	Threads.@threads for i in 1:n_trees
		inds = rand(rngs[i], 1:t_samples, num_samples)

		X_slice = ModalLogic.slice_dataset(X, inds; return_view = true)
		Y_slice = @view Y[inds]
		# v_weights = @views W[inds]

		trees[i] = build_tree(
			X_slice
			, Y_slice
			# , v_weights
			;
			####
			loss_function        = loss_function,
			max_depth            = max_depth,
			min_samples_leaf     = min_samples_leaf,
			min_purity_increase  = min_purity_increase,
			min_loss_at_leaf     = min_loss_at_leaf,
			####
			n_subrelations       = n_subrelations,
			n_subfeatures        = n_subfeatures,
			initConditions       = initConditions,
			useRelationAll       = useRelationAll,
			####
			rng                  = rngs[i])

		# grab out-of-bag indices
		oob_samples[i] = setdiff(1:t_samples, inds)

		tree_preds = apply_tree(trees[i], ModalLogic.slice_dataset(X, oob_samples[i]; return_view = true))
		cms[i] = confusion_matrix(Y[oob_samples[i]], tree_preds)
	end

	oob_classified = Vector{Bool}()
	# For each observation z_i, construct its random forest
	# predictor by averaging (or majority voting) only those 
	# trees corresponding to boot-strap samples in which z_i did not appear.
	Threads.@threads for i in 1:t_samples
		selected_trees = fill(false, n_trees)

		# pick every tree trained without i-th sample
		for j in 1:n_trees
			if i in oob_samples[j] # if i is present in the j-th tree, selecte thi tree
				selected_trees[j] = true
			end
		end
		
		index_of_trees_to_test_with = findall(selected_trees)

		if length(index_of_trees_to_test_with) == 0
			continue
		end

		v_features = ModalLogic.slice_dataset(X, [i]; return_view = true)
		v_labels = @views Y[[i]]

		# TODO: optimization - no need to pass through ConfusionMatrix
		pred = apply_forest(trees[index_of_trees_to_test_with], v_features)
		cm = confusion_matrix(v_labels, pred)

		push!(oob_classified, cm.overall_accuracy > 0.5)
	end

	oob_error = 1.0 - (length(findall(oob_classified)) / length(oob_classified))

	return Forest{S}(trees, cms, oob_error)
end
