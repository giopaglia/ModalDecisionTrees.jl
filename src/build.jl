export build_stump, build_tree, build_forest

using StructuredArrays # , FillArrays # TODO choose one

include("tree.jl")

################################################################################
########################## Matricial Dataset ###################################
################################################################################

# # # Build models on (multi-dimensional) arrays
# function build_stump(
#   bare_dataset  :: MatricialDataset{T,D},
#   labels        :: AbstractVector{String},
#   weights       :: Union{Nothing,AbstractVector{U}} = nothing;
#   ontology      :: Ontology = ModalLogic.getIntervalOntologyOfDim(Val(D-2)),
#   kwargs...) where {T, D, U}
#   build_stump(OntologicalDataset{T,D-2}(bare_dataset, ontology, TODO...), labels, weights; kwargs...)
# end

# function build_tree(
#   bare_dataset  :: MatricialDataset{T,D},
#   labels        :: AbstractVector{String},
#   weights       :: Union{Nothing,AbstractVector{U}} = nothing;
#   ontology      :: Ontology = ModalLogic.getIntervalOntologyOfDim(Val(D-2)),
#   kwargs...) where {T, D, U}
#   build_tree(OntologicalDataset{T,D-2}(bare_dataset, ontology, TODO...), labels, weights; kwargs...)
# end

# function build_forest(
#   bare_dataset  :: MatricialDataset{T,D};
#   labels        :: AbstractVector{String},
#   # weights       :: Union{Nothing,AbstractVector{U}} = nothing TODO
#   ontology      :: Ontology = ModalLogic.getIntervalOntologyOfDim(Val(D-2)),
#   kwargs...) where {T, D, U}
#   # build_forest(OntologicalDataset{T,D-2}(ontology,bare_dataset), labels, weights; kwargs...)
#   build_forest(OntologicalDataset{T,D-2}(bare_dataset, ontology, TODO...), labels; kwargs...)
# end

################################################################################
########################## Modal Dataset #######################################
################################################################################

# # Build models on (multi-dimensional) arrays
function build_stump(X :: SingleFrameGenericDataset, args, kwargs...)
    build_stump(MultiFrameModalDataset(X), args...; kwargs...)
end

function build_tree(X :: SingleFrameGenericDataset, args, kwargs...)
    build_tree(MultiFrameModalDataset(X), args...; kwargs...)
end

function build_forest(X :: SingleFrameGenericDataset, args, kwargs...)
    build_forest(MultiFrameModalDataset(X), args...; kwargs...)
end

################################################################################
########################## Actual Build Funcs ##################################
################################################################################

# Build a stump (tree with depth 1)
function build_stump(
        X                 :: MultiFrameModalDataset,
        Y                 :: AbstractVector{L},
        W                 :: Union{Nothing,AbstractVector{U}} = nothing;
        kwargs...) where {L<:Label, U}
    params = NamedTuple(kwargs)
    @assert !haskey(params, :max_depth) || params.max_depth == 1 "build_stump doesn't allow max_depth != 1"
    build_tree(X, Y, W; max_depth = 1, kwargs...)
end

# TODO set default pruning arguments for tree, and make sure that forests override these
# Build a tree
function build_tree(
    X                   :: MultiFrameModalDataset,
    Y                   :: AbstractVector{L},
    W                   :: Union{Nothing,AbstractVector{U}}   = nothing;
    ##############################################################################
    loss_function       :: Union{Nothing,Function}            = nothing,
    max_depth           :: Int64                              = typemax(Int64),
    min_samples_leaf    :: Int64                              = 1,
    min_purity_increase :: AbstractFloat                      = -Inf,
    max_purity_at_leaf  :: AbstractFloat                      = Inf,
    ##############################################################################
    n_subrelations      :: Union{Function,AbstractVector{<:Function}}             = identity,
    n_subfeatures       :: Union{Function,AbstractVector{<:Function}}             = identity,
    initConditions      :: Union{_initCondition,AbstractVector{<:_initCondition}} = startWithRelationGlob,
    allowRelationGlob   :: Union{Bool,AbstractVector{Bool}}                       = true,
    ##############################################################################
    perform_consistency_check :: Bool = true,
    ##############################################################################
    rng                 :: Random.AbstractRNG = Random.GLOBAL_RNG) where {L<:Label, U}
    
    if isnothing(W)
        W = UniformVector{Int64}(1,n_samples(X))
    end
    
    if isnothing(loss_function)
        loss_function = default_loss_function(L)
    end
    
    if allowRelationGlob isa Bool
        allowRelationGlob = fill(allowRelationGlob, n_frames(X))
    end
    if n_subrelations isa Function
        n_subrelations = fill(n_subrelations, n_frames(X))
    end
    if n_subfeatures isa Function
        n_subfeatures  = fill(n_subfeatures, n_frames(X))
    end
    if initConditions isa _initCondition
        initConditions = fill(initConditions, n_frames(X))
    end

    if max_depth == -1
        max_depth = typemax(Int64)
    end
    @assert max_depth > 0

    if any(map(f->f isa MatricialDataset, frames(X)))
        @error "Cannot learn from MatricialDataset! Please use OntologicalDataset, FeatModalDataset or StumpFeatModalDataset."
    end

    # TODO figure out what to do here. Maybe it can be helpful to make rng either an rng or a seed, and then mk_rng transforms it into an rng
    fit(X, Y, initConditions, W
        ;###########################################################################
        loss_function       = loss_function,
        max_depth           = max_depth,
        min_samples_leaf    = min_samples_leaf,
        min_purity_increase = min_purity_increase,
        max_purity_at_leaf  = max_purity_at_leaf,
        ############################################################################
        n_subrelations      = n_subrelations,
        n_subfeatures       = [ n_subfeatures[i](n_features(frame)) for (i,frame) in enumerate(frames(X)) ],
        allowRelationGlob   = allowRelationGlob,
        ############################################################################
        perform_consistency_check = perform_consistency_check,
        ############################################################################
        rng                 = rng)
end


function build_forest(
    X                   :: MultiFrameModalDataset,
    Y                   :: AbstractVector{L},
    # Use unary weights if no weight is supplied
    W                   :: AbstractVector{U} = UniformVector{Int64}(1,n_samples(X)); # from StructuredArrays
    # W                   :: AbstractVector{U} = Ones{Int64}(n_samples(X));      # from FillArrays
    ##############################################################################
    # Forest logic-agnostic parameters
    n_trees             = 100,
    partial_sampling    = 0.7,      # portion of instances sampled (without replacement) by each tree
    ##############################################################################
    # Tree logic-agnostic parameters
    loss_function       :: Union{Nothing,Function}          = nothing,
    max_depth           :: Int64                            = typemax(Int64),
    min_samples_leaf    :: Int64                            = 1,
    min_purity_increase :: AbstractFloat                    = -Inf,
    max_purity_at_leaf  :: AbstractFloat                    = Inf,
    ##############################################################################
    # Modal parameters
    n_subrelations      :: Union{Function,AbstractVector{<:Function}}             = identity,
    n_subfeatures       :: Union{Function,AbstractVector{<:Function}}             = x -> ceil(Int64, sqrt(x)),
    initConditions      :: Union{_initCondition,AbstractVector{<:_initCondition}} = startWithRelationGlob,
    allowRelationGlob   :: Union{Bool,AbstractVector{Bool}}                       = true,
    ##############################################################################
    perform_consistency_check :: Bool = true,
    ##############################################################################
    rng                 :: Random.AbstractRNG = Random.GLOBAL_RNG) where {L<:Label, U}

    if n_subrelations isa Function
        n_subrelations = fill(n_subrelations, n_frames(X))
    end
    if n_subfeatures isa Function
        n_subfeatures  = fill(n_subfeatures, n_frames(X))
    end
    if initConditions isa _initCondition
        initConditions = fill(initConditions, n_frames(X))
    end
    if allowRelationGlob isa Bool
        allowRelationGlob = fill(allowRelationGlob, n_frames(X))
    end

    if n_trees < 1
        throw_n_log("the number of trees must be >= 1")
    end
    
    if !(0.0 < partial_sampling <= 1.0)
        throw_n_log("partial_sampling must be in the range (0,1]")
    end
    
    if any(map(f->f isa FeatModalDataset, frames(X)))
        @warn "Warning! StumpFeatModalDataset is recommended for performance, instead of FeatModalDataset."
    end

    tot_samples = n_samples(X)
    num_samples = floor(Int64, partial_sampling * tot_samples)

    trees = Vector{DTree{L}}(undef, n_trees)
    oob_metrics = Vector{PerformanceStruct}(undef, n_trees)
    oob_samples = Vector{Vector{Integer}}(undef, n_trees)

    rngs = [util.spawn_rng(rng) for i_tree in 1:n_trees]

    if W isa UniformVector
        W_one_slice = UniformVector{Int64}(1,num_samples)
    end

    get_W_slice(W::UniformVector, inds) = W_one_slice
    get_W_slice(W::Any, inds) = @view W[inds]

    # TODO improve naming (at least)
    _get_weights(W::UniformVector, inds) = nothing
    _get_weights(W::Any, inds) = @view W[inds]

    Threads.@threads for i_tree in 1:n_trees
        inds = rand(rngs[i_tree], 1:tot_samples, num_samples)

        X_slice = ModalLogic.slice_dataset(X, inds; return_view = true)
        Y_slice = @view Y[inds]

        trees[i_tree] = build_tree(
            X_slice
            , Y_slice
            , get_W_slice(W, inds)
            ;
            ####
            loss_function        = loss_function,
            max_depth            = max_depth,
            min_samples_leaf     = min_samples_leaf,
            min_purity_increase  = min_purity_increase,
            max_purity_at_leaf   = max_purity_at_leaf,
            ####
            n_subrelations       = n_subrelations,
            n_subfeatures        = n_subfeatures,
            initConditions       = initConditions,
            allowRelationGlob    = allowRelationGlob,
            ####
            perform_consistency_check = perform_consistency_check,
            ####
            rng                  = rngs[i_tree])

        # grab out-of-bag indices
        oob_samples[i_tree] = setdiff(1:tot_samples, inds)

        tree_preds = apply_tree(trees[i_tree], ModalLogic.slice_dataset(X, oob_samples[i_tree]; return_view = true))
        oob_metrics[i_tree] = confusion_matrix(Y[oob_samples[i_tree]], tree_preds, _get_weights(W, inds))
    end

    metrics = (;
        # oob_metrics = oob_metrics,
    )

    if L<:CLabel
        # For each sample, construct its random forest predictor
        #  by averaging (or majority voting) only those
        #  trees corresponding to boot-strap samples in which the sample did not appear
        oob_classified = Vector{Bool}()
        Threads.@threads for i in 1:tot_samples
          selected_trees = fill(false, n_trees)

          # pick every tree trained without i-th sample
          for i_tree in 1:n_trees
              if i in oob_samples[i_tree] # if i is present in the i_tree-th tree, selecte thi tree
                  selected_trees[i_tree] = true
              end
          end
            
          index_of_trees_to_test_with = findall(selected_trees)

          if length(index_of_trees_to_test_with) == 0
              continue
          end

          X_slice = ModalLogic.slice_dataset(X, [i]; return_view = true)
          Y_slice = [Y[i]]

          pred = apply_trees(trees[index_of_trees_to_test_with], X_slice)

          push!(oob_classified, Y_slice[1] == pred[1])
        end
        oob_error = 1.0 - (sum(W[findall(oob_classified)]) / sum(W))
        metrics = merge(metrics, (
            oob_error = oob_error,
        ))
    end

    DForest{L}(trees, metrics)
end
