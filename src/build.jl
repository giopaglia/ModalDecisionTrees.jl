export build_stump, build_tree, build_forest

include("fit_tree.jl")

################################################################################
###################### Single-frame active datasets ############################
################################################################################

# # Build models on (multi-dimensional) arrays
function build_stump(X :: ActiveFeaturedDataset, args...; kwargs...)
    build_stump(ActiveMultiFrameModalDataset(X), args...; kwargs...)
end

function build_tree(X :: ActiveFeaturedDataset, args...; kwargs...)
    build_tree(ActiveMultiFrameModalDataset(X), args...; kwargs...)
end

function build_forest(X :: ActiveFeaturedDataset, args...; kwargs...)
    build_forest(ActiveMultiFrameModalDataset(X), args...; kwargs...)
end

################################################################################
####################### Multi-frame active datasets ############################
################################################################################

# Build a stump (tree with depth 1)
function build_stump(
        X                 :: ActiveMultiFrameModalDataset,
        Y                 :: AbstractVector{L},
        W                 :: Union{Nothing,AbstractVector{U},Symbol} = nothing;
        kwargs...) where {L<:Label, U}
    params = NamedTuple(kwargs)
    @assert !haskey(params, :max_depth) || params.max_depth == 1 "build_stump doesn't allow max_depth != 1"
    build_tree(X, Y, W; max_depth = 1, kwargs...)
end

# TODO set default pruning arguments for tree, and make sure that forests override these
# Build a tree
function build_tree(
        X                   :: ActiveMultiFrameModalDataset,
        Y                   :: AbstractVector{L},
        W                   :: Union{Nothing,AbstractVector{U},Symbol}   = default_weights(Y);
        ##############################################################################
        loss_function       :: Union{Nothing,Function}            = nothing,
        max_depth           :: Int64                              = default_max_depth,
        min_samples_leaf    :: Int64                              = default_min_samples_leaf,
        min_purity_increase :: AbstractFloat                      = default_min_purity_increase,
        max_purity_at_leaf  :: AbstractFloat                      = default_max_purity_at_leaf,
        ##############################################################################
        n_subrelations      :: Union{Function,AbstractVector{<:Function}}             = identity,
        n_subfeatures       :: Union{Function,AbstractVector{<:Function}}             = identity,
        init_conditions     :: Union{InitCondition,AbstractVector{<:InitCondition}}   = start_without_world,
        allow_global_splits :: Union{Bool,AbstractVector{Bool}}                       = true,
        ##############################################################################
        perform_minification      :: Bool = false,
        perform_consistency_check :: Bool = true,
        ##############################################################################
        rng                 :: Random.AbstractRNG = Random.GLOBAL_RNG,
        print_progress      :: Bool = true,
    ) where {L<:Label, U}
    
    @assert W isa AbstractVector || W in [nothing, :rebalance, :default]

    W = if isnothing(W) || W == :default
        default_weights(Y)
    elseif W == :rebalance
        balanced_weights(Y)
    else
        W
    end

    @assert all(W .>= 0) "Sample weights must be non-negative."

    @assert nsamples(X) == length(Y) == length(W) "Mismatching number of samples in X, Y & W: $(nsamples(X)), $(length(Y)), $(length(W))"
    
    if isnothing(loss_function)
        loss_function = default_loss_function(L)
    end
    
    if allow_global_splits isa Bool
        allow_global_splits = fill(allow_global_splits, nframes(X))
    end
    if n_subrelations isa Function
        n_subrelations = fill(n_subrelations, nframes(X))
    end
    if n_subfeatures isa Function
        n_subfeatures  = fill(n_subfeatures, nframes(X))
    end
    if init_conditions isa InitCondition
        init_conditions = fill(init_conditions, nframes(X))
    end

    @assert max_depth > 0

    # if any(map(f->f isa AbstractDimensionalDataset, frames(X)))
    #     @error "Cannot learn from AbstractDimensionalDataset! Please use DimensionalFeaturedDataset, FeaturedDataset or SupportedFeaturedDataset."
    # end

    # TODO figure out what to do here. Maybe it can be helpful to make rng either an rng or a seed, and then mk_rng transforms it into an rng
    fit_tree(X, Y, init_conditions, W
        ;###########################################################################
        loss_function       = loss_function,
        max_depth           = max_depth,
        min_samples_leaf    = min_samples_leaf,
        min_purity_increase = min_purity_increase,
        max_purity_at_leaf  = max_purity_at_leaf,
        ############################################################################
        n_subrelations      = n_subrelations,
        n_subfeatures       = [ n_subfeatures[i](nfeatures(frame)) for (i,frame) in enumerate(frames(X)) ],
        allow_global_splits = allow_global_splits,
        ############################################################################
        perform_minification      = perform_minification,
        perform_consistency_check = perform_consistency_check,
        ############################################################################
        rng                 = rng,
        print_progress                 = print_progress,
    )
end


function build_forest(
        X                   :: ActiveMultiFrameModalDataset,
        Y                   :: AbstractVector{L},
        # Use unary weights if no weight is supplied
        W                   :: Union{Nothing,AbstractVector{U},Symbol} = default_weights(Y);
        ##############################################################################
        # Forest logic-agnostic parameters
        n_trees             = 100,
        partial_sampling    = 0.7,      # portion of sub-sampled samples (without replacement) by each tree
        ##############################################################################
        # Tree logic-agnostic parameters
        loss_function       :: Union{Nothing,Function}          = nothing,
        max_depth           :: Int64                            = default_max_depth,
        min_samples_leaf    :: Int64                            = default_min_samples_leaf,
        min_purity_increase :: AbstractFloat                    = default_min_purity_increase,
        max_purity_at_leaf  :: AbstractFloat                    = default_max_purity_at_leaf,
        ##############################################################################
        # Modal parameters
        n_subrelations      :: Union{Function,AbstractVector{<:Function}}             = identity,
        n_subfeatures       :: Union{Function,AbstractVector{<:Function}}             = x -> ceil(Int64, sqrt(x)),
        init_conditions     :: Union{InitCondition,AbstractVector{<:InitCondition}}   = start_without_world,
        allow_global_splits :: Union{Bool,AbstractVector{Bool}}                       = true,
        ##############################################################################
        perform_minification      :: Bool = false,
        perform_consistency_check :: Bool = true,
        ##############################################################################
        rng                 :: Random.AbstractRNG = Random.GLOBAL_RNG,
        print_progress :: Bool = true,
        suppress_parity_warning :: Bool = false,
    ) where {L<:Label, U}

    @assert W isa AbstractVector || W in [nothing, :rebalance, :default]

    W = if isnothing(W) || W == :default
        default_weights(Y)
    elseif W == :rebalance
        balanced_weights(Y)
    else
        W
    end

    @assert all(W .>= 0) "Sample weights must be non-negative."

    @assert nsamples(X) == length(Y) == length(W) "Mismatching number of samples in X, Y & W: $(nsamples(X)), $(length(Y)), $(length(W))"

    if n_subrelations isa Function
        n_subrelations = fill(n_subrelations, nframes(X))
    end
    if n_subfeatures isa Function
        n_subfeatures  = fill(n_subfeatures, nframes(X))
    end
    if init_conditions isa InitCondition
        init_conditions = fill(init_conditions, nframes(X))
    end
    if allow_global_splits isa Bool
        allow_global_splits = fill(allow_global_splits, nframes(X))
    end

    if n_trees < 1
        throw_n_log("the number of trees must be >= 1")
    end
    
    if !(0.0 < partial_sampling <= 1.0)
        throw_n_log("partial_sampling must be in the range (0,1]")
    end
    
    if any(map(f->f isa FeaturedDataset, frames(X)))
        @warn "Warning! Consider using the optimized structure SupportedFeaturedDataset, instead of FeaturedDataset."
    end

    tot_samples = nsamples(X)
    num_samples = floor(Int64, partial_sampling * tot_samples)

    trees = Vector{DTree{L}}(undef, n_trees)
    oob_samples = Vector{Vector{Integer}}(undef, n_trees)
    oob_metrics = Vector{NamedTuple}(undef, n_trees)

    rngs = [spawn_rng(rng) for i_tree in 1:n_trees]

    if print_progress
        p = Progress(n_trees, 1, "Computing DForest...")
    end
    Threads.@threads for i_tree in 1:n_trees
        train_idxs = rand(rngs[i_tree], 1:tot_samples, num_samples)

        X_slice = _slice_dataset(X, train_idxs, Val(true))
        Y_slice = @view Y[train_idxs]
        W_slice = SoleModels.slice_weights(W, train_idxs)

        trees[i_tree] = build_tree(
            X_slice
            , Y_slice
            , W_slice
            ;
            ################################################################################
            loss_function        = loss_function,
            max_depth            = max_depth,
            min_samples_leaf     = min_samples_leaf,
            min_purity_increase  = min_purity_increase,
            max_purity_at_leaf   = max_purity_at_leaf,
            ################################################################################
            n_subrelations       = n_subrelations,
            n_subfeatures        = n_subfeatures,
            init_conditions      = init_conditions,
            allow_global_splits  = allow_global_splits,
            ################################################################################
            perform_minification      = perform_minification,
            perform_consistency_check = perform_consistency_check,
            ################################################################################
            rng                  = rngs[i_tree],
            print_progress       = false,
        )

        # grab out-of-bag indices
        oob_samples[i_tree] = setdiff(1:tot_samples, train_idxs)

        tree_preds = apply_tree(trees[i_tree], _slice_dataset(X, oob_samples[i_tree], Val(true)))

        oob_metrics[i_tree] = (;
            actual = Y[oob_samples[i_tree]],
            predicted = tree_preds,
            weights = collect(SoleModels.slice_weights(W, oob_samples[i_tree]))
        )

        !print_progress || next!(p)
    end

    metrics = (;
        oob_metrics = oob_metrics,
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
            
            X_slice = _slice_dataset(X, [i], Val(true))
            Y_slice = [Y[i]]
            
            preds = apply_model(trees[index_of_trees_to_test_with], X_slice; suppress_parity_warning = suppress_parity_warning)
            
            push!(oob_classified, Y_slice[1] == preds[1])
        end
        oob_error = 1.0 - (sum(W[findall(oob_classified)]) / sum(W))
        metrics = merge(metrics, (
            oob_error = oob_error,
        ))
    end

    DForest{L}(trees, metrics)
end
