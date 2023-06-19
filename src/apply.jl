using StatsBase
export apply_tree, apply_forest, apply_model, print_apply, tree_walk_metrics

import MLJ: predict

############################################################################################
############################################################################################
############################################################################################

mm_instance_initialworldset(Xs::MultiLogiset, tree::DTree, i_instance::Integer) = begin
    Ss = Vector{WorldSet}(undef, nmodalities(Xs))
    for (i_modality,X) in enumerate(modalities(Xs))
        Ss[i_modality] = initialworldset(X, i_instance, init_conditions(tree)[i_modality])
    end
    Ss
end

softmax(v::AbstractVector) = exp.(v) ./ sum(exp.(v))
softmax(m::AbstractMatrix) = mapslices(softmax, m; dims=1)

############################################################################################
############################################################################################
############################################################################################

# Patch single-frame _-> multiframe
apply(model::Union{DTree,DForest}, X::AbstractLogiset, args...; kwargs...) =
    apply(model, MultiLogiset(X), args...; kwargs...)

apply_model = apply

# TODO remove these patches
apply_tree   = apply_model
apply_forest = apply_model

############################################################################################

# TODO discriminate between kwargs for apply_tree & print_tree
function print_apply(tree::DTree, X::GenericModalDataset, Y::AbstractVector; kwargs...)
    predictions, new_tree = apply_model(tree, X, Y)
    print_tree(new_tree; kwargs...)
    predictions, new_tree
end


apply_proba(model::Union{DTree,DForest}, X::AbstractLogiset, args...; kwargs...) =
    apply_proba(model, MultiLogiset(X), args...; kwargs...)

# apply_tree_proba   = apply_model_proba
# apply_trees_proba  = apply_model_proba
# apply_forest_proba = apply_model_proba

apply_model_proba = apply_proba

predict(model::Union{DTree,DForest}, X::AbstractLogiset, args...; kwargs...) =
    predict(model, MultiLogiset(X), args...; kwargs...)

################################################################################
# Apply models: predict labels for a new dataset of instances
################################################################################

function apply(leaf::DTLeaf, X::Any, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}; suppress_parity_warning = false)
    prediction(leaf)
end

function apply(leaf::NSDTLeaf, X::Any, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}; suppress_parity_warning = false)
    d = slicedataset(X, [i_instance])
    # println(typeof(d))
    # println(hasmethod(size,   (typeof(d),)) ? size(d)   : nothing)
    # println(hasmethod(length, (typeof(d),)) ? length(d) : nothing)
    preds = leaf.predicting_function(d)
    @assert length(preds) == 1 "Error in apply(::NSDTLeaf, ...) The predicting function returned some malformed output. Expected is a Vector of a single prediction, while the returned value is:\n$(preds)\n$(hasmethod(length, (typeof(preds),)) ? length(preds) : "(length = $(length(preds)))")\n$(hasmethod(size, (typeof(preds),)) ? size(preds) : "(size = $(size(preds)))")"
    # println(preds)
    # println(typeof(preds))
    preds[1]
end

function apply(tree::DTInternal, X::MultiLogiset, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}; kwargs...)
    @logmsg LogDetail "applying branch..."
    @logmsg LogDetail " worlds" worlds
    (satisfied,new_worlds) =
        modalstep(
            frame(X, i_modality(tree)),
            i_instance,
            worlds[i_modality(tree)],
            decision(tree),
    )

    worlds[i_modality(tree)] = new_worlds
    @logmsg LogDetail " ->(satisfied,worlds')" satisfied worlds
    apply((satisfied ? left(tree) : right(tree)), X, i_instance, worlds; kwargs...)
end

# Obtain predictions of a tree on a dataset
function apply(tree::DTree{L}, X::MultiLogiset; kwargs...) where {L}
    @logmsg LogDetail "apply..."
    _n_instances = ninstances(X)
    predictions = Vector{L}(undef, _n_instances)

    for i_instance in 1:_n_instances
        @logmsg LogDetail " instance $i_instance/$_n_instances"
        # TODO figure out: is it better to interpret the whole dataset at once, or instance-by-instance? The first one enables reusing training code

        worlds = mm_instance_initialworldset(X, tree, i_instance)

        predictions[i_instance] = apply(root(tree), X, i_instance, worlds; kwargs...)
    end
    predictions
end

# use an array of trees to test features
function apply(
    trees::AbstractVector{<:DTree{<:L}},
    X::MultiLogiset;
    suppress_parity_warning = false,
    tree_weights::Union{AbstractMatrix{Z},AbstractVector{Z},Nothing} = nothing,
) where {L<:Label,Z<:Real}
    @logmsg LogDetail "apply..."
    ntrees = length(trees)
    _n_instances = ninstances(X)

    if !(tree_weights isa AbstractMatrix)
        if isnothing(tree_weights)
            tree_weights = Ones{Int}(length(trees), ninstances(X)) # TODO optimize?
        elseif tree_weights isa AbstractVector
            tree_weights = hcat([tree_weights for i in 1:ninstances(X)]...)
        else
            @show typef(tree_weights)
            error("Unexpected tree_weights encountered $(tree_weights).")
        end
    end

    @assert length(trees) == size(tree_weights, 1) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."
    @assert ninstances(X) == size(tree_weights, 2) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."

    # apply each tree to the whole dataset
    _predictions = Matrix{L}(undef, ntrees, _n_instances)
    Threads.@threads for i_tree in 1:ntrees
        _predictions[i_tree,:] = apply(trees[i_tree], X; suppress_parity_warning = suppress_parity_warning)
    end

    # for each instance, aggregate the predictions
    predictions = Vector{L}(undef, _n_instances)
    Threads.@threads for i_instance in 1:_n_instances
        predictions[i_instance] = bestguess(
            _predictions[:,i_instance],
            tree_weights[:,i_instance];
            suppress_parity_warning = suppress_parity_warning
        )
    end

    predictions
end

# use a proper forest to test features
function apply(
    forest::DForest,
    X::MultiLogiset;
    suppress_parity_warning = false,
    weight_trees_by::Union{Bool,Symbol,AbstractVector} = false,
)
    if weight_trees_by == false
        apply(trees(forest), X; suppress_parity_warning = suppress_parity_warning)
    elseif isa(weight_trees_by, AbstractVector)
        apply(trees(forest), X; suppress_parity_warning = suppress_parity_warning, tree_weights = weight_trees_by)
    # elseif weight_trees_by == :accuracy
    #   # TODO: choose HOW to weight a tree... overall_accuracy is just an example (maybe can be parameterized)
    #   apply(forest.trees, X; tree_weights = map(cm -> overall_accuracy(cm), get(forest.metrics, :oob_metrics...)))
    else
        @error "Unexpected value for weight_trees_by: $(weight_trees_by)"
    end
end

function apply(
    nsdt::RootLevelNeuroSymbolicHybrid,
    X::MultiLogiset;
    suppress_parity_warning = false,
)
    W = softmax(nsdt.feature_function(X))
    apply(nsdt.trees, X; suppress_parity_warning = suppress_parity_warning, tree_weights = W)
end

################################################################################
# Print+Apply models: apply labels for a new dataset of instances
################################################################################

function _empty_tree_leaves(leaf::DTLeaf{L}) where {L}
    DTLeaf{L}(prediction(leaf), L[])
end

function _empty_tree_leaves(leaf::NSDTLeaf{L}) where {L}
    NSDTLeaf{L}(leaf.predicting_function, L[], leaf.supp_valid_labels, L[], leaf.supp_valid_predictions)
end

function _empty_tree_leaves(node::DTInternal)
    return DTInternal(
        i_modality(node),
        decision(node),
        _empty_tree_leaves(this(node)),
        _empty_tree_leaves(left(node)),
        _empty_tree_leaves(right(node)),
    )
end

function _empty_tree_leaves(tree::DTree)
    return DTree(
        _empty_tree_leaves(root(tree)),
        worldtypes(tree),
        init_conditions(tree),
    )
end


function apply(
    leaf::DTLeaf{L},
    X::MultiLogiset,
    i_instance::Integer,
    worlds::AbstractVector{<:AbstractWorldSet},
    class::L;
    update_labels = false,
    suppress_parity_warning = false,
) where {L<:Label}
    _supp_labels = L[supp_labels(leaf)..., class]

    _prediction =
        if update_labels
            bestguess(supp_labels(leaf))
        else
            prediction(leaf)
        end

    _prediction, DTLeaf(_prediction, _supp_labels)
end

function apply(
    leaf::NSDTLeaf{L},
    X::MultiLogiset,
    i_instance::Integer,
    worlds::AbstractVector{<:AbstractWorldSet},
    class::L;
    update_labels = false,
    suppress_parity_warning = false,
) where {L<:Label}
    _supp_train_labels      = L[leaf.supp_train_labels...,      class]
    _supp_train_predictions = L[leaf.supp_train_predictions..., apply(leaf, X, i_instance, worlds; kwargs...)]

    _predicting_function =
        if update_labels
            error("TODO expand code retrain")
        else
            leaf.predicting_function
        end
    d = slicedataset(X, [i_instance])
    _predicting_function(d)[1], NSDTLeaf{L}(_predicting_function, _supp_train_labels, leaf.supp_valid_labels, _supp_train_predictions, leaf.supp_valid_predictions)
end

function apply(
    tree::DTInternal{L},
    X::MultiLogiset,
    i_instance::Integer,
    worlds::AbstractVector{<:AbstractWorldSet},
    class::L;
    kwargs...,
) where {L}

    (satisfied,new_worlds) = modalstep(frame(X, i_modality(tree)), i_instance, worlds[i_modality(tree)], decision(tree))

    # if satisfied
    #   println("new_worlds: $(new_worlds)")
    # end

    worlds[i_modality(tree)] = new_worlds

    this_prediction, this_leaf = apply(this(tree),  X, i_instance, worlds, class; kwargs...) # TODO test whether this works correctly

    pred, left_leaf, right_leaf =
        if satisfied
            pred, left_leaf = apply(left(tree),  X, i_instance, worlds, class; kwargs...)
            pred, left_leaf, right(tree)
        else
            pred, right_leaf = apply(right(tree), X, i_instance, worlds, class; kwargs...)
            pred, left(tree), right_leaf
        end

    pred, DTInternal(i_modality(tree), decision(tree), this_leaf, left_leaf, right_leaf)
end

function apply(
    tree::DTree{L},
    X::MultiLogiset,
    Y::AbstractVector{<:L};
    reset_leaves = true,
    kwargs...,
) where {L}

    # Reset
    tree = (reset_leaves ? _empty_tree_leaves(tree) : tree)

    predictions = L[]
    _root = root(tree)

    # Propagate instances down the tree
    for i_instance in 1:ninstances(X)
        worlds = mm_instance_initialworldset(X, tree, i_instance)
        pred, _root = apply(_root, X, i_instance, worlds, Y[i_instance]; kwargs...)
        push!(predictions, pred)
    end
    predictions, DTree(_root, worldtypes(tree), init_conditions(tree))
end

# use an array of trees to test features
function apply(
    trees::AbstractVector{<:DTree{<:L}},
    X::MultiLogiset,
    Y::AbstractVector{<:L};
    tree_weights::Union{AbstractMatrix{Z},AbstractVector{Z},Nothing} = nothing,
    suppress_parity_warning = false,
) where {L<:Label,Z<:Real}
    @logmsg LogDetail "apply..."
    trees = deepcopy(trees)
    ntrees = length(trees)
    _n_instances = ninstances(X)

    if !(tree_weights isa AbstractMatrix)
        if isnothing(tree_weights)
            tree_weights = Ones{Int}(length(trees), ninstances(X)) # TODO optimize?
        elseif tree_weights isa AbstractVector
            tree_weights = hcat([tree_weights for i in 1:ninstances(X)]...)
        else
            @show typef(tree_weights)
            error("Unexpected tree_weights encountered $(tree_weights).")
        end
    end

    @assert length(trees) == size(tree_weights, 1) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."
    @assert ninstances(X) == size(tree_weights, 2) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."

    # apply each tree to the whole dataset
    _predictions = Matrix{L}(undef, ntrees, _n_instances)
    Threads.@threads for i_tree in 1:ntrees
        _predictions[i_tree,:], trees[i_tree] = apply(trees[i_tree], X, Y)
    end

    # for each instance, aggregate the predictions
    predictions = Vector{L}(undef, _n_instances)
    Threads.@threads for i_instance in 1:_n_instances
        predictions[i_instance] = bestguess(
            _predictions[:,i_instance],
            tree_weights[:,i_instance];
            suppress_parity_warning = suppress_parity_warning
        )
    end

    predictions, trees
end

# use a proper forest to test features
function apply(
    forest::DForest,
    X::MultiLogiset,
    Y::AbstractVector{<:L};
    weight_trees_by::Union{Bool,Symbol,AbstractVector} = false,
    kwargs...
) where {L<:Label}
    predictions, trees = begin
        if weight_trees_by == false
            apply(trees(forest), X, Y; kwargs...)
        elseif isa(weight_trees_by, AbstractVector)
            apply(trees(forest), X, Y; tree_weights = weight_trees_by, kwargs...)
        # elseif weight_trees_by == :accuracy
        #   # TODO: choose HOW to weight a tree... overall_accuracy is just an example (maybe can be parameterized)
        #   apply(forest.trees, X; tree_weights = map(cm -> overall_accuracy(cm), get(forest.metrics, :oob_metrics...)))
        else
            @error "Unexpected value for weight_trees_by: $(weight_trees_by)"
        end
    end
    predictions, DForest{L}(trees, (;)) # TODO note that the original metrics are lost here
end

function apply(
    nsdt::RootLevelNeuroSymbolicHybrid,
    X::MultiLogiset,
    Y::AbstractVector{<:L};
    suppress_parity_warning = false,
    kwargs...
) where {L<:Label}
    W = softmax(nsdt.feature_function(X))
    predictions, trees = apply(
        nsdt.trees,
        X,
        Y;
        suppress_parity_warning = suppress_parity_warning,
        tree_weights = W,
        kwargs...,
    )
    predictions, RootLevelNeuroSymbolicHybrid(nsdt.feature_function, trees, (;)) # TODO note that the original metrics are lost here
end

# function apply(tree::DTNode{T, L}, X::AbstractDimensionalDataset{T,D}, Y::AbstractVector{<:L}; reset_leaves = true, update_labels = false) where {L,T,D}
#   return apply(DTree(tree, [worldtype(get_interval_ontology(Val(D-2)))], [start_without_world]), X, Y, reset_leaves = reset_leaves, update_labels = update_labels)
# end

############################################################################################

# using Distributions
# using CategoricalDistributions
using CategoricalArrays

function apply_proba(leaf::DTLeaf, X::Any, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet})
    supp_labels(leaf)
end

function apply_proba(tree::DTInternal, X::MultiLogiset, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet})
    @logmsg LogDetail "applying branch..."
    @logmsg LogDetail " worlds" worlds
    (satisfied,new_worlds) =
        modalstep(
            frame(X, i_modality(tree)),
            i_instance,
            worlds[i_modality(tree)],
            decision(tree),
    )

    worlds[i_modality(tree)] = new_worlds
    @logmsg LogDetail " ->(satisfied,worlds')" satisfied worlds
    apply_proba((satisfied ? left(tree) : right(tree)), X, i_instance, worlds)
end

# Obtain predictions of a tree on a dataset
function apply_proba(tree::DTree{L}, X::MultiLogiset, classes) where {L<:CLabel}
    @logmsg LogDetail "apply_proba..."
    _n_instances = ninstances(X)
    prediction_scores = Matrix{Float64}(undef, _n_instances, length(classes))

    for i_instance in 1:_n_instances
        @logmsg LogDetail " instance $i_instance/$_n_instances"
        # TODO figure out: is it better to interpret the whole dataset at once, or instance-by-instance? The first one enables reusing training code

        worlds = mm_instance_initialworldset(X, tree, i_instance)

        this_prediction_scores = apply_proba(root(tree), X, i_instance, worlds)
        # d = Distributions.fit(UnivariateFinite, categorical(this_prediction_scores; levels = classes))
        d = begin
            c = categorical(this_prediction_scores; levels = classes)
            cc = countmap(c)
            s = [cc[cl] for cl in classes(c)]
            UnivariateFinite(classes(c), s ./ sum(s))
        end
        prediction_scores[i_instance, :] .= [pdf(d, c) for c in classes]
    end
    prediction_scores
end

# Obtain predictions of a tree on a dataset
function apply_proba(tree::DTree{L}, X::MultiLogiset) where {L<:RLabel}
    @logmsg LogDetail "apply_proba..."
    _n_instances = ninstances(X)
    prediction_scores = Vector{Vector{Float64}}(undef, _n_instances)

    for i_instance in 1:_n_instances
        @logmsg LogDetail " instance $i_instance/$_n_instances"
        # TODO figure out: is it better to interpret the whole dataset at once, or instance-by-instance? The first one enables reusing training code

        worlds = mm_instance_initialworldset(X, tree, i_instance)

        prediction_scores[i_instance] = apply_proba(tree.root, X, i_instance, worlds)
    end
    prediction_scores
end

# use an array of trees to test features
function apply_proba(
    trees::AbstractVector{<:DTree{<:L}},
    X::MultiLogiset,
    classes;
    tree_weights::Union{AbstractMatrix{Z},AbstractVector{Z},Nothing} = nothing,
) where {L<:CLabel,Z<:Real}
    @logmsg LogDetail "apply_proba..."
    ntrees = length(trees)
    _n_instances = ninstances(X)

    if !(tree_weights isa AbstractMatrix)
        if isnothing(tree_weights)
            tree_weights = Ones{Int}(length(trees), ninstances(X)) # TODO optimize?
        elseif tree_weights isa AbstractVector
            tree_weights = hcat([tree_weights for i in 1:ninstances(X)]...)
        else
            @show typef(tree_weights)
            error("Unexpected tree_weights encountered $(tree_weights).")
        end
    end

    @assert length(trees) == size(tree_weights, 1) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."
    @assert ninstances(X) == size(tree_weights, 2) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."

    # apply each tree to the whole dataset
    _predictions = Array{Float64,3}(undef, _n_instances, ntrees, length(classes))
    Threads.@threads for i_tree in 1:ntrees
        _predictions[:,i_tree,:] = apply_proba(trees[i_tree], X, classes)
    end

    # Average the prediction scores
    if isnothing(tree_weights)
        dropdims(mean(_predictions; dims=2), dims=2)
    else
        tree_weights = tree_weights./sum(tree_weights)
        prediction_scores = Matrix{Float64}(undef, _n_instances, length(classes))
        Threads.@threads for i in 1:_n_instances
            prediction_scores[i,:] .= mean(_predictions[i,:,:] * tree_weights; dims=1)
        end
        prediction_scores
    end
end

# use an array of trees to test features
function apply_proba(
    trees::AbstractVector{<:DTree{<:L}},
    X::MultiLogiset;
    tree_weights::Union{Nothing,AbstractVector{Z}} = nothing,
) where {L<:RLabel,Z<:Real}
    @logmsg LogDetail "apply_proba..."
    ntrees = length(trees)
    _n_instances = ninstances(X)

    if !isnothing(tree_weights)
        @assert length(trees) === length(tree_weights) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."
    end

    # apply each tree to the whole dataset
    _predictions = Matrix{Vector{Float64}}(undef, _n_instances, ntrees)
    Threads.@threads for i_tree in 1:ntrees
        _predictions[:,i_tree] = apply_proba(trees[i_tree], X)
    end

    # Average the prediction scores
    if isnothing(tree_weights)
        Vector{Vector{Float64}}([vcat(_inst_predictions...)
            for _inst_predictions in eachrow(_predictions)])
    else
        error("TODO expand code")
    end
end

# use a proper forest to test features
function apply_proba(
    forest::DForest{L},
    X::MultiLogiset,
    args...;
    weight_trees_by::Union{Bool,Symbol,AbstractVector} = false
) where {L<:Label}
    if weight_trees_by == false
        apply_proba(trees(forest), X, args...)
    elseif isa(weight_trees_by, AbstractVector)
        apply_proba(trees(forest), X, args...; tree_weights = weight_trees_by)
    # elseif weight_trees_by == :accuracy
    #   # TODO: choose HOW to weight a tree... overall_accuracy is just an example (maybe can be parameterized)
    #   apply_proba(forest.trees, X, args...; tree_weights = map(cm -> overall_accuracy(cm), get(forest.metrics, :oob_metrics...)))
    else
        @error "Unexpected value for weight_trees_by: $(weight_trees_by)"
    end
end


############################################################################################

# function tree_walk_metrics(leaf::DTLeaf; n_tot_inst = nothing, best_rule_params = [])
#     if isnothing(n_tot_inst)
#         n_tot_inst = ninstances(leaf)
#     end

#     matches = findall(leaf.supp_labels .== predictions(leaf))

#     n_correct = length(matches)
#     n_inst = length(leaf.supp_labels)

#     metrics = Dict()
#     confidence = n_correct/n_inst

#     metrics["_n_instances"] = n_inst
#     metrics["n_correct"] = n_correct
#     metrics["avg_confidence"] = confidence
#     metrics["best_confidence"] = confidence

#     if !isnothing(n_tot_inst)
#         support = n_inst/n_tot_inst
#         metrics["avg_support"] = support
#         metrics["support"] = support
#         metrics["best_support"] = support

#         for best_rule_p in best_rule_params
#             if (haskey(best_rule_p, :min_confidence) && best_rule_p.min_confidence > metrics["best_confidence"]) ||
#                 (haskey(best_rule_p, :min_support) && best_rule_p.min_support > metrics["best_support"])
#                 metrics["best_rule_t=$(best_rule_p)"] = -Inf
#             else
#                 metrics["best_rule_t=$(best_rule_p)"] = metrics["best_confidence"] * best_rule_p.t + metrics["best_support"] * (1-best_rule_p.t)
#             end
#         end
#     end


#     metrics
# end

# function tree_walk_metrics(tree::DTInternal; n_tot_inst = nothing, best_rule_params = [])
#     if isnothing(n_tot_inst)
#         n_tot_inst = ninstances(tree)
#     end
#     # TODO visit also tree.this
#     metrics_l = tree_walk_metrics(tree.left;  n_tot_inst = n_tot_inst, best_rule_params = best_rule_params)
#     metrics_r = tree_walk_metrics(tree.right; n_tot_inst = n_tot_inst, best_rule_params = best_rule_params)

#     metrics = Dict()

#     # Number of instances passing through the node
#     metrics["_n_instances"] =
#         metrics_l["_n_instances"] + metrics_r["_n_instances"]

#     # Number of correct instances passing through the node
#     metrics["n_correct"] =
#         metrics_l["n_correct"] + metrics_r["n_correct"]

#     # Average confidence of the subtree
#     metrics["avg_confidence"] =
#         (metrics_l["_n_instances"] * metrics_l["avg_confidence"] +
#         metrics_r["_n_instances"] * metrics_r["avg_confidence"]) /
#             (metrics_l["_n_instances"] + metrics_r["_n_instances"])

#     # Average support of the subtree (Note to self: weird...?)
#     metrics["avg_support"] =
#         (metrics_l["_n_instances"] * metrics_l["avg_support"] +
#         metrics_r["_n_instances"] * metrics_r["avg_support"]) /
#             (metrics_l["_n_instances"] + metrics_r["_n_instances"])

#     # Best confidence of the best-confidence path passing through the node
#     metrics["best_confidence"] = max(metrics_l["best_confidence"], metrics_r["best_confidence"])

#     # Support of the current node
#     if !isnothing(n_tot_inst)
#         metrics["support"] = (metrics_l["_n_instances"] + metrics_r["_n_instances"])/n_tot_inst

#         # Best support of the best-support path passing through the node
#         metrics["best_support"] = max(metrics_l["best_support"], metrics_r["best_support"])

#         # Best rule (confidence and support) passing through the node
#         for best_rule_p in best_rule_params
#             metrics["best_rule_t=$(best_rule_p)"] = max(metrics_l["best_rule_t=$(best_rule_p)"], metrics_r["best_rule_t=$(best_rule_p)"])
#         end
#     end

#     metrics
# end

# tree_walk_metrics(tree::DTree; kwargs...) = tree_walk_metrics(tree.root; kwargs...)
