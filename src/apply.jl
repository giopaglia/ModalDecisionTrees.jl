using StatsBase
export apply_tree, apply_forest, apply_model, print_apply, tree_walk_metrics

############################################################################################
############################################################################################
############################################################################################

inst_init_world_sets(Xs::MultiFrameModalDataset, tree::DTree, i_sample::Integer) = begin
    Ss = Vector{WorldSet}(undef, n_frames(Xs))
    for (i_frame,X) in enumerate(frames(Xs))
        Ss[i_frame] = init_world_sets_fun(X, i_sample, tree.world_types[i_frame])(tree.init_conditions[i_frame])
    end
    Ss
end

############################################################################################
############################################################################################
############################################################################################

# ModalDataset -> MultiFrameModalDataset

apply(model::Any, X::ModalDataset, args...; kwargs...) =
    apply(model, MultiFrameModalDataset(X), args...; kwargs...)

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


apply_proba(model::Any, X::ModalDataset, args...; kwargs...) =
    apply_proba(model, MultiFrameModalDataset(X), args...; kwargs...)

# apply_tree_proba   = apply_model_proba
# apply_trees_proba  = apply_model_proba
# apply_forest_proba = apply_model_proba

apply_model_proba = apply_proba

################################################################################
# Apply models: predict labels for a new dataset of instances
################################################################################

function apply(leaf::DTLeaf, X::Any, i_sample::Integer, worlds::AbstractVector{<:AbstractWorldSet})
    prediction(leaf)
end

function apply(leaf::NSDTLeaf, X::Any, i_sample::Integer, worlds::AbstractVector{<:AbstractWorldSet})
    d = slice_dataset(X, [i_sample])
    # println(typeof(d))
    # println(hasmethod(size,   (typeof(d),)) ? size(d)   : nothing)
    # println(hasmethod(length, (typeof(d),)) ? length(d) : nothing)
    preds = leaf.predicting_function(d)
    @assert length(preds) == 1 "Error in apply(::NSDTLeaf, ...) The predicting function returned some malformed output. Expected is a Vector of a single prediction, while the returned value is:\n$(preds)\n$(hasmethod(length, (typeof(preds),)) ? length(preds) : "(length = $(length(preds)))")\n$(hasmethod(size, (typeof(preds),)) ? size(preds) : "(size = $(size(preds)))")"
    # println(preds)
    # println(typeof(preds))
    preds[1]
end

function apply(tree::DTInternal, X::MultiFrameModalDataset, i_sample::Integer, worlds::AbstractVector{<:AbstractWorldSet})
    @logmsg DTDetail "applying branch..."
    @logmsg DTDetail " worlds" worlds
    (satisfied,new_worlds) =
        ModalLogic.modal_step(
            get_frame(X, tree.i_frame),
            i_sample,
            worlds[tree.i_frame],
            tree.decision,
    )

    worlds[tree.i_frame] = new_worlds
    @logmsg DTDetail " ->(satisfied,worlds')" satisfied worlds
    apply((satisfied ? tree.left : tree.right), X, i_sample, worlds)
end

# Obtain predictions of a tree on a dataset
function apply(tree::DTree{L}, X::MultiFrameModalDataset) where {L}
    @logmsg DTDetail "apply..."
    _n_samples = n_samples(X)
    predictions = Vector{L}(undef, _n_samples)
    
    for i_sample in 1:_n_samples
        @logmsg DTDetail " instance $i_sample/$_n_samples"
        # TODO figure out: is it better to interpret the whole dataset at once, or instance-by-instance? The first one enables reusing training code

        worlds = inst_init_world_sets(X, tree, i_sample)

        predictions[i_sample] = apply(tree.root, X, i_sample, worlds)
    end
    predictions
end

# use an array of trees to test features
function apply(
        trees::AbstractVector{<:DTree{<:L}},
        X::MultiFrameModalDataset;
        suppress_parity_warning = false,
        tree_weights::Union{AbstractVector{Z},Nothing} = nothing,
    ) where {L<:Label, Z<:Real}
    @logmsg DTDetail "apply..."
    n_trees = length(trees)
    _n_samples = n_samples(X)

    if !isnothing(tree_weights)
        @assert length(trees) === length(tree_weights) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."
    end

    # apply each tree to the whole dataset
    _predictions = Matrix{L}(undef, n_trees, _n_samples)
    Threads.@threads for i_tree in 1:n_trees
        _predictions[i_tree,:] = apply(trees[i_tree], X)
    end

    # for each instance, aggregate the predictions
    predictions = Vector{L}(undef, _n_samples)
    Threads.@threads for i in 1:_n_samples
        predictions[i] = majority_vote(_predictions[:,i], tree_weights; suppress_parity_warning = suppress_parity_warning)
    end

    predictions
end

# use a proper forest to test features
function apply(
        forest::DForest,
        X::MultiFrameModalDataset;
        suppress_parity_warning = false,
        weight_trees_by::Union{Bool,Symbol,AbstractVector} = false,
    )
    if weight_trees_by == false
        apply(forest.trees, X; suppress_parity_warning = suppress_parity_warning)
    elseif isa(weight_trees_by, AbstractVector)
        apply(forest.trees, X; suppress_parity_warning = suppress_parity_warning, tree_weights = weight_trees_by)
    # elseif weight_trees_by == :accuracy
    #   # TODO: choose HOW to weight a tree... overall_accuracy is just an example (maybe can be parameterized)
    #   apply(forest.trees, X; tree_weights = map(cm -> overall_accuracy(cm), get(forest.metrics, :oob_metrics...)))
    else
        @error "Unexpected value for weight_trees_by: $(weight_trees_by)"
    end
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

function _empty_tree_leaves(node::DTInternal{T, L}) where {T, L}
    return DTInternal(
        node.i_frame,
        node.decision,
        _empty_tree_leaves(node.this),
        _empty_tree_leaves(node.left),
        _empty_tree_leaves(node.right),
    )
end

function _empty_tree_leaves(tree::DTree)
    return DTree(
        _empty_tree_leaves(tree.root),
        tree.world_types,
        tree.init_conditions,
    )
end


function apply(
        leaf::DTLeaf{L},
        X::MultiFrameModalDataset,
        i_sample::Integer,
        worlds::AbstractVector{<:AbstractWorldSet},
        class::L;
        update_labels = false,
    ) where {L<:Label}
    _supp_labels = L[leaf.supp_labels..., class]

    _prediction = 
        if update_labels
            average_label(leaf.supp_labels)
        else
            prediction(leaf)
        end

    _prediction, DTLeaf(_prediction, _supp_labels)
end

function apply(
        leaf::NSDTLeaf{L},
        X::MultiFrameModalDataset,
        i_sample::Integer,
        worlds::AbstractVector{<:AbstractWorldSet},
        class::L;
        update_labels = false,
    ) where {L<:Label}
    _supp_train_labels      = L[leaf.supp_train_labels...,      class]
    _supp_train_predictions = L[leaf.supp_train_predictions..., apply(leaf, X, i_sample, worlds)]

    _predicting_function = 
        if update_labels
            error("TODO expand code retrain")
        else
            leaf.predicting_function
        end
    d = slice_dataset(X, [i_sample])
    _predicting_function(d)[1], NSDTLeaf{L}(_predicting_function, _supp_train_labels, leaf.supp_valid_labels, _supp_train_predictions, leaf.supp_valid_predictions)
end

function apply(
    tree::DTInternal{T, L},
    X::MultiFrameModalDataset,
    i_sample::Integer,
    worlds::AbstractVector{<:AbstractWorldSet},
    class::L;
    update_labels = false,
) where {T, L}
    
    (satisfied,new_worlds) = ModalLogic.modal_step(get_frame(X, tree.i_frame), i_sample, worlds[tree.i_frame], tree.decision)
    
    # if satisfied
    #   println("new_worlds: $(new_worlds)")
    # end

    worlds[tree.i_frame] = new_worlds

    this_prediction, this_leaf = apply(tree.this,  X, i_sample, worlds, class, update_labels = update_labels) # TODO test whether this works correctly 
    
    pred, left_leaf, right_leaf =
        if satisfied
            pred, left_leaf = apply(tree.left,  X, i_sample, worlds, class, update_labels = update_labels)
            pred, left_leaf, tree.right
        else
            pred, right_leaf = apply(tree.right, X, i_sample, worlds, class, update_labels = update_labels)
            pred, tree.left, right_leaf
        end

    pred, DTInternal(tree.i_frame, tree.decision, this_leaf, left_leaf, right_leaf)
end

function apply(
    tree::DTree{L},
    X::MultiFrameModalDataset,
    Y::AbstractVector{<:L};
    reset_leaves = true,
    update_labels = false,
) where {L}

    # Reset 
    tree = (reset_leaves ? _empty_tree_leaves(tree) : tree)

    predictions = L[]
    root = tree.root

    # Propagate instances down the tree
    for i_sample in 1:n_samples(X)
        worlds = inst_init_world_sets(X, tree, i_sample)
        pred, root = apply(root, X, i_sample, worlds, Y[i_sample], update_labels = update_labels)
        push!(predictions, pred)
    end
    predictions, DTree(root, tree.world_types, tree.init_conditions)
end

# function apply(tree::DTNode{T, L}, X::DimensionalDataset{T,D}, Y::AbstractVector{<:L}; reset_leaves = true, update_labels = false) where {L, T, D}
#   return apply(DTree(tree, [world_type(ModalLogic.get_interval_ontology(Val(D-2)))], [start_without_world]), X, Y, reset_leaves = reset_leaves, update_labels = update_labels)
# end

############################################################################################

# using Distributions
# using CategoricalDistributions
using CategoricalArrays

function apply_proba(leaf::DTLeaf, X::Any, i_sample::Integer, worlds::AbstractVector{<:AbstractWorldSet})
    leaf.supp_labels
end

function apply_proba(tree::DTInternal, X::MultiFrameModalDataset, i_sample::Integer, worlds::AbstractVector{<:AbstractWorldSet})
    @logmsg DTDetail "applying branch..."
    @logmsg DTDetail " worlds" worlds
    (satisfied,new_worlds) =
        ModalLogic.modal_step(
            get_frame(X, tree.i_frame),
            i_sample,
            worlds[tree.i_frame],
            tree.decision,
    )

    worlds[tree.i_frame] = new_worlds
    @logmsg DTDetail " ->(satisfied,worlds')" satisfied worlds
    apply_proba((satisfied ? tree.left : tree.right), X, i_sample, worlds)
end

# Obtain predictions of a tree on a dataset
function apply_proba(tree::DTree{L}, X::MultiFrameModalDataset, classes) where {L<:CLabel}
    @logmsg DTDetail "apply_proba..."
    _n_samples = n_samples(X)
    prediction_scores = Matrix{Float64}(undef, _n_samples, length(classes))
    
    for i_sample in 1:_n_samples
        @logmsg DTDetail " instance $i_sample/$_n_samples"
        # TODO figure out: is it better to interpret the whole dataset at once, or instance-by-instance? The first one enables reusing training code

        worlds = inst_init_world_sets(X, tree, i_sample)

        this_prediction_scores = apply_proba(tree.root, X, i_sample, worlds)
        # d = Distributions.fit(UnivariateFinite, categorical(this_prediction_scores; levels = classes))
        d = begin
            c = categorical(this_prediction_scores; levels = classes)
            cc = countmap(c)
            s = [cc[cl] for cl in classes(c)]
            UnivariateFinite(classes(c), s ./ sum(s))
        end
        prediction_scores[i_sample, :] .= [pdf(d, c) for c in classes]
    end
    prediction_scores
end

# use an array of trees to test features
function apply_proba(
        trees::AbstractVector{<:DTree{<:L}},
        X::MultiFrameModalDataset,
        classes;
        tree_weights::Union{AbstractVector{Z},Nothing} = nothing,
    ) where {L<:CLabel, Z<:Real}
    @logmsg DTDetail "apply_proba..."
    n_trees = length(trees)
    _n_samples = n_samples(X)

    if !isnothing(tree_weights)
        @assert length(trees) === length(tree_weights) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."
    end

    # apply each tree to the whole dataset
    _predictions = Array{Float64,3}(undef, _n_samples, n_trees, length(classes))
    Threads.@threads for i_tree in 1:n_trees
        _predictions[:,i_tree,:] = apply_proba(trees[i_tree], X, classes)
    end

    # Average the prediction scores
    if isnothing(tree_weights)
        dropdims(mean(_predictions; dims=2), dims=2)
    else
        tree_weights = tree_weights./sum(tree_weights)
        prediction_scores = Matrix{Float64}(undef, _n_samples, length(classes))
        Threads.@threads for i in 1:_n_samples
            prediction_scores[i,:] .= mean(_predictions[i,:,:] * tree_weights; dims=1)
        end
        prediction_scores
    end
end

# use a proper forest to test features
function apply_proba(forest::DForest{L}, X::MultiFrameModalDataset, classes; weight_trees_by::Union{Bool,Symbol,AbstractVector} = false) where {L<:CLabel}
    if weight_trees_by == false
        apply_proba(forest.trees, X, classes)
    elseif isa(weight_trees_by, AbstractVector)
        apply_proba(forest.trees, X, classes; tree_weights = weight_trees_by)
    # elseif weight_trees_by == :accuracy
    #   # TODO: choose HOW to weight a tree... overall_accuracy is just an example (maybe can be parameterized)
    #   apply_proba(forest.trees, X, classes; tree_weights = map(cm -> overall_accuracy(cm), get(forest.metrics, :oob_metrics...)))
    else
        @error "Unexpected value for weight_trees_by: $(weight_trees_by)"
    end
end


############################################################################################

# function tree_walk_metrics(leaf::DTLeaf; n_tot_inst = nothing, best_rule_params = [])
#     if isnothing(n_tot_inst)
#         n_tot_inst = n_samples(leaf)
#     end
    
#     matches = findall(leaf.supp_labels .== predictions(leaf))

#     n_correct = length(matches)
#     n_inst = length(leaf.supp_labels)

#     metrics = Dict()
#     confidence = n_correct/n_inst
    
#     metrics["_n_samples"] = n_inst
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
#         n_tot_inst = n_samples(tree)
#     end
#     # TODO visit also tree.this
#     metrics_l = tree_walk_metrics(tree.left;  n_tot_inst = n_tot_inst, best_rule_params = best_rule_params)
#     metrics_r = tree_walk_metrics(tree.right; n_tot_inst = n_tot_inst, best_rule_params = best_rule_params)

#     metrics = Dict()

#     # Number of instances passing through the node
#     metrics["_n_samples"] =
#         metrics_l["_n_samples"] + metrics_r["_n_samples"]

#     # Number of correct instances passing through the node
#     metrics["n_correct"] =
#         metrics_l["n_correct"] + metrics_r["n_correct"]
    
#     # Average confidence of the subtree
#     metrics["avg_confidence"] =
#         (metrics_l["_n_samples"] * metrics_l["avg_confidence"] +
#         metrics_r["_n_samples"] * metrics_r["avg_confidence"]) /
#             (metrics_l["_n_samples"] + metrics_r["_n_samples"])
    
#     # Average support of the subtree (Note to self: weird...?)
#     metrics["avg_support"] =
#         (metrics_l["_n_samples"] * metrics_l["avg_support"] +
#         metrics_r["_n_samples"] * metrics_r["avg_support"]) /
#             (metrics_l["_n_samples"] + metrics_r["_n_samples"])
    
#     # Best confidence of the best-confidence path passing through the node
#     metrics["best_confidence"] = max(metrics_l["best_confidence"], metrics_r["best_confidence"])
    
#     # Support of the current node
#     if !isnothing(n_tot_inst)
#         metrics["support"] = (metrics_l["_n_samples"] + metrics_r["_n_samples"])/n_tot_inst
    
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
