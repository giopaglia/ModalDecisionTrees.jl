using StatsBase

export apply_tree, apply_forest, apply_model, printapply, tree_walk_metrics

import SoleModels: apply

############################################################################################
############################################################################################
############################################################################################

function apply end

apply_model = apply

@deprecate apply_tree(x) apply_model(x)
@deprecate apply_forest(x) apply_model(x)

function apply_proba end

apply_model_proba = apply_proba

@deprecate apply_tree_proba(x) apply_model_proba(x)
@deprecate apply_trees_proba(x) apply_model_proba(x)
@deprecate apply_forest_proba(x) apply_model_proba(x)


############################################################################################
############################################################################################
############################################################################################

mm_instance_initialworldset(Xs, tree::DTree, i_instance::Integer) = begin
    Ss = Vector{WorldSet}(undef, nmodalities(Xs))
    for (i_modality,X) in enumerate(eachmodality(Xs))
        Ss[i_modality] = initialworldset(X, i_instance, initconditions(tree)[i_modality])
    end
    Ss
end

softmax(v::AbstractVector) = exp.(v) ./ sum(exp.(v))
softmax(m::AbstractMatrix) = mapslices(softmax, m; dims=1)

############################################################################################
############################################################################################
############################################################################################

printapply(model::SymbolicModel, args...; kwargs...) = printapply(stdout, model, args...; kwargs...)
# printapply_proba(model::SymbolicModel, args...; kwargs...) = printapply_proba(stdout, model, args...; kwargs...)

function printapply(io::IO, model::SymbolicModel, Xs, Y::AbstractVector; kwargs...)
    predictions, newmodel = sprinkle(model, Xs, Y)
    printmodel(io, newmodel; kwargs...)
    predictions, newmodel
end

# function printapply_proba(io::IO, model::SymbolicModel, Xs, Y::AbstractVector; kwargs...)
#     predictions, newmodel = apply_proba(model, Xs, Y TODO)
#     printmodel(io, newmodel; kwargs...)
#     predictions, newmodel
# end

################################################################################
# Apply models: predict labels for a new dataset of instances
################################################################################

function apply(leaf::DTLeaf, Xs, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}; suppress_parity_warning = false)
    prediction(leaf)
end

function apply(leaf::NSDTLeaf, Xs, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}; suppress_parity_warning = false)
    d = slicedataset(Xs, [i_instance])
    # println(typeof(d))
    # println(hasmethod(size,   (typeof(d),)) ? size(d)   : nothing)
    # println(hasmethod(length, (typeof(d),)) ? length(d) : nothing)
    preds = leaf.predicting_function(d)
    @assert length(preds) == 1 "Error in apply(::NSDTLeaf, ...) The predicting function returned some malformed output. Expected is a Vector of a single prediction, while the returned value is:\n$(preds)\n$(hasmethod(length, (typeof(preds),)) ? length(preds) : "(length = $(length(preds)))")\n$(hasmethod(size, (typeof(preds),)) ? size(preds) : "(size = $(size(preds)))")"
    # println(preds)
    # println(typeof(preds))
    preds[1]
end

function apply(tree::DTInternal, Xs, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}; kwargs...)
    @logmsg LogDetail "applying branch..."
    @logmsg LogDetail " worlds" worlds
    (satisfied,new_worlds) =
        modalstep(
            modality(Xs, i_modality(tree)),
            i_instance,
            worlds[i_modality(tree)],
            decision(tree),
    )

    worlds[i_modality(tree)] = new_worlds
    @logmsg LogDetail " ->(satisfied,worlds')" satisfied worlds
    apply((satisfied ? left(tree) : right(tree)), Xs, i_instance, worlds; kwargs...)
end

# Obtain predictions of a tree on a dataset
function apply(tree::DTree{L}, Xs; print_progress = !(Xs isa MultiLogiset), kwargs...) where {L}
    @logmsg LogDetail "apply..."
    _ninstances = ninstances(Xs)
    predictions = Vector{L}(undef, _ninstances)

    if print_progress
        p = Progress(_ninstances, 1, "Applying tree...")
    end
    Threads.@threads for i_instance in 1:_ninstances
        @logmsg LogDetail " instance $i_instance/$_ninstances"
        # TODO figure out: is it better to interpret the whole dataset at once, or instance-by-instance? The first one enables reusing training code

        worlds = mm_instance_initialworldset(Xs, tree, i_instance)

        predictions[i_instance] = apply(root(tree), Xs, i_instance, worlds; kwargs...)
        print_progress && next!(p)
    end
    predictions
end

# use an array of trees to test features
function apply(
    trees::AbstractVector{<:DTree{<:L}},
    Xs;
    print_progress = !(Xs isa MultiLogiset),
    suppress_parity_warning = false,
    tree_weights::Union{AbstractMatrix{Z},AbstractVector{Z},Nothing} = nothing,
) where {L<:Label,Z<:Real}
    @logmsg LogDetail "apply..."
    ntrees = length(trees)
    _ninstances = ninstances(Xs)

    if !(tree_weights isa AbstractMatrix)
        if isnothing(tree_weights)
            tree_weights = Ones{Int}(length(trees), ninstances(Xs)) # TODO optimize?
        elseif tree_weights isa AbstractVector
            tree_weights = hcat([tree_weights for i in 1:ninstances(Xs)]...)
        else
            @show typeof(tree_weights)
            error("Unexpected tree_weights encountered $(tree_weights).")
        end
    end

    @assert length(trees) == size(tree_weights, 1) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."
    @assert ninstances(Xs) == size(tree_weights, 2) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."

    # apply each tree to the whole dataset
    _predictions = Matrix{L}(undef, ntrees, _ninstances)
    if print_progress
        p = Progress(ntrees, 1, "Applying trees...")
    end
    Threads.@threads for i_tree in 1:ntrees
        _predictions[i_tree,:] = apply(trees[i_tree], Xs; print_progress = false, suppress_parity_warning = suppress_parity_warning)
        print_progress && next!(p)
    end

    # for each instance, aggregate the predictions
    predictions = Vector{L}(undef, _ninstances)
    Threads.@threads for i_instance in 1:_ninstances
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
    Xs;
    suppress_parity_warning = false,
    weight_trees_by::Union{Bool,Symbol,AbstractVector} = false,
)
    if weight_trees_by == false
        apply(trees(forest), Xs; suppress_parity_warning = suppress_parity_warning)
    elseif isa(weight_trees_by, AbstractVector)
        apply(trees(forest), Xs; suppress_parity_warning = suppress_parity_warning, tree_weights = weight_trees_by)
    # elseif weight_trees_by == :accuracy
    #   # TODO: choose HOW to weight a tree... overall_accuracy is just an example (maybe can be parameterized)
    #   apply(forest.trees, Xs; tree_weights = map(cm -> overall_accuracy(cm), get(forest.metrics, :oob_metrics...)))
    else
        error("Unexpected value for weight_trees_by: $(weight_trees_by)")
    end
end

function apply(
    nsdt::RootLevelNeuroSymbolicHybrid,
    Xs;
    suppress_parity_warning = false,
)
    W = softmax(nsdt.feature_function(Xs))
    apply(nsdt.trees, Xs; suppress_parity_warning = suppress_parity_warning, tree_weights = W)
end

################################################################################
# Sprinkle: distribute dataset instances throughout a tree
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
        initconditions(tree),
    )
end


function sprinkle(
    leaf::DTLeaf{L},
    Xs,
    i_instance::Integer,
    worlds::AbstractVector{<:AbstractWorldSet},
    y::L;
    update_labels = false,
    suppress_parity_warning = false,
) where {L<:Label}
    _supp_labels = L[supp_labels(leaf)..., y]

    _prediction =
        if update_labels
            bestguess(supp_labels(leaf))
        else
            prediction(leaf)
        end

    _prediction, DTLeaf(_prediction, _supp_labels)
end

function sprinkle(
    leaf::NSDTLeaf{L},
    Xs,
    i_instance::Integer,
    worlds::AbstractVector{<:AbstractWorldSet},
    y::L;
    update_labels = false,
    suppress_parity_warning = false,
) where {L<:Label}
    _supp_train_labels      = L[leaf.supp_train_labels...,      y]
    _supp_train_predictions = L[leaf.supp_train_predictions..., apply(leaf, Xs, i_instance, worlds; kwargs...)]

    _predicting_function =
        if update_labels
            error("TODO expand code retrain")
        else
            leaf.predicting_function
        end
    d = slicedataset(Xs, [i_instance])
    _predicting_function(d)[1], NSDTLeaf{L}(_predicting_function, _supp_train_labels, leaf.supp_valid_labels, _supp_train_predictions, leaf.supp_valid_predictions)
end

function sprinkle(
    tree::DTInternal{L},
    Xs,
    i_instance::Integer,
    worlds::AbstractVector{<:AbstractWorldSet},
    y::L;
    kwargs...,
) where {L}

    (satisfied,new_worlds) =
        modalstep(
            modality(Xs, i_modality(tree)),
            i_instance,
            worlds[i_modality(tree)],
            decision(tree)
    )

    # if satisfied
    #   println("new_worlds: $(new_worlds)")
    # end

    worlds[i_modality(tree)] = new_worlds

    this_prediction, this_leaf = sprinkle(this(tree),  Xs, i_instance, worlds, y; kwargs...) # TODO test whether this works correctly

    pred, left_leaf, right_leaf =
        if satisfied
            pred, left_leaf = sprinkle(left(tree),  Xs, i_instance, worlds, y; kwargs...)
            pred, left_leaf, right(tree)
        else
            pred, right_leaf = sprinkle(right(tree), Xs, i_instance, worlds, y; kwargs...)
            pred, left(tree), right_leaf
        end

    pred, DTInternal(i_modality(tree), decision(tree), this_leaf, left_leaf, right_leaf)
end

function sprinkle(
    tree::DTree{L},
    Xs,
    Y::AbstractVector{<:L};
    print_progress = !(Xs isa MultiLogiset),
    reset_leaves = true,
    kwargs...,
) where {L}

    # Reset
    tree = (reset_leaves ? _empty_tree_leaves(tree) : tree)

    predictions = L[]
    _root = root(tree)

    # Propagate instances down the tree
    if print_progress
        p = Progress(ninstances(Xs), 1, "Applying trees...")
    end
    Threads.@threads for i_instance in 1:ninstances(Xs)
        worlds = mm_instance_initialworldset(Xs, tree, i_instance)
        pred, _root = sprinkle(_root, Xs, i_instance, worlds, Y[i_instance]; kwargs...)
        push!(predictions, pred)
        print_progress && next!(p)
    end
    predictions, DTree(_root, worldtypes(tree), initconditions(tree))
end

# use an array of trees to test features
function sprinkle(
    trees::AbstractVector{<:DTree{<:L}},
    Xs,
    Y::AbstractVector{<:L};
    print_progress = !(Xs isa MultiLogiset),
    tree_weights::Union{AbstractMatrix{Z},AbstractVector{Z},Nothing} = nothing,
    suppress_parity_warning = false,
) where {L<:Label,Z<:Real}
    @logmsg LogDetail "sprinkle..."
    trees = deepcopy(trees)
    ntrees = length(trees)
    _ninstances = ninstances(Xs)

    if !(tree_weights isa AbstractMatrix)
        if isnothing(tree_weights)
            tree_weights = Ones{Int}(length(trees), ninstances(Xs)) # TODO optimize?
        elseif tree_weights isa AbstractVector
            tree_weights = hcat([tree_weights for i in 1:ninstances(Xs)]...)
        else
            @show typeof(tree_weights)
            error("Unexpected tree_weights encountered $(tree_weights).")
        end
    end

    @assert length(trees) == size(tree_weights, 1) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."
    @assert ninstances(Xs) == size(tree_weights, 2) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."

    # apply each tree to the whole dataset
    _predictions = Matrix{L}(undef, ntrees, _ninstances)
    if print_progress
        p = Progress(ntrees, 1, "Applying trees...")
    end
    Threads.@threads for i_tree in 1:ntrees
        _predictions[i_tree,:], trees[i_tree] = sprinkle(trees[i_tree], Xs, Y; print_progress = false)
        print_progress && next!(p)
    end

    # for each instance, aggregate the predictions
    predictions = Vector{L}(undef, _ninstances)
    Threads.@threads for i_instance in 1:_ninstances
        predictions[i_instance] = bestguess(
            _predictions[:,i_instance],
            tree_weights[:,i_instance];
            suppress_parity_warning = suppress_parity_warning
        )
    end

    predictions, trees
end

# use a proper forest to test features
function sprinkle(
    forest::DForest,
    Xs,
    Y::AbstractVector{<:L};
    weight_trees_by::Union{Bool,Symbol,AbstractVector} = false,
    kwargs...
) where {L<:Label}
    predictions, trees = begin
        if weight_trees_by == false
            sprinkle(trees(forest), Xs, Y; kwargs...)
        elseif isa(weight_trees_by, AbstractVector)
            sprinkle(trees(forest), Xs, Y; tree_weights = weight_trees_by, kwargs...)
        # elseif weight_trees_by == :accuracy
        #   # TODO: choose HOW to weight a tree... overall_accuracy is just an example (maybe can be parameterized)
        #   sprinkle(forest.trees, Xs; tree_weights = map(cm -> overall_accuracy(cm), get(forest.metrics, :oob_metrics...)))
        else
            error("Unexpected value for weight_trees_by: $(weight_trees_by)")
        end
    end
    predictions, DForest{L}(trees, (;)) # TODO note that the original metrics are lost here
end

function sprinkle(
    nsdt::RootLevelNeuroSymbolicHybrid,
    Xs,
    Y::AbstractVector{<:L};
    suppress_parity_warning = false,
    kwargs...
) where {L<:Label}
    W = softmax(nsdt.feature_function(Xs))
    predictions, trees = sprinkle(
        nsdt.trees,
        Xs,
        Y;
        suppress_parity_warning = suppress_parity_warning,
        tree_weights = W,
        kwargs...,
    )
    predictions, RootLevelNeuroSymbolicHybrid(nsdt.feature_function, trees, (;)) # TODO note that the original metrics are lost here
end

# function sprinkle(tree::DTNode{T,L}, X::AbstractDimensionalDataset{T,D}, Y::AbstractVector{<:L}; reset_leaves = true, update_labels = false) where {L,T,D}
#   return sprinkle(DTree(tree, [worldtype(get_interval_ontology(Val(D-2)))], [start_without_world]), X, Y, reset_leaves = reset_leaves, update_labels = update_labels)
# end

############################################################################################

using Distributions
using CategoricalDistributions
using CategoricalArrays

function apply_proba(leaf::DTLeaf, Xs, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet})
    supp_labels(leaf)
end

function apply_proba(tree::DTInternal, Xs, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet})
    @logmsg LogDetail "applying branch..."
    @logmsg LogDetail " worlds" worlds
    (satisfied,new_worlds) =
        modalstep(
            modality(Xs, i_modality(tree)),
            i_instance,
            worlds[i_modality(tree)],
            decision(tree),
    )

    worlds[i_modality(tree)] = new_worlds
    @logmsg LogDetail " ->(satisfied,worlds')" satisfied worlds
    apply_proba((satisfied ? left(tree) : right(tree)), Xs, i_instance, worlds)
end

# Obtain predictions of a tree on a dataset
function apply_proba(tree::DTree{L}, Xs, _classes, return_scores = false) where {L<:CLabel}
    @logmsg LogDetail "apply_proba..."
    _classes = string.(_classes)
    _ninstances = ninstances(Xs)
    prediction_scores = Matrix{Float64}(undef, _ninstances, length(_classes))

    for i_instance in 1:_ninstances
        @logmsg LogDetail " instance $i_instance/$_ninstances"
        # TODO figure out: is it better to interpret the whole dataset at once, or instance-by-instance? The first one enables reusing training code

        worlds = mm_instance_initialworldset(Xs, tree, i_instance)

        this_prediction_scores = apply_proba(root(tree), Xs, i_instance, worlds)
        # d = Distributions.fit(UnivariateFinite, categorical(this_prediction_scores; levels = _classes))
        d = begin
            c = categorical(collect(this_prediction_scores); levels = _classes)
            cc = countmap(c)
            s = [get(cc, cl, 0) for cl in classes(c)]
            UnivariateFinite(classes(c), s ./ sum(s))
        end
        prediction_scores[i_instance, :] .= [pdf(d, c) for c in _classes]
    end
    if return_scores
        prediction_scores
    else
        UnivariateFinite(_classes, prediction_scores, pool=missing)
    end
end

# Obtain predictions of a tree on a dataset
function apply_proba(tree::DTree{L}, Xs, _classes = nothing, return_scores = false) where {L<:RLabel}
    @logmsg LogDetail "apply_proba..."
    _ninstances = ninstances(Xs)
    prediction_scores = Vector{Vector{Float64}}(undef, _ninstances)

    for i_instance in 1:_ninstances
        @logmsg LogDetail " instance $i_instance/$_ninstances"
        # TODO figure out: is it better to interpret the whole dataset at once, or instance-by-instance? The first one enables reusing training code

        worlds = mm_instance_initialworldset(Xs, tree, i_instance)

        prediction_scores[i_instance] = apply_proba(tree.root, Xs, i_instance, worlds)
    end
    if return_scores
        prediction_scores
    else
        [Distributions.fit(Normal, sc) for sc in prediction_scores]
    end
end

# use an array of trees to test features
function apply_proba(
    trees::AbstractVector{<:DTree{<:L}},
    Xs,
    classes;
    tree_weights::Union{AbstractMatrix{Z},AbstractVector{Z},Nothing} = nothing,
) where {L<:CLabel,Z<:Real}
    @logmsg LogDetail "apply_proba..."
    ntrees = length(trees)
    _ninstances = ninstances(Xs)

    if !(tree_weights isa AbstractMatrix)
        if isnothing(tree_weights)
            tree_weights = Ones{Int}(length(trees), ninstances(Xs)) # TODO optimize?
        elseif tree_weights isa AbstractVector
            tree_weights = hcat([tree_weights for i in 1:ninstances(Xs)]...)
        else
            @show typeof(tree_weights)
            error("Unexpected tree_weights encountered $(tree_weights).")
        end
    end

    @assert length(trees) == size(tree_weights, 1) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."
    @assert ninstances(Xs) == size(tree_weights, 2) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."

    # apply each tree to the whole dataset
    _predictions = Array{Float64,3}(undef, _ninstances, ntrees, length(classes))
    Threads.@threads for i_tree in 1:ntrees
        _predictions[:,i_tree,:] = apply_proba(trees[i_tree], Xs, classes)
    end

    # Average the prediction scores
    if isnothing(tree_weights)
        dropdims(mean(_predictions; dims=2), dims=2)
    else
        tree_weights = tree_weights./sum(tree_weights)
        prediction_scores = Matrix{Float64}(undef, _ninstances, length(classes))
        Threads.@threads for i in 1:_ninstances
            prediction_scores[i,:] .= mean(_predictions[i,:,:] * tree_weights; dims=1)
        end
        prediction_scores
    end
end

# use an array of trees to test features
function apply_proba(
    trees::AbstractVector{<:DTree{<:L}},
    Xs;
    tree_weights::Union{Nothing,AbstractVector{Z}} = nothing,
) where {L<:RLabel,Z<:Real}
    @logmsg LogDetail "apply_proba..."
    ntrees = length(trees)
    _ninstances = ninstances(Xs)

    if !isnothing(tree_weights)
        @assert length(trees) === length(tree_weights) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."
    end

    # apply each tree to the whole dataset
    _predictions = Matrix{Vector{Float64}}(undef, _ninstances, ntrees)
    Threads.@threads for i_tree in 1:ntrees
        _predictions[:,i_tree] = apply_proba(trees[i_tree], Xs)
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
    Xs,
    args...;
    weight_trees_by::Union{Bool,Symbol,AbstractVector} = false
) where {L<:Label}
    if weight_trees_by == false
        apply_proba(trees(forest), Xs, args...)
    elseif isa(weight_trees_by, AbstractVector)
        apply_proba(trees(forest), Xs, args...; tree_weights = weight_trees_by)
    # elseif weight_trees_by == :accuracy
    #   # TODO: choose HOW to weight a tree... overall_accuracy is just an example (maybe can be parameterized)
    #   apply_proba(forest.trees, Xs, args...; tree_weights = map(cm -> overall_accuracy(cm), get(forest.metrics, :oob_metrics...)))
    else
        error("Unexpected value for weight_trees_by: $(weight_trees_by)")
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

#     metrics["_ninstances"] = n_inst
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
#     metrics["_ninstances"] =
#         metrics_l["_ninstances"] + metrics_r["_ninstances"]

#     # Number of correct instances passing through the node
#     metrics["n_correct"] =
#         metrics_l["n_correct"] + metrics_r["n_correct"]

#     # Average confidence of the subtree
#     metrics["avg_confidence"] =
#         (metrics_l["_ninstances"] * metrics_l["avg_confidence"] +
#         metrics_r["_ninstances"] * metrics_r["avg_confidence"]) /
#             (metrics_l["_ninstances"] + metrics_r["_ninstances"])

#     # Average support of the subtree (Note to self: weird...?)
#     metrics["avg_support"] =
#         (metrics_l["_ninstances"] * metrics_l["avg_support"] +
#         metrics_r["_ninstances"] * metrics_r["avg_support"]) /
#             (metrics_l["_ninstances"] + metrics_r["_ninstances"])

#     # Best confidence of the best-confidence path passing through the node
#     metrics["best_confidence"] = max(metrics_l["best_confidence"], metrics_r["best_confidence"])

#     # Support of the current node
#     if !isnothing(n_tot_inst)
#         metrics["support"] = (metrics_l["_ninstances"] + metrics_r["_ninstances"])/n_tot_inst

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
