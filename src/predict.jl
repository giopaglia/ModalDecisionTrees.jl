export apply_tree, apply_forest, apply_trees, apply_model, print_apply_model, print_apply_tree, tree_walk_metrics


apply_model(tree::DTree,    args...; kwargs...) = apply_tree(tree,     args...; kwargs...)
apply_model(forest::DForest, args...; kwargs...) = apply_forest(forest, args...; kwargs...)

print_apply_model(tree::DTree, args...; kwargs...) = print_apply_tree(tree, args...; kwargs...)


################################################################################
# Apply models: predict labels for a new dataset of instances
################################################################################

# TODO avoid these fallbacks?
inst_init_world_sets(X::SingleFrameGenericDataset, tree::DTree, i_instance::Integer) = 
  inst_init_world_sets(MultiFrameModalDataset(X), tree, i_instance)
print_apply_tree(tree::DTree{S}, X::SingleFrameGenericDataset, Y::Vector{S}; kwargs...) where {S} = 
  print_apply_tree(tree, MultiFrameModalDataset(X), Y; kwargs...)

inst_init_world_sets(Xs::MultiFrameModalDataset, tree::DTree, i_instance::Integer) = begin
  Ss = Vector{WorldSet}(undef, n_frames(Xs))
  for (i_frame,X) in enumerate(ModalLogic.frames(Xs))
    Ss[i_frame] = initws_function(X, i_instance)(tree.initConditions[i_frame])
  end
  Ss
end

apply_tree(leaf::DTLeaf, X::Any, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}) = leaf.label

function apply_tree(tree::DTInternal, X::MultiFrameModalDataset, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet})
  @logmsg DTDetail "applying branch..."
  satisfied = true
  @logmsg DTDetail " worlds" worlds
  (satisfied,new_worlds) =
    ModalLogic.modal_step(
            get_frame(X, tree.i_frame),
            i_instance,
            worlds[tree.i_frame],
            tree.decision)

  worlds[tree.i_frame] = new_worlds
  @logmsg DTDetail " ->(satisfied,worlds')" satisfied worlds
  apply_tree((satisfied ? tree.left : tree.right), X, i_instance, worlds)
end

# Apply tree with initialConditions to a dimensional dataset in matricial form
function apply_tree(tree::DTree{S}, X::GenericDataset) where {S}
  @logmsg DTDetail "apply_tree..."
  n_instances = n_samples(X)
  predictions = Vector{S}(undef, n_instances)
  
  for i_instance in 1:n_instances
    @logmsg DTDetail " instance $i_instance/$n_instances"
    # TODO figure out: is it better to interpret the whole dataset at once, or instance-by-instance? The first one enables reusing training code

    worlds = inst_init_world_sets(X, tree, i_instance)

    predictions[i_instance] = apply_tree(tree.root, X, i_instance, worlds)
  end
  predictions
  # return (if S <: Float64 # TODO remove
  #     Float64.(predictions)
  #   else
  #     predictions
  #   end)
end

# Apply tree to a dimensional dataset in matricial form
# function apply_tree(tree::DTNode, d::MatricialDataset{T,D}) where {T, D}
#   apply_tree(DTree(tree, [world_type(ModalLogic.getIntervalOntologyOfDim(Val(D-2)))], [startWithRelationGlob]), d)
# end

# use an array of trees to test features
function apply_trees(trees::AbstractVector{DTree{S}}, X::GenericDataset; suppress_parity_warning = false, tree_weights::Union{AbstractVector{Z},Nothing} = nothing) where {S, Z<:Real}
  @logmsg DTDetail "apply_trees..."
  n_trees = length(trees)
  n_instances = n_samples(X)

  if !isnothing(tree_weights)
    @assert length(trees) === length(tree_weights) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."
  end

  # apply each tree to the whole dataset
  votes = Matrix{S}(undef, n_trees, n_instances)
  for i_tree in 1:n_trees
    votes[i_tree,:] = apply_tree(trees[i_tree], X)
  end

  # for each instance, aggregate the votes
  predictions = Vector{S}(undef, n_instances)
  Threads.@threads for i in 1:n_instances
    predictions[i] = begin
      if S <: Float64
        # @error "apply_trees need a code expansion. The case is S = $(S) <: Float64"
        if isnothing(tree_weights)
          mean(votes[:,i])
        else
          n_trees = length(votes[:,i])
          sum([votes[j,i] * tree_weights[j] for j in 1:n_trees]) / sum(tree_weights)
        end
      else
        best_score(votes[:,i], tree_weights; suppress_parity_warning = suppress_parity_warning)
      end
    end
  end

  return predictions
end

# use a proper forest to test features
function apply_forest(forest::DForest, X::GenericDataset; weight_trees_by::Union{Bool,Symbol,AbstractVector} = false)
  if weight_trees_by == false
    apply_trees(forest.trees, X)
  elseif isa(weight_trees_by, AbstractVector)
    apply_trees(forest.trees, X; tree_weights = weight_trees_by)
  elseif weight_trees_by == :accuracy
    # TODO: choose HOW to weight a tree... overall_accuracy is just an example (maybe can be parameterized)
    apply_trees(forest.trees, X; tree_weights = map(cm -> overall_accuracy(cm), forest.cm))
  else
    @error "Unexpected value for weight_trees_by: $(weight_trees_by)"
  end
end

################################################################################
# Print+Apply models: predict labels for a new dataset of instances
################################################################################

function _empty_tree_leaves(leaf::DTLeaf{S}) where {S}
    DTLeaf(leaf.label, S[])
end

function _empty_tree_leaves(tree::DTInternal)
  return DTInternal(
    tree.i_frame,
    tree.decision,
    _empty_tree_leaves(tree.left),
    _empty_tree_leaves(tree.right)
  )
end

function _empty_tree_leaves(tree::DTree)
  return DTree(
    _empty_tree_leaves(tree.root),
    tree.worldTypes,
    tree.initConditions
  )
end


function print_apply_tree(leaf::DTLeaf{<:Float64}, X::Any, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, class::S; update_label = false) where {S}
  vals = [leaf.supp_labels..., class]

  label = 
  if update_label
    StatsBase.mean(leaf.supp_labels)
  else
    leaf.label
  end

  return DTLeaf(label, vals)
end

function print_apply_tree(leaf::DTLeaf{S}, X::Any, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, class::S; update_label = false) where {S}
  vals = S[ leaf.supp_labels..., class ] # Note: this works when leaves are reset

  label = 
  if update_label

    # TODO optimize this code
    occur = Dict{S,Int}(v => 0 for v in unique(vals))
    for v in vals
      occur[v] += 1
    end
    cur_maj = vals[1]
    cur_max = occur[vals[1]]
    for v in vals
      if occur[v] > cur_max
        cur_max = occur[v]
        cur_maj = v
      end
    end
    cur_maj
  else
    leaf.label
  end

  return DTLeaf(label, vals)
end

function print_apply_tree(
  tree::DTInternal{T, S},
  X::MultiFrameModalDataset,
  i_instance::Integer,
  worlds::AbstractVector{<:AbstractWorldSet},
  class::S;
  update_label = false,
) where {T, S}
  
  (satisfied,new_worlds) = ModalLogic.modal_step(get_frame(X, tree.i_frame), i_instance, worlds[tree.i_frame], tree.decision)
  
  # if satisfied
  #   println("new_worlds: $(new_worlds)")
  # end

  worlds[tree.i_frame] = new_worlds

  DTInternal(
    tree.i_frame,
    tree.decision,
      satisfied  ? print_apply_tree(tree.left,  X, i_instance, worlds, class, update_label = update_label) : tree.left,
    (!satisfied) ? print_apply_tree(tree.right, X, i_instance, worlds, class, update_label = update_label) : tree.right,
  )
end

function print_apply_tree(
  io::IO,
  tree::DTree{S},
  X::GenericDataset,
  Y::Vector{S};
  reset_leaves = true,
  update_label = false,
  do_print = true,
  print_relative_confidence = false,
) where {S}
  # Reset 
  tree = (reset_leaves ? _empty_tree_leaves(tree) : tree)

  # Propagate instances down the tree
  for i_instance in 1:n_samples(X)

    worlds = inst_init_world_sets(X, tree, i_instance)

    tree = DTree(
      print_apply_tree(tree.root, X, i_instance, worlds, Y[i_instance], update_label = update_label),
      tree.worldTypes,
      tree.initConditions,
    )
  end
  if do_print
    if print_relative_confidence
      print_tree(io, tree; rel_confidence_class_counts = countmap(Y))
    else
      print_tree(io, tree)
    end
    # print(tree)
  end
  return tree
end
print_apply_tree(tree::DTree{S}, X::GenericDataset, Y::Vector{S}; kwargs...) where {S} = print_apply_tree(stdout, tree, X, Y; kwargs...)

# function print_apply_tree(tree::DTNode{T, S}, X::MatricialDataset{T,D}, Y::Vector{S}; reset_leaves = true, update_label = false) where {S, T, D}
#   return print_apply_tree(DTree(tree, [world_type(ModalLogic.getIntervalOntologyOfDim(Val(D-2)))], [startWithRelationGlob]), X, Y, reset_leaves = reset_leaves, update_label = update_label)
# end


function tree_walk_metrics(leaf::DTLeaf; n_tot_inst = nothing, best_rule_params = [])
  if isnothing(n_tot_inst)
    n_tot_inst = n_samples(leaf)
  end
  
  matches = findall(leaf.supp_labels .== leaf.label)

  n_correct = length(matches)
  n_inst = length(leaf.supp_labels)

  metrics = Dict()
  confidence = n_correct/n_inst
  
  metrics["n_instances"] = n_inst
  metrics["n_correct"] = n_correct
  metrics["avg_confidence"] = confidence
  metrics["best_confidence"] = confidence
  
  if !isnothing(n_tot_inst)
    support = n_inst/n_tot_inst
    metrics["avg_support"] = support
    metrics["support"] = support
    metrics["best_support"] = support

    for best_rule_p in best_rule_params
      if (haskey(best_rule_p, :min_confidence) && best_rule_p.min_confidence > metrics["best_confidence"]) ||
        (haskey(best_rule_p, :min_support) && best_rule_p.min_support > metrics["best_support"])
        metrics["best_rule_t=$(best_rule_p)"] = -Inf
      else
        metrics["best_rule_t=$(best_rule_p)"] = metrics["best_confidence"] * best_rule_p.t + metrics["best_support"] * (1-best_rule_p.t)
      end
    end
  end


  metrics
end

function tree_walk_metrics(tree::DTInternal; n_tot_inst = nothing, best_rule_params = [])
  if isnothing(n_tot_inst)
    n_tot_inst = n_samples(tree)
  end
  metrics_l = tree_walk_metrics(tree.left;  n_tot_inst = n_tot_inst, best_rule_params = best_rule_params)
  metrics_r = tree_walk_metrics(tree.right; n_tot_inst = n_tot_inst, best_rule_params = best_rule_params)

  metrics = Dict()

  # Number of instances passing through the node
  metrics["n_instances"] =
    metrics_l["n_instances"] + metrics_r["n_instances"]

  # Number of correct instances passing through the node
  metrics["n_correct"] =
    metrics_l["n_correct"] + metrics_r["n_correct"]
  
  # Average confidence of the subtree
  metrics["avg_confidence"] =
    (metrics_l["n_instances"] * metrics_l["avg_confidence"] +
    metrics_r["n_instances"] * metrics_r["avg_confidence"]) /
      (metrics_l["n_instances"] + metrics_r["n_instances"])
  
  # Average support of the subtree (Note to self: weird...?)
  metrics["avg_support"] =
    (metrics_l["n_instances"] * metrics_l["avg_support"] +
    metrics_r["n_instances"] * metrics_r["avg_support"]) /
      (metrics_l["n_instances"] + metrics_r["n_instances"])
  
  # Best confidence of the best-confidence path passing through the node
  metrics["best_confidence"] = max(metrics_l["best_confidence"], metrics_r["best_confidence"])
  
  # Support of the current node
  if !isnothing(n_tot_inst)
    metrics["support"] = (metrics_l["n_instances"] + metrics_r["n_instances"])/n_tot_inst
  
    # Best support of the best-support path passing through the node
    metrics["best_support"] = max(metrics_l["best_support"], metrics_r["best_support"])
    
    # Best rule (confidence and support) passing through the node
    for best_rule_p in best_rule_params
      metrics["best_rule_t=$(best_rule_p)"] = max(metrics_l["best_rule_t=$(best_rule_p)"], metrics_r["best_rule_t=$(best_rule_p)"])
    end
  end

  metrics
end

tree_walk_metrics(tree::DTree; kwargs...) = tree_walk_metrics(tree.root; kwargs...)


#=
TODO

# Returns a dict ("Label1" => 1, "Label2" => 2, "Label3" => 3, ...)
label_index(labels::AbstractVector{Label}) = Dict(v => k for (k, v) in enumerate(labels))

## Helper function. Counts the votes.
## Returns a vector of probabilities (eg. [0.2, 0.6, 0.2]) which is in the same
## order as get_labels(classifier) (eg. ["versicolor", "setosa", "virginica"])
function compute_probabilities(labels::AbstractVector{Label}, votes::AbstractVector{Label}, weights=1.0)
  label2ind = label_index(labels)
  counts = zeros(Float64, length(label2ind))
  for (i, label) in enumerate(votes)
    if isa(weights, Real)
      counts[label2ind[label]] += weights
    else
      counts[label2ind[label]] += weights[i]
    end
  end
  return counts / sum(counts) # normalize to get probabilities
end

# Applies `row_fun(X_row)::AbstractVector` to each row in X
# and returns a matrix containing the resulting vectors, stacked vertically
function stack_function_results(row_fun::Function, X::AbstractMatrix)
  N = size(X, 1)
  N_cols = length(row_fun(X[1, :])) # gets the number of columns
  out = Array{Float64}(undef, N, N_cols)
  for i in 1:N
    out[i, :] = row_fun(X[i, :])
  end
  return out
end

"""    apply_tree_proba(::Node, features, col_labels::AbstractVector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
apply_tree_proba(leaf::DTLeaf, features::AbstractVector, labels) where =
  compute_probabilities(labels, leaf.supp_labels)

function apply_tree_proba(tree::DTInternal{S, T}, features::AbstractVector{S}, labels) where {S, T}
  if tree.decision.threshold === nothing
    return apply_tree_proba(tree.left, features, labels)
  elseif eval(Expr(:call, tree.decision.test_operator, tree.decision.feature ... , tree.decision.threshold))
    return apply_tree_proba(tree.left, features, labels)
  else
    return apply_tree_proba(tree.right, features, labels)
  end
end

apply_tree_proba(tree::DTNode, features::AbstractMatrix{S}, labels) =
  stack_function_results(row->apply_tree_proba(tree, row, labels), features)

=#
