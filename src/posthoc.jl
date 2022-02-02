################################################################################
################################################################################
# TODO explain
################################################################################
################################################################################

export prune_tree

function prune_tree(tree::DTree; kwargs...)
    DTree(prune_tree(tree.root; depth = 0, kwargs...), tree.worldTypes, tree.initConditions)
end

function prune_tree(leaf::DTLeaf; kwargs...)
    leaf
end

function prune_tree(node::DTInternal{T, L}; kwargs...) where {T, L}

    @assert ! (haskey(kwargs, max_depth) && ! haskey(kwargs, depth)) "Please specify the node depth: prune_tree(node; depth = ...)"

    pruning_params = merge((
        loss_function       = default_loss_function(L) ::Union{Nothing,Function},
        max_depth           = typemax(Int64)           ::Int                    ,
        min_samples_leaf    = 1                        ::Int                    ,
        min_purity_increase = -Inf                     ::AbstractFloat          ,
        max_purity_at_leaf  = Inf                      ::AbstractFloat          ,
    ), NamedTuple(kwargs))
    
    # Honor constraints on the number of instances
    nt = length(node.this.supp_labels)
    nl = length(node.left.supp_labels)
    nr = length(node.right.supp_labels)

    if (pruning_params.max_depth < depth) ||
       (pruning_params.min_samples_leaf > nr)  ||
       (pruning_params.min_samples_leaf > nl)  ||
        return node.this
    end
    
    # Honor purity constraints
    # TODO fix
    purity   = DecisionTree.compute_purity(node.this.supp_labels,  pruning_params.loss_function)
    purity_r = DecisionTree.compute_purity(node.left.supp_labels,  pruning_params.loss_function)
    purity_l = DecisionTree.compute_purity(node.right.supp_labels, pruning_params.loss_function)

    split_purity = (nl * purity_l + nr * purity_r)

    if (purity_r > pruning_params.max_purity_at_leaf) ||
       (purity_l > pruning_params.max_purity_at_leaf) ||
       dishonor_min_purity_increase(L, pruning_params.min_purity_increase, purity, split_purity, nt)
        return node.this
    end

    return DTInternal(
        node.i_frame,
        node.decision,
        node.this,
        prune_tree(node.left;  pruning_params..., depth = depth+1),
        prune_tree(node.right; pruning_params..., depth = depth+1)
    )
end

