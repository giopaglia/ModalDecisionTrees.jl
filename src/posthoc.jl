################################################################################
################################################################################
# TODO explain
################################################################################
################################################################################

export prune_tree

function prune_tree(tree::DTree; kwargs...)
    DTree(prune_tree(tree.root; kwargs..., depth = 0), tree.worldTypes, tree.initConditions)
end

function prune_tree(leaf::DTLeaf; kwargs...)
    leaf
end

function prune_tree(node::DTInternal{T, L}; kwargs..., depth = 0) where {T, L}

    kwargs = merge((
        loss_function       :: Union{Nothing,Function} = default_loss_function(L),
        max_depth           :: Int                     = typemax(Int64),
        min_samples_leaf    :: Int                     = 1,
        min_purity_increase :: AbstractFloat           = -Inf,
        max_purity_at_leaf  :: AbstractFloat           = Inf,
    ), kwargs)
    
    # Honor constraints on the number of instances
    nt = length(node.this.supp_labels)
    nl = length(node.left.supp_labels)
    nr = length(node.right.supp_labels)

    if (kwargs.max_depth < node.depth) ||
       (kwargs.min_samples_leaf > nr)  ||
       (kwargs.min_samples_leaf > nl)  ||
        return node.this
    end
    
    # Honor purity constraints
    purity   = DecisionTree.compute_purity(node.this.supp_labels,  kwargs.loss_function)
    purity_r = DecisionTree.compute_purity(node.left.supp_labels,  kwargs.loss_function)
    purity_l = DecisionTree.compute_purity(node.right.supp_labels, kwargs.loss_function)

    split_purity = (nl * purity_l + nr * purity_r)

    if (purity_r > kwargs.max_purity_at_leaf) ||
       (purity_l > kwargs.max_purity_at_leaf) ||
       dishonor_min_purity_increase(L, kwargs.min_purity_increase, purity, split_purity, nt)
        return node.this
    end

    return DTInternal(
        node.i_frame,
        node.decision,
        node.this,
        prune_tree(node.left; kwargs..., depth = depth+1),
        prune_tree(node.right; kwargs..., depth = depth+1)
    )
end

