################################################################################
################################################################################
# TODO explain
################################################################################
################################################################################

export prune_tree

# TODO fix this using specified purity
function prune_tree(tree::DTNode, max_purity_threshold::AbstractFloat = 1.0)
    if max_purity_threshold >= 1.0
        return tree
    end
    # Prune the tree once TODO make more efficient (avoid copying so many nodes.)
    function _prune_run(tree::DTNode)
        N = length(tree)
        if N == 1        ## a DTLeaf
            return tree
        elseif N == 2    ## a stump
            all_labels = [tree.left.values; tree.right.values]
            majority = majority_vote(all_labels)
            purity = sum(all_labels .== majority) / length(all_labels)
            if purity >= max_purity_threshold
                return DTLeaf(majority, all_labels)
            else
                return tree
            end
        else
            # TODO also associate an Internal node with values and majority (all_labels, majority)
            return DTInternal(tree.i_frame, tree.relation, tree.feature, tree.test_operator, tree.threshold,
                        _prune_run(tree.left),
                        _prune_run(tree.right))
        end
    end

    # Keep pruning until "convergence"
    pruned = _prune_run(tree)
    while true
        length(pruned) < length(tree) || break
        pruned = _prune_run(tree)
        tree = pruned
    end
    return pruned
end

function prune_tree(tree::DTree, max_purity_threshold::AbstractFloat = 1.0)
    DTree(prune_tree(tree.root), tree.worldTypes, tree.initConditions)
end

