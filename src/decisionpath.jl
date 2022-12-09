
export DecisionPath, DecisionPathNode,
            get_path_in_tree, get_internalnode_dirname,
            mk_tree_path, get_tree_path_as_dirpath

struct DecisionPathNode
    taken         :: Bool
    feature       :: ModalFeature
    test_operator :: TestOperatorFun
    threshold     :: T where T
    worlds        :: AbstractWorldSet
end

const DecisionPath = Vector{DecisionPathNode}

_get_path_in_tree(leaf::DTLeaf, X::Any, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, i_frame::Integer, paths::Vector{DecisionPath})::AbstractWorldSet = return worlds[i_frame]
function _get_path_in_tree(tree::DTInternal, X::MultiFrameModalDataset, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, i_frame::Integer, paths::Vector{DecisionPath})::AbstractWorldSet
    satisfied = true
    (satisfied,new_worlds,worlds_map) =
        ModalLogic.modal_step(
                        get_frame(X, tree.i_frame),
                        i_instance,
                        worlds[tree.i_frame],
                        tree.decision,
                        Val(true)
                    )

    worlds[tree.i_frame] = new_worlds
    survivors = _get_path_in_tree((satisfied ? tree.left : tree.right), X, i_instance, worlds, tree.i_frame, paths)

    # if survivors of next step are in the list of worlds viewed by one
    # of the just accumulated "new_worlds" then that world is a survivor
    # for this step
    new_survivors::AbstractWorldSet = Vector{AbstractWorld}()
    for curr_w in keys(worlds_map)
        if length(intersect(worlds_map[curr_w], survivors)) > 0
            push!(new_survivors, curr_w)
        end
    end

    pushfirst!(paths[i_instance], DecisionPathNode(satisfied, tree.decision.feature, tree.decision.test_operator, tree.decision.threshold, deepcopy(new_survivors)))

    return new_survivors
end
function get_path_in_tree(tree::DTree{S}, X::GenericDataset)::Vector{DecisionPath} where {S}
    n_instances = n_samples(X)
    paths::Vector{DecisionPath} = [ DecisionPath() for i in 1:n_instances ]
    for i_instance in 1:n_instances
        worlds = DecisionTree.inst_init_world_sets(X, tree, i_instance)
        _get_path_in_tree(tree.root, X, i_instance, worlds, 1, paths)
    end
    paths
end

function get_internalnode_dirname(node::DTInternal)::String
    replace(display_decision(node), " " => "_")
end

mk_tree_path(leaf::DTLeaf; path::String) = touch(path * "/" * string(leaf.prediction) * ".txt")
function mk_tree_path(node::DTInternal; path::String)
    dir_name = get_internalnode_dirname(node)
    mkpath(path * "/Y_" * dir_name)
    mkpath(path * "/N_" * dir_name)
    mk_tree_path(node.left; path = path * "/Y_" * dir_name)
    mk_tree_path(node.right; path = path * "/N_" * dir_name)
end
function mk_tree_path(tree_hash::String, tree::DTree; path::String)
    mkpath(path * "/" * tree_hash)
    mk_tree_path(tree.root; path = path * "/" * tree_hash)
end

function get_tree_path_as_dirpath(tree_hash::String, tree::DTree, decpath::DecisionPath; path::String)::String
    current = tree.root
    result = path * "/" * tree_hash
    for node in decpath
        if current isa DTLeaf break end
        result *= "/" * (node.taken ? "Y" : "N") * "_" * get_internalnode_dirname(current)
        current = node.taken ? current.left : current.right
    end
    result
end
