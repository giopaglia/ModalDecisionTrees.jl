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
        loss_function       = default_loss_function(L)      ::Union{Nothing,Function},
        max_depth           = default_max_depth             ::Int                    ,
        min_samples_leaf    = default_min_samples_leaf      ::Int                    ,
        min_purity_increase = default_min_purity_increase   ::AbstractFloat          ,
        max_purity_at_leaf  = default_max_purity_at_leaf    ::AbstractFloat          ,
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

    DTInternal(
        node.i_frame,
        node.decision,
        node.this,
        prune_tree(node.left;  pruning_params..., depth = depth+1),
        prune_tree(node.right; pruning_params..., depth = depth+1)
    )
end

function prune_forest(forest::DForest{L}, kwargs...) where {L}
    pruning_params = merge((
        n_trees             = default_n_trees                      ::Integer                ,
    ), NamedTuple(kwargs))

    # Remove trees
    if pruning_params.n_trees != default_n_trees
        perm = Random.randperm(rng, length(forest.trees))[1:pruning_params.n_trees]
        forest = slice_forest(forest, perm)
    end
    pruning_params = Base.structdiff(pruning_params, (;
        n_trees           = nothing,
    ))

    # Prune trees
    if parametrization_is_going_to_prune(pruning_params)
        v_trees = map((t)->prune_tree(t, pruning_params), v_trees)
        # Note: metrics are lost
        forest = DForest{L}(v_trees)
    end

    forest
end

function slice_forest(forest::DForest{L}, perm::AbstractVector{<:Integer}) where {L}
    # Note: can't compute oob_error
    v_trees = @views forest.trees[perm]
    v_metrics = (;)
    if haskey(forest.metrics, :oob_metrics)
        v_metrics.oob_metrics = @views forest.metrics.oob_metrics[perm]
    end
    DForest{L}(v_trees, v_metrics)
end

# When training many trees with different pruning parametrizations, it can be beneficial to find the non-dominated set of parametrizations,
#  train a single tree per each non-dominated parametrization, and prune it afterwards x times. This hack can help save cpu time
function nondominated_pruning_parametrizations(args::AbstractVector{<:NamedTuple}; do_it_or_not = true, return_perm = false)
    nondominated_pars, perm =
        if do_it_or_not
            
            # To be optimized
            to_opt = [
                # tree & forest
                :max_depth,
                :min_samples_leaf,
                :min_purity_increase,
                :max_purity_at_leaf,
                # forest
                :n_trees,
            ]
            
            # To be matched
            to_match = [
                :loss_function,
                :n_subrelations,
                :n_subfeatures,
                :initConditions,
                :allowRelationGlob,
                :rng,
                :partial_sampling,
                :perform_consistency_check,
            ]

            # To be left so that they are used for pruning
            to_leave = [
                to_opt...,
                :loss_function,
            ]

            dominating = OrderedDict()
            @assert all((a)->length(setdiff(collect(keys(a)), [to_opt..., to_match..., to_leave...])) == 0, args) "Got unexpected model parameters in: $(args)"

            # Note: this optimizatio assumes that parameters are defaulted to their bottom value
            polarity(::Val{:max_depth})           = max
            polarity(::Val{:min_samples_leaf})    = min
            polarity(::Val{:min_purity_increase}) = min
            polarity(::Val{:max_purity_at_leaf})  = max

            bottom(::Val{:max_depth})           = typemin(Int)
            bottom(::Val{:min_samples_leaf})    = typemax(Int)
            bottom(::Val{:min_purity_increase}) = Inf
            bottom(::Val{:max_purity_at_leaf})  = -Inf

            perm = []
            # Find non-dominated parameter set
            for this_args in args

                base_args = Base.structdiff(this_args, (;
                    max_depth           = nothing,
                    min_samples_leaf    = nothing,
                    min_purity_increase = nothing,
                    max_purity_at_leaf  = nothing,
                ))

                dominating[base_args] = ((
                    max_depth           = polarity(Val(:max_depth          ))((haskey(this_args, :max_depth          ) ? this_args.max_depth           : bottom(Val(:max_depth          ))),(haskey(dominating, base_args) ? dominating[base_args][1].max_depth           : bottom(Val(:max_depth          )))),
                    min_samples_leaf    = polarity(Val(:min_samples_leaf   ))((haskey(this_args, :min_samples_leaf   ) ? this_args.min_samples_leaf    : bottom(Val(:min_samples_leaf   ))),(haskey(dominating, base_args) ? dominating[base_args][1].min_samples_leaf    : bottom(Val(:min_samples_leaf   )))),
                    min_purity_increase = polarity(Val(:min_purity_increase))((haskey(this_args, :min_purity_increase) ? this_args.min_purity_increase : bottom(Val(:min_purity_increase))),(haskey(dominating, base_args) ? dominating[base_args][1].min_purity_increase : bottom(Val(:min_purity_increase)))),
                    max_purity_at_leaf  = polarity(Val(:max_purity_at_leaf ))((haskey(this_args, :max_purity_at_leaf ) ? this_args.max_purity_at_leaf  : bottom(Val(:max_purity_at_leaf ))),(haskey(dominating, base_args) ? dominating[base_args][1].max_purity_at_leaf  : bottom(Val(:max_purity_at_leaf )))),
                ),[(haskey(dominating, base_args) ? dominating[base_args][2] : [])..., this_args])

                outer_idx = findfirst((x)->x==base_args, (keys(dominating)))|>collect
                inner_idx = length(dominating[base_args])
                push!(perm, (outer_idx, inner_idx))
            end

            [
                begin
                    if (rep_args.max_depth           == bottom(Val(:max_depth))          ) rep_args = Base.structdiff(rep_args, (; max_depth           = nothing)) end
                    if (rep_args.min_samples_leaf    == bottom(Val(:min_samples_leaf))   ) rep_args = Base.structdiff(rep_args, (; min_samples_leaf    = nothing)) end
                    if (rep_args.min_purity_increase == bottom(Val(:min_purity_increase))) rep_args = Base.structdiff(rep_args, (; min_purity_increase = nothing)) end
                    if (rep_args.max_purity_at_leaf  == bottom(Val(:max_purity_at_leaf)) ) rep_args = Base.structdiff(rep_args, (; max_purity_at_leaf  = nothing)) end
                    
                    this_args = merge(base_args, rep_args)
                    (this_args, [begin
                        ks = intersect(to_leave, keys(post_pruning_args))
                        (; zip(ks, [x[k] for k in ks])...)
                    end for post_pruning_args in post_pruning_argss])
                end for (i_model, (base_args,(rep_args, post_pruning_argss))) in enumerate(dominating)
            ], perm
        else
            zip(args, Iterators.repeated([(;)])) |> collect, zip(1:length(args), Iterators.repeated(1)) |> collect
        end

    if return_perm
        nondominated_pars, perm
    else
        nondominated_pars
    end
end

# 

function train_functional_leaves(
        tree::DTree,
        args...;
        kwargs...,
    )
    error("TODO expand code
    worlds = inst_init_world_sets(TODO reverted, X, tree.initConditions)
    DTree(train_functional_leaves(tree.root, worlds, args...; kwargs...), tree.worldTypes, tree.initConditions)
    ")
end

function train_functional_leaves(
        node::DTInternal{T, L},
        worlds::AbstractVector{<:AbstractVector{<:AbstractWorldSet}},
        datasets::AbstractVector{Tuple{AbstractVector{<:GenericDataset},AbstractVector}},
        args...;
        kwargs...,
    ) where {T, L}
    
    satisfied_idxs   = []
    unsatisfied_idxs = []

    datasets_l = Tuple{AbstractVector{<:GenericDataset},AbstractVector}[]
    datasets_r = Tuple{AbstractVector{<:GenericDataset},AbstractVector}[]

    error("TODO expand each instance")
    for i_instance in 1:10
        (satisfied,new_worlds) = ModalLogic.modal_step(get_frame(X, tree.i_frame), i_instance, worlds[tree.i_frame][i_instance], tree.decision)

        if satisfied
            push!(satisfied_idxs, i_instance)
        else
            push!(unsatisfied_idxs, i_instance)
        end

        worlds[tree.i_frame][i_instance] = new_worlds
    end

    error("TODO expand code
    ...TODO satisfied_idxs
    push!(datasets_l, )

    ...TODO unsatisfied_idxs
    push!(datasets_r, )
    ")

    DTInternal(
        tree.i_frame,
        tree.decision,
        train_functional_leaves(tree.this,  worlds, datasets,   args...; kwargs...), # TODO test whether this works correctly
        train_functional_leaves(tree.left,  worlds, datasets_l, args...; kwargs...),
        train_functional_leaves(tree.right, worlds, datasets_r, args...; kwargs...),
    )
end

function train_functional_leaves(
        leaf::DTLeaf,
        worlds::AbstractVector{<:AbstractVector{<:AbstractWorldSet}},
        datasets::AbstractVector{Tuple{AbstractVector{<:GenericDataset},AbstractVector}},
        rng::Random.AbstractRNG;
        callback_fun::Function,
    )
    error("TODO expand code callback_fun(datasets)")
    leaf
end

