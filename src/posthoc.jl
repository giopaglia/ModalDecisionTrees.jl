################################################################################
################################################################################
# TODO explain
################################################################################
################################################################################

export prune_tree, prune_forest

using DataStructures

function prune_tree(tree::DTree; kwargs...)
    DTree(prune_tree(tree.root; depth = 0, kwargs...), tree.worldTypes, tree.initConditions)
end

function prune_tree(leaf::AbstractDecisionLeaf; kwargs...)
    leaf
end

function prune_tree(node::DTInternal{T, L}; depth = nothing, kwargs...) where {T, L}

    @assert ! (haskey(kwargs, :max_depth) && isnothing(depth)) "Please specify the node depth: prune_tree(node; depth = ...)"

    kwargs = NamedTuple(kwargs)

    if haskey(kwargs, :loss_function) && isnothing(kwargs.loss_function)
        ks = filter((k)->k != :loss_function, collect(keys(kwargs)))
        kwargs = (; zip(ks,kwargs[k] for k in ks)...)
    end

    pruning_params = merge((
        loss_function       = default_loss_function(L)      ::Union{Nothing,Function},
        max_depth           = default_max_depth             ::Int                    ,
        # min_samples_leaf    = default_min_samples_leaf      ::Int                    ,
        min_purity_increase = default_min_purity_increase   ::AbstractFloat          ,
        max_purity_at_leaf  = default_max_purity_at_leaf    ::AbstractFloat          ,
    ), NamedTuple(kwargs))
    
    @assert all(map((x)->(isa(x, DTInternal) || isa(x, DTLeaf)), [node.this, node.left, node.right]))

    # Honor constraints on the number of instances
    nt = length(supp_labels(node.this))
    nl = length(supp_labels(node.left))
    nr = length(supp_labels(node.right))

    if (pruning_params.max_depth < depth)
        # ||
       # (pruning_params.min_samples_leaf > nr)  ||
       # (pruning_params.min_samples_leaf > nl)  ||
        return node.this
    end
    
    # Honor purity constraints
    # TODO fix missing weights!!
    purity   = ModalDecisionTrees.compute_purity(supp_labels(node.this);  loss_function = pruning_params.loss_function)
    purity_r = ModalDecisionTrees.compute_purity(supp_labels(node.left);  loss_function = pruning_params.loss_function)
    purity_l = ModalDecisionTrees.compute_purity(supp_labels(node.right); loss_function = pruning_params.loss_function)

    split_purity_times_nt = (nl * purity_l + nr * purity_r)

    # println("purity: ", purity)
    # println("purity_r: ", purity_r)
    # println("purity_l: ", purity_l)

    if (purity_r > pruning_params.max_purity_at_leaf) ||
       (purity_l > pruning_params.max_purity_at_leaf) ||
       dishonor_min_purity_increase(L, pruning_params.min_purity_increase, purity, split_purity_times_nt, nt)
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

function prune_forest(forest::DForest{L}, rng::Random.AbstractRNG = Random.GLOBAL_RNG; kwargs...) where {L}
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
    v_metrics = Dict()
    if haskey(forest.metrics, :oob_metrics)
        v_metrics[:oob_metrics] = @views forest.metrics.oob_metrics[perm]
    end
    v_metrics = NamedTuple(v_metrics)
    DForest{L}(v_trees, v_metrics)
end

# When training many trees with different pruning parametrizations, it can be beneficial to find the non-dominated set of parametrizations,
#  train a single tree per each non-dominated parametrization, and prune it afterwards x times. This hack can help save cpu time
function nondominated_pruning_parametrizations(args::AbstractVector; do_it_or_not = true, return_perm = false)
    args = convert(Vector{NamedTuple}, args)
    nondominated_pars, perm =
        if do_it_or_not
            
            # To be optimized
            to_opt = [
                # tree & forest
                :max_depth,
                # :min_samples_leaf,
                :min_purity_increase,
                :max_purity_at_leaf,
                # forest
                :n_trees,
            ]
            
            # To be matched
            to_match = [
                :min_samples_leaf,
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
            # polarity(::Val{:min_samples_leaf})    = min
            polarity(::Val{:min_purity_increase}) = min
            polarity(::Val{:max_purity_at_leaf})  = max

            bottom(::Val{:max_depth})           = typemin(Int)
            # bottom(::Val{:min_samples_leaf})    = typemax(Int)
            bottom(::Val{:min_purity_increase}) = Inf
            bottom(::Val{:max_purity_at_leaf})  = -Inf

            perm = []
            # Find non-dominated parameter set
            for this_args in args

                base_args = Base.structdiff(this_args, (;
                    max_depth           = nothing,
                    # min_samples_leaf    = nothing,
                    min_purity_increase = nothing,
                    max_purity_at_leaf  = nothing,
                ))

                dominating[base_args] = ((
                    max_depth           = polarity(Val(:max_depth          ))((haskey(this_args, :max_depth          ) ? this_args.max_depth           : bottom(Val(:max_depth          ))),(haskey(dominating, base_args) ? dominating[base_args][1].max_depth           : bottom(Val(:max_depth          )))),
                    # min_samples_leaf    = polarity(Val(:min_samples_leaf   ))((haskey(this_args, :min_samples_leaf   ) ? this_args.min_samples_leaf    : bottom(Val(:min_samples_leaf   ))),(haskey(dominating, base_args) ? dominating[base_args][1].min_samples_leaf    : bottom(Val(:min_samples_leaf   )))),
                    min_purity_increase = polarity(Val(:min_purity_increase))((haskey(this_args, :min_purity_increase) ? this_args.min_purity_increase : bottom(Val(:min_purity_increase))),(haskey(dominating, base_args) ? dominating[base_args][1].min_purity_increase : bottom(Val(:min_purity_increase)))),
                    max_purity_at_leaf  = polarity(Val(:max_purity_at_leaf ))((haskey(this_args, :max_purity_at_leaf ) ? this_args.max_purity_at_leaf  : bottom(Val(:max_purity_at_leaf ))),(haskey(dominating, base_args) ? dominating[base_args][1].max_purity_at_leaf  : bottom(Val(:max_purity_at_leaf )))),
                ),[(haskey(dominating, base_args) ? dominating[base_args][2] : [])..., this_args])

                outer_idx = findfirst((k)->k==base_args, collect(keys(dominating)))
                inner_idx = length(dominating[base_args][2])
                push!(perm, (outer_idx, inner_idx))
            end

            [
                begin
                    if (rep_args.max_depth           == bottom(Val(:max_depth))          ) rep_args = Base.structdiff(rep_args, (; max_depth           = nothing)) end
                    # if (rep_args.min_samples_leaf    == bottom(Val(:min_samples_leaf))   ) rep_args = Base.structdiff(rep_args, (; min_samples_leaf    = nothing)) end
                    if (rep_args.min_purity_increase == bottom(Val(:min_purity_increase))) rep_args = Base.structdiff(rep_args, (; min_purity_increase = nothing)) end
                    if (rep_args.max_purity_at_leaf  == bottom(Val(:max_purity_at_leaf)) ) rep_args = Base.structdiff(rep_args, (; max_purity_at_leaf  = nothing)) end
                    
                    this_args = merge(base_args, rep_args)
                    (this_args, [begin
                        ks = intersect(to_leave, keys(post_pruning_args))
                        (; zip(ks, [post_pruning_args[k] for k in ks])...)
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
        datasets::AbstractVector{Tuple{GenericDataset,AbstractVector}},
        args...;
        kwargs...,
    )
    # World sets for (dataset, frame, instance)
    worlds = Vector{Vector{Vector{<:WST} where {WorldType<:AbstractWorld,WST<:WorldSet{WorldType}}}}([
        init_world_sets(X, tree.initConditions)
    for (X,Y) in datasets])
    DTree(train_functional_leaves(tree.root, worlds, datasets, args...; kwargs...), tree.worldTypes, tree.initConditions)
end

# At internal nodes, a functional model is trained by calling a callback function, and the leaf is created
function train_functional_leaves(
        node::DTInternal{T, L},
        worlds::AbstractVector{<:AbstractVector{<:AbstractVector{<:AbstractWorldSet}}},
        datasets::AbstractVector{Tuple{GenericDataset,AbstractVector}},
        args...;
        kwargs...,
    ) where {T, L}
    
    # Each dataset is sliced, and two subsets are derived (left and right)
    datasets_l = Tuple{GenericDataset,AbstractVector}[]
    datasets_r = Tuple{GenericDataset,AbstractVector}[]
    
    worlds_l = AbstractVector{<:AbstractVector{<:AbstractWorldSet}}[]
    worlds_r = AbstractVector{<:AbstractVector{<:AbstractWorldSet}}[]

    for (i_dataset,(X,Y)) in enumerate(datasets)

        satisfied_idxs   = Integer[]
        unsatisfied_idxs = Integer[]

        for i_instance in 1:n_samples(X)
            (satisfied,new_worlds) = ModalLogic.modal_step(get_frame(X, node.i_frame), i_instance, worlds[i_dataset][node.i_frame][i_instance], node.decision)

            if satisfied
                push!(satisfied_idxs, i_instance)
            else
                push!(unsatisfied_idxs, i_instance)
            end

            worlds[i_dataset][node.i_frame][i_instance] = new_worlds
        end

        push!(datasets_l, ModalLogic.slice_dataset((X,Y), satisfied_idxs;   allow_no_instances = true))
        push!(datasets_r, ModalLogic.slice_dataset((X,Y), unsatisfied_idxs; allow_no_instances = true))

        push!(worlds_l, [frame_worlds[satisfied_idxs]   for frame_worlds in worlds[i_dataset]])
        push!(worlds_r, [frame_worlds[unsatisfied_idxs] for frame_worlds in worlds[i_dataset]])
    end
    
    DTInternal(
        node.i_frame,
        node.decision,
        # train_functional_leaves(node.this,  worlds,   datasets,   args...; kwargs...), # TODO test whether this makes sense and works correctly
        node.this,
        train_functional_leaves(node.left,  worlds_l, datasets_l, args...; kwargs...),
        train_functional_leaves(node.right, worlds_r, datasets_r, args...; kwargs...),
    )
end

# At leaves, a functional model is trained by calling a callback function, and the leaf is created
function train_functional_leaves(
        leaf::AbstractDecisionLeaf{L},
        worlds::AbstractVector{<:AbstractVector{<:AbstractVector{<:AbstractWorldSet}}},
        datasets::AbstractVector{Tuple{GenericDataset,AbstractVector}};
        train_callback::Function,
    ) where {L<:Label}
    functional_model = train_callback(datasets)
    
    @assert length(datasets) == 2 "TODO expand code: $(length(datasets))"
    (train_X, train_Y), (valid_X, valid_Y) = datasets[1], datasets[2]

    supp_train_labels = train_Y
    supp_valid_labels = valid_Y
    supp_train_predictions = functional_model(train_X) # TODO conversion here?
    supp_valid_predictions = functional_model(valid_X) # TODO conversion here?

    NSDTLeaf{L}(functional_model, supp_train_labels, supp_valid_labels, supp_train_predictions, supp_valid_predictions)
end

