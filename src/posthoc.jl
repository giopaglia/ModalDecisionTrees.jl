################################################################################
################################################################################
# TODO explain
################################################################################
################################################################################

export prune_tree, prune_forest

using DataStructures

function prune_tree(tree::DTree; kwargs...)
    DTree(prune_tree(root(tree); depth = 0, kwargs...), world_types(tree), init_conditions(tree))
end

function prune_tree(leaf::AbstractDecisionLeaf; kwargs...)
    leaf
end

function prune_tree(node::DTInternal{L}; depth = nothing, kwargs...) where {L}

    @assert ! (haskey(kwargs, :max_depth) && isnothing(depth)) "Please specify the node depth: prune_tree(node; depth = ...)"

    kwargs = NamedTuple(kwargs)

    if haskey(kwargs, :loss_function) && isnothing(kwargs.loss_function)
        ks = filter((k)->k != :loss_function, collect(keys(kwargs)))
        kwargs = (; zip(ks,kwargs[k] for k in ks)...)
    end

    pruning_params = merge((
        loss_function       = default_loss_function(L)      ::Union{Nothing,Function},
        max_depth           = default_max_depth             ::Int                    ,
        min_samples_leaf    = default_min_samples_leaf      ::Int                    ,
        min_purity_increase = default_min_purity_increase   ::AbstractFloat          ,
        max_purity_at_leaf  = default_max_purity_at_leaf    ::AbstractFloat          ,
    ), NamedTuple(kwargs))

    @assert all(map((x)->(isa(x, DTInternal) || isa(x, DTLeaf)), [this(node), left(node), right(node)]))

    # Honor constraints on the number of instances
    nt = length(supp_labels(this(node)))
    nl = length(supp_labels(left(node)))
    nr = length(supp_labels(right(node)))

    if (pruning_params.max_depth < depth) ||
       (pruning_params.min_samples_leaf > nr) ||
       (pruning_params.min_samples_leaf > nl)
        return this(node)
    end

    # Honor purity constraints
    # TODO fix missing weights!!
    purity   = ModalDecisionTrees.compute_purity(supp_labels(this(node));  loss_function = pruning_params.loss_function)
    purity_r = ModalDecisionTrees.compute_purity(supp_labels(left(node));  loss_function = pruning_params.loss_function)
    purity_l = ModalDecisionTrees.compute_purity(supp_labels(right(node)); loss_function = pruning_params.loss_function)

    split_purity_times_nt = (nl * purity_l + nr * purity_r)

    # println("purity: ", purity)
    # println("purity_r: ", purity_r)
    # println("purity_l: ", purity_l)

    if (purity_r > pruning_params.max_purity_at_leaf) ||
       (purity_l > pruning_params.max_purity_at_leaf) ||
       dishonor_min_purity_increase(L, pruning_params.min_purity_increase, purity, split_purity_times_nt, nt)
        return this(node)
    end

    DTInternal(
        i_frame(node),
        decision(node),
        this(node),
        prune_tree(left(node);  pruning_params..., depth = depth+1),
        prune_tree(right(node); pruning_params..., depth = depth+1)
    )
end

function prune_forest(forest::DForest{L}, rng::Random.AbstractRNG = Random.GLOBAL_RNG; kwargs...) where {L}
    pruning_params = merge((
        n_trees             = default_n_trees                      ::Integer                ,
    ), NamedTuple(kwargs))

    # Remove trees
    if pruning_params.n_trees != default_n_trees
        perm = Random.randperm(rng, length(trees(forest)))[1:pruning_params.n_trees]
        forest = slice_forest(forest, perm)
    end
    pruning_params = Base.structdiff(pruning_params, (;
        n_trees           = nothing,
    ))

    # Prune trees
    # if parametrization_is_going_to_prune(pruning_params)
    v_trees = map((t)->prune_tree(t; pruning_params...), trees(forest))
    # Note: metrics are lost
    forest = DForest{L}(v_trees)
    # end

    forest
end

function slice_forest(forest::DForest{L}, perm::AbstractVector{<:Integer}) where {L}
    # Note: can't compute oob_error
    v_trees = @views trees(forest)[perm]
    v_metrics = Dict()
    if haskey(metrics(forest), :oob_metrics)
        v_metrics[:oob_metrics] = @views metrics(forest).oob_metrics[perm]
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
                :init_conditions,
                :allow_global_splits,
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
        datasets::AbstractVector{Tuple{GenericModalDataset,AbstractVector}},
        args...;
        kwargs...,
    )
    # World sets for (dataset, frame, instance)
    worlds = Vector{Vector{Vector{<:WST} where {WorldType<:AbstractWorld,WST<:WorldSet{WorldType}}}}([
        ModalDecisionTrees.initialworldsets(X, init_conditions(tree))
    for (X,Y) in datasets])
    DTree(train_functional_leaves(root(tree), worlds, datasets, args...; kwargs...), world_types(tree), init_conditions(tree))
end

# At internal nodes, a functional model is trained by calling a callback function, and the leaf is created
function train_functional_leaves(
        node::DTInternal{L},
        worlds::AbstractVector{<:AbstractVector{<:AbstractVector{<:AbstractWorldSet}}},
        datasets::AbstractVector{Tuple{GenericModalDataset,AbstractVector}},
        args...;
        kwargs...,
    ) where {L}

    # Each dataset is sliced, and two subsets are derived (left and right)
    datasets_l = Tuple{GenericModalDataset,AbstractVector}[]
    datasets_r = Tuple{GenericModalDataset,AbstractVector}[]

    worlds_l = AbstractVector{<:AbstractVector{<:AbstractWorldSet}}[]
    worlds_r = AbstractVector{<:AbstractVector{<:AbstractWorldSet}}[]

    for (i_dataset,(X,Y)) in enumerate(datasets)

        satisfied_idxs   = Integer[]
        unsatisfied_idxs = Integer[]

        for i_sample in 1:nsamples(X)
            (satisfied,new_worlds) = modalstep(get_frame(X, i_frame(node)), i_sample, worlds[i_dataset][i_frame(node)][i_sample], decision(node))

            if satisfied
                push!(satisfied_idxs, i_sample)
            else
                push!(unsatisfied_idxs, i_sample)
            end

            worlds[i_dataset][i_frame(node)][i_sample] = new_worlds
        end

        push!(datasets_l, slice_dataset((X,Y), satisfied_idxs;   allow_no_instances = true))
        push!(datasets_r, slice_dataset((X,Y), unsatisfied_idxs; allow_no_instances = true))

        push!(worlds_l, [frame_worlds[satisfied_idxs]   for frame_worlds in worlds[i_dataset]])
        push!(worlds_r, [frame_worlds[unsatisfied_idxs] for frame_worlds in worlds[i_dataset]])
    end

    DTInternal(
        i_frame(node),
        decision(node),
        # train_functional_leaves(node.this,  worlds,   datasets,   args...; kwargs...), # TODO test whether this makes sense and works correctly
        this(node),
        train_functional_leaves(left(node),  worlds_l, datasets_l, args...; kwargs...),
        train_functional_leaves(right(node), worlds_r, datasets_r, args...; kwargs...),
    )
end

# At leaves, a functional model is trained by calling a callback function, and the leaf is created
function train_functional_leaves(
        leaf::AbstractDecisionLeaf{L},
        worlds::AbstractVector{<:AbstractVector{<:AbstractVector{<:AbstractWorldSet}}},
        datasets::AbstractVector{Tuple{GenericModalDataset,AbstractVector}};
        train_callback::Function,
    ) where {L<:Label}
    functional_model = train_callback(datasets)

    @assert length(datasets) == 2 "TODO expand code: $(length(datasets))"
    (train_X, train_Y), (valid_X, valid_Y) = datasets[1], datasets[2]

    # println("train_functional_leaves")
    # println(typeof(train_X))
    # println(hasmethod(size,   (typeof(train_X),)) ? size(train_X)   : nothing)
    # println(hasmethod(length, (typeof(train_X),)) ? length(train_X) : nothing)
    # println(nsamples(train_X))

    # println(typeof(valid_X))
    # println(hasmethod(size,   (typeof(valid_X),)) ? size(valid_X)   : nothing)
    # println(hasmethod(length, (typeof(valid_X),)) ? length(valid_X) : nothing)

    # println(nsamples(valid_X))

    supp_train_labels = train_Y
    supp_valid_labels = valid_Y
    supp_train_predictions = functional_model(train_X)
    supp_valid_predictions = functional_model(valid_X)

    function predicting_function(X)::Vector{L} # TODO avoid this definition, just return the model
        functional_model(X)
    end
    NSDTLeaf{L}(predicting_function, supp_train_labels, supp_valid_labels, supp_train_predictions, supp_valid_predictions)
end


############################################################################################
############################################################################################
############################################################################################

function _variable_countmap(leaf::AbstractDecisionLeaf{L}; weighted = false) where {L<:Label}
    []
end

function _variable_countmap(node::DTInternal{L}; weighted = false) where {L<:Label}
    th = begin
        d = decision(node)
        f = feature(d)
        (f isa SingleAttributeFeature) ? [((i_frame(node), f.i_attribute), (weighted ? length(supp_labels) : 1)),] : []
    end
    [th..., _variable_countmap(left(node); weighted = weighted)..., _variable_countmap(right(node); weighted = weighted)...]
end

function variable_countmap(tree::DTree{L}; weighted = false) where {L<:Label}
    vals = _variable_countmap(root(tree); weighted = weighted)
    if !weighted
        countmap(first.(vals))
    else
        c = Dict([attr => 0 for attr in unique(first.(vals))])
        for (attr, weight) in vals
            c[attr] += weight
        end
        Dict([attr => count/sum(values(c)) for (attr, count) in c])
    end
end

function variable_countmap(forest::DForest{L}; weighted = false) where {L<:Label}
    vals = [_variable_countmap(root(t); weighted = weighted) for t in trees(forest)] |> Iterators.flatten
    if !weighted
        countmap(first.(vals))
    else
        c = Dict([attr => 0 for attr in unique(first.(vals))])
        for (attr, weight) in vals
            c[attr] += weight
        end
        Dict([attr => count/sum(values(c)) for (attr, count) in c])
    end
end


############################################################################################
############################################################################################
############################################################################################

function merge_into_leaf(nodes::AbstractVector{<:DTNode})
    merge_into_leaf(map((n)->(n isa AbstractDecisionLeaf ? n : this(n)), nodes))
end

function merge_into_leaf(leaves::AbstractVector{<:DTLeaf{L}}) where {L}
    dtleaf_types = typeof.(leaves)
    @assert length(unique(dtleaf_types)) == 1 "Can't aggregate different leaf types: $(dtleaf_types)"
    dtleaf_type = dtleaf_types[1]
    dtleaf_type(L.(collect(Iterators.flatten(map((leaf)->supp_labels(leaf), leaves)))))
end

function merge_into_leaf(leaves::AbstractVector{<:NSDTLeaf{L}}) where {L}
    # dtleaf_types = typeof.(leaves)
    # @assert length(unique(dtleaf_types)) == 1 "Can't aggregate different leaf types: $(dtleaf_types)"
    # dtleaf_type = dtleaf_types[1]
    dtleaf_type = DTLeaf{L}
    supp_train_labels      = L.(collect(Iterators.flatten(map((leaf)->leaf.supp_train_labels, leaves))))
    supp_valid_labels      = L.(collect(Iterators.flatten(map((leaf)->leaf.supp_valid_labels, leaves))))
    supp_train_predictions = L.(collect(Iterators.flatten(map((leaf)->leaf.supp_train_predictions, leaves))))
    supp_valid_predictions = L.(collect(Iterators.flatten(map((leaf)->leaf.supp_valid_predictions, leaves))))
    supp_labels = [supp_train_labels..., supp_valid_labels..., supp_train_predictions..., supp_valid_predictions...]
    predicting_function = (args...; kwargs...)->(best_guess(supp_labels))
    dtleaf_type(
        predicting_function,
        supp_train_labels,
        supp_valid_labels,
        supp_train_predictions,
        supp_valid_predictions,
    )
end
