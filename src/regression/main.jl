include("tree.jl")
import Random
import Distributed

# Convenience functions - make a Random Number Generator object
mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(seed::T) where T <: Integer = Random.MersenneTwister(seed)

function _convert(node::treeregressor.NodeMeta{S}, labels::Array{T}) where {S, T <: Float64}
    if node.is_leaf
        return Leaf{T}(node.label, labels[node.region])
    else
        left = _convert(node.l, labels)
        right = _convert(node.r, labels)
        return Node{S, T}(node.feature, node.threshold, left, right)
    end
end

function build_stump(labels::Vector{T}, features::Matrix{S}; rng = Random.GLOBAL_RNG) where {S, T <: Float64}
    return build_tree(labels, features, 1, 0, 1)
end

function build_tree(
        labels             :: Vector{T},
        features           :: Matrix{S},
        min_samples_leaf    = 5,
        n_subfeatures       = 0,
        max_depth           = -1,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}

    rng = mk_rng(rng)::Random.AbstractRNG
    if max_depth < -1
        error("Unexpected value for max_depth: $(max_depth) "
            * "(expected: max_depth >= 0, or max_depth = -1 for infinite depth)")
    end

    if max_depth == -1
        max_depth = typemax(Int)
    end

    if n_subfeatures == 0
        n_subfeatures = size(features, 2)
    end

    t = treeregressor.fit(
        X                   = features,
        Y                   = labels,
        W                   = nothing,
        max_features        = Int(n_subfeatures),
        max_depth           = Int(max_depth),
        min_samples_leaf    = Int(min_samples_leaf),
        min_samples_split   = Int(min_samples_split),
        min_purity_increase = Float64(min_purity_increase),
        rng                 = rng)

    return _convert(t.root, labels[t.labels])
end

function build_forest(
        labels              :: Vector{T},
        features            :: Matrix{S},
        n_subfeatures       = 0,
        n_trees             = 10,
        partial_sampling    = 0.7,
        max_depth           = -1,
        min_samples_leaf    = 1,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}

    rng = mk_rng(rng)::Random.AbstractRNG
    partial_sampling = max(1.0, partial_sampling)
    rngs = Vector{Random.AbstractRNG}(undef, n_trees)
    for i in 1:n_trees
        rngs[i] = mk_rng(rand(rng, UInt))
    end

    t_samples = length(labels)
    n_samples = floor(Int, partial_sampling * t_samples)

    forest = Distributed.@distributed (vcat) for i in 1:n_trees
        inds = rand(rngs[i], 1:t_samples, n_samples)
        build_tree(
            labels[inds],
            features[inds,:],
            min_samples_leaf,
            n_subfeatures,
            max_depth,
            min_samples_split,
            min_purity_increase,
            rng = rngs[i])
    end

    return Ensemble{S, T}(forest)
end


function prune_tree(tree::LeafOrNode{S, T}, purity_thresh=0.0) where {S, T <: Float64}

    function recurse(leaf :: Leaf{T}, purity_thresh :: Float64)
        tssq = 0.0
        tsum = 0.0
        for v in leaf.values
            tssq += v*v
            tsum += v
        end

        return tssq, tsum, leaf
    end

    function recurse(node :: Node{S, T}, purity_thresh :: Float64)

        lssq, lsum, l = recurse(node.left, purity_thresh)
        rssq, rsum, r = recurse(node.right, purity_thresh)

        if is_leaf(l) && is_leaf(r)
            n_samples = length(l.values) + length(r.values)
            tsum = lsum + rsum
            tssq = lssq + rssq
            tavg = tsum / n_samples
            purity = tavg * tavg - tssq / n_samples
            if purity > purity_thresh
                return tsum, tssq, Leaf{T}(tavg, [l.values; r.values])
            end

        end

        return 0.0, 0.0, Node{S, T}(node.featid, node.featval, l, r)
    end


    _, _, node = recurse(tree, purity_thresh)
    return node
end