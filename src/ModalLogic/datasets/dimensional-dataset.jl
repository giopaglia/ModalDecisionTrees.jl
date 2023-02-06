
# DimensionalDataset bindings

get_gamma(X::DimensionalDataset, i_sample::Integer, w::AbstractWorld, f::AbstractFeature) = _get_gamma(X, i_sample, w, f)
Base.@propagate_inbounds @inline function _get_gamma(X::DimensionalDataset{T,N}, i_sample::Integer, w::AbstractWorld, f::AbstractFeature, args...) where {T,N}
    w_values = interpret_world(w, get_instance(X, i_sample))::AbstractDimensionalInstance{T,N-1}
    compute_feature(f, w_values)::T
end

Base.@propagate_inbounds @inline function _get_modal_gamma(X::DimensionalDataset{T,N}, i_sample::Integer, w::AbstractWorld, r::AbstractRelation, f::AbstractFeature, aggr::Aggregator, args...) where {T,N}
    aggr([
        aggregator_bottom(aggr, T),
        [get_gamma(X, i_sample, w2, f) for w2 in representatives(X, i_sample, w, r, f, aggr)]...
    ])
end

Base.@propagate_inbounds @inline function _get_global_gamma(X::DimensionalDataset{T,N}, i_sample::Integer, f::AbstractFeature, aggr::Aggregator) where {T,N}
    aggr([
        aggregator_bottom(aggr, T),
        [get_gamma(X, i_sample, w2, f) for w2 in representatives(X, i_sample, RelationGlob, f, aggr)]...
    ])
end


initialworldset(X::UniformDimensionalDataset, i_sample, args...) = initialworldset(FullDimensionalFrame(channel_size(X)), args...)
accessibles(X::UniformDimensionalDataset, i_sample, args...) = accessibles(FullDimensionalFrame(channel_size(X)), args...)
representatives(X::UniformDimensionalDataset, i_sample, args...) = representatives(FullDimensionalFrame(channel_size(X)), args...)
allworlds(X::UniformDimensionalDataset, i_sample, args...) = allworlds(FullDimensionalFrame(channel_size(X)), args...)
