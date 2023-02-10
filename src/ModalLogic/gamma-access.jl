
@inline function onestep_accessible_aggregation(X::PassiveDimensionalDataset{N,W}, i_sample::Integer, w::W, r::AbstractRelation, f::AbstractFeature{V}, aggr::Aggregator, args...) where {N,V,W<:AbstractWorld}
    vs = [X[i_sample, w2, f] for w2 in representatives(X, i_sample, w, r, f, aggr)]
    return (length(vs) == 0 ? aggregator_bottom(aggr, V) : aggr(vs))
end

@inline function onestep_accessible_aggregation(X::PassiveDimensionalDataset{N,W}, i_sample::Integer, r::_RelationGlob, f::AbstractFeature{V}, aggr::Aggregator, args...) where {N,V,W<:AbstractWorld}
    vs = [X[i_sample, w2, f] for w2 in representatives(X, i_sample, r, f, aggr)]
    return (length(vs) == 0 ? aggregator_bottom(aggr, V) : aggr(vs))
end

############################################################################################

@inline function onestep_accessible_aggregation(X::DimensionalFeaturedDataset{VV,N,W}, i_sample::Integer, w::W, r::AbstractRelation, f::AbstractFeature{V}, aggr::Aggregator, args...) where {VV,N,V<:VV,W<:AbstractWorld}
    onestep_accessible_aggregation(domain(X), i_sample, w, r, f, aggr, args...)
end
@inline function onestep_accessible_aggregation(X::DimensionalFeaturedDataset{VV,N,W}, i_sample::Integer, r::_RelationGlob, f::AbstractFeature{V}, aggr::Aggregator, args...) where {VV,N,V<:VV,W<:AbstractWorld}
    onestep_accessible_aggregation(domain(X), i_sample, r, f, aggr, args...)
end

############################################################################################

@inline function onestep_accessible_aggregation(X::FeaturedDataset{VV,W}, i_sample::Integer, w::W, r::AbstractRelation, f::AbstractFeature{V}, aggr::Aggregator, args...) where {VV,V<:VV,W<:AbstractWorld}
    vs = [X[i_sample, w2, f] for w2 in representatives(X, i_sample, w, r, f, aggr)]
    return (length(vs) == 0 ? aggregator_bottom(aggr, V) : aggr(vs))
end

@inline function onestep_accessible_aggregation(X::FeaturedDataset{VV,W}, i_sample::Integer, r::_RelationGlob, f::AbstractFeature{V}, aggr::Aggregator, args...) where {VV,V<:VV,W<:AbstractWorld}
    vs = [X[i_sample, w2, f] for w2 in representatives(X, i_sample, r, f, aggr)]
    return (length(vs) == 0 ? aggregator_bottom(aggr, V) : aggr(vs))
end

############################################################################################

function onestep_accessible_aggregation(
    X::SupportedFeaturedDataset{VV,W},
    i_sample::Integer,
    w::W,
    relation::AbstractRelation,
    feature::AbstractFeature{V},
    aggregator::Aggregator,
    i_featsnaggr::Union{Nothing,Integer} = nothing,
    i_relation::Integer = find_relation_id(X, relation),
) where {VV,V<:VV,W<:AbstractWorld}
    if isnothing(i_featsnaggr)
        compute_modal_gamma(support(X), emd(X), i_sample, w, relation, feature, aggregator, i_relation)
    else
        _compute_modal_gamma(support(X), emd(X), i_sample, w, relation, feature, aggregator, i_featsnaggr, i_relation)
    end
end

@inline function onestep_accessible_aggregation(
    X::SupportedFeaturedDataset{VV,W},
    i_sample::Integer,
    r::_RelationGlob,
    f::AbstractFeature{V},
    aggr::Aggregator,
    args...
) where {VV,V<:VV,W<:AbstractWorld}
    compute_global_gamma(support(X), emd(X), i_sample, f, aggr, args...)
end

############################################################################################

function fwd_slice_compute_global_gamma(emd::FeaturedDataset, i_sample, cur_fwd_slice::FWDFeatureSlice, feature, aggr)
    # accessible_worlds = allworlds(emd, i_sample)
    accessible_worlds = representatives(emd, i_sample, RelationGlob, feature, aggr)
    threshold = apply_aggregator(cur_fwd_slice, accessible_worlds, aggr)
end

function fwd_slice_compute_modal_gamma(emd::FeaturedDataset, i_sample, cur_fwd_slice::FWDFeatureSlice, w, relation, feature, aggr)
    # accessible_worlds = accessibles(emd, i_sample, w, relation)
    accessible_worlds = representatives(emd, i_sample, w, relation, feature, aggr)
    threshold = apply_aggregator(cur_fwd_slice, accessible_worlds, aggr)
end

############################################################################################

