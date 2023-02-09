
Base.@propagate_inbounds @inline function get_modal_gamma(emd::ActiveModalDataset{T,W}, i_sample::Integer, w::W, r::AbstractRelation, f::AbstractFeature, test_operator::TestOperatorFun) where {T,W<:AbstractWorld}
    aggr = existential_aggregator(test_operator)
    _get_modal_gamma(emd, i_sample, w, r, f, aggr)
end

############################################################################################


Base.@propagate_inbounds @inline function _get_modal_gamma(X::PassiveDimensionalDataset{T,N}, i_sample::Integer, w::AbstractWorld, r::AbstractRelation, f::AbstractFeature, aggr::Aggregator, args...) where {T,N}
    aggr([
        aggregator_bottom(aggr, T),
        [X[i_sample, w2, f] for w2 in representatives(X, i_sample, w, r, f, aggr)]...
    ])
end

Base.@propagate_inbounds @inline function _get_modal_gamma(X::PassiveDimensionalDataset{T,N}, i_sample::Integer, w::AbstractWorld, r::_RelationGlob, f::AbstractFeature, aggr::Aggregator, args...) where {T,N}
    aggr([
        aggregator_bottom(aggr, T),
        [X[i_sample, w2, f] for w2 in representatives(X, i_sample, w, r, f, aggr)]...
    ])
end

Base.@propagate_inbounds @inline function _get_global_gamma(X::PassiveDimensionalDataset{T,N}, i_sample::Integer, f::AbstractFeature, aggr::Aggregator) where {T,N}
    aggr([
        aggregator_bottom(aggr, T),
        [X[i_sample, w2, f] for w2 in representatives(X, i_sample, RelationGlob, f, aggr)]...
    ])
end

############################################################################################

Base.@propagate_inbounds @inline _get_modal_gamma(imd::InterpretedModalDataset, args...) = _get_modal_gamma(domain(imd), args...)
Base.@propagate_inbounds @inline _get_global_gamma(imd::InterpretedModalDataset, args...) = _get_global_gamma(domain(imd), args...)

############################################################################################

Base.@propagate_inbounds @inline function _get_modal_gamma(X::ExplicitModalDataset{T,W}, i_sample::Integer, w::W, r::AbstractRelation, f::AbstractFeature, aggr::Aggregator, args...) where {T,W<:AbstractWorld}
    aggr([
        aggregator_bottom(aggr, T),
        [X[i_sample, w2, f] for w2 in representatives(X, i_sample, w, r, f, aggr)]...
    ])
end

# TODO simply remove?
Base.@propagate_inbounds @inline function _get_modal_gamma(X::ExplicitModalDataset{T,W}, i_sample::Integer, w::W, r::_RelationGlob, f::AbstractFeature, aggr::Aggregator, args...) where {T,W<:AbstractWorld}
    aggr([
        aggregator_bottom(aggr, T),
        [X[i_sample, w2, f] for w2 in representatives(X, i_sample, RelationGlob, f, aggr)]...
    ])
end

Base.@propagate_inbounds @inline function _get_global_gamma(X::ExplicitModalDataset{T,W}, i_sample::Integer, f::AbstractFeature, aggr::Aggregator) where {T,W<:AbstractWorld}
    aggr([
        aggregator_bottom(aggr, T),
        [X[i_sample, w2, f] for w2 in representatives(X, i_sample, RelationGlob, f, aggr)]...
    ])
end

############################################################################################

function _get_global_gamma(
    X::ExplicitModalDatasetS{T,W},
    i_sample::Integer,
    feature::AbstractFeature,
    aggregator::Aggregator
) where {T,W<:AbstractWorld}
    compute_global_gamma(support(X), emd(X), i_sample, feature, aggregator)
end

function _get_modal_gamma(
    X::ExplicitModalDatasetS{T,W},
    i_sample::Integer,
    w::W,
    relation::AbstractRelation,
    feature::AbstractFeature,
    aggregator::Aggregator,
    i_featsnaggr::Union{Nothing,Integer} = nothing,
    i_relation::Integer = find_relation_id(X, relation),
) where {T,W<:AbstractWorld}
    if relation isa _RelationGlob
        _get_global_gamma(X, i_sample, feature, aggregator)
    else
        if isnothing(i_featsnaggr)
            compute_modal_gamma(support(X), emd(X), i_sample, w, relation, feature, aggregator, i_relation)
        else
            _compute_modal_gamma(support(X), emd(X), i_sample, w, relation, feature, aggregator, i_featsnaggr, i_relation)
        end
    end
end

############################################################################################

function fwd_slice_compute_global_gamma(emd::ExplicitModalDataset, i_sample, cur_fwd_slice::FWDFeatureSlice, feature, aggr)
    # accessible_worlds = allworlds(emd, i_sample)
    accessible_worlds = representatives(emd, i_sample, RelationGlob, feature, aggr)
    threshold = apply_aggregator(cur_fwd_slice, accessible_worlds, aggr)
end

function fwd_slice_compute_modal_gamma(emd::ExplicitModalDataset, i_sample, cur_fwd_slice::FWDFeatureSlice, w, relation, feature, aggr)
    # accessible_worlds = accessibles(emd, i_sample, w, relation)
    accessible_worlds = representatives(emd, i_sample, w, relation, feature, aggr)
    threshold = apply_aggregator(cur_fwd_slice, accessible_worlds, aggr)
end

############################################################################################

