
############################################################################################

hasnans(d::DimensionalDataset) = any(_isnan.(d))

Base.@propagate_inbounds @inline function get_gamma(d::DimensionalDataset{T,N}, i_sample::Integer, w::AbstractWorld, f::AbstractFeature) where {T,N}
    w_values = interpret_world(w, get_instance(d, i_sample))::AbstractDimensionalInstance{T,N-1}
    compute_feature(f, w_values)::T
end

Base.@propagate_inbounds @inline function get_modal_gamma(d::DimensionalDataset{T,N}, i_sample::Integer, w::AbstractWorld, r::AbstractRelation, f::AbstractFeature, test_operator::TestOperatorFun) where {T,N}
    aggr = existential_aggregator(test_operator)
    _get_modal_gamma(d, i_sample, w, r, f, aggr)
end

Base.@propagate_inbounds @inline function get_global_gamma(d::DimensionalDataset{T,N}, i_sample::Integer, f::AbstractFeature, test_operator::TestOperatorFun) where {T,N}
    aggr = existential_aggregator(test_operator)
    _get_global_gamma(d, i_sample, f, aggr)
end

Base.@propagate_inbounds @inline function _get_modal_gamma(d::DimensionalDataset{T,N}, i_sample::Integer, w::AbstractWorld, r::AbstractRelation, f::AbstractFeature, aggr::Aggregator) where {T,N}
    aggr([
        aggregator_bottom(aggr, T),
        [get_gamma(d, i_sample, w2, f) for w2 in representatives(FullDimensionalFrame(max_channel_size(d)), w, r, f, aggr)]...
    ])
end

Base.@propagate_inbounds @inline function _get_global_gamma(d::DimensionalDataset{T,N}, i_sample::Integer, f::AbstractFeature, aggr::Aggregator) where {T,N}
    aggr([
        aggregator_bottom(aggr, T),
        [get_gamma(d, i_sample, w2, f) for w2 in representatives(FullDimensionalFrame(max_channel_size(d)), RelationGlob, f, aggr)]...
    ])
end


initialworldset(d::DimensionalDataset,  i_sample::Integer, iC) =
    ModalDecisionTrees.initialworldset(FullDimensionalFrame(max_channel_size(d)), iC)

############################################################################################
