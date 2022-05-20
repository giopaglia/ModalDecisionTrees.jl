
############################################################################################

Base.@propagate_inbounds @inline function get_gamma(d::DimensionalDataset{T,N}, i_sample::Integer, w::AbstractWorld, feature::ModalFeature) where {T,N}
    w_values = interpret_world(w, get_instance(d, i_sample))::AbstractDimensionalInstance{T,N-1}
    interpret_feature(feature, w_values)::T
end

init_world_sets_fun(d::DimensionalDataset,  i_sample::Integer, WorldType::Type{<:AbstractWorld}) =
    (iC)->ModalDecisionTrees.init_world_set(iC, WorldType, max_channel_size(d))

############################################################################################

# For convenience, `accessibles` & `accessibles_aggr` work with domains OR their dimensions
accessibles(S::AbstractWorldSet, r::AbstractRelation, channel::DimensionalChannel) = accessibles(S, r, size(channel)...)
accessibles_aggr(f::ModalFeature, a::Aggregator, Sw::Any, r::AbstractRelation, channel::DimensionalChannel) = accessibles_aggr(f, a, Sw, r, size(channel)...)

############################################################################################
