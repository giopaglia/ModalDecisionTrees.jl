
############################################################################################

hasnans(d::DimensionalDataset) = any(_isnan.(d))

Base.@propagate_inbounds @inline function get_gamma(d::DimensionalDataset{T,N}, i_sample::Integer, w::AbstractWorld, feature::AbstractFeature) where {T,N}
    w_values = interpret_world(w, get_instance(d, i_sample))::AbstractDimensionalInstance{T,N-1}
    compute_feature(feature, w_values)::T
end

init_world_sets_fun(d::DimensionalDataset,  i_sample::Integer, W::Type{<:AbstractWorld}) =
    (iC)->ModalDecisionTrees.init_world_set(iC, FullDimensionalFrame(max_channel_size(d)))

############################################################################################
