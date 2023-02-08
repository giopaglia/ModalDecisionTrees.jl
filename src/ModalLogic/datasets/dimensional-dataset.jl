
# DimensionalDataset bindings

Base.@propagate_inbounds @inline function get_gamma(
    X::DimensionalDataset{T,N},
    i_sample::Integer,
    w::AbstractWorld,
    f::AbstractFeature,
    args...
) where {T,N}
    w_values = interpret_world(w, get_instance(X, i_sample))::AbstractDimensionalInstance{T,N-1}
    compute_feature(f, w_values)::T
end

initialworldset(X::UniformDimensionalDataset, i_sample, args...) = initialworldset(FullDimensionalFrame(channel_size(X)), args...)
accessibles(X::UniformDimensionalDataset, i_sample, args...) = accessibles(FullDimensionalFrame(channel_size(X)), args...)
representatives(X::UniformDimensionalDataset, i_sample, args...) = representatives(FullDimensionalFrame(channel_size(X)), args...)
allworlds(X::UniformDimensionalDataset, i_sample, args...) = allworlds(FullDimensionalFrame(channel_size(X)), args...)
