using SoleData
using SoleData: AbstractDimensionalDataset,
                AbstractDimensionalInstance,
                AbstractDimensionalChannel,
                UniformDimensionalDataset,
                DimensionalInstance,
                DimensionalChannel


# A modal dataset can be *active* or *passive*.
# 
# A passive modal dataset is one that you can interpret decisions on, but cannot necessarily
#  enumerate decisions for, as it doesn't have objects for storing the logic (relations, features, etc.).
# Dimensional datasets are passive.

# function get_interval_worldtype(N::Integer)
#     if N == 0
#         OneWorld
#     elseif N == 0
#         Interval
#     elseif N == 0
#         Interval2D
#     else
#         error("TODO")
#     end
# end

struct PassiveDimensionalDataset{T,N,W<:AbstractWorld,D,DOM<:AbstractDimensionalDataset{T,D},FR<:AbstractDimensionalFrame{N,W,TruthValue}} <: AbstractConditionalDataset{W,AbstractCondition,TruthValue,FR}
    
    d::DOM

    function PassiveDimensionalDataset{T,N,W,D,DOM,FR}(
        d::DOM,
    ) where {T,N,W<:AbstractWorld,D,DOM<:AbstractDimensionalDataset{T,D},FR<:AbstractDimensionalFrame{N,W,TruthValue}}
        @assert D == (N+1+1) "ERROR! Dimensionality mismatch: can't instantiate PassiveDimensionalDataset{$(T),$(N),$(W),$(DOM)} with underlying structure $(DOM). $(D) == ($(N)+1+1) should hold."
        @assert SoleLogics.goeswith_dim(W, N) "ERROR! Dimensionality mismatch: can't interpret worldtype $(W) on PassiveDimensionalDataset of dimensionality = $(N)"
        new{T,N,W,D,DOM,FR}(d)
    end
    
    function PassiveDimensionalDataset{T,N,W,D,DOM}(
        d::DOM,
    ) where {T,N,W<:AbstractWorld,D,DOM<:AbstractDimensionalDataset{T,D}}
        PassiveDimensionalDataset{T,N,W,D,DOM,AbstractDimensionalFrame{N,W,TruthValue}}(d)
    end

    function PassiveDimensionalDataset{T,N,W}(
        d::DOM,
    ) where {T,N,W<:AbstractWorld,D,DOM<:AbstractDimensionalDataset{T,D}}
        PassiveDimensionalDataset{T,N,W,D,DOM}(d)
    end

    function PassiveDimensionalDataset(
        d::DOM,
        worldtype::Type{<:AbstractWorld},
        # worldtype = get_interval_worldtype(D-1-1) TODO default?
    ) where {T,D,DOM<:AbstractDimensionalDataset{T,D}}
        PassiveDimensionalDataset{T,N,worldtype}(d)
    end
end

Base.@propagate_inbounds @inline function Base.getindex(
    X::PassiveDimensionalDataset{T,N,W},
    i_sample::Integer,
    w::W,
    f::AbstractFeature,
    args...,
) where {T,N,W<:AbstractWorld}
    w_values = interpret_world(w, get_instance(X.d, i_sample))::AbstractDimensionalInstance{T,N+1}
    compute_feature(f, w_values)::T
end

Base.size(X::PassiveDimensionalDataset)                 = Base.size(X.d)

nattributes(X::PassiveDimensionalDataset)               = SoleData.nattributes(X.d)
nsamples(X::PassiveDimensionalDataset)                  = SoleData.nsamples(X.d)
channel_size(X::PassiveDimensionalDataset)              = SoleData.channel_size(X.d)
max_channel_size(X::PassiveDimensionalDataset)          = SoleData.max_channel_size(X.d)
dimensionality(X::PassiveDimensionalDataset)            = SoleData.dimensionality(X.d)

get_instance(X::PassiveDimensionalDataset, args...)     = get_instance(X.d, args...)

_slice_dataset(X::PassiveDimensionalDataset{T,N,W}, inds::AbstractVector{<:Integer}, args...; kwargs...) where {T,N,W} =
    PassiveDimensionalDataset{T,N,W}(_slice_dataset(X.d, inds, args...; kwargs...))

hasnans(X::PassiveDimensionalDataset) = hasnans(X.d)

worldtype(X::PassiveDimensionalDataset{T,N,W}) where {T,N,W} = W

initialworldset(X::PassiveDimensionalDataset, args...) = _initialworldset(X.d, args...)
accessibles(X::PassiveDimensionalDataset, args...) = _accessibles(X.d, args...)
representatives(X::PassiveDimensionalDataset, args...) = _representatives(X.d, args...)
allworlds(X::PassiveDimensionalDataset, args...) = _allworlds(X.d, args...)

############################################################################################

_initialworldset(X::UniformDimensionalDataset, i_sample, args...) = initialworldset(FullDimensionalFrame(channel_size(X)), args...)
_accessibles(X::UniformDimensionalDataset, i_sample, args...) = accessibles(FullDimensionalFrame(channel_size(X)), args...)
_representatives(X::UniformDimensionalDataset, i_sample, args...) = representatives(FullDimensionalFrame(channel_size(X)), args...)
_allworlds(X::UniformDimensionalDataset, i_sample, args...) = allworlds(FullDimensionalFrame(channel_size(X)), args...)
