############################################################################################
############################################################################################
# world-specific FWD implementations
############################################################################################
############################################################################################

abstract type UniformFullDimensionalFWD{T,N,W<:AbstractWorld} <: AbstractFWD{T,W,FR where FR<:FullDimensionalFrame{N,W,Bool}} end

channel_size(fwd::UniformFullDimensionalFWD) = error("TODO add message inviting to add channel_size")
initialworldset(fwd::UniformFullDimensionalFWD, i_sample, args...) = initialworldset(FullDimensionalFrame(channel_size(fwd)), args...)
accessibles(fwd::UniformFullDimensionalFWD, i_sample, args...) = accessibles(FullDimensionalFrame(channel_size(fwd)), args...)
representatives(fwd::UniformFullDimensionalFWD, i_sample, args...) = representatives(FullDimensionalFrame(channel_size(fwd)), args...)
allworlds(fwd::UniformFullDimensionalFWD{T,W}, i_sample::Integer, args...) where {T,W} = allworlds(FullDimensionalFrame(channel_size(fwd)), args...)

############################################################################################
# FWD, OneWorld: 2D array (nsamples × nfeatures)
############################################################################################

struct OneWorldFWD{T} <: UniformFullDimensionalFWD{T,0,OneWorld}
    d :: Array{T,2}
end

channel_size(fwd::OneWorldFWD) = ()
goeswith(::Type{OneWorldFWD}, ::Type{OneWorld}) = true
default_fwd_type(::Type{OneWorld}) = OneWorldFWD

nsamples(fwd::OneWorldFWD)  = size(fwd.d, 1)
nfeatures(fwd::OneWorldFWD) = size(fwd.d, 2)
Base.size(fwd::OneWorldFWD, args...) = size(fwd.d, args...)

function fwd_init(::Type{OneWorldFWD}, X::InterpretedModalDataset{T,0,OneWorld}) where {T}
    OneWorldFWD{T}(Array{T,2}(undef, nsamples(X), nfeatures(X)))
end

function fwd_init_world_slice(fwd::OneWorldFWD, i_sample::Integer, w::AbstractWorld)
    nothing
end

hasnans(fwd::OneWorldFWD) = any(_isnan.(fwd.d))

Base.@propagate_inbounds @inline fwd_get(
    fwd         :: OneWorldFWD{T},
    i_sample    :: Integer,
    w           :: OneWorld,
    i_feature   :: Integer) where {T} = fwd.d[i_sample, i_feature]

Base.@propagate_inbounds @inline function fwd_set(fwd::OneWorldFWD{T}, w::OneWorld, i_sample::Integer, i_feature::Integer, threshold::T) where {T}
    fwd.d[i_sample, i_feature] = threshold
end

Base.@propagate_inbounds @inline function fwd_set_feature_slice(fwd::OneWorldFWD{T}, i_feature::Integer, feature_fwd::Array{T,1}) where {T}
    fwd.d[:, i_feature] = feature_fwd
end

function slice_dataset(fwd::OneWorldFWD{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    OneWorldFWD{T}(if return_view @view fwd.d[inds,:] else fwd.d[inds,:] end)
end

Base.@propagate_inbounds @inline fwd_get_channel(fwd::OneWorldFWD{T}, i_sample::Integer, i_feature::Integer) where {T} =
    fwd.d[i_sample, i_feature]
const OneWorldFeaturedChannel{T} = T
fwd_channel_interpret_world(fwc::T #=Note: should be OneWorldFeaturedChannel{T}, but it throws error =#, w::OneWorld) where {T} = fwc

############################################################################################
# FWD, Interval: 4D array (x × y × nsamples × nfeatures)
############################################################################################

struct IntervalFWD{T} <: UniformFullDimensionalFWD{T,1,Interval}
    d :: Array{T,4}
end

channel_size(fwd::IntervalFWD) = (size(fwd.d, 1),)
goeswith(::Type{IntervalFWD}, ::Type{Interval}) = true
default_fwd_type(::Type{Interval}) = IntervalFWD

nsamples(fwd::IntervalFWD)  = size(fwd.d, 3)
nfeatures(fwd::IntervalFWD) = size(fwd.d, 4)
Base.size(fwd::IntervalFWD, args...) = size(fwd.d, args...)

function fwd_init(::Type{IntervalFWD}, X::InterpretedModalDataset{T,1,Interval}) where {T}
    IntervalFWD{T}(Array{T,4}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, nsamples(X), nfeatures(X)))
end

function fwd_init_world_slice(fwd::IntervalFWD, i_sample::Integer, w::AbstractWorld)
    nothing
end

hasnans(fwd::IntervalFWD) = begin
    # @show ([hasnans(fwd.d[x,y,:,:]) for x in 1:size(fwd.d, 1) for y in (x+1):size(fwd.d, 2)])
    any([hasnans(fwd.d[x,y,:,:]) for x in 1:size(fwd.d, 1) for y in (x+1):size(fwd.d, 2)])
end

Base.@propagate_inbounds @inline fwd_get(
    fwd         :: IntervalFWD{T},
    i_sample    :: Integer,
    w           :: Interval,
    i_feature   :: Integer) where {T} = fwd.d[w.x, w.y, i_sample, i_feature]

Base.@propagate_inbounds @inline function fwd_set(fwd::IntervalFWD{T}, w::Interval, i_sample::Integer, i_feature::Integer, threshold::T) where {T}
    fwd.d[w.x, w.y, i_sample, i_feature] = threshold
end

Base.@propagate_inbounds @inline function fwd_set_feature_slice(fwd::IntervalFWD{T}, i_feature::Integer, feature_fwd::Array{T,3}) where {T}
    fwd.d[:, :, :, i_feature] = feature_fwd
end

function slice_dataset(fwd::IntervalFWD{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    IntervalFWD{T}(if return_view @view fwd.d[:,:,inds,:] else fwd.d[:,:,inds,:] end)
end
Base.@propagate_inbounds @inline fwd_get_channel(fwd::IntervalFWD{T}, i_sample::Integer, i_feature::Integer) where {T} =
    @views fwd.d[:,:,i_sample, i_feature]
const IntervalFeaturedChannel{T} = AbstractArray{T,2}
fwd_channel_interpret_world(fwc::IntervalFeaturedChannel{T}, w::Interval) where {T} =
    fwc[w.x, w.y]

############################################################################################
# FWD, Interval: 6D array (x.x × x.y × y.x × y.y × nsamples × nfeatures)
############################################################################################

struct Interval2DFWD{T} <: UniformFullDimensionalFWD{T,2,Interval2D}
    d :: Array{T,6}
end

channel_size(fwd::Interval2DFWD) = (size(fwd.d, 1),size(fwd.d, 3))
goeswith(::Type{Interval2DFWD}, ::Type{Interval2D}) = true
default_fwd_type(::Type{Interval2D}) = Interval2DFWD

nsamples(fwd::Interval2DFWD)  = size(fwd.d, 5)
nfeatures(fwd::Interval2DFWD) = size(fwd.d, 6)
Base.size(fwd::Interval2DFWD, args...) = size(fwd.d, args...)


function fwd_init(::Type{Interval2DFWD}, X::InterpretedModalDataset{T,2,Interval2D}) where {T}
    Interval2DFWD{T}(Array{T,6}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, max_channel_size(X)[2], max_channel_size(X)[2]+1, nsamples(X), nfeatures(X)))
end

function fwd_init_world_slice(fwd::Interval2DFWD, i_sample::Integer, w::AbstractWorld)
    nothing
end

hasnans(fwd::Interval2DFWD) = begin
    # @show ([hasnans(fwd.d[xx,xy,yx,yy,:,:]) for xx in 1:size(fwd.d, 1) for xy in (xx+1):size(fwd.d, 2) for yx in 1:size(fwd.d, 3) for yy in (yx+1):size(fwd.d, 4)])
    any([hasnans(fwd.d[xx,xy,yx,yy,:,:]) for xx in 1:size(fwd.d, 1) for xy in (xx+1):size(fwd.d, 2) for yx in 1:size(fwd.d, 3) for yy in (yx+1):size(fwd.d, 4)])
end

Base.@propagate_inbounds @inline fwd_get(
    fwd         :: Interval2DFWD{T},
    i_sample    :: Integer,
    w           :: Interval2D,
    i_feature   :: Integer) where {T} = fwd.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_feature]

Base.@propagate_inbounds @inline function fwd_set(fwd::Interval2DFWD{T}, w::Interval2D, i_sample::Integer, i_feature::Integer, threshold::T) where {T}
    fwd.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_feature] = threshold
end

Base.@propagate_inbounds @inline function fwd_set_feature_slice(fwd::Interval2DFWD{T}, i_feature::Integer, feature_fwd::Array{T,5}) where {T}
    fwd.d[:, :, :, :, :, i_feature] = feature_fwd
end

function slice_dataset(fwd::Interval2DFWD{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    Interval2DFWD{T}(if return_view @view fwd.d[:,:,:,:,inds,:] else fwd.d[:,:,:,:,inds,:] end)
end
Base.@propagate_inbounds @inline fwd_get_channel(fwd::Interval2DFWD{T}, i_sample::Integer, i_feature::Integer) where {T} =
    @views fwd.d[:,:,:,:,i_sample, i_feature]
const Interval2DFeaturedChannel{T} = AbstractArray{T,4}
fwd_channel_interpret_world(fwc::Interval2DFeaturedChannel{T}, w::Interval2D) where {T} =
    fwc[w.x.x, w.x.y, w.y.x, w.y.y]

############################################################################################

const FWDFeatureSlice{T} = Union{
    # FWDFeatureSlice(InterpretedModalDataset{T where T,0,ModalLogic.OneWorld})
    T, # Note: should be, but it throws error OneWorldFeaturedChannel{T},
    IntervalFeaturedChannel{T},
    Interval2DFeaturedChannel{T},
    # FWDFeatureSlice(InterpretedModalDataset{T where T,2,Interval2D})
}

############################################################################################
############################################################################################


# TODO add AbstractWorldSet type
apply_aggregator(fwd_feature_slice::FWDFeatureSlice{T}, worlds::Any, aggregator::Agg) where {T,Agg<:Aggregator} = begin
    
    # TODO try reduce(aggregator, worlds; init=ModalLogic.bottom(aggregator, T))
    # TODO remove this aggregator_to_binary...
    
    if length(worlds |> collect) == 0
        aggregator_bottom(aggregator, T)
    else
        aggregator((w)->fwd_channel_interpret_world(fwd_feature_slice, w), worlds)
    end

    # opt = aggregator_to_binary(aggregator)
    # threshold = ModalLogic.bottom(aggregator, T)
    # for w in worlds
    #   e = fwd_channel_interpret_world(fwd_feature_slice, w)
    #   threshold = opt(threshold,e)
    # end
    # threshold
end

