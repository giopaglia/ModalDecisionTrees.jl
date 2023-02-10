import Base: size, ndims, getindex

############################################################################################
############################################################################################
# world-specific FWD implementations
############################################################################################
############################################################################################

abstract type AbstractUniformFullDimensionalFWD{T,N,W<:AbstractWorld,FR<:FullDimensionalFrame{N,W,Bool}} <: AbstractFWD{T,W,FR} end

channel_size(fwd::AbstractUniformFullDimensionalFWD) = error("TODO add message inviting to add channel_size")
initialworldset(fwd::AbstractUniformFullDimensionalFWD, i_sample, args...) = initialworldset(FullDimensionalFrame(channel_size(fwd)), args...)
accessibles(fwd::AbstractUniformFullDimensionalFWD, i_sample, args...) = accessibles(FullDimensionalFrame(channel_size(fwd)), args...)
representatives(fwd::AbstractUniformFullDimensionalFWD, i_sample, args...) = representatives(FullDimensionalFrame(channel_size(fwd)), args...)
allworlds(fwd::AbstractUniformFullDimensionalFWD, i_sample::Integer, args...) = allworlds(FullDimensionalFrame(channel_size(fwd)), args...)

############################################################################################
############################################################################################

struct UniformFullDimensionalFWD{
    T,
    W<:AbstractWorld,
    N,
    D<:AbstractArray{T},
    FR<:FullDimensionalFrame{N,W,Bool}
} <: AbstractUniformFullDimensionalFWD{T,N,W,FR}

    d :: D

    function UniformFullDimensionalFWD{T,W,N,D,FR}(d::D) where {T,W<:AbstractWorld,N,D<:AbstractArray{T},FR<:FullDimensionalFrame{N,W,Bool}}
        new{T,W,N,D,FR}(d)
    end

    function UniformFullDimensionalFWD{T,W,N}(d::D) where {T,W<:AbstractWorld,N,D<:AbstractArray{T}}
        new{T,W,N,D,FullDimensionalFrame{N,W,Bool}}(d)
    end

############################################################################################

    function UniformFullDimensionalFWD(X::DimensionalFeaturedDataset{T,N,W}) where {T,N,W<:OneWorld}
        UniformFullDimensionalFWD{T,W,N}(Array{T,2}(undef, nsamples(X), nfeatures(X)))
    end
    function UniformFullDimensionalFWD(X::DimensionalFeaturedDataset{T,N,W}) where {T,N,W<:Interval}
        UniformFullDimensionalFWD{T,W,N}(Array{T,4}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, nsamples(X), nfeatures(X)))
    end
    function UniformFullDimensionalFWD(X::DimensionalFeaturedDataset{T,N,W}) where {T,N,W<:Interval2D}
        UniformFullDimensionalFWD{T,W,N}(Array{T,6}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, max_channel_size(X)[2], max_channel_size(X)[2]+1, nsamples(X), nfeatures(X)))
    end

end

Base.size(fwd::UniformFullDimensionalFWD, args...) = size(fwd.d, args...)
Base.ndims(fwd::UniformFullDimensionalFWD, args...) = ndims(fwd.d, args...)

nsamples(fwd::UniformFullDimensionalFWD)  = size(fwd, ndims(fwd)-1)
nfeatures(fwd::UniformFullDimensionalFWD) = size(fwd, ndims(fwd))

############################################################################################

channel_size(fwd::UniformFullDimensionalFWD{T,OneWorld}) where {T} = ()
channel_size(fwd::UniformFullDimensionalFWD{T,Interval}) where {T} = (size(fwd, 1),)
channel_size(fwd::UniformFullDimensionalFWD{T,Interval2D}) where {T} = (size(fwd, 1),size(fwd, 3))

############################################################################################

function capacity(fwd::UniformFullDimensionalFWD{T,OneWorld}) where {T}
    prod(size(fwd))
end
function capacity(fwd::UniformFullDimensionalFWD{T,Interval}) where {T}
    prod([
        nsamples(fwd),
        nfeatures(fwd),
        div(size(fwd, 1)*(size(fwd, 2)),2),
    ])
end
function capacity(fwd::UniformFullDimensionalFWD{T,Interval2D}) where {T}
    prod([
        nsamples(fwd),
        nfeatures(fwd),
        div(size(fwd, 1)*(size(fwd, 2)),2),
        div(size(fwd, 3)*(size(fwd, 4)),2),
    ])
end

############################################################################################

function hasnans(fwd::UniformFullDimensionalFWD{T,OneWorld}) where {T}
    any(_isnan.(fwd.d))
end
function hasnans(fwd::UniformFullDimensionalFWD{T,Interval}) where {T}
    any([hasnans(fwd.d[x,y,:,:])
        for x in 1:size(fwd, 1) for y in (x+1):size(fwd, 2)])
end
function hasnans(fwd::UniformFullDimensionalFWD{T,Interval2D}) where {T}
    any([hasnans(fwd.d[xx,xy,yx,yy,:,:])
        for xx in 1:size(fwd, 1) for xy in (xx+1):size(fwd, 2)
        for yx in 1:size(fwd, 3) for yy in (yx+1):size(fwd, 4)])
end

############################################################################################

function fwd_init_world_slice(fwd::UniformFullDimensionalFWD, i_sample::Integer, w::AbstractWorld)
    nothing
end

############################################################################################

@inline function Base.getindex(
    fwd         :: UniformFullDimensionalFWD{T,OneWorld},
    i_sample    :: Integer,
    w           :: OneWorld,
    i_feature   :: Integer
) where {T}
    fwd.d[i_sample, i_feature]
end

@inline function Base.getindex(
    fwd         :: UniformFullDimensionalFWD{T,Interval},
    i_sample    :: Integer,
    w           :: Interval,
    i_feature   :: Integer
) where {T}
    fwd.d[w.x, w.y, i_sample, i_feature]
end

@inline function Base.getindex(
    fwd         :: UniformFullDimensionalFWD{T,Interval2D},
    i_sample    :: Integer,
    w           :: Interval2D,
    i_feature   :: Integer
) where {T}
    fwd.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_feature]
end

############################################################################################

Base.@propagate_inbounds @inline function fwd_set(fwd::UniformFullDimensionalFWD{T,OneWorld}, w::OneWorld, i_sample::Integer, i_feature::Integer, threshold::T) where {T}
    fwd.d[i_sample, i_feature] = threshold
end

Base.@propagate_inbounds @inline function fwd_set(fwd::UniformFullDimensionalFWD{T,Interval}, w::Interval, i_sample::Integer, i_feature::Integer, threshold::T) where {T}
    fwd.d[w.x, w.y, i_sample, i_feature] = threshold
end

Base.@propagate_inbounds @inline function fwd_set(fwd::UniformFullDimensionalFWD{T,Interval2D}, w::Interval2D, i_sample::Integer, i_feature::Integer, threshold::T) where {T}
    fwd.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_feature] = threshold
end

############################################################################################

function _slice_dataset(fwd::UniformFullDimensionalFWD{T,W,N}, inds::AbstractVector{<:Integer}, return_view::Val = Val(false)) where {T,W<:OneWorld,N}
    UniformFullDimensionalFWD{T,W,N}(if return_view == Val(true) @view fwd.d[inds,:] else fwd.d[inds,:] end)
end

function _slice_dataset(fwd::UniformFullDimensionalFWD{T,W,N}, inds::AbstractVector{<:Integer}, return_view::Val = Val(false)) where {T,W<:Interval,N}
    UniformFullDimensionalFWD{T,W,N}(if return_view == Val(true) @view fwd.d[:,:,inds,:] else fwd.d[:,:,inds,:] end)
end

function _slice_dataset(fwd::UniformFullDimensionalFWD{T,W,N}, inds::AbstractVector{<:Integer}, return_view::Val = Val(false)) where {T,W<:Interval2D,N}
    UniformFullDimensionalFWD{T,W,N}(if return_view == Val(true) @view fwd.d[:,:,:,:,inds,:] else fwd.d[:,:,:,:,inds,:] end)
end

############################################################################################

# TODO fix

Base.@propagate_inbounds @inline function fwd_set_feature_slice(fwd::UniformFullDimensionalFWD{T,OneWorld}, i_feature::Integer, feature_fwd::Array{T,1}) where {T}
    fwd.d[:, i_feature] = feature_fwd
end

Base.@propagate_inbounds @inline fwdread_channel(fwd::UniformFullDimensionalFWD{T,OneWorld}, i_sample::Integer, i_feature::Integer) where {T} =
    fwd.d[i_sample, i_feature]
const OneWorldFeaturedChannel{T} = T
fwd_channel_interpret_world(fwc::T #=Note: should be OneWorldFeaturedChannel{T}, but it throws error =#, w::OneWorld) where {T} = fwc

Base.@propagate_inbounds @inline function fwd_set_feature_slice(fwd::UniformFullDimensionalFWD{T,Interval}, i_feature::Integer, feature_fwd::Array{T,3}) where {T}
    fwd.d[:, :, :, i_feature] = feature_fwd
end
Base.@propagate_inbounds @inline fwdread_channel(fwd::UniformFullDimensionalFWD{T,Interval}, i_sample::Integer, i_feature::Integer) where {T} =
    @views fwd.d[:,:,i_sample, i_feature]
const IntervalFeaturedChannel{T} = AbstractArray{T,2}
fwd_channel_interpret_world(fwc::IntervalFeaturedChannel{T}, w::Interval) where {T} =
    fwc[w.x, w.y]


Base.@propagate_inbounds @inline function fwd_set_feature_slice(fwd::UniformFullDimensionalFWD{T,Interval2D}, i_feature::Integer, feature_fwd::Array{T,5}) where {T}
    fwd.d[:, :, :, :, :, i_feature] = feature_fwd
end
Base.@propagate_inbounds @inline fwdread_channel(fwd::UniformFullDimensionalFWD{T,Interval2D}, i_sample::Integer, i_feature::Integer) where {T} =
    @views fwd.d[:,:,:,:,i_sample, i_feature]
const Interval2DFeaturedChannel{T} = AbstractArray{T,4}
fwd_channel_interpret_world(fwc::Interval2DFeaturedChannel{T}, w::Interval2D) where {T} =
    fwc[w.x.x, w.x.y, w.y.x, w.y.y]

const FWDFeatureSlice{T} = Union{
    # FWDFeatureSlice(DimensionalFeaturedDataset{T where T,0,ModalLogic.OneWorld})
    T, # Note: should be, but it throws error OneWorldFeaturedChannel{T},
    IntervalFeaturedChannel{T},
    Interval2DFeaturedChannel{T},
    # FWDFeatureSlice(DimensionalFeaturedDataset{T where T,2,Interval2D})
}


############################################################################################


# channel_size(fwd::OneWorldFWD) = ()

# nsamples(fwd::OneWorldFWD)  = size(fwd.d, 1)
# nfeatures(fwd::OneWorldFWD) = size(fwd.d, 2)

# function fwd_init(::Type{OneWorldFWD}, X::DimensionalFeaturedDataset{T,0,OneWorld}) where {T}
#     OneWorldFWD{T}(Array{T,2}(undef, nsamples(X), nfeatures(X)))
# end

# function fwd_init_world_slice(fwd::OneWorldFWD, i_sample::Integer, w::AbstractWorld)
#     nothing
# end

# hasnans(fwd::OneWorldFWD) = any(_isnan.(fwd.d))

# Base.@propagate_inbounds @inline fwdread(
#     fwd         :: OneWorldFWD{T},
#     i_sample    :: Integer,
#     w           :: OneWorld,
#     i_feature   :: Integer) where {T} = fwd.d[i_sample, i_feature]

# Base.@propagate_inbounds @inline function fwd_set(fwd::OneWorldFWD{T}, w::OneWorld, i_sample::Integer, i_feature::Integer, threshold::T) where {T}
#     fwd.d[i_sample, i_feature] = threshold
# end

# function _slice_dataset(fwd::OneWorldFWD{T}, inds::AbstractVector{<:Integer}, return_view::Val = Val(false)) where {T}
#     OneWorldFWD{T}(if return_view == Val(true) @view fwd.d[inds,:] else fwd.d[inds,:] end)
# end

# Base.@propagate_inbounds @inline function fwd_set_feature_slice(fwd::OneWorldFWD{T}, i_feature::Integer, feature_fwd::Array{T,1}) where {T}
#     fwd.d[:, i_feature] = feature_fwd
# end

# Base.@propagate_inbounds @inline fwdread_channel(fwd::OneWorldFWD{T}, i_sample::Integer, i_feature::Integer) where {T} =
#     fwd.d[i_sample, i_feature]
# const OneWorldFeaturedChannel{T} = T
# fwd_channel_interpret_world(fwc::T #=Note: should be OneWorldFeaturedChannel{T}, but it throws error =#, w::OneWorld) where {T} = fwc

############################################################################################
# FWD, Interval: 4D array (x × y × nsamples × nfeatures)
############################################################################################

# struct IntervalFWD{T} <: UniformFullDimensionalFWD{T,1,Interval}
#     d :: Array{T,4}
# end

# channel_size(fwd::IntervalFWD) = (size(fwd, 1),)

# nsamples(fwd::IntervalFWD)  = size(fwd, 3)
# nfeatures(fwd::IntervalFWD) = size(fwd, 4)

# function fwd_init(::Type{IntervalFWD}, X::DimensionalFeaturedDataset{T,1,Interval}) where {T}
#     IntervalFWD{T}(Array{T,4}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, nsamples(X), nfeatures(X)))
# end

# function fwd_init_world_slice(fwd::IntervalFWD, i_sample::Integer, w::AbstractWorld)
#     nothing
# end

# function hasnans(fwd::IntervalFWD)
#     # @show ([hasnans(fwd.d[x,y,:,:]) for x in 1:size(fwd, 1) for y in (x+1):size(fwd, 2)])
#     any([hasnans(fwd.d[x,y,:,:]) for x in 1:size(fwd, 1) for y in (x+1):size(fwd, 2)])
# end

# Base.@propagate_inbounds @inline fwdread(
#     fwd         :: IntervalFWD{T},
#     i_sample    :: Integer,
#     w           :: Interval,
#     i_feature   :: Integer) where {T} = fwd.d[w.x, w.y, i_sample, i_feature]

# Base.@propagate_inbounds @inline function fwd_set(fwd::IntervalFWD{T}, w::Interval, i_sample::Integer, i_feature::Integer, threshold::T) where {T}
#     fwd.d[w.x, w.y, i_sample, i_feature] = threshold
# end

# Base.@propagate_inbounds @inline function fwd_set_feature_slice(fwd::IntervalFWD{T}, i_feature::Integer, feature_fwd::Array{T,3}) where {T}
#     fwd.d[:, :, :, i_feature] = feature_fwd
# end

# function _slice_dataset(fwd::IntervalFWD{T}, inds::AbstractVector{<:Integer}, return_view::Val = Val(false)) where {T}
#     IntervalFWD{T}(if return_view == Val(true) @view fwd.d[:,:,inds,:] else fwd.d[:,:,inds,:] end)
# end
# Base.@propagate_inbounds @inline fwdread_channel(fwd::IntervalFWD{T}, i_sample::Integer, i_feature::Integer) where {T} =
#     @views fwd.d[:,:,i_sample, i_feature]
# const IntervalFeaturedChannel{T} = AbstractArray{T,2}
# fwd_channel_interpret_world(fwc::IntervalFeaturedChannel{T}, w::Interval) where {T} =
#     fwc[w.x, w.y]

############################################################################################
# FWD, Interval: 6D array (x.x × x.y × y.x × y.y × nsamples × nfeatures)
############################################################################################

# struct Interval2DFWD{T} <: UniformFullDimensionalFWD{T,2,Interval2D}
#     d :: Array{T,6}
# end

# channel_size(fwd::Interval2DFWD) = (size(fwd, 1),size(fwd, 3))

# nsamples(fwd::Interval2DFWD)  = size(fwd, 5)
# nfeatures(fwd::Interval2DFWD) = size(fwd, 6)


# function fwd_init(::Type{Interval2DFWD}, X::DimensionalFeaturedDataset{T,2,Interval2D}) where {T}
#     Interval2DFWD{T}(Array{T,6}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, max_channel_size(X)[2], max_channel_size(X)[2]+1, nsamples(X), nfeatures(X)))
# end

# function fwd_init_world_slice(fwd::Interval2DFWD, i_sample::Integer, w::AbstractWorld)
#     nothing
# end

# function hasnans(fwd::Interval2DFWD)
#     # @show ([hasnans(fwd.d[xx,xy,yx,yy,:,:]) for xx in 1:size(fwd, 1) for xy in (xx+1):size(fwd, 2) for yx in 1:size(fwd, 3) for yy in (yx+1):size(fwd, 4)])
#     any([hasnans(fwd.d[xx,xy,yx,yy,:,:]) for xx in 1:size(fwd, 1) for xy in (xx+1):size(fwd, 2) for yx in 1:size(fwd, 3) for yy in (yx+1):size(fwd, 4)])
# end

# Base.@propagate_inbounds @inline fwdread(
#     fwd         :: Interval2DFWD{T},
#     i_sample    :: Integer,
#     w           :: Interval2D,
#     i_feature   :: Integer) where {T} = fwd.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_feature]

# Base.@propagate_inbounds @inline function fwd_set(fwd::Interval2DFWD{T}, w::Interval2D, i_sample::Integer, i_feature::Integer, threshold::T) where {T}
#     fwd.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_feature] = threshold
# end

# Base.@propagate_inbounds @inline function fwd_set_feature_slice(fwd::Interval2DFWD{T}, i_feature::Integer, feature_fwd::Array{T,5}) where {T}
#     fwd.d[:, :, :, :, :, i_feature] = feature_fwd
# end

# function _slice_dataset(fwd::Interval2DFWD{T}, inds::AbstractVector{<:Integer}, return_view::Val = Val(false)) where {T}
#     Interval2DFWD{T}(if return_view == Val(true) @view fwd.d[:,:,:,:,inds,:] else fwd.d[:,:,:,:,inds,:] end)
# end
# Base.@propagate_inbounds @inline fwdread_channel(fwd::Interval2DFWD{T}, i_sample::Integer, i_feature::Integer) where {T} =
#     @views fwd.d[:,:,:,:,i_sample, i_feature]
# const Interval2DFeaturedChannel{T} = AbstractArray{T,4}
# fwd_channel_interpret_world(fwc::Interval2DFeaturedChannel{T}, w::Interval2D) where {T} =
#     fwc[w.x.x, w.x.y, w.y.x, w.y.y]

############################################################################################

############################################################################################
############################################################################################


# TODO add AbstractWorldSet type
function apply_aggregator(fwd_feature_slice::FWDFeatureSlice{T}, worlds::Any, aggregator::Agg) where {T,Agg<:Aggregator}
    
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
