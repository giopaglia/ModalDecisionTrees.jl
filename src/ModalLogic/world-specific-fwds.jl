############################################################################################
############################################################################################
# world-specific FWD implementations
############################################################################################
############################################################################################

############################################################################################
# FWD, OneWorld: 2D array (nsamples × nfeatures)
############################################################################################

struct OneWorldFWD{T} <: AbstractFWD{T, OneWorld}
    d :: Array{T, 2}
end

goes_with(::Type{OneWorldFWD}, ::Type{OneWorld}) = true
default_fwd_type(::Type{OneWorld}) = OneWorldFWD

nsamples(fwd::OneWorldFWD{T}) where {T}  = size(fwd, 1)
nfeatures(fwd::OneWorldFWD{T}) where {T} = size(fwd, 2)
Base.size(fwd::OneWorldFWD{T}, args...) where {T} = size(fwd.d, args...)

function fwd_init(::Type{OneWorldFWD}, X::InterpretedModalDataset{T, 0, OneWorld}) where {T}
    OneWorldFWD{T}(Array{T, 2}(undef, nsamples(X), nfeatures(X)))
end

function fwd_init_world_slice(fwd::OneWorldFWD{T}, args...) where {T}
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

Base.@propagate_inbounds @inline function fwd_set_feature_slice(fwd::OneWorldFWD{T}, i_feature::Integer, feature_fwd::Array{T, 1}) where {T}
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

struct IntervalFWD{T} <: AbstractFWD{T, Interval}
    d :: Array{T, 4}
end

goes_with(::Type{IntervalFWD}, ::Type{Interval}) = true
default_fwd_type(::Type{Interval}) = IntervalFWD

nsamples(fwd::IntervalFWD{T}) where {T}  = size(fwd, 3)
nfeatures(fwd::IntervalFWD{T}) where {T} = size(fwd, 4)
Base.size(fwd::IntervalFWD{T}, args...) where {T} = size(fwd.d, args...)

function fwd_init(::Type{IntervalFWD}, X::InterpretedModalDataset{T, 1, Interval}) where {T}
    IntervalFWD{T}(Array{T, 4}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, nsamples(X), nfeatures(X)))
end

function fwd_init_world_slice(fwd::IntervalFWD{T}, args...) where {T}
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

Base.@propagate_inbounds @inline function fwd_set_feature_slice(fwd::IntervalFWD{T}, i_feature::Integer, feature_fwd::Array{T, 3}) where {T}
    fwd.d[:, :, :, i_feature] = feature_fwd
end

function slice_dataset(fwd::IntervalFWD{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    IntervalFWD{T}(if return_view @view fwd.d[:,:,inds,:] else fwd.d[:,:,inds,:] end)
end
Base.@propagate_inbounds @inline fwd_get_channel(fwd::IntervalFWD{T}, i_sample::Integer, i_feature::Integer) where {T} =
    @views fwd.d[:,:,i_sample, i_feature]
const IntervalFeaturedChannel{T} = Array{T, 2}
fwd_channel_interpret_world(fwc::IntervalFeaturedChannel{T}, w::Interval) where {T} =
    fwc[w.x, w.y]

############################################################################################
# FWD, Interval: 6D array (x.x × x.y × y.x × y.y × nsamples × nfeatures)
############################################################################################

struct Interval2DFWD{T} <: AbstractFWD{T, Interval2D}
    d :: Array{T, 6}
end

goes_with(::Type{Interval2DFWD}, ::Type{Interval2D}) = true
default_fwd_type(::Type{Interval2D}) = Interval2DFWD

nsamples(fwd::Interval2DFWD{T}) where {T}  = size(fwd, 5)
nfeatures(fwd::Interval2DFWD{T}) where {T} = size(fwd, 6)
Base.size(fwd::Interval2DFWD{T}, args...) where {T} = size(fwd.d, args...)


function fwd_init(::Type{Interval2DFWD}, X::InterpretedModalDataset{T, 2, Interval2D}) where {T}
    Interval2DFWD{T}(Array{T, 6}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, max_channel_size(X)[2], max_channel_size(X)[2]+1, nsamples(X), nfeatures(X)))
end

function fwd_init_world_slice(fwd::Interval2DFWD{T}, args...) where {T}
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

Base.@propagate_inbounds @inline function fwd_set_feature_slice(fwd::Interval2DFWD{T}, i_feature::Integer, feature_fwd::Array{T, 5}) where {T}
    fwd.d[:, :, :, :, :, i_feature] = feature_fwd
end

function slice_dataset(fwd::Interval2DFWD{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    Interval2DFWD{T}(if return_view @view fwd.d[:,:,:,:,inds,:] else fwd.d[:,:,:,:,inds,:] end)
end
Base.@propagate_inbounds @inline fwd_get_channel(fwd::Interval2DFWD{T}, i_sample::Integer, i_feature::Integer) where {T} =
    @views fwd.d[:,:,:,:,i_sample, i_feature]
const Interval2DFeaturedChannel{T} = Array{T, 4}
fwd_channel_interpret_world(fwc::Interval2DFeaturedChannel{T}, w::Interval2D) where {T} =
    fwc[w.x.x, w.x.y, w.y.x, w.y.y]

############################################################################################

const FWDFeatureSlice{T} = Union{
    # FWDFeatureSlice(InterpretedModalDataset{T where T, 0, ModalLogic.OneWorld})
    T, # Note: should be, but it throws error OneWorldFeaturedChannel{T},
    IntervalFeaturedChannel{T},
    Interval2DFeaturedChannel{T},
    # FWDFeatureSlice(InterpretedModalDataset{T where T, 2, Interval2D})
}

############################################################################################
############################################################################################


# TODO add AbstractWorldSet type
compute_modal_gamma(fwd_feature_slice::FWDFeatureSlice{T}, worlds::Any, aggregator::Agg) where {T, Agg<:Aggregator} = begin
    
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


############################################################################################
############################################################################################
# world-specific FWD supports implementations
############################################################################################
############################################################################################

############################################################################################
# FWD support, OneWorld: 3D array (nsamples × nfeatsnaggrs × nrelations)
############################################################################################

struct OneWorldFWD_RS{T} <: AbstractRelationalSupport{T, OneWorld}
    d :: Array{T, 3}
end

nsamples(emds::OneWorldFWD_RS{T}) where {T}     = size(emds, 1)
nfeatsnaggrs(emds::OneWorldFWD_RS{T}) where {T} = size(emds, 2)
nrelations(emds::OneWorldFWD_RS{T}) where {T}   = size(emds, 3)
Base.@propagate_inbounds @inline Base.getindex(
    emds         :: OneWorldFWD_RS{T},
    i_sample     :: Integer,
    w            :: OneWorld,
    i_featsnaggr :: Integer,
    i_relation   :: Integer) where {T} = emds.d[i_sample, i_featsnaggr, i_relation]
Base.size(emds::OneWorldFWD_RS{T}, args...) where {T} = size(emds.d, args...)
goes_with(::Type{OneWorldFWD_RS}, ::Type{OneWorld}) = true

hasnans(emds::OneWorldFWD_RS) = any(_isnan.(emds.d))

fwd_rs_init(emd::ExplicitModalDataset{T, OneWorld}, nfeatsnaggrs::Integer, nrelations::Integer; perform_initialization = false) where {T} = begin
    if perform_initialization
        _fwd_rs = fill!(Array{Union{T,Nothing}, 3}(undef, nsamples(emd), nfeatsnaggrs, nrelations), nothing)
        OneWorldFWD_RS{Union{T,Nothing}}(_fwd_rs)
    else
        _fwd_rs = Array{T, 3}(undef, nsamples(emd), nfeatsnaggrs, nrelations)
        OneWorldFWD_RS{T}(_fwd_rs)
    end
end
fwd_rs_init_world_slice(emds::OneWorldFWD_RS, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) =
    nothing
Base.@propagate_inbounds @inline fwd_rs_set(emds::OneWorldFWD_RS{T}, i_sample::Integer, w::OneWorld, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
    emds.d[i_sample, i_featsnaggr, i_relation] = threshold
function slice_dataset(emds::OneWorldFWD_RS{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    OneWorldFWD_RS{T}(if return_view @view emds.d[inds,:,:] else emds.d[inds,:,:] end)
end

############################################################################################
# FWD support, Interval: 5D array (x × y × nsamples × nfeatsnaggrs × nrelations)
############################################################################################


struct IntervalFWD_RS{T} <: AbstractRelationalSupport{T, Interval}
    d :: Array{T, 5}
end

nsamples(emds::IntervalFWD_RS{T}) where {T}     = size(emds, 3)
nfeatsnaggrs(emds::IntervalFWD_RS{T}) where {T} = size(emds, 4)
nrelations(emds::IntervalFWD_RS{T}) where {T}   = size(emds, 5)
Base.@propagate_inbounds @inline Base.getindex(
    emds         :: IntervalFWD_RS{T},
    i_sample     :: Integer,
    w            :: Interval,
    i_featsnaggr :: Integer,
    i_relation   :: Integer) where {T} = emds.d[w.x, w.y, i_sample, i_featsnaggr, i_relation]
Base.size(emds::IntervalFWD_RS{T}, args...) where {T} = size(emds.d, args...)
goes_with(::Type{IntervalFWD_RS}, ::Type{Interval}) = true


hasnans(emds::IntervalFWD_RS) = begin
    # @show [hasnans(emds.d[x,y,:,:,:]) for x in 1:size(emds.d, 1) for y in (x+1):size(emds.d, 2)]
    any([hasnans(emds.d[x,y,:,:,:]) for x in 1:size(emds.d, 1) for y in (x+1):size(emds.d, 2)])
end

# Note: assuming default_fwd_type(::Type{Interval}) = IntervalFWD
fwd_rs_init(emd::ExplicitModalDataset{T, Interval}, nfeatsnaggrs::Integer, nrelations::Integer; perform_initialization = false) where {T} = begin
    if perform_initialization
        _fwd_rs = fill!(Array{Union{T,Nothing}, 5}(undef, size(emd.fwd, 1), size(emd.fwd, 2), nsamples(emd), nfeatsnaggrs, nrelations), nothing)
        IntervalFWD_RS{Union{T,Nothing}}(_fwd_rs)
    else
        _fwd_rs = Array{T, 5}(undef, size(emd.fwd, 1), size(emd.fwd, 2), nsamples(emd), nfeatsnaggrs, nrelations)
        IntervalFWD_RS{T}(_fwd_rs)
    end
end
fwd_rs_init_world_slice(emds::IntervalFWD_RS, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) =
    nothing
Base.@propagate_inbounds @inline fwd_rs_set(emds::IntervalFWD_RS{T}, i_sample::Integer, w::Interval, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
    emds.d[w.x, w.y, i_sample, i_featsnaggr, i_relation] = threshold
function slice_dataset(emds::IntervalFWD_RS{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    IntervalFWD_RS{T}(if return_view @view emds.d[:,:,inds,:,:] else emds.d[:,:,inds,:,:] end)
end

############################################################################################
# FWD support, Interval2D: 7D array (x.x × x.y × y.x × y.y × nsamples × nfeatsnaggrs × nrelations)
############################################################################################

# struct Interval2DFWD_RS{T} <: AbstractRelationalSupport{T, Interval2D}
#   d :: Array{T, 7}
# end

# nsamples(emds::Interval2DFWD_RS{T}) where {T}     = size(emds, 5)
# nfeatsnaggrs(emds::Interval2DFWD_RS{T}) where {T} = size(emds, 6)
# nrelations(emds::Interval2DFWD_RS{T}) where {T}   = size(emds, 7)
# Base.@propagate_inbounds @inline Base.getindex(
#   emds         :: Interval2DFWD_RS{T},
#   i_sample     :: Integer,
#   w            :: Interval2D,
#   i_featsnaggr :: Integer,
#   i_relation   :: Integer) where {T} = emds.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_featsnaggr, i_relation]
# size(emds::Interval2DFWD_RS{T}, args...) where {T} = size(emds.d, args...)
# goes_with(::Type{Interval2DFWD_RS}, ::Type{Interval2D}) = true

# TODO... hasnans(emds::Interval2DFWD_RS) = any(_isnan.(emds.d))
# TODO...? hasnans(emds::Interval2DFWD_RS) = any([hasnans(emds.d[xx,xy,yx,yy,:,:]) for xx in 1:size(emds.d, 1) for xy in (xx+1):size(emds.d, 2) for yx in 1:size(emds.d, 3) for yy in (yx+1):size(emds.d, 4)])

# fwd_rs_init(emd::ExplicitModalDataset{T, Interval2D}, nfeatsnaggrs::Integer, nrelations::Integer; perform_initialization = false) where {T} = begin
#   if perform_initialization
#       _fwd_rs = fill!(Array{Union{T,Nothing}, 7}(undef, size(emd.fwd, 1), size(emd.fwd, 2), size(emd.fwd, 3), size(emd.fwd, 4), nsamples(emd), nfeatsnaggrs, nrelations), nothing)
#       Interval2DFWD_RS{Union{T,Nothing}}(_fwd_rs)
#   else
#       _fwd_rs = Array{T, 7}(undef, size(emd.fwd, 1), size(emd.fwd, 2), size(emd.fwd, 3), size(emd.fwd, 4), nsamples(emd), nfeatsnaggrs, nrelations)
#       Interval2DFWD_RS{T}(_fwd_rs)
#   end
# end
# fwd_rs_init_world_slice(emds::Interval2DFWD_RS, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) =
#   nothing
# Base.@propagate_inbounds @inline fwd_rs_set(emds::Interval2DFWD_RS{T}, i_sample::Integer, w::Interval2D, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
#   emds.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_featsnaggr, i_relation] = threshold
# function slice_dataset(emds::Interval2DFWD_RS{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
# @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
#   Interval2DFWD_RS{T}(if return_view @view emds.d[:,:,:,:,inds,:,:] else emds.d[:,:,:,:,inds,:,:] end)
# end


############################################################################################
# FWD support, Interval2D: 7D array (linearized(x) × linearized(y) × nsamples × nfeatsnaggrs × nrelations)
############################################################################################

struct Interval2DFWD_RS{T} <: AbstractRelationalSupport{T, Interval2D}
    d :: Array{T, 5}
end

nsamples(emds::Interval2DFWD_RS{T}) where {T}     = size(emds, 3)
nfeatsnaggrs(emds::Interval2DFWD_RS{T}) where {T} = size(emds, 4)
nrelations(emds::Interval2DFWD_RS{T}) where {T}   = size(emds, 5)
Base.@propagate_inbounds @inline Base.getindex(
    emds         :: Interval2DFWD_RS{T},
    i_sample     :: Integer,
    w            :: Interval2D,
    i_featsnaggr :: Integer,
    i_relation   :: Integer) where {T} = emds.d[w.x.x+div((w.x.y-2)*(w.x.y-1),2), w.y.x+div((w.y.y-2)*(w.y.y-1),2), i_sample, i_featsnaggr, i_relation]
Base.size(emds::Interval2DFWD_RS{T}, args...) where {T} = size(emds.d, args...)
goes_with(::Type{Interval2DFWD_RS}, ::Type{Interval2D}) = true

hasnans(emds::Interval2DFWD_RS) = any(_isnan.(emds.d))

fwd_rs_init(emd::ExplicitModalDataset{T, Interval2D}, nfeatsnaggrs::Integer, nrelations::Integer; perform_initialization = false) where {T} = begin
    if perform_initialization
        _fwd_rs = fill!(Array{Union{T,Nothing}, 5}(undef, div(size(emd.fwd, 1)*size(emd.fwd, 2),2), div(size(emd.fwd, 3)*size(emd.fwd, 4),2), nsamples(emd), nfeatsnaggrs, nrelations), nothing)
        Interval2DFWD_RS{Union{T,Nothing}}(_fwd_rs)
    else
        _fwd_rs = Array{T, 5}(undef, div(size(emd.fwd, 1)*size(emd.fwd, 2),2), div(size(emd.fwd, 3)*size(emd.fwd, 4),2), nsamples(emd), nfeatsnaggrs, nrelations)
        Interval2DFWD_RS{T}(_fwd_rs)
    end
end
fwd_rs_init_world_slice(emds::Interval2DFWD_RS, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) =
    nothing
Base.@propagate_inbounds @inline fwd_rs_set(emds::Interval2DFWD_RS{T}, i_sample::Integer, w::Interval2D, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
    emds.d[w.x.x+div((w.x.y-2)*(w.x.y-1),2), w.y.x+div((w.y.y-2)*(w.y.y-1),2), i_sample, i_featsnaggr, i_relation] = threshold
function slice_dataset(emds::Interval2DFWD_RS{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    Interval2DFWD_RS{T}(if return_view @view emds.d[:,:,inds,:,:] else emds.d[:,:,inds,:,:] end)
end
