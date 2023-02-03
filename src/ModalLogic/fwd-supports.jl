
############################################################################################
############################################################################################
# world-specific FWD supports implementations
############################################################################################
############################################################################################

abstract type UniformFullDimensionalRelationalSupport{T,W<:AbstractWorld} <: AbstractRelationalSupport{T,W} end

nnothing(fwd_rs::UniformFullDimensionalRelationalSupport) = count(isnothing, fwd_rs.d)
function nonnothingshare(fwd_rs::UniformFullDimensionalRelationalSupport)
    isinf(capacity(fwd_rs)) ? (0/Inf) : (1-(nnothing(fwd_rs)  / capacity(fwd_rs)))
end

############################################################################################
# FWD support, OneWorld: 3D array (nsamples × nfeatsnaggrs × nrelations)
############################################################################################

struct OneWorldFWD_RS{T} <: UniformFullDimensionalRelationalSupport{T,OneWorld}
    d :: Array{T,3}
end

nsamples(emds::OneWorldFWD_RS)     = size(emds, 1)
nfeatsnaggrs(emds::OneWorldFWD_RS) = size(emds, 2)
nrelations(emds::OneWorldFWD_RS)   = size(emds, 3)
capacity(emds::OneWorldFWD_RS)     = prod(size(emds.d))

Base.@propagate_inbounds @inline Base.getindex(
    emds         :: OneWorldFWD_RS{T},
    i_sample     :: Integer,
    w            :: OneWorld,
    i_featsnaggr :: Integer,
    i_relation   :: Integer) where {T} = emds.d[i_sample, i_featsnaggr, i_relation]
Base.size(emds::OneWorldFWD_RS, args...) = size(emds.d, args...)
goeswith(::Type{OneWorldFWD_RS}, ::Type{OneWorld}) = true

hasnans(emds::OneWorldFWD_RS) = any(_isnan.(emds.d))

fwd_rs_init(emd::ExplicitModalDataset{T,OneWorld}, nfeatsnaggrs::Integer, nrelations::Integer; perform_initialization = false) where {T} = begin
    if perform_initialization
        _fwd_rs = fill!(Array{Union{T,Nothing}, 3}(undef, nsamples(emd), nfeatsnaggrs, nrelations), nothing)
        OneWorldFWD_RS{Union{T,Nothing}}(_fwd_rs)
    else
        _fwd_rs = Array{T,3}(undef, nsamples(emd), nfeatsnaggrs, nrelations)
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


struct IntervalFWD_RS{T} <: UniformFullDimensionalRelationalSupport{T,Interval}
    d :: Array{T,5}
end

nsamples(emds::IntervalFWD_RS)     = size(emds, 3)
nfeatsnaggrs(emds::IntervalFWD_RS) = size(emds, 4)
nrelations(emds::IntervalFWD_RS)   = size(emds, 5)
capacity(emds::IntervalFWD_RS)     =
    prod([nsamples(emds.d), nfeatsnaggrs(emds), nrelations(emds), nsamples(emds), div(size(emds.d, 1)*(size(emds.d, 1)+1),2)])

Base.@propagate_inbounds @inline Base.getindex(
    emds         :: IntervalFWD_RS{T},
    i_sample     :: Integer,
    w            :: Interval,
    i_featsnaggr :: Integer,
    i_relation   :: Integer) where {T} = emds.d[w.x, w.y, i_sample, i_featsnaggr, i_relation]
Base.size(emds::IntervalFWD_RS, args...) = size(emds.d, args...)
goeswith(::Type{IntervalFWD_RS}, ::Type{Interval}) = true


hasnans(emds::IntervalFWD_RS) = begin
    # @show [hasnans(emds.d[x,y,:,:,:]) for x in 1:size(emds.d, 1) for y in (x+1):size(emds.d, 2)]
    any([hasnans(emds.d[x,y,:,:,:]) for x in 1:size(emds.d, 1) for y in (x+1):size(emds.d, 2)])
end

# Note: assuming default_fwd_type(::Type{Interval}) = IntervalFWD
fwd_rs_init(emd::ExplicitModalDataset{T,Interval}, nfeatsnaggrs::Integer, nrelations::Integer; perform_initialization = false) where {T} = begin
    if perform_initialization
        _fwd_rs = fill!(Array{Union{T,Nothing}, 5}(undef, size(emd.fwd, 1), size(emd.fwd, 2), nsamples(emd), nfeatsnaggrs, nrelations), nothing)
        IntervalFWD_RS{Union{T,Nothing}}(_fwd_rs)
    else
        _fwd_rs = Array{T,5}(undef, size(emd.fwd, 1), size(emd.fwd, 2), nsamples(emd), nfeatsnaggrs, nrelations)
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

# struct Interval2DFWD_RS{T} <: UniformFullDimensionalRelationalSupport{T,Interval2D}
#   d :: Array{T,7}
# end

# nsamples(emds::Interval2DFWD_RS)     = size(emds, 5)
# nfeatsnaggrs(emds::Interval2DFWD_RS) = size(emds, 6)
# nrelations(emds::Interval2DFWD_RS)   = size(emds, 7)
# Base.@propagate_inbounds @inline Base.getindex(
#   emds         :: Interval2DFWD_RS{T},
#   i_sample     :: Integer,
#   w            :: Interval2D,
#   i_featsnaggr :: Integer,
#   i_relation   :: Integer) where {T} = emds.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_featsnaggr, i_relation]
# size(emds::Interval2DFWD_RS) = size(emds.d, args...)
# goeswith(::Type{Interval2DFWD_RS}, ::Type{Interval2D}) = true

# TODO... hasnans(emds::Interval2DFWD_RS) = any(_isnan.(emds.d))
# TODO...? hasnans(emds::Interval2DFWD_RS) = any([hasnans(emds.d[xx,xy,yx,yy,:,:]) for xx in 1:size(emds.d, 1) for xy in (xx+1):size(emds.d, 2) for yx in 1:size(emds.d, 3) for yy in (yx+1):size(emds.d, 4)])

# fwd_rs_init(emd::ExplicitModalDataset{T,Interval2D}, nfeatsnaggrs::Integer, nrelations::Integer; perform_initialization = false) where {T} = begin
#   if perform_initialization
#       _fwd_rs = fill!(Array{Union{T,Nothing}, 7}(undef, size(emd.fwd, 1), size(emd.fwd, 2), size(emd.fwd, 3), size(emd.fwd, 4), nsamples(emd), nfeatsnaggrs, nrelations), nothing)
#       Interval2DFWD_RS{Union{T,Nothing}}(_fwd_rs)
#   else
#       _fwd_rs = Array{T,7}(undef, size(emd.fwd, 1), size(emd.fwd, 2), size(emd.fwd, 3), size(emd.fwd, 4), nsamples(emd), nfeatsnaggrs, nrelations)
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

struct Interval2DFWD_RS{T} <: UniformFullDimensionalRelationalSupport{T,Interval2D}
    d :: Array{T,5}
end

nsamples(emds::Interval2DFWD_RS)     = size(emds, 3)
nfeatsnaggrs(emds::Interval2DFWD_RS) = size(emds, 4)
nrelations(emds::Interval2DFWD_RS)   = size(emds, 5)
capacity(emds::Interval2DFWD_RS)     = prod(size(emds.d))

Base.@propagate_inbounds @inline Base.getindex(
    emds         :: Interval2DFWD_RS{T},
    i_sample     :: Integer,
    w            :: Interval2D,
    i_featsnaggr :: Integer,
    i_relation   :: Integer) where {T} = emds.d[w.x.x+div((w.x.y-2)*(w.x.y-1),2), w.y.x+div((w.y.y-2)*(w.y.y-1),2), i_sample, i_featsnaggr, i_relation]
Base.size(emds::Interval2DFWD_RS, args...) = size(emds.d, args...)
goeswith(::Type{Interval2DFWD_RS}, ::Type{Interval2D}) = true

hasnans(emds::Interval2DFWD_RS) = any(_isnan.(emds.d))

fwd_rs_init(emd::ExplicitModalDataset{T,Interval2D}, nfeatsnaggrs::Integer, nrelations::Integer; perform_initialization = false) where {T} = begin
    if perform_initialization
        _fwd_rs = fill!(Array{Union{T,Nothing}, 5}(undef, div(size(emd.fwd, 1)*size(emd.fwd, 2),2), div(size(emd.fwd, 3)*size(emd.fwd, 4),2), nsamples(emd), nfeatsnaggrs, nrelations), nothing)
        Interval2DFWD_RS{Union{T,Nothing}}(_fwd_rs)
    else
        _fwd_rs = Array{T,5}(undef, div(size(emd.fwd, 1)*size(emd.fwd, 2),2), div(size(emd.fwd, 3)*size(emd.fwd, 4),2), nsamples(emd), nfeatsnaggrs, nrelations)
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
