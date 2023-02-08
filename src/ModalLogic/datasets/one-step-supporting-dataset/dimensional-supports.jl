import Base: size, ndims, getindex


############################################################################################
############################################################################################
# world-specific FWD supports implementations
############################################################################################
############################################################################################

abstract type AbstractUniformFullDimensionalRelationalSupport{T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}} <: AbstractRelationalSupport{T,W,FR} end

# TODO switch from nothing to missing?
usesmemo(fwd_rs::AbstractUniformFullDimensionalRelationalSupport) = Nothing <: eltype(fwd_rs.d)
nnothing(fwd_rs::AbstractUniformFullDimensionalRelationalSupport) = count(isnothing, fwd_rs.d)
function nonnothingshare(fwd_rs::AbstractUniformFullDimensionalRelationalSupport)
    isinf(capacity(fwd_rs)) ? (0/Inf) : (1-(nnothing(fwd_rs)  / capacity(fwd_rs)))
end

############################################################################################
# FWD relational support for uniform full dimensional frames:
#  a (nsamples × nfeatsnaggrs × nrelations) structure for each world.
#  Each world is linearized, resulting in a (3+N*2)-D array
############################################################################################

struct UniformFullDimensionalRelationalSupport{
    T,
    W<:AbstractWorld,
    N,
    D<:Array{TT} where TT<:Union{T,Nothing},
} <: AbstractUniformFullDimensionalRelationalSupport{T,W,FullDimensionalFrame{N,W,Bool}}
    d :: D
end

Base.size(support::UniformFullDimensionalRelationalSupport, args...) = size(support.d, args...)
Base.ndims(support::UniformFullDimensionalRelationalSupport, args...) = ndims(support.d, args...)

nsamples(support::UniformFullDimensionalRelationalSupport)     = size(support, ndims(support)-2)
nfeatsnaggrs(support::UniformFullDimensionalRelationalSupport) = size(support, ndims(support)-1)
nrelations(support::UniformFullDimensionalRelationalSupport)   = size(support, ndims(support))

############################################################################################

function capacity(support::UniformFullDimensionalRelationalSupport{T,OneWorld}) where {T}
    prod(size(support))
end
function capacity(support::UniformFullDimensionalRelationalSupport{T,Interval}) where {T}
    prod([
        nsamples(support),
        nfeatsnaggrs(support),
        nrelations(support),
        div(size(support.d, 1)*(size(support.d, 1)+1),2),
    ])
end
function capacity(support::UniformFullDimensionalRelationalSupport{T,Interval2D}) where {T}
    prod([
        nsamples(support),
        nfeatsnaggrs(support),
        nrelations(support),
        div(size(support.d, 1)*(size(support.d, 1)+1),2),
        div(size(support.d, 3)*(size(support.d, 3)+1),2),
    ])
end

############################################################################################

function hasnans(support::UniformFullDimensionalRelationalSupport{T,OneWorld}) where {T}
    any(_isnan.(support.d))
end
function hasnans(support::UniformFullDimensionalRelationalSupport{T,Interval}) where {T}
    any([hasnans(support.d[x,y,:,:,:])
        for x in 1:size(support.d, 1) for y in (x+1):size(support.d, 2)])
end
function hasnans(support::UniformFullDimensionalRelationalSupport{T,Interval2D}) where {T}
    any([hasnans(support.d[xx,xy,yx,yy,:,:,:])
        for xx in 1:size(support.d, 1) for xy in (xx+1):size(support.d, 2)
        for yx in 1:size(support.d, 3) for yy in (yx+1):size(support.d, 4)])
end

############################################################################################

function fwd_rs_init_world_slice(
    support::UniformFullDimensionalRelationalSupport,
    i_sample::Integer,
    i_featsnaggr::Integer,
    i_relation::Integer
)
    nothing
end

############################################################################################

function fwd_rs_init(
    emd::ExplicitModalDataset,
    perform_initialization::Bool
)
    nfeatsnaggrs = sum(length.(grouped_featsnaggrs(emd)))
    fwd_rs_init(fwd(emd), nfeatsnaggrs, nrelations(emd), perform_initialization)
end

function fwd_rs_init(
    fwd::UniformFullDimensionalFWD{T,0,W},
    nfeatsnaggrs::Integer,
    nrelations::Integer,
    perform_initialization::Bool
) where {T,W<:OneWorld}
    # error("TODO actually, using a relational or a global support with a OneWorld frame makes no sense. Figure out what to do here!")
    _fwd_rs = begin
        if perform_initialization
            _fwd_rs = Array{Union{T,Nothing}, 3}(undef, nsamples(fwd), nfeatsnaggrs, nrelations)
            fill!(_fwd_rs, nothing)
        else
            Array{T,3}(undef, nsamples(fwd), nfeatsnaggrs, nrelations)
        end
    end
    UniformFullDimensionalRelationalSupport{T,W,0,typeof(_fwd_rs)}(_fwd_rs)
end
function fwd_rs_init(
    fwd::UniformFullDimensionalFWD{T,1,W},
    nfeatsnaggrs::Integer,
    nrelations::Integer,
    perform_initialization::Bool
) where {T,W<:Interval}
    _fwd_rs = begin
        if perform_initialization
            _fwd_rs = Array{Union{T,Nothing}, 5}(undef, size(fwd, 1), size(fwd, 2), nsamples(fwd), nfeatsnaggrs, nrelations)
            fill!(_fwd_rs, nothing)
        else
            Array{T,5}(undef, size(fwd, 1), size(fwd, 2), nsamples(fwd), nfeatsnaggrs, nrelations)
        end
    end
    UniformFullDimensionalRelationalSupport{T,W,1,typeof(_fwd_rs)}(_fwd_rs)
end
function fwd_rs_init(
    fwd::UniformFullDimensionalFWD{T,2,W},
    nfeatsnaggrs::Integer,
    nrelations::Integer,
    perform_initialization::Bool
) where {T,W<:Interval2D}
    _fwd_rs = begin
        if perform_initialization
            _fwd_rs = Array{Union{T,Nothing}, 5}(undef, size(fwd, 1), size(fwd, 2), size(fwd, 3), size(fwd, 4), nsamples(fwd), nfeatsnaggrs, nrelations)
            fill!(_fwd_rs, nothing)
        else
            Array{T,5}(undef, size(fwd, 1), size(fwd, 2), size(fwd, 3), size(fwd, 4), nsamples(fwd), nfeatsnaggrs, nrelations)
        end
    end
    UniformFullDimensionalRelationalSupport{T,W,2,typeof(_fwd_rs)}(_fwd_rs)
end

############################################################################################

Base.@propagate_inbounds @inline function Base.getindex(
    support      :: UniformFullDimensionalRelationalSupport{T,W},
    i_sample     :: Integer,
    w            :: W,
    i_featsnaggr :: Integer,
    i_relation   :: Integer
) where {T,W<:OneWorld}
    support.d[i_sample, i_featsnaggr, i_relation]
end
Base.@propagate_inbounds @inline function Base.getindex(
    support      :: UniformFullDimensionalRelationalSupport{T,W},
    i_sample     :: Integer,
    w            :: W,
    i_featsnaggr :: Integer,
    i_relation   :: Integer
) where {T,W<:Interval}
    support.d[w.x, w.y, i_sample, i_featsnaggr, i_relation]
end
Base.@propagate_inbounds @inline function Base.getindex(
    support      :: UniformFullDimensionalRelationalSupport{T,W},
    i_sample     :: Integer,
    w            :: W,
    i_featsnaggr :: Integer,
    i_relation   :: Integer
) where {T,W<:Interval2D}
    support.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_featsnaggr, i_relation]
end

############################################################################################

Base.@propagate_inbounds @inline function fwd_rs_set(
    support::UniformFullDimensionalRelationalSupport{T,OneWorld},
    i_sample::Integer,
    w::OneWorld,
    i_featsnaggr::Integer,
    i_relation::Integer,
    threshold::T
) where {T}
    support.d[i_sample, i_featsnaggr, i_relation] = threshold
end

Base.@propagate_inbounds @inline function fwd_rs_set(
    support::UniformFullDimensionalRelationalSupport{T,Interval},
    i_sample::Integer,
    w::Interval,
    i_featsnaggr::Integer,
    i_relation::Integer,
    threshold::T
) where {T}
    support.d[w.x, w.y, i_sample, i_featsnaggr, i_relation] = threshold
end

Base.@propagate_inbounds @inline function fwd_rs_set(
    support::UniformFullDimensionalRelationalSupport{T,Interval2D},
    i_sample::Integer,
    w::Interval2D,
    i_featsnaggr::Integer,
    i_relation::Integer,
    threshold::T
) where {T}
    support.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_featsnaggr, i_relation] = threshold
end

############################################################################################

function _slice_dataset(
    support::RS,
    inds::AbstractVector{<:Integer},
    return_view::Val = Val(false)
) where {T,RS<:UniformFullDimensionalRelationalSupport{T,OneWorld}}
    RS(if return_view == Val(true) @view support.d[inds,:,:] else support.d[inds,:,:] end)
end
function _slice_dataset(
    support::RS,
    inds::AbstractVector{<:Integer},
    return_view::Val = Val(false)
) where {T,RS<:UniformFullDimensionalRelationalSupport{T,Interval}}
    RS(if return_view == Val(true) @view support.d[:,:,inds,:,:] else support.d[:,:,inds,:,:] end)
end
function _slice_dataset(
    support::RS,
    inds::AbstractVector{<:Integer},
    return_view::Val = Val(false)
) where {T,RS<:UniformFullDimensionalRelationalSupport{T,Interval2D}}
    RS(if return_view == Val(true) @view support.d[:,:,:,:,inds,:,:] else support.d[:,:,:,:,inds,:,:] end)
end

############################################################################################
# FWD support, OneWorld: 3D array (nsamples × nfeatsnaggrs × nrelations)
############################################################################################

# struct OneWorldFWD_RS{T} <: AbstractUniformFullDimensionalRelationalSupport{T,OneWorld}
#     d :: Array{T,3}
# end

# nsamples(support::OneWorldFWD_RS)     = size(support, 1)
# nfeatsnaggrs(support::OneWorldFWD_RS) = size(support, 2)
# nrelations(support::OneWorldFWD_RS)   = size(support, 3)
# capacity(support::OneWorldFWD_RS)     = prod(size(support.d))

# Base.@propagate_inbounds @inline Base.getindex(
#     support      :: OneWorldFWD_RS{T},
#     i_sample     :: Integer,
#     w            :: OneWorld,
#     i_featsnaggr :: Integer,
#     i_relation   :: Integer) where {T} = support.d[i_sample, i_featsnaggr, i_relation]
# Base.size(support::OneWorldFWD_RS, args...) = size(support.d, args...)

# hasnans(support::OneWorldFWD_RS) = any(_isnan.(support.d))

# function fwd_rs_init(emd::ExplicitModalDataset{T,OneWorld}, nfeatsnaggrs::Integer, nrelations::Integer, perform_initialization::Bool) where {T}
#     if perform_initialization
#         _fwd_rs = fill!(Array{Union{T,Nothing}, 3}(undef, nsamples(emd), nfeatsnaggrs, nrelations), nothing)
#         OneWorldFWD_RS{Union{T,Nothing}}(_fwd_rs)
#     else
#         _fwd_rs = Array{T,3}(undef, nsamples(emd), nfeatsnaggrs, nrelations)
#         OneWorldFWD_RS{T}(_fwd_rs)
#     end
# end
# fwd_rs_init_world_slice(support::OneWorldFWD_RS, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) =
#     nothing
# Base.@propagate_inbounds @inline fwd_rs_set(support::OneWorldFWD_RS{T}, i_sample::Integer, w::OneWorld, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
#     support.d[i_sample, i_featsnaggr, i_relation] = threshold
# function _slice_dataset(support::OneWorldFWD_RS{T}, inds::AbstractVector{<:Integer}, return_view::Val = Val(false)) where {T}
#     OneWorldFWD_RS{T}(if return_view == Val(true) @view support.d[inds,:,:] else support.d[inds,:,:] end)
# end

############################################################################################
# FWD support, Interval: 5D array (x × y × nsamples × nfeatsnaggrs × nrelations)
############################################################################################


# struct IntervalFWD_RS{T} <: AbstractUniformFullDimensionalRelationalSupport{T,Interval}
#     d :: Array{T,5}
# end

# nsamples(support::IntervalFWD_RS)     = size(support, 3)
# nfeatsnaggrs(support::IntervalFWD_RS) = size(support, 4)
# nrelations(support::IntervalFWD_RS)   = size(support, 5)
# capacity(support::IntervalFWD_RS)     =
#     prod([nsamples(support), nfeatsnaggrs(support), nrelations(support), div(size(support.d, 1)*(size(support.d, 1)+1),2)])

# Base.@propagate_inbounds @inline Base.getindex(
#     support      :: IntervalFWD_RS{T},
#     i_sample     :: Integer,
#     w            :: Interval,
#     i_featsnaggr :: Integer,
#     i_relation   :: Integer) where {T} = support.d[w.x, w.y, i_sample, i_featsnaggr, i_relation]
# Base.size(support::IntervalFWD_RS, args...) = size(support.d, args...)


# function hasnans(support::IntervalFWD_RS)
#     # @show [hasnans(support.d[x,y,:,:,:]) for x in 1:size(support.d, 1) for y in (x+1):size(support.d, 2)]
#     any([hasnans(support.d[x,y,:,:,:]) for x in 1:size(support.d, 1) for y in (x+1):size(support.d, 2)])
# end

# # Note: assuming default_fwd_type(::Type{Interval}) = IntervalFWD
# function fwd_rs_init(emd::ExplicitModalDataset{T,Interval}, nfeatsnaggrs::Integer, nrelations::Integer, perform_initialization::Bool) where {T}
#     _fwd = emd.fwd
#     if perform_initialization
#         _fwd_rs = fill!(Array{Union{T,Nothing}, 5}(undef, size(_fwd, 1), size(_fwd, 2), nsamples(emd), nfeatsnaggrs, nrelations), nothing)
#         IntervalFWD_RS{Union{T,Nothing}}(_fwd_rs)
#     else
#         _fwd_rs = Array{T,5}(undef, size(_fwd, 1), size(_fwd, 2), nsamples(emd), nfeatsnaggrs, nrelations)
#         IntervalFWD_RS{T}(_fwd_rs)
#     end
# end
# fwd_rs_init_world_slice(support::IntervalFWD_RS, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) =
#     nothing
# Base.@propagate_inbounds @inline fwd_rs_set(support::IntervalFWD_RS{T}, i_sample::Integer, w::Interval, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
#     support.d[w.x, w.y, i_sample, i_featsnaggr, i_relation] = threshold
# function _slice_dataset(support::IntervalFWD_RS{T}, inds::AbstractVector{<:Integer}, return_view::Val = Val(false)) where {T}
#     IntervalFWD_RS{T}(if return_view == Val(true) @view support.d[:,:,inds,:,:] else support.d[:,:,inds,:,:] end)
# end

############################################################################################
# FWD support, Interval2D: 7D array (x.x × x.y × y.x × y.y × nsamples × nfeatsnaggrs × nrelations)
############################################################################################

# struct Interval2DFWD_RS{T} <: AbstractUniformFullDimensionalRelationalSupport{T,Interval2D}
#   d :: Array{T,7}
# end

# nsamples(support::Interval2DFWD_RS)     = size(support, 5)
# nfeatsnaggrs(support::Interval2DFWD_RS) = size(support, 6)
# nrelations(support::Interval2DFWD_RS)   = size(support, 7)
# Base.@propagate_inbounds @inline Base.getindex(
#   support      :: Interval2DFWD_RS{T},
#   i_sample     :: Integer,
#   w            :: Interval2D,
#   i_featsnaggr :: Integer,
#   i_relation   :: Integer) where {T} = support.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_featsnaggr, i_relation]
# size(support::Interval2DFWD_RS) = size(support.d, args...)

# TODO... hasnans(support::Interval2DFWD_RS) = any(_isnan.(support.d))
# TODO...? hasnans(support::Interval2DFWD_RS) = any([hasnans(support.d[xx,xy,yx,yy,:,:,:]) for xx in 1:size(support.d, 1) for xy in (xx+1):size(support.d, 2) for yx in 1:size(support.d, 3) for yy in (yx+1):size(support.d, 4)])

# fwd_rs_init(emd::ExplicitModalDataset{T,Interval2D}, nfeatsnaggrs::Integer, nrelations::Integer, perform_initialization::Bool) where {T} = begin
#   _fwd = emd.fwd
#   if perform_initialization
#       _fwd_rs = fill!(Array{Union{T,Nothing}, 7}(undef, size(_fwd, 1), size(_fwd, 2), size(_fwd, 3), size(_fwd, 4), nsamples(emd), nfeatsnaggrs, nrelations), nothing)
#       Interval2DFWD_RS{Union{T,Nothing}}(_fwd_rs)
#   else
#       _fwd_rs = Array{T,7}(undef, size(_fwd, 1), size(_fwd, 2), size(_fwd, 3), size(_fwd, 4), nsamples(emd), nfeatsnaggrs, nrelations)
#       Interval2DFWD_RS{T}(_fwd_rs)
#   end
# end
# fwd_rs_init_world_slice(support::Interval2DFWD_RS, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) =
#   nothing
# Base.@propagate_inbounds @inline fwd_rs_set(support::Interval2DFWD_RS{T}, i_sample::Integer, w::Interval2D, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
#   support.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_featsnaggr, i_relation] = threshold
# function _slice_dataset(support::Interval2DFWD_RS{T}, inds::AbstractVector{<:Integer}, return_view::Val = Val(false)) where {T}
#   Interval2DFWD_RS{T}(if return_view == Val(true) @view support.d[:,:,:,:,inds,:,:] else support.d[:,:,:,:,inds,:,:] end)
# end


############################################################################################
# FWD support, Interval2D: 7D array (linearized(x) × linearized(y) × nsamples × nfeatsnaggrs × nrelations)
############################################################################################

# # TODO rewrite

# struct Interval2DFWD_RS{T} <: AbstractUniformFullDimensionalRelationalSupport{T,Interval2D}
#     d :: Array{T,5}
# end

# nsamples(support::Interval2DFWD_RS)     = size(support, 3)
# nfeatsnaggrs(support::Interval2DFWD_RS) = size(support, 4)
# nrelations(support::Interval2DFWD_RS)   = size(support, 5)
# capacity(support::Interval2DFWD_RS)     = prod(size(support.d))

# Base.@propagate_inbounds @inline Base.getindex(
#     support      :: Interval2DFWD_RS{T},
#     i_sample     :: Integer,
#     w            :: Interval2D,
#     i_featsnaggr :: Integer,
#     i_relation   :: Integer) where {T} = support.d[w.x.x+div((w.x.y-2)*(w.x.y-1),2), w.y.x+div((w.y.y-2)*(w.y.y-1),2), i_sample, i_featsnaggr, i_relation]
# Base.size(support::Interval2DFWD_RS, args...) = size(support.d, args...)

# hasnans(support::Interval2DFWD_RS) = any(_isnan.(support.d))

# function fwd_rs_init(emd::ExplicitModalDataset{T,Interval2D}, nfeatsnaggrs::Integer, nrelations::Integer, perform_initialization::Bool) where {T}
#     _fwd = emd.fwd
#     if perform_initialization
#         _fwd_rs = fill!(Array{Union{T,Nothing}, 5}(undef, div(size(_fwd, 1)*size(_fwd, 2),2), div(size(_fwd, 3)*size(_fwd, 4),2), nsamples(emd), nfeatsnaggrs, nrelations), nothing)
#         Interval2DFWD_RS{Union{T,Nothing}}(_fwd_rs)
#     else
#         _fwd_rs = Array{T,5}(undef, div(size(_fwd, 1)*size(_fwd, 2),2), div(size(_fwd, 3)*size(_fwd, 4),2), nsamples(emd), nfeatsnaggrs, nrelations)
#         Interval2DFWD_RS{T}(_fwd_rs)
#     end
# end
# fwd_rs_init_world_slice(support::Interval2DFWD_RS, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) =
#     nothing
# Base.@propagate_inbounds @inline fwd_rs_set(support::Interval2DFWD_RS{T}, i_sample::Integer, w::Interval2D, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
#     support.d[w.x.x+div((w.x.y-2)*(w.x.y-1),2), w.y.x+div((w.y.y-2)*(w.y.y-1),2), i_sample, i_featsnaggr, i_relation] = threshold
# function _slice_dataset(support::Interval2DFWD_RS{T}, inds::AbstractVector{<:Integer}, return_view::Val = Val(false)) where {T}
#     Interval2DFWD_RS{T}(if return_view == Val(true) @view support.d[:,:,inds,:,:] else support.d[:,:,inds,:,:] end)
# end
