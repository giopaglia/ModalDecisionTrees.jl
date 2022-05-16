############################################################################################
############################################################################################
# world-specific FWD implementations
############################################################################################
############################################################################################

############################################################################################
# FWD, OneWorld: 2D array (n_samples × n_features)
############################################################################################

struct OneWorldFWD{T} <: AbstractFWD{T, OneWorld}
    d :: Array{T, 2}
end

goes_with(::Type{OneWorldFWD}, ::Type{OneWorld}) = true
default_fwd_type(::Type{OneWorld}) = OneWorldFWD

n_samples(fwd::OneWorldFWD{T}) where {T}  = size(fwd, 1)
n_features(fwd::OneWorldFWD{T}) where {T} = size(fwd, 2)
Base.size(fwd::OneWorldFWD{T}, args...) where {T} = size(fwd.d, args...)

function fwd_init(::Type{OneWorldFWD}, X::InterpretedModalDataset{T, 0, OneWorld}) where {T}
    OneWorldFWD{T}(Array{T, 2}(undef, n_samples(X), n_features(X)))
end

function fwd_init_world_slice(fwd::OneWorldFWD{T}, args...) where {T}
    nothing
end

fwd_get(
    fwd         :: OneWorldFWD{T},
    i_sample  :: Integer,
    w           :: OneWorld,
    i_feature   :: Integer) where {T} = fwd.d[i_sample, i_feature]

function fwd_set(fwd::OneWorldFWD{T}, w::OneWorld, i_sample::Integer, i_feature::Integer, threshold::T) where {T}
    fwd.d[i_sample, i_feature] = threshold
end

function fwd_set_feature_slice(fwd::OneWorldFWD{T}, i_feature::Integer, feature_fwd::Array{T, 1}) where {T}
    fwd.d[:, i_feature] = feature_fwd
end

function slice_dataset(fwd::OneWorldFWD{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    OneWorldFWD{T}(if return_view @view fwd.d[inds,:] else fwd.d[inds,:] end)
end

fwd_get_channel(fwd::OneWorldFWD{T}, i_sample::Integer, i_feature::Integer) where {T} =
    fwd.d[i_sample, i_feature]
const OneWorldFeaturedChannel{T} = T
fwd_channel_interpret_world(fwc::T #=Note: should be OneWorldFeaturedChannel{T}, but it throws error =#, w::OneWorld) where {T} = fwc

############################################################################################
# FWD, Interval: 4D array (x × y × n_samples × n_features)
############################################################################################

struct IntervalFWD{T} <: AbstractFWD{T, Interval}
    d :: Array{T, 4}
end

goes_with(::Type{IntervalFWD}, ::Type{Interval}) = true
default_fwd_type(::Type{Interval}) = IntervalFWD

n_samples(fwd::IntervalFWD{T}) where {T}  = size(fwd, 3)
n_features(fwd::IntervalFWD{T}) where {T} = size(fwd, 4)
Base.size(fwd::IntervalFWD{T}, args...) where {T} = size(fwd.d, args...)

function fwd_init(::Type{IntervalFWD}, X::InterpretedModalDataset{T, 1, Interval}) where {T}
    IntervalFWD{T}(Array{T, 4}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_features(X)))
end

function fwd_init_world_slice(fwd::IntervalFWD{T}, args...) where {T}
    nothing
end

fwd_get(
    fwd         :: IntervalFWD{T},
    i_sample  :: Integer,
    w           :: Interval,
    i_feature   :: Integer) where {T} = fwd.d[w.x, w.y, i_sample, i_feature]

function fwd_set(fwd::IntervalFWD{T}, w::Interval, i_sample::Integer, i_feature::Integer, threshold::T) where {T}
    fwd.d[w.x, w.y, i_sample, i_feature] = threshold
end

function fwd_set_feature_slice(fwd::IntervalFWD{T}, i_feature::Integer, feature_fwd::Array{T, 3}) where {T}
    fwd.d[:, :, :, i_feature] = feature_fwd
end

function slice_dataset(fwd::IntervalFWD{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    IntervalFWD{T}(if return_view @view fwd.d[:,:,inds,:] else fwd.d[:,:,inds,:] end)
end
fwd_get_channel(fwd::IntervalFWD{T}, i_sample::Integer, i_feature::Integer) where {T} =
    @views fwd.d[:,:,i_sample, i_feature]
const IntervalFeaturedChannel{T} = AbstractArray{T, 2}
fwd_channel_interpret_world(fwc::IntervalFeaturedChannel{T}, w::Interval) where {T} =
    fwc[w.x, w.y]

############################################################################################
# FWD, Interval: 6D array (x.x × x.y × y.x × y.y × n_samples × n_features)
############################################################################################

struct Interval2DFWD{T} <: AbstractFWD{T, Interval2D}
    d :: Array{T, 6}
end

goes_with(::Type{Interval2DFWD}, ::Type{Interval2D}) = true
default_fwd_type(::Type{Interval2D}) = Interval2DFWD

n_samples(fwd::Interval2DFWD{T}) where {T}  = size(fwd, 5)
n_features(fwd::Interval2DFWD{T}) where {T} = size(fwd, 6)
Base.size(fwd::Interval2DFWD{T}, args...) where {T} = size(fwd.d, args...)


function fwd_init(::Type{Interval2DFWD}, X::InterpretedModalDataset{T, 2, Interval2D}) where {T}
    Interval2DFWD{T}(Array{T, 6}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, max_channel_size(X)[2], max_channel_size(X)[2]+1, n_samples(X), n_features(X)))
end

function fwd_init_world_slice(fwd::Interval2DFWD{T}, args...) where {T}
    nothing
end

fwd_get(
    fwd         :: Interval2DFWD{T},
    i_sample  :: Integer,
    w           :: Interval2D,
    i_feature   :: Integer) where {T} = fwd.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_feature]

function fwd_set(fwd::Interval2DFWD{T}, w::Interval2D, i_sample::Integer, i_feature::Integer, threshold::T) where {T}
    fwd.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_feature] = threshold
end

function fwd_set_feature_slice(fwd::Interval2DFWD{T}, i_feature::Integer, feature_fwd::Array{T, 5}) where {T}
    fwd.d[:, :, :, :, :, i_feature] = feature_fwd
end

function slice_dataset(fwd::Interval2DFWD{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    Interval2DFWD{T}(if return_view @view fwd.d[:,:,:,:,inds,:] else fwd.d[:,:,:,:,inds,:] end)
end
fwd_get_channel(fwd::Interval2DFWD{T}, i_sample::Integer, i_feature::Integer) where {T} =
    @views fwd.d[:,:,:,:,i_sample, i_feature]
const Interval2DFeaturedChannel{T} = AbstractArray{T, 4}
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
# FWD support, OneWorld: 3D array (n_samples × n_featsnaggrs × n_relations)
############################################################################################

struct OneWorldFWD_RS{T} <: AbstractRelationalSupport{T, OneWorld}
    d :: AbstractArray{T, 3}
end

n_samples(emds::OneWorldFWD_RS{T}) where {T}     = size(emds, 1)
n_featsnaggrs(emds::OneWorldFWD_RS{T}) where {T} = size(emds, 2)
n_relations(emds::OneWorldFWD_RS{T}) where {T}   = size(emds, 3)
Base.getindex(
    emds         :: OneWorldFWD_RS{T},
    i_sample     :: Integer,
    w            :: OneWorld,
    i_featsnaggr :: Integer,
    i_relation   :: Integer) where {T} = emds.d[i_sample, i_featsnaggr, i_relation]
Base.size(emds::OneWorldFWD_RS{T}, args...) where {T} = size(emds.d, args...)
goes_with(::Type{OneWorldFWD_RS}, ::Type{OneWorld}) = true

fwd_rs_init(emd::ExplicitModalDataset{T, OneWorld}, n_featsnaggrs::Integer, n_relations::Integer; perform_initialization = false) where {T} = begin
    if perform_initialization
        _fwd_rs = fill!(Array{Union{T,Nothing}, 3}(undef, n_samples(emd), n_featsnaggrs, n_relations), nothing)
        OneWorldFWD_RS{Union{T,Nothing}}(_fwd_rs)
    else
        _fwd_rs = Array{T, 3}(undef, n_samples(emd), n_featsnaggrs, n_relations)
        OneWorldFWD_RS{T}(_fwd_rs)
    end
end
fwd_rs_init_world_slice(emds::OneWorldFWD_RS, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) =
    nothing
fwd_rs_set(emds::OneWorldFWD_RS{T}, i_sample::Integer, w::OneWorld, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
    emds.d[i_sample, i_featsnaggr, i_relation] = threshold
function slice_dataset(emds::OneWorldFWD_RS{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    OneWorldFWD_RS{T}(if return_view @view emds.d[inds,:,:] else emds.d[inds,:,:] end)
end

############################################################################################
# FWD support, Interval: 5D array (x × y × n_samples × n_featsnaggrs × n_relations)
############################################################################################


struct IntervalFWD_RS{T} <: AbstractRelationalSupport{T, Interval}
    d :: AbstractArray{T, 5}
end

n_samples(emds::IntervalFWD_RS{T}) where {T}     = size(emds, 3)
n_featsnaggrs(emds::IntervalFWD_RS{T}) where {T} = size(emds, 4)
n_relations(emds::IntervalFWD_RS{T}) where {T}   = size(emds, 5)
Base.getindex(
    emds         :: IntervalFWD_RS{T},
    i_sample     :: Integer,
    w            :: Interval,
    i_featsnaggr :: Integer,
    i_relation   :: Integer) where {T} = emds.d[w.x, w.y, i_sample, i_featsnaggr, i_relation]
Base.size(emds::IntervalFWD_RS{T}, args...) where {T} = size(emds.d, args...)
goes_with(::Type{IntervalFWD_RS}, ::Type{Interval}) = true

# Note: assuming default_fwd_type(::Type{Interval}) = IntervalFWD
fwd_rs_init(emd::ExplicitModalDataset{T, Interval}, n_featsnaggrs::Integer, n_relations::Integer; perform_initialization = false) where {T} = begin
    if perform_initialization
        _fwd_rs = fill!(Array{Union{T,Nothing}, 5}(undef, size(emd.fwd, 1), size(emd.fwd, 2), n_samples(emd), n_featsnaggrs, n_relations), nothing)
        IntervalFWD_RS{Union{T,Nothing}}(_fwd_rs)
    else
        _fwd_rs = Array{T, 5}(undef, size(emd.fwd, 1), size(emd.fwd, 2), n_samples(emd), n_featsnaggrs, n_relations)
        IntervalFWD_RS{T}(_fwd_rs)
    end
end
fwd_rs_init_world_slice(emds::IntervalFWD_RS, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) =
    nothing
fwd_rs_set(emds::IntervalFWD_RS{T}, i_sample::Integer, w::Interval, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
    emds.d[w.x, w.y, i_sample, i_featsnaggr, i_relation] = threshold
function slice_dataset(emds::IntervalFWD_RS{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    IntervalFWD_RS{T}(if return_view @view emds.d[:,:,inds,:,:] else emds.d[:,:,inds,:,:] end)
end

############################################################################################
# FWD support, Interval2D: 7D array (x.x × x.y × y.x × y.y × n_samples × n_featsnaggrs × n_relations)
############################################################################################

# struct Interval2DFWD_RS{T} <: AbstractRelationalSupport{T, Interval2D}
#   d :: AbstractArray{T, 7}
# end

# n_samples(emds::Interval2DFWD_RS{T}) where {T}     = size(emds, 5)
# n_featsnaggrs(emds::Interval2DFWD_RS{T}) where {T} = size(emds, 6)
# n_relations(emds::Interval2DFWD_RS{T}) where {T}   = size(emds, 7)
# getindex(
#   emds         :: Interval2DFWD_RS{T},
#   i_sample     :: Integer,
#   w            :: Interval2D,
#   i_featsnaggr :: Integer,
#   i_relation   :: Integer) where {T} = emds.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_featsnaggr, i_relation]
# size(emds::Interval2DFWD_RS{T}, args...) where {T} = size(emds.d, args...)
# goes_with(::Type{Interval2DFWD_RS}, ::Type{Interval2D}) = true

# fwd_rs_init(emd::ExplicitModalDataset{T, Interval2D}, n_featsnaggrs::Integer, n_relations::Integer; perform_initialization = false) where {T} = begin
#   if perform_initialization
#       _fwd_rs = fill!(Array{Union{T,Nothing}, 7}(undef, size(emd.fwd, 1), size(emd.fwd, 2), size(emd.fwd, 3), size(emd.fwd, 4), n_samples(emd), n_featsnaggrs, n_relations), nothing)
#       Interval2DFWD_RS{Union{T,Nothing}}(_fwd_rs)
#   else
#       _fwd_rs = Array{T, 7}(undef, size(emd.fwd, 1), size(emd.fwd, 2), size(emd.fwd, 3), size(emd.fwd, 4), n_samples(emd), n_featsnaggrs, n_relations)
#       Interval2DFWD_RS{T}(_fwd_rs)
#   end
# end
# fwd_rs_init_world_slice(emds::Interval2DFWD_RS, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) =
#   nothing
# fwd_rs_set(emds::Interval2DFWD_RS{T}, i_sample::Integer, w::Interval2D, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
#   emds.d[w.x.x, w.x.y, w.y.x, w.y.y, i_sample, i_featsnaggr, i_relation] = threshold
# function slice_dataset(emds::Interval2DFWD_RS{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
# @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
#   Interval2DFWD_RS{T}(if return_view @view emds.d[:,:,:,:,inds,:,:] else emds.d[:,:,:,:,inds,:,:] end)
# end


############################################################################################
# FWD support, Interval2D: 7D array (linearized(x) × linearized(y) × n_samples × n_featsnaggrs × n_relations)
############################################################################################

struct Interval2DFWD_RS{T} <: AbstractRelationalSupport{T, Interval2D}
    d :: AbstractArray{T, 5}
end

n_samples(emds::Interval2DFWD_RS{T}) where {T}     = size(emds, 3)
n_featsnaggrs(emds::Interval2DFWD_RS{T}) where {T} = size(emds, 4)
n_relations(emds::Interval2DFWD_RS{T}) where {T}   = size(emds, 5)
Base.getindex(
    emds         :: Interval2DFWD_RS{T},
    i_sample     :: Integer,
    w            :: Interval2D,
    i_featsnaggr :: Integer,
    i_relation   :: Integer) where {T} = emds.d[w.x.x+div((w.x.y-2)*(w.x.y-1),2), w.y.x+div((w.y.y-2)*(w.y.y-1),2), i_sample, i_featsnaggr, i_relation]
Base.size(emds::Interval2DFWD_RS{T}, args...) where {T} = size(emds.d, args...)
goes_with(::Type{Interval2DFWD_RS}, ::Type{Interval2D}) = true

fwd_rs_init(emd::ExplicitModalDataset{T, Interval2D}, n_featsnaggrs::Integer, n_relations::Integer; perform_initialization = false) where {T} = begin
    if perform_initialization
        _fwd_rs = fill!(Array{Union{T,Nothing}, 5}(undef, div(size(emd.fwd, 1)*size(emd.fwd, 2),2), div(size(emd.fwd, 3)*size(emd.fwd, 4),2), n_samples(emd), n_featsnaggrs, n_relations), nothing)
        Interval2DFWD_RS{Union{T,Nothing}}(_fwd_rs)
    else
        _fwd_rs = Array{T, 5}(undef, div(size(emd.fwd, 1)*size(emd.fwd, 2),2), div(size(emd.fwd, 3)*size(emd.fwd, 4),2), n_samples(emd), n_featsnaggrs, n_relations)
        Interval2DFWD_RS{T}(_fwd_rs)
    end
end
fwd_rs_init_world_slice(emds::Interval2DFWD_RS, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) =
    nothing
fwd_rs_set(emds::Interval2DFWD_RS{T}, i_sample::Integer, w::Interval2D, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
    emds.d[w.x.x+div((w.x.y-2)*(w.x.y-1),2), w.y.x+div((w.y.y-2)*(w.y.y-1),2), i_sample, i_featsnaggr, i_relation] = threshold
function slice_dataset(emds::Interval2DFWD_RS{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    Interval2DFWD_RS{T}(if return_view @view emds.d[:,:,inds,:,:] else emds.d[:,:,inds,:,:] end)
end
