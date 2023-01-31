
abstract type AbstractFWD{T<:Number,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}} end

# Any implementation for a fwd must indicate their compatible world types via `goeswith`.
# Fallback:
goeswith(::Type{<:AbstractFWD}, ::Type{<:AbstractWorld}) = false

# # A function for getting a threshold value from the lookup table
Base.getindex(
    fwd         :: AbstractFWD{T},
    i_sample    :: Integer,
    w           :: AbstractWorld,
    i_feature   :: Integer) where {T} = fwd_get(fwd, i_sample, w, i_feature)

# Any world type must also specify their default fwd constructor, which must accept a type
#  parameter for the data type {T}, via:
# default_fwd_type(::Type{<:AbstractWorld})

# 
# Actually, the interface for AbstractFWD's is a bit tricky; the most straightforward
#  way of learning it is by considering the fallback fwd structure defined as follows.
# TODO oh, but the implementation is broken due to a strange error (see https://discourse.julialang.org/t/tricky-too-many-parameters-for-type-error/25182 )

# # The most generic fwd structure is a matrix of dictionaries of size (nsamples × nfeatures)
# struct GenericFWD{T,W} <: AbstractFWD{T,W}
#   d :: AbstractVector{<:AbstractDict{W,AbstractVector{T,1}},1}
#   nfeatures :: Integer
# end

# # It goes for any world type
# goeswith(::Type{<:GenericFWD}, ::Type{<:AbstractWorld}) = true

# # And it is the default fwd structure for an world type
# default_fwd_type(::Type{<:AbstractWorld}) = GenericFWD

# nsamples(fwd::GenericFWD{T}) where {T}  = size(fwd, 1)
# nfeatures(fwd::GenericFWD{T}) where {T} = fwd.d
# Base.size(fwd::GenericFWD{T}, args...) where {T} = size(fwd.d, args...)

# # The matrix is initialized with #undef values
# function fwd_init(::Type{GenericFWD}, imd::InterpretedModalDataset{T}) where {T}
#     d = Array{Dict{W,T}, 2}(undef, nsamples(imd))
#     for i in 1:nsamples
#         d[i] = Dict{W,Array{T,1}}()
#     end
#     GenericFWD{T}(d, nfeatures(imd))
# end

# # A function for initializing individual world slices
# function fwd_init_world_slice(fwd::GenericFWD{T}, i_sample::Integer, w::AbstractWorld) where {T}
#     fwd.d[i_sample][w] = Array{T,1}(undef, fwd.nfeatures)
# end

# # A function for getting a threshold value from the lookup table
# Base.@propagate_inbounds @inline fwd_get(
#     fwd         :: GenericFWD{T},
#     i_sample    :: Integer,
#     w           :: AbstractWorld,
#     i_feature   :: Integer) where {T} = fwd.d[i_sample][w][i_feature]

# # A function for setting a threshold value in the lookup table
# Base.@propagate_inbounds @inline function fwd_set(fwd::GenericFWD{T}, w::AbstractWorld, i_sample::Integer, i_feature::Integer, threshold::T) where {T}
#     fwd.d[i_sample][w][i_feature] = threshold
# end

# # A function for setting threshold values for a single feature (from a feature slice, experimental)
# Base.@propagate_inbounds @inline function fwd_set_feature(fwd::GenericFWD{T}, i_feature::Integer, fwd_feature_slice::Any) where {T}
#     throw_n_log("Warning! fwd_set_feature with GenericFWD is not yet implemented!")
#     for ((i_sample,w),threshold::T) in read_fwd_feature_slice(fwd_feature_slice)
#         fwd.d[i_sample][w][i_feature] = threshold
#     end
# end

# # A function for slicing the dataset
# function slice_dataset(fwd::GenericFWD{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
#     @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
#     GenericFWD{T}(if return_view @view fwd.d[inds] else fwd.d[inds] end, fwd.nfeatures)
# end

# Others...
# Base.@propagate_inbounds @inline fwd_get_channel(fwd::GenericFWD{T}, i_sample::Integer, i_feature::Integer) where {T} = TODO
# const GenericFeaturedChannel{T} = TODO
# fwd_channel_interpret_world(fwc::GenericFeaturedChannel{T}, w::AbstractWorld) where {T} = TODO
