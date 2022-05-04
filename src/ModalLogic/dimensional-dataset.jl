############################################################################################
# Dimensional dataset
############################################################################################
# 
# An D-dimensional dataset is a multi-dimensional array representing a set of
#  (multi-attribute) D-dimensional instances:
# The size of the array is {X × Y × ...} × n_attributes × n_samples
# The dimensionality of the channel is denoted as N = D-1-1 (e.g. 1 for time series,
#  2 for images), and its dimensions are denoted as X, Y, Z, etc.
# 
# Note: It'd be nice to define these with N being the dimensionality of the channel:
#   e.g. const DimensionalDataset{T,N} = AbstractArray{T,N+1+1}
# Unfortunately, this is not currently allowed ( see https://github.com/JuliaLang/julia/issues/8322 )
# 
# Note: This implementation assumes that all instances have uniform channel size (e.g. time
#  series with same number of points, or images of same width and height)
############################################################################################

const DimensionalDataset{T<:Number,D}     = Array{T,D}
const DimensionalChannel{T<:Number,N}     = Array{T,N}
const DimensionalInstance{T<:Number,MN}   = Array{T,MN}

############################################################################################

n_samples(d::DimensionalDataset{T,D})        where {T,D} = size(d, D)::Int64
n_attributes(d::DimensionalDataset{T,D})     where {T,D} = size(d, D-1)::Int64
max_channel_size(d::DimensionalDataset{T,D}) where {T,D} = size(d)[1:end-2]

get_instance(d::DimensionalDataset{T,2},     idx::Integer) where T = @views d[:, idx]         # N=0
get_instance(d::DimensionalDataset{T,3},     idx::Integer) where T = @views d[:, :, idx]      # N=1
get_instance(d::DimensionalDataset{T,4},     idx::Integer) where T = @views d[:, :, :, idx]   # N=2

function slice_dataset(d::DimensionalDataset{T,2}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where T # N=0
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    if return_view @views d[:, inds]       else d[:, inds]    end
end
function slice_dataset(d::DimensionalDataset{T,3}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where T # N=1
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    if return_view @views d[:, :, inds]    else d[:, :, inds] end
end
function slice_dataset(d::DimensionalDataset{T,4}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where T # N=2
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    if return_view @views d[:, :, :, inds] else d[:, :, :, inds] end
end

concat_datasets(d1::DimensionalDataset{T,N}, d2::DimensionalDataset{T,N}) where {T,N} = cat(d1, d2; dims=N)

function get_gamma(d::DimensionalDataset{T,N}, i_instance::Integer, w::AbstractWorld, feature::ModalFeature) where {T,N}
    get_interpretation_function(feature)(interpret_world(w, get_instance(d, i_instance))::DimensionalChannel{T,N-1-1})::T
end

init_world_sets_fun(d::DimensionalDataset,  i_instance::Integer, WorldType::Type{<:AbstractWorld}) =
    (iC)->ModalDecisionTrees.init_world_set(iC, WorldType, max_channel_size(d))

############################################################################################

instance_channel_size(inst::DimensionalInstance{T,MN}) where {T,MN} = size(inst)[1:end-1]

get_instance_attribute(inst::DimensionalInstance{T,1}, idx_a::Integer) where T = @views inst[      idx_a]::T                       # N=0
get_instance_attribute(inst::DimensionalInstance{T,2}, idx_a::Integer) where T = @views inst[:,    idx_a]::DimensionalChannel{T,1} # N=1
get_instance_attribute(inst::DimensionalInstance{T,3}, idx_a::Integer) where T = @views inst[:, :, idx_a]::DimensionalChannel{T,2} # N=2

# For convenience, `accessibles` & `accessibles_aggr` work with domains OR their dimensions
accessibles(S::AbstractWorldSet, r::AbstractRelation, channel::DimensionalChannel) = accessibles(S, r, size(channel)...)
accessibles_aggr(f::ModalFeature, a::Aggregator, Sw::Any, r::AbstractRelation, channel::DimensionalChannel) = accessibles_aggr(f, a, Sw, r, size(channel)...)

############################################################################################
