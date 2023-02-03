
struct GenericRelationalSupport{T,W} <: AbstractRelationalSupport{T,W}
    d :: AbstractArray{Dict{W,T}, 3}
end

goeswith(::Type{GenericRelationalSupport}, ::Type{<:AbstractWorld}) = true
# default_fwd_rs_type(::Type{<:AbstractWorld}) = GenericRelationalSupport # TODO implement similar pattern used for fwd

hasnans(emds::GenericRelationalSupport) = begin
    # @show any(map(d->(any(_isnan.(collect(values(d))))), emds.d))
    any(map(d->(any(_isnan.(collect(values(d))))), emds.d))
end

nsamples(emds::GenericRelationalSupport)     = size(emds, 1)
nfeatsnaggrs(emds::GenericRelationalSupport) = size(emds, 2)
nrelations(emds::GenericRelationalSupport)   = size(emds, 3)
capacity(emds::GenericRelationalSupport)     = Inf

Base.getindex(
    emds         :: GenericRelationalSupport{T,W},
    i_sample     :: Integer,
    w            :: W,
    i_featsnaggr :: Integer,
    i_relation   :: Integer) where {T,W<:AbstractWorld} = emds.d[i_sample, i_featsnaggr, i_relation][w]
Base.size(emds::GenericRelationalSupport, args...) = size(emds.d, args...)

fwd_rs_init(emd::ExplicitModalDataset{T,W}, nfeatsnaggrs::Integer, nrelations::Integer; perform_initialization = false) where {T,W} = begin
    if perform_initialization
        _fwd_rs = fill!(Array{Dict{W,Union{T,Nothing}}, 3}(undef, nsamples(emd), nfeatsnaggrs, nrelations), nothing)
        GenericRelationalSupport{Union{T,Nothing}, W}(_fwd_rs)
    else
        _fwd_rs = Array{Dict{W,T}, 3}(undef, nsamples(emd), nfeatsnaggrs, nrelations)
        GenericRelationalSupport{T,W}(_fwd_rs)
    end
end
fwd_rs_init_world_slice(emds::GenericRelationalSupport{T,W}, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) where {T,W} =
    emds.d[i_sample, i_featsnaggr, i_relation] = Dict{W,T}()
fwd_rs_set(emds::GenericRelationalSupport{T,W}, i_sample::Integer, w::AbstractWorld, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T,W} =
    emds.d[i_sample, i_featsnaggr, i_relation][w] = threshold
function slice_dataset(emds::GenericRelationalSupport{T,W}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T,W}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    GenericRelationalSupport{T,W}(if return_view @view emds.d[inds,:,:] else emds.d[inds,:,:] end)
end

############################################################################################

# Note: the global support is world-agnostic
struct GenericGlobalSupport{T} <: AbstractGlobalSupport{T}
    d :: AbstractArray{T,2}
end

goeswith(::Type{AbstractGlobalSupport}, ::Type{<:AbstractWorld}) = true
# default_fwd_gs_type(::Type{<:AbstractWorld}) = GenericGlobalSupport # TODO implement similar pattern used for fwd

hasnans(emds::GenericGlobalSupport) = begin
    # @show any(_isnan.(emds.d))
    any(_isnan.(emds.d))
end

nsamples(emds::GenericGlobalSupport{T}) where {T}  = size(emds, 1)
nfeatsnaggrs(emds::GenericGlobalSupport{T}) where {T} = size(emds, 2)
Base.getindex(
    emds         :: GenericGlobalSupport{T},
    i_sample     :: Integer,
    i_featsnaggr  :: Integer) where {T} = emds.d[i_sample, i_featsnaggr]
Base.size(emds::GenericGlobalSupport{T}, args...) where {T} = size(emds.d, args...)

fwd_gs_init(emd::ExplicitModalDataset{T}, nfeatsnaggrs::Integer) where {T} =
    GenericGlobalSupport{T}(Array{T,2}(undef, nsamples(emd), nfeatsnaggrs))
fwd_gs_set(emds::GenericGlobalSupport{T}, i_sample::Integer, i_featsnaggr::Integer, threshold::T) where {T} =
    emds.d[i_sample, i_featsnaggr] = threshold
function slice_dataset(emds::GenericGlobalSupport{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    GenericGlobalSupport{T}(if return_view @view emds.d[inds,:] else emds.d[inds,:] end)
end
