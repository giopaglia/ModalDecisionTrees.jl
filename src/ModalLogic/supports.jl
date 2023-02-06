
abstract type SupportingModalDataset{T<:Number,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}} end

isminifiable(X::SupportingModalDataset) = false

############################################################################################

include("one-step-support.jl")

############################################################################################
############################################################################################
############################################################################################

struct GenericRelationalSupport{
    T,
    W<:AbstractWorld,
    FR<:AbstractFrame{W,Bool},
    D<:AbstractArray{Dict{W,TT}, 3} where TT<:Union{T,Nothing},
} <: AbstractRelationalSupport{T,W,FR}
    d :: D
end

goeswith(::Type{GenericRelationalSupport}, ::Type{<:AbstractWorld}) = true
# default_fwd_rs_type(::Type{<:AbstractWorld}) = GenericRelationalSupport # TODO implement similar pattern used for fwd

function hasnans(support::GenericRelationalSupport)
    # @show any(map(d->(any(_isnan.(collect(values(d))))), support.d))
    any(map(d->(any(_isnan.(collect(values(d))))), support.d))
end

nsamples(support::GenericRelationalSupport)     = size(support, 1)
nfeatsnaggrs(support::GenericRelationalSupport) = size(support, 2)
nrelations(support::GenericRelationalSupport)   = size(support, 3)
capacity(support::GenericRelationalSupport)     = Inf

Base.getindex(
    support      :: GenericRelationalSupport{T,W},
    i_sample     :: Integer,
    w            :: W,
    i_featsnaggr :: Integer,
    i_relation   :: Integer) where {T,W<:AbstractWorld} = support.d[i_sample, i_featsnaggr, i_relation][w]
Base.size(support::GenericRelationalSupport, args...) = size(support.d, args...)

function fwd_rs_init(emd::ExplicitModalDataset{T,W,FR}, perform_initialization = false) where {T,W,FR}
    nfeatsnaggrs = sum(length.(grouped_featsnaggrs(emd)))
    _fwd_rs = begin
        if perform_initialization
            _fwd_rs = Array{Dict{W,Union{T,Nothing}}, 3}(undef, nsamples(emd), nfeatsnaggrs, nrelations(emd))
            fill!(_fwd_rs, nothing)
        else
            Array{Dict{W,T}, 3}(undef, nsamples(emd), nfeatsnaggrs, nrelations(emd))
        end
    end
    GenericRelationalSupport{T,W,FR,typeof(_fwd_rs)}(_fwd_rs)
end
fwd_rs_init_world_slice(support::GenericRelationalSupport{T,W}, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) where {T,W} =
    support.d[i_sample, i_featsnaggr, i_relation] = Dict{W,T}()
fwd_rs_set(support::GenericRelationalSupport{T,W}, i_sample::Integer, w::AbstractWorld, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T,W} =
    support.d[i_sample, i_featsnaggr, i_relation][w] = threshold
function slice_dataset(support::GenericRelationalSupport{T,W}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T,W}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    GenericRelationalSupport{T,W}(if return_view @view support.d[inds,:,:] else support.d[inds,:,:] end)
end

############################################################################################

# Note: the global support is world-agnostic
struct GenericGlobalSupport{T} <: AbstractGlobalSupport{T}
    d :: AbstractArray{T,2}
end

goeswith(::Type{AbstractGlobalSupport}, ::Type{<:AbstractWorld}) = true
# default_fwd_gs_type(::Type{<:AbstractWorld}) = GenericGlobalSupport # TODO implement similar pattern used for fwd

function hasnans(support::GenericGlobalSupport)
    # @show any(_isnan.(support.d))
    any(_isnan.(support.d))
end

nsamples(support::GenericGlobalSupport)  = size(support, 1)
nfeatsnaggrs(support::GenericGlobalSupport) = size(support, 2)
Base.getindex(
    support      :: GenericGlobalSupport,
    i_sample     :: Integer,
    i_featsnaggr  :: Integer) = support.d[i_sample, i_featsnaggr]
Base.size(support::GenericGlobalSupport{T}, args...) where {T} = size(support.d, args...)

function fwd_gs_init(emd::ExplicitModalDataset{T}) where {T}
    @assert world_type(emd) != OneWorld "TODO adjust this note: note that you should not use a global support when not using global decisions"
    nfeatsnaggrs = sum(length.(grouped_featsnaggrs(emd)))
    GenericGlobalSupport{T}(Array{T,2}(undef, nsamples(emd), nfeatsnaggrs))
end
fwd_gs_set(support::GenericGlobalSupport{T}, i_sample::Integer, i_featsnaggr::Integer, threshold::T) where {T} =
    support.d[i_sample, i_featsnaggr] = threshold
function slice_dataset(support::GenericGlobalSupport{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    GenericGlobalSupport{T}(if return_view @view support.d[inds,:] else support.d[inds,:] end)
end
