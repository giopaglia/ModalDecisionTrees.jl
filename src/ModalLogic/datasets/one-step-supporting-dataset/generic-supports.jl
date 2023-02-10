
############################################################################################
############################################################################################
############################################################################################

struct GenericRelationalSupport{
    V,
    W<:AbstractWorld,
    FR<:AbstractFrame{W,Bool},
    D<:AbstractArray{Dict{W,VV}, 3} where VV<:Union{V,Nothing},
} <: AbstractRelationalSupport{V,W,FR}
    d :: D

    function GenericRelationalSupport(emd::FeaturedDataset{V,W,FR}, perform_initialization = false) where {V,W,FR}
        _nfeatsnaggrs = nfeatsnaggrs(emd)
        _fwd_rs = begin
            if perform_initialization
                _fwd_rs = Array{Dict{W,Union{V,Nothing}}, 3}(undef, nsamples(emd), _nfeatsnaggrs, nrelations(emd))
                fill!(_fwd_rs, nothing)
            else
                Array{Dict{W,V}, 3}(undef, nsamples(emd), _nfeatsnaggrs, nrelations(emd))
            end
        end
        new{V,W,FR,typeof(_fwd_rs)}(_fwd_rs)
    end
end

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
    support      :: GenericRelationalSupport{V,W},
    i_sample     :: Integer,
    w            :: W,
    i_featsnaggr :: Integer,
    i_relation   :: Integer) where {V,W<:AbstractWorld} = support.d[i_sample, i_featsnaggr, i_relation][w]
Base.size(support::GenericRelationalSupport, args...) = size(support.d, args...)

fwd_rs_init_world_slice(support::GenericRelationalSupport{V,W}, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) where {V,W} =
    support.d[i_sample, i_featsnaggr, i_relation] = Dict{W,V}()
fwd_rs_set(support::GenericRelationalSupport{V,W}, i_sample::Integer, w::AbstractWorld, i_featsnaggr::Integer, i_relation::Integer, threshold::V) where {V,W} =
    support.d[i_sample, i_featsnaggr, i_relation][w] = threshold
function _slice_dataset(support::GenericRelationalSupport{V,W}, inds::AbstractVector{<:Integer}, return_view::Val = Val(false)) where {V,W}
    GenericRelationalSupport{V,W}(if return_view == Val(true) @view support.d[inds,:,:] else support.d[inds,:,:] end)
end

############################################################################################

# Note: the global support is world-agnostic
struct GenericGlobalSupport{V} <: AbstractGlobalSupport{V}
    d :: AbstractArray{V,2}

    function GenericGlobalSupport(emd::FeaturedDataset{V}) where {V}
        @assert worldtype(emd) != OneWorld "TODO adjust this note: note that you should not use a global support when not using global decisions"
        _nfeatsnaggrs = nfeatsnaggrs(emd)
        new{V}(Array{V,2}(undef, nsamples(emd), _nfeatsnaggrs))
    end
end

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
Base.size(support::GenericGlobalSupport{V}, args...) where {V} = size(support.d, args...)

fwd_gs_set(support::GenericGlobalSupport{V}, i_sample::Integer, i_featsnaggr::Integer, threshold::V) where {V} =
    support.d[i_sample, i_featsnaggr] = threshold
function _slice_dataset(support::GenericGlobalSupport{V}, inds::AbstractVector{<:Integer}, return_view::Val = Val(false)) where {V}
    GenericGlobalSupport{V}(if return_view == Val(true) @view support.d[inds,:] else support.d[inds,:] end)
end
