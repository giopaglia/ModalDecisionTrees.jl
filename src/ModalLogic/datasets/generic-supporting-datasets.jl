
struct GenericSupportingDataset{
    W<:AbstractWorld,
    FR<:AbstractFrame{W,Bool},
    M<:AbstractVector{<:AbstractDict{Formula,Vector{W}}},
} <: SupportingDataset{W,FR}
    
    memo::M

    function GenericSupportingDataset{W,FR,M}(
        memo :: M,
    ) where {W<:AbstractWorld,FR<:AbstractFrame{W,Bool},M<:AbstractVector{<:AbstractDict{Formula,Vector{W}}}}
        new{W,FR,M}(memo)
    end

    function GenericSupportingDataset(
        emd :: FeaturedDataset{V,W,FR},
    ) where {V,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        memo = Vector{Dict{Formula,Vector{W}}}(undef, nsamples(emd))
        for i_sample in 1:nsamples(emd)
            memo[i_sample] = Dict{Formula,Vector{W}}()
        end
        GenericSupportingDataset{W,FR,typeof(memo)}(memo)
    end
end

usesmemo(X::GenericSupportingDataset) = true

Base.size(X::GenericSupportingDataset) = ()
capacity(X::GenericSupportingDataset) = Inf
nmemoizedvalues(X::GenericSupportingDataset) = sum(length.(X.d))

function _slice_dataset(X::GFSD, inds::AbstractVector{<:Integer}, args...; kwargs...) where {GFSD<:GenericSupportingDataset}
    GFSD(X.w0, _slice_dataset(X.memo[inds], args...; kwargs...))
end

hasnans(X::GenericSupportingDataset) = false # TODO double check that this is intended

isminifiable(X::GenericSupportingDataset) = false # TODO double check that this is intended

struct ChainedFeaturedSupportingDataset{
    V<:Number,
    W<:AbstractWorld,
    FR<:AbstractFrame{W,Bool},
    M<:AbstractVector{<:AbstractDict{Formula,V}},
} <: SupportingDataset{W,FR}
    
    w0::W

    memo::M

    function ChainedFeaturedSupportingDataset{V,W,FR,M}(
        memo :: M,
    ) where {V,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},M<:AbstractVector{<:AbstractDict{Formula,V}}}
        new{V,W,FR,M}(memo)
    end

    function ChainedFeaturedSupportingDataset(
        emd :: FeaturedDataset{V,W,FR},
    ) where {V,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        memo = Vector{Dict{Formula,Vector{W}}}(undef, nsamples(emd))
        for i_sample in 1:nsamples(emd)
            memo[i_sample] = Dict{Formula,Vector{W}}()
        end
        ChainedFeaturedSupportingDataset{V,W,FR,typeof(memo)}(memo)
    end
end

usesmemo(X::ChainedFeaturedSupportingDataset) = true

Base.size(X::ChainedFeaturedSupportingDataset) = ()
capacity(X::ChainedFeaturedSupportingDataset) = Inf
nmemoizedvalues(X::ChainedFeaturedSupportingDataset) = sum(length.(X.d))

function _slice_dataset(X::CFSD, inds::AbstractVector{<:Integer}, args...; kwargs...) where {CFSD<:ChainedFeaturedSupportingDataset}
    CFSD(X.w0, _slice_dataset(X.memo[inds], args...; kwargs...))
end

hasnans(X::ChainedFeaturedSupportingDataset) = false # TODO double check that this is intended

isminifiable(X::ChainedFeaturedSupportingDataset) = false # TODO double check that this is intended
