
############################################################################################
# Featured world dataset
############################################################################################
# 
# In the most general case, the representation of a modal dataset is based on a
#  multi-dimensional lookup table, referred to as *propositional lookup table*,
#  or *featured world dataset* (abbreviated into fwd).
# 
# This structure, is such that the value at fwd[i, w, f], referred to as *gamma*,
#  is the value of feature f on world w on the i-th instance, and can be used to answer the
#  question whether a proposition (e.g., minimum(A1) ≥ 10) holds onto a given world and instance;
#  however, an fwd table can be implemented in many ways, mainly depending on the world type.
# 
# Note that this structure does not constitute a ActiveFeaturedDataset (see FeaturedDataset a few lines below)
# 
############################################################################################

abstract type AbstractFWD{V<:Number,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}} end

# # A function for getting a threshold value from the lookup table
# Maybe TODO: but fails with ArgumentError: invalid index: − of type SoleLogics.OneWorld:
fwdread(fwd::AbstractFWD, args...) = Base.getindex(fwd, args...)
@inline function Base.getindex(
    fwd         :: AbstractFWD{V,W},
    i_sample    :: Integer,
    w           :: W,
    i_feature   :: Integer
) where {V,W<:AbstractWorld}
    error("TODO provide...")
end

# 
# Actually, the interface for AbstractFWD's is a bit tricky; the most straightforward
#  way of learning it is by considering the fallback fwd structure defined as follows.
# TODO oh, but the implementation is broken due to a strange error (see https://discourse.julialang.org/t/tricky-too-many-parameters-for-type-error/25182 )

# # The most generic fwd structure is a matrix of dictionaries of size (nsamples × nfeatures)
# struct GenericFWD{V,W} <: AbstractFWD{V,W}
#   d :: AbstractVector{<:AbstractDict{W,AbstractVector{V,1}},1}
#   nfeatures :: Integer
# end

# nsamples(fwd::GenericFWD{V}) where {V}  = size(fwd, 1)
# nfeatures(fwd::GenericFWD{V}) where {V} = fwd.d
# Base.size(fwd::GenericFWD{V}, args...) where {V} = size(fwd.d, args...)

# # The matrix is initialized with #undef values
# function fwd_init(::Type{GenericFWD}, imd::DimensionalFeaturedDataset{V}) where {V}
#     d = Array{Dict{W,V}, 2}(undef, nsamples(imd))
#     for i in 1:nsamples
#         d[i] = Dict{W,Array{V,1}}()
#     end
#     GenericFWD{V}(d, nfeatures(imd))
# end

# # A function for initializing individual world slices
# function fwd_init_world_slice(fwd::GenericFWD{V}, i_sample::Integer, w::AbstractWorld) where {V}
#     fwd.d[i_sample][w] = Array{V,1}(undef, fwd.nfeatures)
# end

# # A function for getting a threshold value from the lookup table
# Base.@propagate_inbounds @inline fwdread(
#     fwd         :: GenericFWD{V},
#     i_sample    :: Integer,
#     w           :: AbstractWorld,
#     i_feature   :: Integer) where {V} = fwd.d[i_sample][w][i_feature]

# # A function for setting a threshold value in the lookup table
# Base.@propagate_inbounds @inline function fwd_set(fwd::GenericFWD{V}, w::AbstractWorld, i_sample::Integer, i_feature::Integer, threshold::V) where {V}
#     fwd.d[i_sample][w][i_feature] = threshold
# end

# # A function for setting threshold values for a single feature (from a feature slice, experimental)
# Base.@propagate_inbounds @inline function fwd_set_feature(fwd::GenericFWD{V}, i_feature::Integer, fwd_feature_slice::Any) where {V}
#     throw_n_log("Warning! fwd_set_feature with GenericFWD is not yet implemented!")
#     for ((i_sample,w),threshold::V) in read_fwd_feature_slice(fwd_feature_slice)
#         fwd.d[i_sample][w][i_feature] = threshold
#     end
# end

# # A function for slicing the dataset
# function _slice_dataset(fwd::GenericFWD{V}, inds::AbstractVector{<:Integer}, return_view::Val = Val(false)) where {V}
#     GenericFWD{V}(if return_view == Val(true) @view fwd.d[inds] else fwd.d[inds] end, fwd.nfeatures)
# end

# Others...
# Base.@propagate_inbounds @inline fwdread_channeaoeu(fwd::GenericFWD{V}, i_sample::Integer, i_feature::Integer) where {V} = TODO
# const GenericFeaturedChannel{V} = TODO
# fwd_channel_interpret_world(fwc::GenericFeaturedChannel{V}, w::AbstractWorld) where {V} = TODO

isminifiable(::AbstractFWD) = true

function minify(fwd::AbstractFWD)
    minify(fwd.d) #TODO improper
end

############################################################################################
# Explicit modal dataset
# 
# An explicit modal dataset is the generic form of a modal dataset, and consists of
#  a wrapper around an fwd lookup table. The information it adds are the relation set,
#  a few functions for enumerating worlds (`accessibles`, `representatives`),
#  and a world set initialization function representing initial conditions (initializing world sets).
# 
############################################################################################

struct FeaturedDataset{
    V<:Number,
    W<:AbstractWorld,
    FR<:AbstractFrame{W,Bool},
    FT<:AbstractFeature{V},
    FWD<:AbstractFWD{V,W,FR},
    G1<:AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}},
    G2<:AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
} <: ActiveFeaturedDataset{V,W,FR,FT}
    
    # Core data (fwd lookup table)
    fwd                     :: FWD

    ## Modal frame:
    # Accessibility relations
    relations               :: AbstractVector{<:AbstractRelation}
    
    # Features
    features                :: Vector{FT}

    # Test operators associated with each feature, grouped by their respective aggregator
    grouped_featsaggrsnops  :: G1
    
    grouped_featsnaggrs     :: G2
    
    function FeaturedDataset{V,W,FR,FT,FWD}(
        fwd                     :: FWD,
        relations               :: AbstractVector{<:AbstractRelation},
        features                :: AbstractVector{FT},
        grouped_featsaggrsnops  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}};
        allow_no_instances = false,
    ) where {V,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},FWD<:AbstractFWD{V,W,FR},FT<:AbstractFeature{V}}
        features = collect(features)
        ty = "FeaturedDataset{$(V),$(W),$(FR),$(FT)}"
        @assert allow_no_instances || nsamples(fwd) > 0     "Can't instantiate $(ty) with no instance. (fwd's type $(typeof(fwd)))"
        @assert length(grouped_featsaggrsnops) > 0 && sum(length.(grouped_featsaggrsnops)) > 0 && sum(vcat([[length(test_ops) for test_ops in aggrs] for aggrs in grouped_featsaggrsnops]...)) > 0 "Can't instantiate $(ty) with no test operator: grouped_featsaggrsnops"
        @assert nfeatures(fwd) == length(features)          "Can't instantiate $(ty) with different numbers of instances $(nsamples(fwd)) and of features $(length(features))."
        grouped_featsnaggrs = features_grouped_featsaggrsnops2grouped_featsnaggrs(features, grouped_featsaggrsnops)
        new{V,W,FR,FT,FWD,typeof(grouped_featsaggrsnops),typeof(grouped_featsnaggrs)}(fwd, relations, features, grouped_featsaggrsnops, grouped_featsnaggrs)
    end

    function FeaturedDataset{V,W,FR}(
        fwd                     :: FWD,
        relations               :: AbstractVector{<:AbstractRelation},
        features                :: AbstractVector{<:AbstractFeature},
        args...;
        kwargs...
    ) where {V,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},FWD<:AbstractFWD{V,W,FR}}
        features = collect(features)
        FT = Union{typeof.(features)...}
        features = Vector{FT}(features)
        FeaturedDataset{V,W,FR,FT,FWD}(fwd, relations, features, args...; kwargs...)
    end

    function FeaturedDataset{V,W}(
        fwd                     :: AbstractFWD{V,W,FR},
        args...;
        kwargs...
    ) where {V,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        FeaturedDataset{V,W,FR}(fwd, args...; kwargs...)
    end

    function FeaturedDataset(
        fwd                    :: AbstractFWD{V,W},
        relations              :: AbstractVector{<:AbstractRelation},
        features               :: AbstractVector{<:AbstractFeature},
        grouped_featsaggrsnops_or_featsnops, # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        args...;
        kwargs...,
    ) where {V,W}
        FeaturedDataset{V,W}(fwd, relations, features, grouped_featsaggrsnops_or_featsnops, args...; kwargs...)
    end

    function FeaturedDataset(
        fwd                    :: AbstractFWD{V,W},
        relations              :: AbstractVector{<:AbstractRelation},
        features               :: AbstractVector{<:AbstractFeature},
        grouped_featsnops      :: AbstractVector{<:AbstractVector{<:TestOperatorFun}},
        args...;
        kwargs...,
    ) where {V,W<:AbstractWorld}
        grouped_featsaggrsnops = grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops)
        FeaturedDataset(fwd, relations, features, grouped_featsaggrsnops, args...; kwargs...)
    end

    _default_fwd_type(::Type{<:AbstractWorld}) = error("No GenericFWD has been implemented yet. Please provide a `fwd_type` parameter, as in: FeaturedDataset(X, IntervalFWD)")
    _default_fwd_type(::Type{<:Union{OneWorld,Interval,Interval2D}}) = UniformFullDimensionalFWD

    # Quite importantly, an fwd can be computed from a dataset in implicit form (domain + ontology + features)
    Base.@propagate_inbounds function FeaturedDataset(
        X          :: DimensionalFeaturedDataset{V,N,W},
        # fwd_type   :: Type{<:AbstractFWD} = _default_fwd_type(W), # TODO
        fwd_type   = _default_fwd_type(W),
        args...;
        kwargs...,
    ) where {V,N,W<:AbstractWorld}

        fwd = begin

            # Initialize the fwd structure
            fwd = fwd_type(X)

            # @logmsg LogOverview "DimensionalFeaturedDataset -> FeaturedDataset"

            _features = features(X)

            _n_samples = nsamples(X)

            # Load any (possible) external features
            if any(isa.(_features, ExternalFWDFeature))
                i_external_features = first.(filter(((i_feature,is_external_fwd),)->(is_external_fwd), collect(enumerate(isa.(_features, ExternalFWDFeature)))))
                for i_feature in i_external_features
                    feature = _features[i_feature]
                    fwd_set_feature_slice(fwd, i_feature, feature.fwd)
                end
            end

            # Load any internal features
            i_features = first.(filter(((i_feature,is_external_fwd),)->!(is_external_fwd), collect(enumerate(isa.(_features, ExternalFWDFeature)))))
            enum_features = zip(i_features, _features[i_features])

            # Compute features
            # p = Progress(_n_samples, 1, "Computing EMD...")
            @inbounds Threads.@threads for i_sample in 1:_n_samples
                @logmsg LogDebug "Instance $(i_sample)/$(_n_samples)"

                # if i_sample == 1 || ((i_sample+1) % (floor(Int, ((_n_samples)/4))+1)) == 0
                #     @logmsg LogOverview "Instance $(i_sample)/$(_n_samples)"
                # end

                for w in allworlds(X, i_sample)
                    
                    fwd_init_world_slice(fwd, i_sample, w)

                    @logmsg LogDebug "World" w

                    for (i_feature,feature) in enum_features

                        threshold = X[i_sample, w, feature, i_feature]

                        @logmsg LogDebug "Feature $(i_feature)" threshold

                        fwd_set(fwd, w, i_sample, i_feature, threshold)

                    end
                end
                # next!(p)
            end
            fwd
        end

        FeaturedDataset(fwd, relations(X), _features, grouped_featsaggrsnops(X), args...; kwargs...)
    end

end


@inline function Base.getindex(
    X::FeaturedDataset{V,W},
    i_sample::Integer,
    w::W,
    feature::AbstractFeature,
    args...
) where {V,W<:AbstractWorld}
    i_feature = find_feature_id(X, feature)
    X[i_sample, w, feature, i_feature, args...]::V
end

@inline function Base.getindex(
    X::FeaturedDataset{V,W},
    i_sample::Integer,
    w::W,
    feature::AbstractFeature,
    i_feature::Integer,
    args...
) where {V,W<:AbstractWorld}
    fwd(X)[i_sample, w, i_feature, args...]::V
end

Base.size(X::FeaturedDataset)              = Base.size(fwd(X))

fwd(X::FeaturedDataset)                    = X.fwd
relations(X::FeaturedDataset)              = X.relations
features(X::FeaturedDataset)               = X.features
grouped_featsaggrsnops(X::FeaturedDataset) = X.grouped_featsaggrsnops
grouped_featsnaggrs(X::FeaturedDataset)    = X.grouped_featsnaggrs

nfeatures(X::FeaturedDataset)              = length(features(X))
nrelations(X::FeaturedDataset)             = length(relations(X))
nsamples(X::FeaturedDataset)               = nsamples(fwd(X))
worldtype(X::FeaturedDataset{V,W}) where {V,W<:AbstractWorld} = W

nfeatsnaggrs(X::FeaturedDataset)            = sum(length.(grouped_featsnaggrs(X)))

initialworldset(X::FeaturedDataset, i_sample, args...) = initialworldset(fwd(X), i_sample, args...)
accessibles(X::FeaturedDataset, i_sample, args...) = accessibles(fwd(X), i_sample, args...)
representatives(X::FeaturedDataset, i_sample, args...) = representatives(fwd(X), i_sample, args...)
allworlds(X::FeaturedDataset, i_sample, args...) = allworlds(fwd(X), i_sample, args...)


function _slice_dataset(X::FeaturedDataset, inds::AbstractVector{<:Integer}, args...; kwargs...)
    FeaturedDataset(
        _slice_dataset(fwd(X), inds, args...; kwargs...),
        relations(X),
        features(X),
        grouped_featsaggrsnops(X)
    )
end


function display_structure(emd::FeaturedDataset; indent_str = "")
    out = "$(typeof(emd))\t$(Base.summarysize(emd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(emd))))\t$(relations(emd))\n"
    out *= indent_str * "└ fwd: \t$(typeof(fwd(emd)))\t$(Base.summarysize(fwd(emd)) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out
end


find_feature_id(X::FeaturedDataset{V,W}, feature::AbstractFeature) where {V,W} =
    findall(x->x==feature, features(X))[1]
find_relation_id(X::FeaturedDataset{V,W}, relation::AbstractRelation) where {V,W} =
    findall(x->x==relation, relations(X))[1]

function hasnans(emd::FeaturedDataset)
    # @show hasnans(fwd(emd))
    hasnans(fwd(emd))
end


isminifiable(::FeaturedDataset) = true

function minify(X::FeaturedDataset)
    new_fwd, backmap = minify(fwd(X))
    X = FeaturedDataset(
        new_fwd,
        relations(X),
        features(X),
        grouped_featsaggrsnops(X),
    )
    X, backmap
end

############################################################################################
############################################################################################
############################################################################################

# World-specific featured world datasets and supports
include("dimensional-fwds.jl")
