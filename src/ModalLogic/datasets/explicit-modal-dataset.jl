
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
# Note that this structure does not constitute a ActiveModalDataset (see ExplicitModalDataset a few lines below)
# 
############################################################################################

abstract type AbstractFWD{T<:Number,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}} end

# Any implementation for a fwd must indicate their compatible world types via `goeswith`.
# Fallback:
goeswith(::Type{<:AbstractFWD}, ::Type{<:AbstractWorld}) = false

# # A function for getting a threshold value from the lookup table
# Maybe TODO: but fails with ArgumentError: invalid index: − of type SoleLogics.OneWorld:
# Base.getindex(fwd::AbstractFWD, args...) = fwd_get(fwd, args...)
Base.getindex(
    fwd         :: AbstractFWD,
    i_sample    :: Integer,
    w           :: AbstractWorld,
    i_feature   :: Integer) = fwd_get(fwd, i_sample, w, i_feature)

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
# function _slice_dataset(fwd::GenericFWD{T}, inds::AbstractVector{<:Integer}, return_view::Val = Val(false)) where {T}
#     GenericFWD{T}(if return_view == Val(true) @view fwd.d[inds] else fwd.d[inds] end, fwd.nfeatures)
# end

# Others...
# Base.@propagate_inbounds @inline fwd_get_channeaoeu(fwd::GenericFWD{T}, i_sample::Integer, i_feature::Integer) where {T} = TODO
# const GenericFeaturedChannel{T} = TODO
# fwd_channel_interpret_world(fwc::GenericFeaturedChannel{T}, w::AbstractWorld) where {T} = TODO

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

struct ExplicitModalDataset{
    T<:Number,
    W<:AbstractWorld,
    FR<:AbstractFrame{W,Bool},
    FWD<:AbstractFWD{T,W,FR},
    G1<:AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}},
    G2<:AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
} <: ActiveModalDataset{T,W,FR}
    
    # Core data (fwd lookup table)
    fwd                :: FWD

    ## Modal frame:
    # Accessibility relations
    relations          :: AbstractVector{<:AbstractRelation}
    
    # Features
    features           :: AbstractVector{<:AbstractFeature}

    # Test operators associated with each feature, grouped by their respective aggregator
    grouped_featsaggrsnops  :: G1
    
    grouped_featsnaggrs :: G2
    
    function ExplicitModalDataset{T,W,FR,FWD}(
        fwd                     :: FWD,
        relations               :: AbstractVector{<:AbstractRelation},
        features                :: AbstractVector{<:AbstractFeature},
        grouped_featsaggrsnops  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}};
        allow_no_instances = false,
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},FWD<:AbstractFWD{T,W,FR}}
        @assert allow_no_instances || nsamples(fwd) > 0     "Can't instantiate ExplicitModalDataset{$(T), $(W)} with no instance. (fwd's type $(typeof(fwd)))"
        @assert length(grouped_featsaggrsnops) > 0 && sum(length.(grouped_featsaggrsnops)) > 0 && sum(vcat([[length(test_ops) for test_ops in aggrs] for aggrs in grouped_featsaggrsnops]...)) > 0 "Can't instantiate ExplicitModalDataset{$(T), $(W)} with no test operator: grouped_featsaggrsnops"
        @assert nfeatures(fwd) == length(features)          "Can't instantiate ExplicitModalDataset{$(T), $(W)} with different numbers of instances $(nsamples(fwd)) and of features $(length(features))."
        grouped_featsnaggrs = features_grouped_featsaggrsnops2grouped_featsnaggrs(features, grouped_featsaggrsnops)
        new{T,W,FR,FWD,typeof(grouped_featsaggrsnops),typeof(grouped_featsnaggrs)}(fwd, relations, features, grouped_featsaggrsnops, grouped_featsnaggrs)
    end

    function ExplicitModalDataset{T,W,FR}(
        fwd                     :: FWD,
        args...;
        kwargs...
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},FWD<:AbstractFWD{T,W,FR}}
        ExplicitModalDataset{T,W,FR,FWD}(fwd, args...; kwargs...)
    end

    function ExplicitModalDataset{T,W}(
        fwd                     :: AbstractFWD{T,W,FR},
        args...;
        kwargs...
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        ExplicitModalDataset{T,W,FR}(fwd, args...; kwargs...)
    end

    ExplicitModalDataset(
        fwd                    :: AbstractFWD{T,W},
        relations              :: AbstractVector{<:AbstractRelation},
        features               :: AbstractVector{<:AbstractFeature},
        grouped_featsaggrsnops_or_featsnops, # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        args...;
        kwargs...,
    ) where {T,W} = begin ExplicitModalDataset{T,W}(fwd, relations, features, grouped_featsaggrsnops_or_featsnops, args...; kwargs...) end

    function ExplicitModalDataset(
        fwd                    :: AbstractFWD{T,W},
        relations              :: AbstractVector{<:AbstractRelation},
        features               :: AbstractVector{<:AbstractFeature},
        grouped_featsnops      :: AbstractVector{<:AbstractVector{<:TestOperatorFun}},
        args...;
        kwargs...,
    ) where {T,W<:AbstractWorld}

        grouped_featsaggrsnops = grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops)
 
        ExplicitModalDataset(fwd, relations, features, grouped_featsaggrsnops, args...; kwargs...)
    end

    # Quite importantly, an fwd can be computed from a dataset in implicit form (domain + ontology + features)
    Base.@propagate_inbounds function ExplicitModalDataset(
        imd                  :: InterpretedModalDataset{T,N,W},
        # FWD                  ::Type{<:AbstractFWD{T,W}} = default_fwd_type(W),
        FWD                  ::Type = default_fwd_type(W),
        args...;
        kwargs...,
    ) where {T,N,W<:AbstractWorld}

        fwd = begin

            # @logmsg LogOverview "InterpretedModalDataset -> ExplicitModalDataset"

            _features = features(imd)

            _n_samples = nsamples(imd)

            @assert goeswith(FWD, W)

            # Initialize the fwd structure
            fwd = fwd_init(FWD, imd)

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

                # instance = get_instance(imd, i_sample)
                # @logmsg LogDebug "instance" instance

                for w in allworlds(imd, i_sample)
                    
                    fwd_init_world_slice(fwd, i_sample, w)

                    @logmsg LogDebug "World" w

                    for (i_feature,feature) in enum_features

                        threshold = get_gamma(imd, i_sample, w, feature)

                        @logmsg LogDebug "Feature $(i_feature)" threshold

                        fwd_set(fwd, w, i_sample, i_feature, threshold)

                    end
                end
                # next!(p)
            end
            fwd
        end

        ExplicitModalDataset(fwd, relations(imd), _features, grouped_featsaggrsnops(imd), args...; kwargs...)
    end

end

Base.getindex(X::ExplicitModalDataset, args...) = Base.getindex(fwd(X), args...)
Base.size(X::ExplicitModalDataset)              = Base.size(fwd(X))

fwd(X::ExplicitModalDataset)                    = X.fwd
relations(X::ExplicitModalDataset)              = X.relations
features(X::ExplicitModalDataset)               = X.features
grouped_featsaggrsnops(X::ExplicitModalDataset) = X.grouped_featsaggrsnops
grouped_featsnaggrs(X::ExplicitModalDataset)    = X.grouped_featsnaggrs

nfeatures(X::ExplicitModalDataset)              = length(features(X))
nrelations(X::ExplicitModalDataset)             = length(relations(X))
nsamples(X::ExplicitModalDataset)               = nsamples(fwd(X))
worldtype(X::ExplicitModalDataset{T,W}) where {T,W<:AbstractWorld} = W


initialworldset(X::ExplicitModalDataset, i_sample, args...) = initialworldset(fwd(X), i_sample, args...)
accessibles(X::ExplicitModalDataset, i_sample, args...) = accessibles(fwd(X), i_sample, args...)
representatives(X::ExplicitModalDataset, i_sample, args...) = representatives(fwd(X), i_sample, args...)
allworlds(X::ExplicitModalDataset, i_sample, args...) = allworlds(fwd(X), i_sample, args...)


function _slice_dataset(X::ExplicitModalDataset, inds::AbstractVector{<:Integer}, args...; kwargs...)
    ExplicitModalDataset(
        _slice_dataset(fwd(X), inds, args...; kwargs...),
        relations(X),
        features(X),
        grouped_featsaggrsnops(X)
    )
end


function display_structure(emd::ExplicitModalDataset; indent_str = "")
    out = "$(typeof(emd))\t$(Base.summarysize(emd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(emd))))\t$(relations(emd))\n"
    out *= indent_str * "└ fwd: \t$(typeof(fwd(emd)))\t$(Base.summarysize(fwd(emd)) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out
end


find_feature_id(X::ExplicitModalDataset{T,W}, feature::AbstractFeature) where {T,W} =
    findall(x->x==feature, features(X))[1]
find_relation_id(X::ExplicitModalDataset{T,W}, relation::AbstractRelation) where {T,W} =
    findall(x->x==relation, relations(X))[1]

function hasnans(emd::ExplicitModalDataset)
    # @show hasnans(fwd(emd))
    hasnans(fwd(emd))
end


isminifiable(::ExplicitModalDataset) = true

function minify(X::ExplicitModalDataset)
    new_fwd, backmap = minify(fwd(X))
    X = ExplicitModalDataset(
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


Base.@propagate_inbounds @inline get_gamma(
        X::ExplicitModalDataset{T,W},
        i_sample::Integer,
        w::W,
        feature::AbstractFeature) where {T,W<:AbstractWorld} = begin
    i_feature = find_feature_id(X, feature)
    X[i_sample, w, i_feature]
end

# World-specific featured world datasets and supports
include("dimensional-fwds.jl")

# TODO remove
default_fwd_type(::Type{OneWorld}) = OneWorldFWD
default_fwd_type(::Type{Interval}) = IntervalFWD
default_fwd_type(::Type{Interval2D}) = Interval2DFWD
