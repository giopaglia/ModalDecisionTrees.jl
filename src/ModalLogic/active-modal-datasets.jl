using ProgressMeter

using SoleModels: CanonicalFeatureGeq, CanonicalFeatureGeqSoft, CanonicalFeatureLeq, CanonicalFeatureLeqSoft
using SoleModels: evaluate_thresh_decision, existential_aggregator, aggregator_bottom, aggregator_to_binary

import SoleData: get_instance, instance, max_channel_size, channel_size, nattributes, nsamples, slice_dataset

using SoleLogics: goeswith_dim

# decision.jl
using ..ModalDecisionTrees: is_propositional_decision, display_decision

############################################################################################

# Convenience function
function grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops::AbstractVector{<:AbstractVector{<:TestOperatorFun}})::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
    grouped_featsaggrsnops = Dict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}[]
    for (i_feature, test_operators) in enumerate(grouped_featsnops)
        aggrsnops = Dict{Aggregator,AbstractVector{<:TestOperatorFun}}()
        for test_operator in test_operators
            aggregator = existential_aggregator(test_operator)
            if (!haskey(aggrsnops, aggregator))
                aggrsnops[aggregator] = TestOperatorFun[]
            end
            push!(aggrsnops[aggregator], test_operator)
        end
        push!(grouped_featsaggrsnops, aggrsnops)
    end
    grouped_featsaggrsnops
end

############################################################################################
# Interpreted modal dataset
############################################################################################
# 
# A modal dataset can be instantiated in *implicit* form, from a dimensional domain, and a few
#  objects inducing an interpretation on the domain; mainly, an ontology (determining worlds and
#  relations), and structures for interpreting features onto the domain.
# 
############################################################################################

@computed struct InterpretedModalDataset{T<:Number,N,W<:AbstractWorld} <: ActiveModalDataset{T,W}

    # Core data (a dimensional domain)
    domain                  :: DimensionalDataset{T,N+1+1}
    
    # Worlds & Relations
    ontology                :: Ontology{W} # Union{Nothing,}
    
    # Features
    features                :: AbstractVector{AbstractFeature}

    # Test operators associated with each feature, grouped by their respective aggregator
    # Note: currently, cannot specify the full type (probably due to @computed)
    grouped_featsaggrsnops  :: AbstractVector # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}

    function InterpretedModalDataset(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{W},
        mixed_features::AbstractVector{<:MixedFeature},
    ) where {T,D,W<:AbstractWorld}
        InterpretedModalDataset{T}(domain, ontology, mixed_features)
    end

    function InterpretedModalDataset{T}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{W},
        mixed_features::AbstractVector{<:MixedFeature},
    ) where {T,D,W<:AbstractWorld}
        InterpretedModalDataset{T,D-1-1}(domain, ontology, mixed_features)
    end

    function InterpretedModalDataset{T,N}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{W},
        mixed_features::AbstractVector{<:MixedFeature},
    ) where {T,N,D,W<:AbstractWorld}
        InterpretedModalDataset{T,N,W}(domain, ontology, mixed_features)
    end
    
    function InterpretedModalDataset{T,N,W}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{W}, # default to get_interval_ontology(Val(D-1-1)) ?
        mixed_features::AbstractVector{<:MixedFeature},
    ) where {T,N,D,W<:AbstractWorld}
        _features, featsnops = begin
            _features = AbstractFeature[]
            featsnops = Vector{<:TestOperatorFun}[]

            # readymade features
            cnv_feat(cf::AbstractFeature) = ([≥, ≤], cf)
            cnv_feat(cf::Tuple{TestOperatorFun,AbstractFeature}) = ([cf[1]], cf[2])
            # single-attribute features
            cnv_feat(cf::Any) = cf
            cnv_feat(cf::Function) = ([≥, ≤], cf)
            cnv_feat(cf::Tuple{TestOperatorFun,Function}) = ([cf[1]], cf[2])

            mixed_features = cnv_feat.(mixed_features)

            readymade_cfs          = filter(x->isa(x, Tuple{<:AbstractVector{<:TestOperatorFun},AbstractFeature}), mixed_features)
            attribute_specific_cfs = filter(x->isa(x, CanonicalFeature) || isa(x, Tuple{<:AbstractVector{<:TestOperatorFun},Function}), mixed_features)

            @assert length(readymade_cfs) + length(attribute_specific_cfs) == length(mixed_features) "Unexpected mixed_features: $(filter(x->(! (x in readymade_cfs) && ! (x in attribute_specific_cfs)), mixed_features))"

            for (test_ops,cf) in readymade_cfs
                push!(_features, cf)
                push!(featsnops, test_ops)
            end

            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic.CanonicalFeatureGeq) = ([≥],ModalDecisionTrees.SingleAttributeMin{T}(i_attr))
            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic.CanonicalFeatureLeq) = ([≤],ModalDecisionTrees.SingleAttributeMax{T}(i_attr))
            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic.CanonicalFeatureGeqSoft) = ([≥],ModalDecisionTrees.SingleAttributeSoftMin{T}(i_attr, cf.alpha))
            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic.CanonicalFeatureLeqSoft) = ([≤],ModalDecisionTrees.SingleAttributeSoftMax{T}(i_attr, cf.alpha))
            single_attr_feats_n_featsnops(i_attr,(test_ops,cf)::Tuple{<:AbstractVector{<:TestOperatorFun},typeof(minimum)}) = (test_ops,SingleAttributeMin{T}(i_attr))
            single_attr_feats_n_featsnops(i_attr,(test_ops,cf)::Tuple{<:AbstractVector{<:TestOperatorFun},typeof(maximum)}) = (test_ops,SingleAttributeMax{T}(i_attr))
            # TODO: assuming Float64 function. Also allow functionwrappers?
            single_attr_feats_n_featsnops(i_attr,(test_ops,cf)::Tuple{<:AbstractVector{<:TestOperatorFun},Function})        = (test_ops,SingleAttributeGenericFeature{Float64}(i_attr, (x)->(Float64(cf(x)))))
            single_attr_feats_n_featsnops(i_attr,::Any) = throw_n_log("Unknown mixed_feature type: $(cf), $(typeof(cf))")

            for i_attr in 1:nattributes(domain)
                for (test_ops,cf) in map((cf)->single_attr_feats_n_featsnops(i_attr,cf),attribute_specific_cfs)
                    push!(featsnops, test_ops)
                    push!(_features, cf)
                end
            end
            _features, featsnops
        end
        InterpretedModalDataset{T,N,world_type(ontology)}(domain, ontology, _features, featsnops)
    end

    function InterpretedModalDataset(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{W},
        features::AbstractVector{<:AbstractFeature},
        grouped_featsaggrsnops_or_featsnops; # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        kwargs...,
    ) where {T,D,W<:AbstractWorld}
        InterpretedModalDataset{T}(domain, ontology, features, grouped_featsaggrsnops_or_featsnops; kwargs...)
    end

    function InterpretedModalDataset{T}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{W},
        features::AbstractVector{<:AbstractFeature},
        grouped_featsaggrsnops_or_featsnops; # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        kwargs...,
    ) where {T,D,W<:AbstractWorld}
        InterpretedModalDataset{T,D-1-1}(domain, ontology, features, grouped_featsaggrsnops_or_featsnops; kwargs...)
    end

    function InterpretedModalDataset{T,N}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{W},
        features::AbstractVector{<:AbstractFeature},
        grouped_featsaggrsnops_or_featsnops; # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        kwargs...,
    ) where {T,N,D,W<:AbstractWorld}
        InterpretedModalDataset{T,N,W}(domain, ontology, features, grouped_featsaggrsnops_or_featsnops; kwargs...)
    end
    
    function InterpretedModalDataset{T,N,W}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{W},
        features::AbstractVector{<:AbstractFeature},
        grouped_featsnops  :: AbstractVector{<:AbstractVector{<:TestOperatorFun}};
        kwargs...,
    ) where {T,N,D,W<:AbstractWorld}

        grouped_featsaggrsnops = grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops)
        
        InterpretedModalDataset{T,N,W}(domain, ontology, features, grouped_featsaggrsnops; kwargs...)
    end
    function InterpretedModalDataset{T,N,W}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{W},
        features::AbstractVector{<:AbstractFeature},
        grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}};
        allow_no_instances = false,
    ) where {T,N,D,W<:AbstractWorld}

        @assert allow_no_instances || nsamples(domain) > 0 "Can't instantiate InterpretedModalDataset{$(T), $(N), $(W)} with no instance. (domain's type $(typeof(domain)))"
        @assert goeswith_dim(W, N) "ERROR! Dimensionality mismatch: can't interpret W $(W) on DimensionalDataset of dimensionality = $(N)"
        @assert D == (N+1+1) "ERROR! Dimensionality mismatch: can't instantiate InterpretedModalDataset{$(T), $(N)} with DimensionalDataset{$(T),$(D)}"
        @assert length(features) == length(grouped_featsaggrsnops) "Can't instantiate InterpretedModalDataset{$(T), $(N), $(W)} with mismatching length(features) == length(grouped_featsaggrsnops): $(length(features)) != $(length(grouped_featsaggrsnops))"
        # @assert length(grouped_featsaggrsnops) > 0 && sum(length.(grouped_featsaggrsnops)) > 0 && sum(vcat([[length(test_ops) for test_ops in aggrs] for aggrs in grouped_featsaggrsnops]...)) > 0 "Can't instantiate ExplicitModalDataset{$(T), $(W)} with no test operator: $(grouped_featsaggrsnops)"

        # if prod(max_channel_size(domain)) == 1
        #   TODO throw warning
        # end
        
        new{T,N,W}(domain, ontology, features, grouped_featsaggrsnops)
    end
end

Base.size(imd::InterpretedModalDataset)              = size(imd.domain)
features(imd::InterpretedModalDataset)               = imd.features
grouped_featsaggrsnops(imd::InterpretedModalDataset) = imd.grouped_featsaggrsnops
nattributes(imd::InterpretedModalDataset)           = nattributes(imd.domain)::Int64
nfeatures(imd::InterpretedModalDataset)             = length(features(imd))::Int64
nrelations(imd::InterpretedModalDataset)            = length(relations(imd))::Int64
nsamples(imd::InterpretedModalDataset)              = nsamples(imd.domain)::Int64
relations(imd::InterpretedModalDataset)              = relations(imd.ontology)
world_type(imd::InterpretedModalDataset{T,N,WT}) where {T,N,WT} = WT

initialworldset(imd::InterpretedModalDataset,  i_sample::Integer, args...) =
    ModalDecisionTrees.initialworldset(FullDimensionalFrame(instance_channel_size(imd.domain, i_sample)), args...)
accessibles(imd::InterpretedModalDataset, i_sample, args...) = accessibles(FullDimensionalFrame(instance_channel_size(imd.domain, i_sample)), args...)
representatives(imd::InterpretedModalDataset, i_sample, args...) = representatives(FullDimensionalFrame(instance_channel_size(imd.domain, i_sample)), args...)
allworlds(imd::InterpretedModalDataset, i_sample, args...) = allworlds(FullDimensionalFrame(instance_channel_size(imd.domain, i_sample)), args...)

# Note: Can't define Base.length(::DimensionalDataset) & Base.iterate(::DimensionalDataset)
Base.length(imd::InterpretedModalDataset)                = nsamples(imd)
Base.iterate(imd::InterpretedModalDataset, state=1) = state > length(imd) ? nothing : (get_instance(imd, state), state+1) # Base.iterate(imd.domain, state=state)
max_channel_size(imd::InterpretedModalDataset)          = max_channel_size(imd.domain)

get_instance(imd::InterpretedModalDataset, args...)     = get_instance(imd.domain, args...)

slice_dataset(imd::InterpretedModalDataset, inds::AbstractVector{<:Integer}, args...; allow_no_instances = false, kwargs...)    =
    InterpretedModalDataset(slice_dataset(imd.domain, inds, args...; allow_no_instances = allow_no_instances, kwargs...), imd.ontology, features(imd), imd.grouped_featsaggrsnops; allow_no_instances = allow_no_instances)


function display_structure(imd::InterpretedModalDataset; indent_str = "")
    out = "$(typeof(imd))\t$(Base.summarysize(imd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(imd))))\t$(relations(imd))\n"
    out *= indent_str * "├ domain shape\t$(Base.size(imd.domain))\n"
    out *= indent_str * "└ max_channel_size\t$(max_channel_size(imd))"
    out
end

hasnans(imd::InterpretedModalDataset) = begin
    # @show hasnans(imd.domain)
    hasnans(imd.domain)
end

Base.@propagate_inbounds @inline get_gamma(imd::InterpretedModalDataset, args...) = get_gamma(imd.domain, args...)

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

abstract type AbstractFWD{T<:Number,W<:AbstractWorld} end

# Any implementation for a fwd must indicate their compatible world types via `goeswith`.
# Fallback:
goeswith(::Type{<:AbstractFWD}, ::Type{<:AbstractWorld}) = false

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

Base.getindex(
    fwd         :: AbstractFWD{T},
    i_sample    :: Integer,
    w           :: AbstractWorld,
    i_feature   :: Integer) where {T} = fwd_get(fwd, i_sample, w, i_feature)

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

############################################################################################
# Explicit modal dataset
# 
# An explicit modal dataset is the generic form of a modal dataset, and consists of
#  a wrapper around an fwd lookup table. The information it adds are the relation set,
#  a few functions for enumerating worlds (`accessibles`, `representatives`),
#  and a world set initialization function representing initial conditions (initializing world sets).
# 
############################################################################################

struct ExplicitModalDataset{T<:Number,W<:AbstractWorld} <: ActiveModalDataset{T,W}
    
    # Core data (fwd lookup table)
    fwd                :: AbstractFWD{T,W} # TODO this goes in the type parameters
    
    ## Modal frame:
    # Accessibility relations
    relations          :: AbstractVector{<:AbstractRelation}
    
    # Features
    features           :: AbstractVector{<:AbstractFeature}

    # Test operators associated with each feature, grouped by their respective aggregator
    grouped_featsaggrsnops  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}

    ExplicitModalDataset(
        fwd                    :: AbstractFWD{T,W},
        relations              :: AbstractVector{<:AbstractRelation},
        features               :: AbstractVector{<:AbstractFeature},
        grouped_featsaggrsnops_or_featsnops, # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        args...;
        kwargs...,
    ) where {T,W} = begin ExplicitModalDataset{T,W}(fwd, relations, features, grouped_featsaggrsnops_or_featsnops, args...; kwargs...) end

    function ExplicitModalDataset{T,W}(
        fwd                     :: AbstractFWD{T,W},
        relations               :: AbstractVector{<:AbstractRelation},
        features                :: AbstractVector{<:AbstractFeature},
        grouped_featsaggrsnops  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}};
        allow_no_instances = false,
    ) where {T,W<:AbstractWorld}
        @assert allow_no_instances || nsamples(fwd) > 0     "Can't instantiate ExplicitModalDataset{$(T), $(W)} with no instance. (fwd's type $(typeof(fwd)))"
        @assert length(grouped_featsaggrsnops) > 0 && sum(length.(grouped_featsaggrsnops)) > 0 && sum(vcat([[length(test_ops) for test_ops in aggrs] for aggrs in grouped_featsaggrsnops]...)) > 0 "Can't instantiate ExplicitModalDataset{$(T), $(W)} with no test operator: grouped_featsaggrsnops"
        @assert nfeatures(fwd) == length(features)          "Can't instantiate ExplicitModalDataset{$(T), $(W)} with different numbers of instances $(nsamples(fwd)) and of features $(length(features))."
        new{T,W}(fwd, relations, features, grouped_featsaggrsnops)
    end

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

                        # threshold = computePropositionalThreshold(feature, w, instance)
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

Base.getindex(X::ExplicitModalDataset{T,W}, args...) where {T,W} = getindex(X.fwd, args...)
Base.size(X::ExplicitModalDataset)              = size(X.fwd) # TODO fix not always defined?
features(X::ExplicitModalDataset)               = X.features
grouped_featsaggrsnops(X::ExplicitModalDataset) = X.grouped_featsaggrsnops
nfeatures(X::ExplicitModalDataset)              = length(X.features)
nrelations(X::ExplicitModalDataset)             = length(X.relations)
nsamples(X::ExplicitModalDataset)               = nsamples(X.fwd)::Int64
relations(X::ExplicitModalDataset)              = X.relations
world_type(X::ExplicitModalDataset{T,W}) where {T,W<:AbstractWorld} = W


initialworldset(X::ExplicitModalDataset, i_sample, args...) = initialworldset(X.fwd, i_sample, args...)
accessibles(X::ExplicitModalDataset, i_sample, args...) = accessibles(X.fwd, i_sample, args...)
representatives(X::ExplicitModalDataset, i_sample, args...) = representatives(X.fwd, i_sample, args...)
allworlds(X::ExplicitModalDataset, i_sample, args...) = allworlds(X.fwd, i_sample, args...)


slice_dataset(X::ExplicitModalDataset{T,W}, inds::AbstractVector{<:Integer}, args...; allow_no_instances = false, kwargs...) where {T,W} =
    ExplicitModalDataset{T,W}(
        slice_dataset(X.fwd, inds, args...; allow_no_instances = allow_no_instances, kwargs...),
        X.relations,
        X.features,
        X.grouped_featsaggrsnops;
        allow_no_instances = allow_no_instances
    )


function display_structure(emd::ExplicitModalDataset; indent_str = "")
    out = "$(typeof(emd))\t$(Base.summarysize(emd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(emd))))\t$(relations(emd))\n"
    out *= indent_str * "└ fwd: \t$(typeof(emd.fwd))\t$(Base.summarysize(emd.fwd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out
end


find_feature_id(X::ExplicitModalDataset{T,W}, feature::AbstractFeature) where {T,W} =
    findall(x->x==feature, features(X))[1]
find_relation_id(X::ExplicitModalDataset{T,W}, relation::AbstractRelation) where {T,W} =
    findall(x->x==relation, relations(X))[1]

hasnans(emd::ExplicitModalDataset) = begin
    # @show hasnans(emd.fwd)
    hasnans(emd.fwd)
end

Base.@propagate_inbounds @inline get_gamma(
        X::ExplicitModalDataset{T,W},
        i_sample::Integer,
        w::W,
        feature::AbstractFeature) where {T,W<:AbstractWorld} = begin
    i_feature = find_feature_id(X, feature)
    X[i_sample, w, i_feature]
end

isminifiable(::ExplicitModalDataset) = true

function minify(X::ExplicitModalDataset)
    new_fwd, backmap = minify(X.fwd)
    X = ExplicitModalDataset(
        new_fwd,
        X.relations,
        X.features,
        X.grouped_featsaggrsnops,
    )
    X, backmap
end

############################################################################################
# Explicit modal dataset with support
###########################################################################################

# The lookup table (fwd) in a featured modal dataset provides a quick answer on the truth of
#  propositional decisions; as for answering modal decisions (e.g., ⟨L⟩ (minimum(A2) ≥ 10) )
#  with an fwd, one must enumerate the accessible worlds, compute the truth on each world,
#  and aggregate the answer (by means of all/any). This process is costly; instead, it is
#  sometimes more convenient to initially spend more time computing the truth of any decision,
#  and store this information in a *support* lookup table. Similarly, one can decide to deploy
#  memoization on this table (instead of computing everything at the beginning, compute it on
#  the fly and store it for later calls).
# 
# We define an abstract type for explicit modal dataset with support lookup tables
abstract type ExplicitModalDatasetWithSupport{T,W} <: ActiveModalDataset{T,W} end
# And an abstract type for support lookup tables
abstract type AbstractSupport{T,W} end
# 
# In general, one can use lookup (with or without memoization) for any decision, even the
#  more complex ones, for example:
#  ⟨G⟩ (minimum(A2) ≥ 10 ∧ (⟨O⟩ (maximum(A3) > 2) ∨ (minimum(A1) < 0)))
# 
# In practice, decision trees only ask about simple decisions such as ⟨L⟩ (minimum(A2) ≥ 10),
#  or ⟨G⟩ (maximum(A2) ≤ 50). Because the global operator G behaves differently from other
#  relations, it is natural to differentiate between global and relational support tables:
# 
abstract type AbstractRelationalSupport{T,W} <: AbstractSupport{T,W}     end
abstract type AbstractGlobalSupport{T}                <: AbstractSupport{T,AbstractWorld} end
#
# Be an *fwd_rs* an fwd relational support, and a *fwd_gs* an fwd global support,
#  for simple support tables like these, it is convenient to store, again, modal *gamma* values.
# Similarly to fwd, gammas are basically values on the verge of truth, that can straightforwardly
#  anser simple modal questions.
# Consider the decision (w ⊨ <R> f ⋈ a) on the i-th instance, for a given feature f,
#  world w, relation R and test operator ⋈, and let gamma (γ) be:
#  - fwd_rs[i, f, a, R, w] if R is a regular relation, or
#  - fwd_gs[i, f, a]       if R is the global relation G,
#  where a = aggregator(⋈). In this context, γ is the unique value for which w ⊨ <R> f ⋈ γ holds and:
#  - if aggregator(⋈) = minimum:     ∀ a > γ:   (w ⊨ <R> f ⋈ a) does not hold
#  - if aggregator(⋈) = maximum:     ∀ a < γ:   (w ⊨ <R> f ⋈ a) does not hold
# 
# Let us define the world type-agnostic implementations for fwd_rs and fwd_gs (note that any fwd_gs
#  is actually inherently world agnostic); world type-specific implementations can be defined
#  in a similar way.

############################################################################################
############################################################################################

isminifiable(::Union{AbstractFWD,AbstractRelationalSupport,AbstractGlobalSupport}) = true

function minify(fwd_or_support::Union{AbstractFWD,AbstractRelationalSupport,AbstractGlobalSupport})
    util.minify(fwd_or_support.d)
end

############################################################################################
############################################################################################

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

# A function that computes fwd_rs and fwd_gs from an explicit modal dataset
Base.@propagate_inbounds function compute_fwd_supports(
        emd                 :: ExplicitModalDataset{T,W},
        grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}};
        compute_relation_glob = false,
        simply_init_modal = false,
    ) where {T,W<:AbstractWorld}

    # @logmsg LogOverview "ExplicitModalDataset -> ExplicitModalDatasetS "

    fwd = emd.fwd
    _features = features(emd)
    _relations = relations(emd)

    compute_fwd_gs = begin
        if RelationGlob in _relations
            throw_n_log("RelationGlob in relations: $(_relations)")
            _relations = filter!(l->l≠RelationGlob, _relations)
            true
        elseif compute_relation_glob
            true
        else
            false
        end
    end

    _n_samples = nsamples(emd)
    nrelations = length(_relations)
    nfeatsnaggrs = sum(length.(grouped_featsnaggrs))

    # println(_n_samples)
    # println(nrelations)
    # println(nfeatsnaggrs)
    # println(grouped_featsnaggrs)

    # Prepare fwd_rs
    fwd_rs = fwd_rs_init(emd, nfeatsnaggrs, nrelations; perform_initialization = simply_init_modal)

    # Prepare fwd_gs
    fwd_gs = begin
        if compute_fwd_gs
            fwd_gs_init(emd, nfeatsnaggrs)
        else
            nothing
        end
    end

    # p = Progress(_n_samples, 1, "Computing EMD supports...")
    Threads.@threads for i_sample in 1:_n_samples
        @logmsg LogDebug "Instance $(i_sample)/$(_n_samples)"

        # if i_sample == 1 || ((i_sample+1) % (floor(Int, ((_n_samples)/4))+1)) == 0
        #     @logmsg LogOverview "Instance $(i_sample)/$(_n_samples)"
        # end

        for (i_feature,aggregators) in enumerate(grouped_featsnaggrs)

            @logmsg LogDebug "Feature $(i_feature)"

            cur_fwd_slice = fwd_get_channel(fwd, i_sample, i_feature)

            @logmsg LogDebug cur_fwd_slice

            # Global relation (independent of the current world)
            if compute_fwd_gs
                @logmsg LogDebug "RelationGlob"

                # TODO optimize: all aggregators are likely reading the same raw values.
                for (i_featsnaggr,aggregator) in aggregators
                # Threads.@threads for (i_featsnaggr,aggregator) in aggregators
                    
                    # accessible_worlds = allworlds(emd, i_sample)
                    accessible_worlds = allworlds_aggr(emd, i_sample, _features[i_feature], aggregator)

                    threshold = compute_modal_gamma(cur_fwd_slice, accessible_worlds, aggregator)

                    @logmsg LogDebug "Aggregator[$(i_featsnaggr)]=$(aggregator)  -->  $(threshold)"

                    # @logmsg LogDebug "Aggregator" aggregator threshold

                    fwd_gs_set(fwd_gs, i_sample, i_featsnaggr, threshold)
                end
            end
            # readline()

            if !simply_init_modal
                # Other relations
                for (i_relation,relation) in enumerate(_relations)

                    @logmsg LogDebug "Relation $(i_relation)/$(nrelations)"

                    for (i_featsnaggr,aggregator) in aggregators
                        fwd_rs_init_world_slice(fwd_rs, i_sample, i_featsnaggr, i_relation)
                    end

                    for w in allworlds(emd, i_sample)

                        @logmsg LogDebug "World" w

                        # TODO optimize: all aggregators are likely reading the same raw values.
                        for (i_featsnaggr,aggregator) in aggregators
                            
                            # accessible_worlds = accessibles(emd, i_sample, w, relation)
                            accessible_worlds = representatives(emd, i_sample, w, relation, _features[i_feature], aggregator)
                        
                            threshold = compute_modal_gamma(cur_fwd_slice, accessible_worlds, aggregator)

                            # @logmsg LogDebug "Aggregator" aggregator threshold

                            fwd_rs_set(fwd_rs, i_sample, w, i_featsnaggr, i_relation, threshold)
                        end
                    end
                end
            end
        end
        # next!(p)
    end
    fwd_rs, fwd_gs
end

############################################################################################
# Finally, let us define two implementations for explicit modal dataset with support, one
#  without memoization and one with memoization
# TODO avoid code duplication
############################################################################################

struct ExplicitModalDatasetS{T<:Number,W<:AbstractWorld} <: ExplicitModalDatasetWithSupport{T,W}

    # Core dataset
    emd                 :: ExplicitModalDataset{T,W}

    # Relational and global support
    fwd_rs              :: AbstractRelationalSupport{T,W}
    fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing}

    # Features and Aggregators
    featsnaggrs         :: AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}}
    grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}}

    function ExplicitModalDatasetS{T,W}(
        emd                 :: ExplicitModalDataset{T,W},
        fwd_rs              :: AbstractRelationalSupport{T,W},
        fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing},
        featsnaggrs         :: AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}},
        grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
    ) where {T,W<:AbstractWorld}
        @assert nsamples(emd) == nsamples(fwd_rs)                               "Can't instantiate ExplicitModalDatasetS{$(T), $(W)} with unmatching nsamples for emd and fwd_rs support: $(nsamples(emd)) and $(nsamples(fwd_rs))"
        @assert nrelations(emd) == nrelations(fwd_rs)                           "Can't instantiate ExplicitModalDatasetS{$(T), $(W)} with unmatching nrelations for emd and fwd_rs support: $(nrelations(emd)) and $(nrelations(fwd_rs))"
        @assert sum(length.(grouped_featsnaggrs)) == length(featsnaggrs)          "Can't instantiate ExplicitModalDatasetS{$(T), $(W)} with unmatching nfeatsnaggrs (grouped vs flattened structure): $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == length(featsnaggrs)   "Can't instantiate ExplicitModalDatasetS{$(T), $(W)} with unmatching nfeatsnaggrs for emd and provided featsnaggrs: $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == nfeatsnaggrs(fwd_rs) "Can't instantiate ExplicitModalDatasetS{$(T), $(W)} with unmatching nfeatsnaggrs for emd and fwd_rs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(nfeatsnaggrs(fwd_rs))"

        if fwd_gs != nothing
            @assert nsamples(emd) == nsamples(fwd_gs) "Can't instantiate ExplicitModalDatasetS{$(T), $(W)} with unmatching nsamples for emd and fwd_gs support: $(nsamples(emd)) and $(nsamples(fwd_gs))"
            # @assert somethinglike(emd) == nfeatsnaggrs(fwd_gs) "Can't instantiate ExplicitModalDatasetS{$(T), $(W)} with unmatching somethinglike for emd and fwd_gs support: $(somethinglike(emd)) and $(nfeatsnaggrs(fwd_gs))"
            @assert sum(length.(emd.grouped_featsaggrsnops)) == nfeatsnaggrs(fwd_gs) "Can't instantiate ExplicitModalDatasetS{$(T), $(W)} with unmatching nfeatsnaggrs for emd and fwd_gs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(nfeatsnaggrs(fwd_gs))"
        end

        new{T,W}(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs)
    end

    function ExplicitModalDatasetS(
        emd                 :: ExplicitModalDataset{T,W};
        compute_relation_glob :: Bool = true,
    ) where {T,W<:AbstractWorld}
        ExplicitModalDatasetS{T,W}(emd, compute_relation_glob = compute_relation_glob)
    end

    function ExplicitModalDatasetS{T,W}(
        emd                   :: ExplicitModalDataset{T,W};
        compute_relation_glob :: Bool = true,
    ) where {T,W<:AbstractWorld}
        
        featsnaggrs = Tuple{<:AbstractFeature,<:Aggregator}[]
        grouped_featsnaggrs = AbstractVector{Tuple{<:Integer,<:Aggregator}}[]

        i_featsnaggr = 1
        for (feat,aggrsnops) in zip(emd.features,emd.grouped_featsaggrsnops)
            aggrs = []
            for aggr in keys(aggrsnops)
                push!(featsnaggrs, (feat,aggr))
                push!(aggrs, (i_featsnaggr,aggr))
                i_featsnaggr += 1
            end
            push!(grouped_featsnaggrs, aggrs)
        end

        # Compute modal dataset propositions and 1-modal decisions
        fwd_rs, fwd_gs = compute_fwd_supports(emd, grouped_featsnaggrs, compute_relation_glob = compute_relation_glob);

        ExplicitModalDatasetS{T,W}(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs)
    end

    function ExplicitModalDatasetS(
        X                   :: InterpretedModalDataset{T,N,W};
        compute_relation_glob :: Bool = true,
    ) where {T,N,W<:AbstractWorld}
        ExplicitModalDatasetS{T,W}(X, compute_relation_glob = compute_relation_glob)
    end

    function ExplicitModalDatasetS{T,W}(
        X                   :: InterpretedModalDataset{T,N,W};
        compute_relation_glob :: Bool = true,
    ) where {T,N,W<:AbstractWorld}

        # Compute modal dataset propositions
        emd = ExplicitModalDataset(X);

        ExplicitModalDatasetS{T,W}(emd, compute_relation_glob = compute_relation_glob)
    end
end

mutable struct ExplicitModalDatasetSMemo{T<:Number,W<:AbstractWorld} <: ExplicitModalDatasetWithSupport{T,W}

    # Core dataset
    emd                 :: ExplicitModalDataset{T,W}

    # Relational and global support
    fwd_rs              :: AbstractRelationalSupport{<:Union{T,Nothing}, W}
    fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing} # TODO maybe nothing is not needed here

    # Features and Aggregators
    featsnaggrs         :: AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}}
    grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}}

    function ExplicitModalDatasetSMemo{T,W}(
        emd                 :: ExplicitModalDataset{T,W},
        fwd_rs              :: AbstractRelationalSupport{<:Union{T,Nothing}, W},
        fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing},
        featsnaggrs         :: AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}},
        grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
    ) where {T,W<:AbstractWorld}
        @assert nsamples(emd) == nsamples(fwd_rs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching nsamples for emd and fwd_rs support: $(nsamples(emd)) and $(nsamples(fwd_rs))"
        @assert nrelations(emd) == nrelations(fwd_rs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching nrelations for emd and fwd_rs support: $(nrelations(emd)) and $(nrelations(fwd_rs))"
        @assert sum(length.(grouped_featsnaggrs)) == length(featsnaggrs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching nfeatsnaggrs (grouped vs flattened structure): $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == length(featsnaggrs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching nfeatsnaggrs for emd and provided featsnaggrs: $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == nfeatsnaggrs(fwd_rs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching nfeatsnaggrs for emd and fwd_rs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(nfeatsnaggrs(fwd_rs))"

        if fwd_gs != nothing
            @assert nsamples(emd) == nsamples(fwd_gs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching nsamples for emd and fwd_gs support: $(nsamples(emd)) and $(nsamples(fwd_gs))"
            # @assert somethinglike(emd) == nfeatsnaggrs(fwd_gs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching somethinglike for emd and fwd_gs support: $(somethinglike(emd)) and $(nfeatsnaggrs(fwd_gs))"
            @assert sum(length.(emd.grouped_featsaggrsnops)) == nfeatsnaggrs(fwd_gs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching nfeatsnaggrs for emd and fwd_gs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(nfeatsnaggrs(fwd_gs))"
        end

        new{T,W}(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs)
    end

    function ExplicitModalDatasetSMemo(
        emd                 :: ExplicitModalDataset{T,W};
        compute_relation_glob :: Bool = true,
    ) where {T,W<:AbstractWorld}
        ExplicitModalDatasetSMemo{T,W}(emd, compute_relation_glob = compute_relation_glob)
    end

    function ExplicitModalDatasetSMemo{T,W}(
        emd                   :: ExplicitModalDataset{T,W};
        compute_relation_glob :: Bool = true,
    ) where {T,W<:AbstractWorld}
        featsnaggrs = Tuple{<:AbstractFeature,<:Aggregator}[]
        grouped_featsnaggrs = AbstractVector{Tuple{<:Integer,<:Aggregator}}[]

        i_featsnaggr = 1
        for (feat,aggrsnops) in zip(emd.features,emd.grouped_featsaggrsnops)
            aggrs = []
            for aggr in keys(aggrsnops)
                push!(featsnaggrs, (feat,aggr))
                push!(aggrs, (i_featsnaggr,aggr))
                i_featsnaggr += 1
            end
            push!(grouped_featsnaggrs, aggrs)
        end

        # Compute modal dataset propositions and 1-modal decisions
        fwd_rs, fwd_gs = compute_fwd_supports(emd, grouped_featsnaggrs, compute_relation_glob = compute_relation_glob, simply_init_modal = true);

        ExplicitModalDatasetSMemo{T,W}(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs)
    end

    function ExplicitModalDatasetSMemo(
        X                   :: InterpretedModalDataset{T,N,W};
        compute_relation_glob :: Bool = true,
    ) where {T,N,W<:AbstractWorld}
        ExplicitModalDatasetSMemo{T,W}(X, compute_relation_glob = compute_relation_glob)
    end

    function ExplicitModalDatasetSMemo{T,W}(
        X                   :: InterpretedModalDataset{T,N,W};
        compute_relation_glob :: Bool = true,
    ) where {T,N,W<:AbstractWorld}

        # Compute modal dataset propositions
        emd = ExplicitModalDataset(X);

        ExplicitModalDatasetSMemo{T,W}(emd, compute_relation_glob = compute_relation_glob)
    end
end

# getindex(X::ExplicitModalDatasetWithSupport{T,W}, args...) where {T,W} = getindex(X.emd, args...)
Base.size(X::ExplicitModalDatasetWithSupport)                                      =  (size(X.emd), size(X.fwd_rs), (isnothing(X.fwd_gs) ? nothing : size(X.fwd_gs)))
featsnaggrs(X::ExplicitModalDatasetWithSupport)                                    = X.featsnaggrs
features(X::ExplicitModalDatasetWithSupport)                                       = features(X.emd)
grouped_featsaggrsnops(X::ExplicitModalDatasetWithSupport)                         = grouped_featsaggrsnops(X.emd)
grouped_featsnaggrs(X::ExplicitModalDatasetWithSupport)                            = X.grouped_featsnaggrs
nfeatures(X::ExplicitModalDatasetWithSupport)                                      = nfeatures(X.emd)
nrelations(X::ExplicitModalDatasetWithSupport)                                     = nrelations(X.emd)
nsamples(X::ExplicitModalDatasetWithSupport)                                       = nsamples(X.emd)::Int64
relations(X::ExplicitModalDatasetWithSupport)                                      = relations(X.emd)
world_type(X::ExplicitModalDatasetWithSupport{T,W}) where {T,W}    = W

initialworldset(X::ExplicitModalDatasetWithSupport,  args...) = initialworldset(X.emd, args...)
accessibles(X::ExplicitModalDatasetWithSupport,     args...) = accessibles(X.emd, args...)
representatives(X::ExplicitModalDatasetWithSupport, args...) = representatives(X.emd, args...)
allworlds(X::ExplicitModalDatasetWithSupport,  args...) = allworlds(X.emd, args...)

function slice_dataset(X::ExplicitModalDatasetWithSupport, inds::AbstractVector{<:Integer}, args...; kwargs...)
    typeof(X)(
        slice_dataset(X.emd, inds, args...; kwargs...),
        slice_dataset(X.fwd_rs, inds, args...; kwargs...),
        (isnothing(X.fwd_gs) ? nothing : slice_dataset(X.fwd_gs, inds, args...; kwargs...)),
        X.featsnaggrs,
        X.grouped_featsnaggrs)
end

find_feature_id(X::ExplicitModalDatasetWithSupport, feature::AbstractFeature) =
    findall(x->x==feature, features(X))[1]
find_relation_id(X::ExplicitModalDatasetWithSupport, relation::AbstractRelation) =
    findall(x->x==relation, relations(X))[1]
find_featsnaggr_id(X::ExplicitModalDatasetWithSupport, feature::AbstractFeature, aggregator::Aggregator) =
    findall(x->x==(feature, aggregator), featsnaggrs(X))[1]

hasnans(X::ExplicitModalDatasetWithSupport) = begin
    # @show hasnans(X.emd)
    # @show hasnans(X.fwd_rs)
    # @show (!isnothing(X.fwd_gs) && hasnans(X.fwd_gs))
    hasnans(X.emd) || hasnans(X.fwd_rs) || (!isnothing(X.fwd_gs) && hasnans(X.fwd_gs))
end

Base.@propagate_inbounds @inline get_gamma(
        X::ExplicitModalDatasetWithSupport{T,W},
        i_sample::Integer,
        w::W,
        feature::AbstractFeature) where {T,W<:AbstractWorld} = get_gamma(X.emd, i_sample, w, feature)

isminifiable(::ExplicitModalDatasetWithSupport) = true

function minify(X::EMD) where {EMD<:ExplicitModalDatasetWithSupport}
    (new_emd, new_fwd_rs, new_fwd_gs), backmap =
        util.minify([
            X.emd,
            X.fwd_rs,
            X.fwd_gs,
        ])

    X = EMD(
        new_emd,
        new_fwd_rs,
        new_fwd_gs,
        featsnaggrs,
        grouped_featsnaggrs,
    )
    X, backmap
end
############################################################################################

get_global_gamma(
        X::ExplicitModalDatasetWithSupport{T,W},
        i_sample::Integer,
        feature::AbstractFeature,
        test_operator::TestOperatorFun) where {T,W<:AbstractWorld} = begin
            # @assert !isnothing(X.fwd_gs) "Error. ExplicitModalDatasetWithSupport must be built with compute_relation_glob = true for it to be ready to test global decisions."
            i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
            X.fwd_gs[i_sample, i_featsnaggr]
end

get_modal_gamma(
        X::ExplicitModalDatasetS{T,W},
        i_sample::Integer,
        w::W,
        relation::AbstractRelation,
        feature::AbstractFeature,
        test_operator::TestOperatorFun) where {T,W<:AbstractWorld} = begin
            i_relation = find_relation_id(X, relation)
            i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
            X.fwd_rs[i_sample, w, i_featsnaggr, i_relation]
end

function display_structure(X::ExplicitModalDatasetS; indent_str = "")
    out = "$(typeof(X))\t$((Base.summarysize(X.emd) + Base.summarysize(X.fwd_rs) + Base.summarysize(X.fwd_gs)) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(X.emd))))\t$(relations(X.emd))\n"
    out *= indent_str * "├ emd\t$(Base.summarysize(X.emd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t(shape $(Base.size(X.emd)))\n"
    out *= indent_str * "├ fwd_rs\t$(Base.summarysize(X.fwd_rs) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t(shape $(Base.size(X.fwd_rs)))\n"
    out *= indent_str * "└ fwd_gs\t$(Base.summarysize(X.fwd_gs) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t"
    if !isnothing(X.fwd_gs)
        out *= "\t(shape $(Base.size(X.fwd_gs)))"
    else
        out *= "\t−"
    end
    out
end

function display_structure(X::ExplicitModalDatasetSMemo; indent_str = "")
    out = "$(typeof(X))\t$((Base.summarysize(X.emd) + Base.summarysize(X.fwd_rs) + Base.summarysize(X.fwd_gs)) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(X.emd))))\t$(relations(X.emd))\n"
    out *= indent_str * "├ emd\t$(Base.summarysize(X.emd) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
        out *= "\t(shape $(Base.size(X.emd.fwd)))\n"
        # out *= "\t(shape $(Base.size(X.emd.fwd)), $(n_nothing) nothings, $((1-(n_nothing    / size(X.emd)))*100)%)\n"
    n_nothing_m = count(isnothing, X.fwd_rs.d)
    out *= indent_str * "├ fwd_rs\t$(Base.summarysize(X.fwd_rs) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
        out *= "\t(shape $(Base.size(X.fwd_rs)), $(n_nothing_m) nothings)\n" # , $((1-(n_nothing_m  / n_v(X.fwd_rs)))*100)%)\n"
    out *= indent_str * "└ fwd_gs\t$(Base.summarysize(X.fwd_gs) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t"
    if !isnothing(X.fwd_gs)
        # n_nothing_g = count(isnothing, X.fwd_gs)
        # out *= "\t(shape $(Base.size(X.fwd_gs)), $(n_nothing_g) nothings, $((1-(n_nothing_g  / size(X.fwd_gs)))*100)%)"
        out *= "\t(shape $(Base.size(X.fwd_gs)))"
    else
        out *= "\t−"
    end
    out
end

############################################################################################
############################################################################################
############################################################################################

test_decision(
        X::ExplicitModalDatasetWithSupport{T,W},
        i_sample::Integer,
        w::W,
        decision::ExistentialDimensionalDecision) where {T,W<:AbstractWorld} = begin
    if is_propositional_decision(decision)
        test_decision(X, i_sample, w, feature(decision), test_operator(decision), threshold(decision))
    else
        gamma = begin
            if relation(decision) isa _RelationGlob
                get_global_gamma(X, i_sample, feature(decision), test_operator(decision))
            else
                get_modal_gamma(X, i_sample, w, relation(decision), feature(decision), test_operator(decision))
            end
        end
        evaluate_thresh_decision(test_operator(decision), gamma, threshold(decision))
    end
end


Base.@propagate_inbounds @resumable function generate_propositional_feasible_decisions(
        emd::Union{ExplicitModalDataset{T,W},InterpretedModalDataset{T,W}},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{W}},
        features_inds::AbstractVector{<:Integer},
        ) where {T,W<:AbstractWorld}
    relation = RelationId
    _n_samples = length(instances_inds)

    # For each feature
    @inbounds for i_feature in features_inds
        feature = features(emd)[i_feature]
        @logmsg LogDebug "Feature $(i_feature): $(feature)"

        # operators for each aggregator
        aggrsnops = grouped_featsaggrsnops(emd)[i_feature]
        # Vector of aggregators
        aggregators = keys(aggrsnops) # Note: order-variant, but that's ok here
        
        # dict->vector
        # aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

        # Initialize thresholds with the bottoms
        thresholds = Array{T,2}(undef, length(aggregators), _n_samples)
        for (i_aggr,aggr) in enumerate(aggregators)
            thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
        end

        # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
        for (instance_idx,i_sample) in enumerate(instances_inds)
            @logmsg LogDetail " Instance $(instance_idx)/$(_n_samples)"
            worlds = Sf[instance_idx]

            # TODO also try this instead
            # values = [X.emd[i_sample, w, i_feature] for w in worlds]
            # thresholds[:,instance_idx] = map(aggr->aggr(values), aggregators)
            
            for w in worlds
                gamma = begin
                    if emd isa ExplicitModalDataset{T,W}
                        fwd_get(emd.fwd, i_sample, w, i_feature) # faster but equivalent to get_gamma(emd, i_sample, w, feature)
                    elseif emd isa InterpretedModalDataset{T,W}
                        get_gamma(emd, i_sample, w, feature)
                    else
                        error("generate_propositional_feasible_decisions is broken.")
                    end
                end
                for (i_aggr,aggr) in enumerate(aggregators)
                    thresholds[i_aggr,instance_idx] = aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,instance_idx])
                end
            end
        end
        
        # tested_test_operator = TestOperatorFun[]

        # @logmsg LogDebug "thresholds: " thresholds
        # For each aggregator
        for (i_aggr,aggr) in enumerate(aggregators)
            aggr_thresholds = thresholds[i_aggr,:]
            aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))
            for (i_test_operator,test_operator) in enumerate(aggrsnops[aggr])
                # TODO figure out a solution to this issue: ≥ and ≤ in a propositional condition can find more or less the same optimum, so no need to check both; but which one of them should be the one on the left child, the one that makes the modal step?
                # if dual_test_operator(test_operator) in tested_test_operator
                #   throw_n_log("Double-check this part of the code: there's a foundational issue here to settle!")
                #   println("Found $(test_operator)'s dual $(dual_test_operator(test_operator)) in tested_test_operator = $(tested_test_operator)")
                #   continue
                # end
                @logmsg LogDetail " Test operator $(test_operator)"
                # Look for the best threshold 'a', as in propositions like "feature >= a"
                for threshold in aggr_domain
                    decision = ExistentialDimensionalDecision(relation, feature, test_operator, threshold)
                    @logmsg LogDebug " Testing decision: $(display_decision(decision))"
                    @yield decision, aggr_thresholds
                end # for threshold
                # push!(tested_test_operator, test_operator)
            end # for test_operator
        end # for aggregator
    end # for feature
end

Base.@propagate_inbounds @resumable function generate_propositional_feasible_decisions(
        X::ExplicitModalDatasetWithSupport{T,W},
        args...
        ) where {T,W<:AbstractWorld}
        for decision in generate_propositional_feasible_decisions(X.emd, args...)
            @yield decision
        end
end

Base.@propagate_inbounds @resumable function generate_global_feasible_decisions(
        X::ExplicitModalDatasetWithSupport{T,W},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{W}},
        features_inds::AbstractVector{<:Integer},
        ) where {T,W<:AbstractWorld}
    relation = RelationGlob
    _n_samples = length(instances_inds)

    
    @assert !isnothing(X.fwd_gs) "Error. ExplicitModalDatasetWithSupport must be built with compute_relation_glob = true for it to be ready to generate global decisions."

    # For each feature
    for i_feature in features_inds
        feature = features(X)[i_feature]
        @logmsg LogDebug "Feature $(i_feature): $(feature)"

        # operators for each aggregator
        aggrsnops = grouped_featsaggrsnops(X)[i_feature]
        # println(aggrsnops)
        # Vector of aggregators
        aggregators_with_ids = grouped_featsnaggrs(X)[i_feature]
        # println(aggregators_with_ids)

        # dict->vector
        # aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

        # # TODO use this optimized version:
        #   thresholds can in fact be directly given by slicing fwd_gs and permuting the two dimensions
        # aggregators_ids = fst.(aggregators_with_ids)
        # thresholds = transpose(X.fwd_gs[instances_inds, aggregators_ids])

        # Initialize thresholds with the bottoms
        thresholds = Array{T,2}(undef, length(aggregators_with_ids), _n_samples)
        for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)
            thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
        end
        
        # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
        for (instance_id,i_sample) in enumerate(instances_inds)
            @logmsg LogDetail " Instance $(instance_id)/$(_n_samples)"
            for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
                gamma = X.fwd_gs[i_sample, i_featsnaggr]
                thresholds[i_aggr,instance_id] = aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,instance_id])
                # println(gamma)
                # println(thresholds[i_aggr,instance_id])
            end
        end

        # println(thresholds)
        @logmsg LogDebug "thresholds: " thresholds

        # For each aggregator
        for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)

            # println(aggr)

            aggr_thresholds = thresholds[i_aggr,:]
            aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))

            for (i_test_operator,test_operator) in enumerate(aggrsnops[aggr])
                @logmsg LogDetail " Test operator $(test_operator)"

                # Look for the best threshold 'a', as in propositions like "feature >= a"
                for threshold in aggr_domain
                    decision = ExistentialDimensionalDecision(relation, feature, test_operator, threshold)
                    @logmsg LogDebug " Testing decision: $(display_decision(decision))"
                    @yield decision, aggr_thresholds
                end # for threshold
            end # for test_operator
        end # for aggregator
    end # for feature
end


Base.@propagate_inbounds @resumable function generate_modal_feasible_decisions(
        X::ExplicitModalDatasetS{T,W},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{W}},
        modal_relations_inds::AbstractVector{<:Integer},
        features_inds::AbstractVector{<:Integer},
        ) where {T,W<:AbstractWorld}
    _n_samples = length(instances_inds)

    # For each relational operator
    for i_relation in modal_relations_inds
        relation = relations(X)[i_relation]
        @logmsg LogDebug "Relation $(relation)..."

        # For each feature
        for i_feature in features_inds
            feature = features(X)[i_feature]
            @logmsg LogDebug "Feature $(i_feature): $(feature)"

            # operators for each aggregator
            aggrsnops = grouped_featsaggrsnops(X)[i_feature]
            # Vector of aggregators
            aggregators_with_ids = grouped_featsnaggrs(X)[i_feature]

            # dict->vector
            # aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

            # Initialize thresholds with the bottoms
            thresholds = Array{T,2}(undef, length(aggregators_with_ids), _n_samples)
            for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)
                thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
            end

            # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
                for (i_sample,instance_id) in enumerate(instances_inds)
                @logmsg LogDetail " Instance $(i_sample)/$(_n_samples)"
                worlds = Sf[i_sample] # TODO could also use representativess here?

                # TODO also try this instead (TODO fix first)
                # values = [X.fwd_rs[instance_id, w, i_feature] for w in worlds]
                # thresholds[:,i_sample] = map((_,aggr)->aggr(values), aggregators_with_ids)
                    
                for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
                    for w in worlds
                        gamma = X.fwd_rs[instance_id, w, i_featsnaggr, i_relation]
                        thresholds[i_aggr,i_sample] = aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,i_sample])
                    end
                end
            end

            @logmsg LogDebug "thresholds: " thresholds

            # For each aggregator
            for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)

                aggr_thresholds = thresholds[i_aggr,:]
                aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))

                for (i_test_operator,test_operator) in enumerate(aggrsnops[aggr])
                    @logmsg LogDetail " Test operator $(test_operator)"

                    # Look for the best threshold 'a', as in propositions like "feature >= a"
                    for threshold in aggr_domain
                        decision = ExistentialDimensionalDecision(relation, feature, test_operator, threshold)
                        @logmsg LogDebug " Testing decision: $(display_decision(decision))"
                        @yield decision, aggr_thresholds
                    end # for threshold
                end # for test_operator
            end # for aggregator
        end # for feature
    end # for relation
end

# Memoization for global gammas
# get_global_gamma(
#       X::ExplicitModalDatasetSMemo{T,W},
#       i_sample::Integer,
#       feature::AbstractFeature,
#       test_operator::TestOperatorFun) where {T,W<:AbstractWorld} = begin
#   @assert !isnothing(X.fwd_gs) "Error. ExplicitModalDatasetSMemo must be built with compute_relation_glob = true for it to be ready to test global decisions."
#   i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
#   # if !isnothing(X.fwd_gs[i_sample, i_featsnaggr])
#   X.fwd_gs[i_sample, i_featsnaggr]
#   # else
#   #   i_feature = find_feature_id(X, feature)
#   #   aggregator = existential_aggregator(test_operator)
#   #   fwd_feature_slice = fwd_get_channel(X.emd.fwd, i_sample, i_feature)
#   #   accessible_worlds = allworlds_aggr(X.emd, i_sample, feature, aggregator)
#   #   gamma = compute_modal_gamma(fwd_feature_slice, accessible_worlds, aggregator)
#   #   fwd_gs_set(X.fwd_gs, i_sample, i_featsnaggr, gamma)
#   # end
# end

# TODO scan this value for an example problem and different number of threads

# using Random
# coin_flip_memoiz_rng = Random.default_rng()

# cfnls_max = 0.8
# # cfnls_k = 5.9
# cfnls_k = 30
# coin_flip_no_look_ExplicitModalDatasetSWithMemoization_value = cfnls_max*cfnls_k/((Threads.nthreads())-1+cfnls_k)
# coin_flip_no_look_ExplicitModalDatasetSWithMemoization() = (rand(coin_flip_memoiz_rng) >= coin_flip_no_look_ExplicitModalDatasetSWithMemoization_value)
# coin_flip_no_look_ExplicitModalDatasetSWithMemoization() = false

get_modal_gamma(
        X::ExplicitModalDatasetSMemo{T,W},
        i_sample::Integer,
        w::W,
        relation::AbstractRelation,
        feature::AbstractFeature,
        test_operator::TestOperatorFun) where {T,W<:AbstractWorld} = begin
    i_relation = find_relation_id(X, relation)
    aggregator = existential_aggregator(test_operator)
    i_featsnaggr = find_featsnaggr_id(X, feature, aggregator)
    # if coin_flip_no_look_ExplicitModalDatasetSWithMemoization() || 
    if false || 
            isnothing(X.fwd_rs[i_sample, w, i_featsnaggr, i_relation])
        i_feature = find_feature_id(X, feature)
        fwd_feature_slice = fwd_get_channel(X.emd.fwd, i_sample, i_feature)
        accessible_worlds = representatives(X.emd, i_sample, w, relation, feature, aggregator)
        gamma = compute_modal_gamma(fwd_feature_slice, accessible_worlds, aggregator)
        fwd_rs_set(X.fwd_rs, i_sample, w, i_featsnaggr, i_relation, gamma)
    else
        X.fwd_rs[i_sample, w, i_featsnaggr, i_relation]
    end
end


Base.@propagate_inbounds @resumable function generate_modal_feasible_decisions(
        X::ExplicitModalDatasetSMemo{T,W},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{W}},
        modal_relations_inds::AbstractVector{<:Integer},
        features_inds::AbstractVector{<:Integer},
        ) where {T,W<:AbstractWorld}
    _n_samples = length(instances_inds)

    # For each relational operator
    for i_relation in modal_relations_inds
        relation = relations(X)[i_relation]
        @logmsg LogDebug "Relation $(relation)..."

        # For each feature
        for i_feature in features_inds
            feature = features(X)[i_feature]
            @logmsg LogDebug "Feature $(i_feature): $(feature)"

            # operators for each aggregator
            aggrsnops = grouped_featsaggrsnops(X)[i_feature]
            # Vector of aggregators
            aggregators_with_ids = grouped_featsnaggrs(X)[i_feature]

            # dict->vector
            # aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

            # Initialize thresholds with the bottoms
            thresholds = Array{T,2}(undef, length(aggregators_with_ids), _n_samples)
            for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)
                thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
            end

            # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
            for (instance_id,i_sample) in enumerate(instances_inds)
                @logmsg LogDetail " Instance $(instance_id)/$(_n_samples)"
                worlds = Sf[instance_id] # TODO could also use representativess here?

                # TODO also try this instead (TODO fix first)
                # values = [X.fwd_rs[i_sample, w, i_feature] for w in worlds]
                # thresholds[:,instance_id] = map((_,aggr)->aggr(values), aggregators_with_ids)
                    
                for (i_aggr,(i_featsnaggr,aggregator)) in enumerate(aggregators_with_ids)
                    for w in worlds
                        gamma = 
                            # if coin_flip_no_look_ExplicitModalDatasetSWithMemoization() || 
                            if false || 
                                isnothing(X.fwd_rs[i_sample, w, i_featsnaggr, i_relation])
                                fwd_feature_slice = fwd_get_channel(X.emd.fwd, i_sample, i_feature)
                                accessible_worlds = representatives(X.emd, i_sample, w, relation, feature, aggregator)
                                gamma = compute_modal_gamma(fwd_feature_slice, accessible_worlds, aggregator)
                                fwd_rs_set(X.fwd_rs, i_sample, w, i_featsnaggr, i_relation, gamma)
                            else
                                X.fwd_rs[i_sample, w, i_featsnaggr, i_relation]
                            end
                        thresholds[i_aggr,instance_id] = aggregator_to_binary(aggregator)(gamma, thresholds[i_aggr,instance_id])
                    end
                end
            end

            @logmsg LogDebug "thresholds: " thresholds

            # For each aggregator
            for (i_aggr,(_,aggregator)) in enumerate(aggregators_with_ids)

                aggr_thresholds = thresholds[i_aggr,:]
                aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))

                for (i_test_operator,test_operator) in enumerate(aggrsnops[aggregator])
                    @logmsg LogDetail " Test operator $(test_operator)"

                    # Look for the best threshold 'a', as in propositions like "feature >= a"
                    for threshold in aggr_domain
                        decision = ExistentialDimensionalDecision(relation, feature, test_operator, threshold)
                        @logmsg LogDebug " Testing decision: $(display_decision(decision))"
                        @yield decision, aggr_thresholds
                    end # for threshold
                end # for test_operator
            end # for aggregator
        end # for feature
    end # for relation
end

############################################################################################
############################################################################################
############################################################################################

# Perform the modal step, that is, evaluate a modal formula
#  on a domain, and eventually compute the new world set.
function modal_step(
        X::Union{ActiveModalDataset{T,W},InterpretedModalDataset{T,N,W}},
        i_sample::Integer,
        worlds::WorldSetType,
        decision::ExistentialDimensionalDecision{T},
        returns_survivors::Union{Val{true},Val{false}} = Val(false)
    ) where {T, N, W<:AbstractWorld, WorldSetType<:AbstractWorldSet{W}}
    @logmsg LogDetail "modal_step" worlds display_decision(decision)

    satisfied = false
    
    # TODO space for optimization here: with some relations (e.g. IA_A, IA_L) can be made smaller

    if returns_survivors isa Val{true}
        worlds_map = Dict{W,AbstractWorldSet{W}}()
    end
    if length(worlds) == 0
        # If there are no neighboring worlds, then the modal decision is not met
        @logmsg LogDetail "   No accessible world"
    else
        # Otherwise, check whether at least one of the accessible worlds witnesses truth of the decision.
        # TODO rewrite with new_worlds = map(...acc_worlds)
        # Initialize new worldset
        new_worlds = WorldSetType()

        # List all accessible worlds
        acc_worlds = 
            if returns_survivors isa Val{true}
                Threads.@threads for curr_w in worlds
                    l = ReentrantLock()
                    acc = accessibles(X, i_sample, curr_w, relation(decision)) |> collect
                    lock(l)
                    worlds_map[curr_w] = acc
                    unlock(l)
                end
                unique(cat([ worlds_map[k] for k in keys(worlds_map) ]...; dims = 1))
            else
                accessibles(X, i_sample, worlds, relation(decision))
            end

        for w in acc_worlds
            if test_decision(X, i_sample, w, feature(decision), test_operator(decision), threshold(decision))
                # @logmsg LogDetail " Found world " w ch_readWorld ... ch_readWorld(w, channel)
                satisfied = true
                push!(new_worlds, w)
            end
        end

        if satisfied == true
            worlds = new_worlds
        else
            # If none of the neighboring worlds satisfies the decision, then 
            #  the new set is left unchanged
        end
    end
    if satisfied
        @logmsg LogDetail "   YES" worlds
    else
        @logmsg LogDetail "   NO"
    end
    if returns_survivors isa Val{true}
        return (satisfied, worlds, worlds_map)
    else
        return (satisfied, worlds)
    end
end

test_decision(
        X::ModalDataset{T},
        i_sample::Integer,
        w::AbstractWorld,
        feature::AbstractFeature,
        test_operator::TestOperatorFun,
        threshold::T) where {T} = begin
    gamma = get_gamma(X, i_sample, w, feature)
    evaluate_thresh_decision(test_operator, gamma, threshold)
end

test_decision(
        X::ModalDataset{T},
        i_sample::Integer,
        w::AbstractWorld,
        decision::ExistentialDimensionalDecision{T}) where {T} = begin
    instance = get_instance(X, i_sample)

    aggregator = existential_aggregator(test_operator(decision))
    
    worlds = representatives(FullDimensionalFrame(instance_channel_size(instance)), w, relation(decision), feature(decision), aggregator)
    gamma = if length(worlds |> collect) == 0
        aggregator_bottom(aggregator, T)
    else
        aggregator((w)->get_gamma(X, i_sample, w, feature(decision)), worlds)
    end

    evaluate_thresh_decision(test_operator(decision), gamma, threshold(decision))
end


export generate_feasible_decisions
                # ,
                # generate_propositional_feasible_decisions,
                # generate_global_feasible_decisions,
                # generate_modal_feasible_decisions

Base.@propagate_inbounds @resumable function generate_feasible_decisions(
        X::ActiveModalDataset{T,W},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{W}},
        allow_propositional_decisions::Bool,
        allow_modal_decisions::Bool,
        allow_global_decisions::Bool,
        modal_relations_inds::AbstractVector{<:Integer},
        features_inds::AbstractVector{<:Integer},
        ) where {T,W<:AbstractWorld}
    # Propositional splits
    if allow_propositional_decisions
        for decision in generate_propositional_feasible_decisions(X, instances_inds, Sf, features_inds)
            @yield decision
        end
    end
    # Global splits
    if allow_global_decisions
        for decision in generate_global_feasible_decisions(X, instances_inds, Sf, features_inds)
            @yield decision
        end
    end
    # Modal splits
    if allow_modal_decisions
        for decision in generate_modal_feasible_decisions(X, instances_inds, Sf, modal_relations_inds, features_inds)
            @yield decision
        end
    end
end


# function slice_dataset(x::Any, dataset_slice::AbstractVector{<:Integer}; allow_no_instances = false, kwargs...)
#     @assert (allow_no_instances || length(dataset_slice) > 0) "Can't apply empty slice to dataset."
#     slice_dataset(x, dataset_slice; kwargs...)
# end
