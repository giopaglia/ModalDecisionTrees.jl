using ProgressMeter

using ..ModalDecisionTrees: CanonicalFeatureGeq, CanonicalFeatureGeqSoft, CanonicalFeatureLeq, CanonicalFeatureLeqSoft
using ..ModalDecisionTrees: evaluate_thresh_decision, existential_aggregator, aggregator_bottom, aggregator_to_binary

const initWorldSetFunction = Function
const accFunction = Function
const accReprFunction = Function

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
#  relations), and structures for interpreting modal features onto the domain.
#
############################################################################################

@computed struct InterpretedModalDataset{T<:Number, N, WorldType<:World} <: ActiveModalDataset{T, WorldType}

    # Core data (a dimensional domain)
    domain                  :: DimensionalDataset{T,N+1+1}

    # Worlds & Relations
    ontology                :: Ontology{WorldType} # Union{Nothing,}

    # Features
    features                :: AbstractVector{ModalFeature}

    # Test operators associated with each feature, grouped by their respective aggregator
    # Note: currently, cannot specify the full type (probably due to @computed)
    grouped_featsaggrsnops  :: AbstractVector # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}

    function InterpretedModalDataset(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        mixed_features::AbstractVector{<:MixedFeature},
    ) where {T, N, D, WorldType<:World}
        InterpretedModalDataset{T}(domain, ontology, mixed_features)
    end

    function InterpretedModalDataset{T}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        mixed_features::AbstractVector{<:MixedFeature},
    ) where {T, N, D, WorldType<:World}
        InterpretedModalDataset{T, D-1-1}(domain, ontology, mixed_features)
    end

    function InterpretedModalDataset{T, N}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        mixed_features::AbstractVector{<:MixedFeature},
    ) where {T, N, D, WorldType<:World}
        InterpretedModalDataset{T, N, WorldType}(domain, ontology, mixed_features)
    end

    function InterpretedModalDataset{T, N, WorldType}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType}, # default to get_interval_ontology(Val(D-1-1)) ?
        mixed_features::AbstractVector{<:MixedFeature},
    ) where {T, N, D, WorldType<:World}
        _features, featsnops = begin
            _features = ModalFeature[]
            featsnops = Vector{<:TestOperatorFun}[]

            # readymade features
            cnv_feat(cf::ModalFeature) = ([≥, ≤], cf)
            cnv_feat(cf::Tuple{TestOperatorFun,ModalFeature}) = ([cf[1]], cf[2])
            # single-attribute features
            cnv_feat(cf::Any) = cf
            cnv_feat(cf::Function) = ([≥, ≤], cf)
            cnv_feat(cf::Tuple{TestOperatorFun,Function}) = ([cf[1]], cf[2])

            mixed_features = cnv_feat.(mixed_features)

            readymade_cfs          = filter(x->isa(x, Tuple{<:AbstractVector{<:TestOperatorFun},ModalFeature}), mixed_features)
            attribute_specific_cfs = filter(x->isa(x, CanonicalFeature) || isa(x, Tuple{<:AbstractVector{<:TestOperatorFun},Function}), mixed_features)

            @assert length(readymade_cfs) + length(attribute_specific_cfs) == length(mixed_features) "Unexpected mixed_features: $(filter(x->(! (x in readymade_cfs) && ! (x in attribute_specific_cfs)), mixed_features))"

            for (test_ops,cf) in readymade_cfs
                push!(_features, cf)
                push!(featsnops, test_ops)
            end

            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic.CanonicalFeatureGeq) = ([≥],ModalDecisionTrees.SingleAttributeMin(i_attr))
            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic.CanonicalFeatureLeq) = ([≤],ModalDecisionTrees.SingleAttributeMax(i_attr))
            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic.CanonicalFeatureGeqSoft) = ([≥],ModalDecisionTrees.SingleAttributeSoftMin(i_attr, cf.alpha))
            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic.CanonicalFeatureLeqSoft) = ([≤],ModalDecisionTrees.SingleAttributeSoftMax(i_attr, cf.alpha))
            single_attr_feats_n_featsnops(i_attr,(test_ops,cf)::Tuple{<:AbstractVector{<:TestOperatorFun},Function})        = (test_ops,SingleAttributeGenericFeature(i_attr, cf))
            single_attr_feats_n_featsnops(i_attr,(test_ops,cf)::Tuple{<:AbstractVector{<:TestOperatorFun},typeof(minimum)}) = (test_ops,SingleAttributeMin(i_attr))
            single_attr_feats_n_featsnops(i_attr,(test_ops,cf)::Tuple{<:AbstractVector{<:TestOperatorFun},typeof(maximum)}) = (test_ops,SingleAttributeMax(i_attr))
            single_attr_feats_n_featsnops(i_attr,::Any) = throw_n_log("Unknown mixed_feature type: $(cf), $(typeof(cf))")

            for i_attr in 1:n_attributes(domain)
                for (test_ops,cf) in map((cf)->single_attr_feats_n_featsnops(i_attr,cf),attribute_specific_cfs)
                    push!(featsnops, test_ops)
                    push!(_features, cf)
                end
            end
            _features, featsnops
        end
        InterpretedModalDataset{T, N, world_type(ontology)}(domain, ontology, _features, featsnops)
    end

    function InterpretedModalDataset(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops_or_featsnops; # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        kwargs...,
    ) where {T, D, WorldType<:World}
        InterpretedModalDataset{T}(domain, ontology, features, grouped_featsaggrsnops_or_featsnops; kwargs...)
    end

    function InterpretedModalDataset{T}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops_or_featsnops; # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        kwargs...,
    ) where {T, D, WorldType<:World}
        InterpretedModalDataset{T, D-1-1}(domain, ontology, features, grouped_featsaggrsnops_or_featsnops; kwargs...)
    end

    function InterpretedModalDataset{T, N}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops_or_featsnops; # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        kwargs...,
    ) where {T, N, D, WorldType<:World}
        InterpretedModalDataset{T, N, WorldType}(domain, ontology, features, grouped_featsaggrsnops_or_featsnops; kwargs...)
    end

    function InterpretedModalDataset{T, N, WorldType}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsnops  :: AbstractVector{<:AbstractVector{<:TestOperatorFun}};
        kwargs...,
    ) where {T, N, D, WorldType<:World}

        grouped_featsaggrsnops = grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops)

        InterpretedModalDataset{T, N, WorldType}(domain, ontology, features, grouped_featsaggrsnops; kwargs...)
    end
    function InterpretedModalDataset{T, N, WorldType}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}};
        allow_no_instances = false,
    ) where {T, N, D, WorldType<:World}

        @assert allow_no_instances || n_samples(domain) > 0 "Can't instantiate InterpretedModalDataset{$(T), $(N), $(WorldType)} with no instance. (domain's type $(typeof(domain)))"
        @assert goes_with_dimensionality(WorldType, N) "ERROR! Dimensionality mismatch: can't interpret WorldType $(WorldType) on DimensionalDataset of dimensionality = $(N)"
        @assert D == (N+1+1) "ERROR! Dimensionality mismatch: can't instantiate InterpretedModalDataset{$(T), $(N)} with DimensionalDataset{$(T),$(D)}"
        @assert length(features) == length(grouped_featsaggrsnops) "Can't instantiate InterpretedModalDataset{$(T), $(N), $(WorldType)} with mismatching length(features) == length(grouped_featsaggrsnops): $(length(features)) != $(length(grouped_featsaggrsnops))"
        # @assert length(grouped_featsaggrsnops) > 0 && sum(length.(grouped_featsaggrsnops)) > 0 && sum(vcat([[length(test_ops) for test_ops in aggrs] for aggrs in grouped_featsaggrsnops]...)) > 0 "Can't instantiate ExplicitModalDataset{$(T), $(WorldType)} with no test operator: $(grouped_featsaggrsnops)"

        # if prod(max_channel_size(domain)) == 1
        #   TODO throw warning
        # end

        new{T, N, WorldType}(domain, ontology, features, grouped_featsaggrsnops)
    end
end

Base.size(imd::InterpretedModalDataset)              = size(imd.domain)
features(imd::InterpretedModalDataset)               = imd.features
grouped_featsaggrsnops(imd::InterpretedModalDataset) = imd.grouped_featsaggrsnops
n_attributes(imd::InterpretedModalDataset)           = n_attributes(imd.domain)::Int64
n_features(imd::InterpretedModalDataset)             = length(features(imd))::Int64
n_relations(imd::InterpretedModalDataset)            = length(relations(imd))::Int64
n_samples(imd::InterpretedModalDataset)              = n_samples(imd.domain)::Int64
relations(imd::InterpretedModalDataset)              = relations(imd.ontology)
world_type(imd::InterpretedModalDataset{T,N,WT}) where {T,N,WT} = WT

init_world_sets_fun(imd::InterpretedModalDataset{T, N, WorldType},  i_sample::Integer) where {T, N, WorldType} =
    (iC)->ModalDecisionTrees.init_world_set(iC, WorldType, instance_channel_size(get_instance(imd, i_sample)))
accessibles_fun(imd::InterpretedModalDataset, i_sample) = (w,R)->accessibles(w,R, instance_channel_size(get_instance(imd, i_sample))...)
all_worlds_fun(imd::InterpretedModalDataset{T, N, WorldType}, i_sample) where {T, N, WorldType} = all_worlds(WorldType, accessibles_fun(imd, i_sample))
accessibles_aggr_fun(imd::InterpretedModalDataset, i_sample)  = (f,a,w,R)->accessibles_aggr(f,a,w,R,instance_channel_size(get_instance(imd, i_sample))...)

# Note: Can't define Base.length(::DimensionalDataset) & Base.iterate(::DimensionalDataset)
Base.length(imd::InterpretedModalDataset)                = n_samples(imd)
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

abstract type AbstractFWD{T<:Number,WorldType<:World} end

# Any implementation for a fwd must indicate their compatible world types via `goes_with`.
# Fallback:
goes_with(::Type{<:AbstractFWD}, ::Type{<:World}) = false

# Any world type must also specify their default fwd constructor, which must accept a type
#  parameter for the data type {T}, via:
# default_fwd_type(::Type{<:World})

#
# Actually, the interface for AbstractFWD's is a bit tricky; the most straightforward
#  way of learning it is by considering the fallback fwd structure defined as follows.
# TODO oh, but the implementation is broken due to a strange error (see https://discourse.julialang.org/t/tricky-too-many-parameters-for-type-error/25182 )

# # The most generic fwd structure is a matrix of dictionaries of size (n_samples × n_features)
# struct GenericFWD{T, WorldType} <: AbstractFWD{T, WorldType}
#   d :: AbstractVector{<:AbstractDict{WorldType,AbstractVector{T,1}},1}
#   n_features :: Integer
# end

# # It goes for any world type
# goes_with(::Type{<:GenericFWD}, ::Type{<:World}) = true

# # And it is the default fwd structure for an world type
# default_fwd_type(::Type{<:World}) = GenericFWD

# n_samples(fwd::GenericFWD{T}) where {T}  = size(fwd, 1)
# n_features(fwd::GenericFWD{T}) where {T} = fwd.d
# Base.size(fwd::GenericFWD{T}, args...) where {T} = size(fwd.d, args...)

# # The matrix is initialized with #undef values
# function fwd_init(::Type{GenericFWD}, imd::InterpretedModalDataset{T}) where {T}
#     d = Array{Dict{WorldType,T}, 2}(undef, n_samples(imd))
#     for i in 1:n_samples
#         d[i] = Dict{WorldType,Array{T,1}}()
#     end
#     GenericFWD{T}(d, n_features(imd))
# end

# # A function for initializing individual world slices
# function fwd_init_world_slice(fwd::GenericFWD{T}, i_sample::Integer, w::World) where {T}
#     fwd.d[i_sample][w] = Array{T,1}(undef, fwd.n_features)
# end

# # A function for getting a threshold value from the lookup table
# Base.@propagate_inbounds @inline fwd_get(
#     fwd         :: GenericFWD{T},
#     i_sample    :: Integer,
#     w           :: World,
#     i_feature   :: Integer) where {T} = fwd.d[i_sample][w][i_feature]

Base.getindex(
    fwd         :: AbstractFWD{T},
    i_sample    :: Integer,
    w           :: World,
    i_feature   :: Integer) where {T} = fwd_get(fwd, i_sample, w, i_feature)

# # A function for setting a threshold value in the lookup table
# Base.@propagate_inbounds @inline function fwd_set(fwd::GenericFWD{T}, w::World, i_sample::Integer, i_feature::Integer, threshold::T) where {T}
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
#     GenericFWD{T}(if return_view @view fwd.d[inds] else fwd.d[inds] end, fwd.n_features)
# end

# Others...
# Base.@propagate_inbounds @inline fwd_get_channel(fwd::GenericFWD{T}, i_sample::Integer, i_feature::Integer) where {T} = TODO
# const GenericFeaturedChannel{T} = TODO
# fwd_channel_interpret_world(fwc::GenericFeaturedChannel{T}, w::World) where {T} = TODO

############################################################################################
# Explicit modal dataset
#
# An explicit modal dataset is the generic form of a modal dataset, and consists of
#  a wrapper around an fwd lookup table. The information it adds are the relation set,
#  a few functions for enumerating worlds (`accessibles`, `accessibles_aggr`),
#  and a world set initialization function representing initial conditions (initializing world sets).
#
############################################################################################

struct ExplicitModalDataset{T<:Number, WorldType<:World} <: ActiveModalDataset{T, WorldType}

    # Core data (fwd lookup table)
    fwd                :: AbstractFWD{T,WorldType}

    ## Modal frame:
    # Accessibility relations
    relations          :: AbstractVector{<:Relation}

    # Worldset initialization functions (one per instance)
    #  with signature (iC::InitCondition) -> vs::AbstractWorldSet{WorldType}
    init_world_sets_funs   :: AbstractVector{<:initWorldSetFunction}
    # Accessibility functions (one per instance)
    #  with signature (w::WorldType/AbstractWorldSet{WorldType}, r::Relation) -> vs::AbstractVector{WorldType}
    accessibles_funs      :: AbstractVector{<:accFunction}
    # Representative accessibility functions (one per instance)
    #  with signature (feature::ModalFeature, aggregator::Aggregator, w::WorldType/AbstractWorldSet{WorldType}, r::Relation) -> vs::AbstractVector{WorldType}
    accessibles_aggr_funs  :: AbstractVector{<:accReprFunction}

    # Features
    features           :: AbstractVector{<:ModalFeature}

    # Test operators associated with each feature, grouped by their respective aggregator
    grouped_featsaggrsnops  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}

    ExplicitModalDataset(
        fwd                    :: AbstractFWD{T,WorldType},
        relations              :: AbstractVector{<:Relation},
        init_world_sets_funs   :: AbstractVector{<:initWorldSetFunction},
        accessibles_funs       :: AbstractVector{<:accFunction},
        accessibles_aggr_funs  :: AbstractVector{<:accReprFunction},
        features               :: AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops_or_featsnops, # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        args...;
        kwargs...,
    ) where {T,WorldType} = begin ExplicitModalDataset{T, WorldType}(fwd, relations, init_world_sets_funs, accessibles_funs, accessibles_aggr_funs, features, grouped_featsaggrsnops_or_featsnops, args...; kwargs...) end

    function ExplicitModalDataset{T, WorldType}(
        fwd                     :: AbstractFWD{T,WorldType},
        relations               :: AbstractVector{<:Relation},
        init_world_sets_funs    :: AbstractVector{<:initWorldSetFunction},
        accessibles_funs        :: AbstractVector{<:accFunction},
        accessibles_aggr_funs   :: AbstractVector{<:accReprFunction},
        features                :: AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}};
        allow_no_instances = false,
    ) where {T,WorldType<:World}
        @assert allow_no_instances || n_samples(fwd) > 0     "Can't instantiate ExplicitModalDataset{$(T), $(WorldType)} with no instance. (fwd's type $(typeof(fwd)))"
        @assert length(grouped_featsaggrsnops) > 0 && sum(length.(grouped_featsaggrsnops)) > 0 && sum(vcat([[length(test_ops) for test_ops in aggrs] for aggrs in grouped_featsaggrsnops]...)) > 0 "Can't instantiate ExplicitModalDataset{$(T), $(WorldType)} with no test operator: grouped_featsaggrsnops"
        @assert  n_samples(fwd) == length(init_world_sets_funs)  "Can't instantiate ExplicitModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of init_world_sets_funs $(length(init_world_sets_funs))."
        @assert  n_samples(fwd) == length(accessibles_funs)     "Can't instantiate ExplicitModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of accessibles_funs $(length(accessibles_funs))."
        @assert  n_samples(fwd) == length(accessibles_aggr_funs) "Can't instantiate ExplicitModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of accessibles_aggr_funs $(length(accessibles_aggr_funs))."
        @assert n_features(fwd) == length(features)          "Can't instantiate ExplicitModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of features $(length(features))."
        new{T, WorldType}(fwd, relations, init_world_sets_funs, accessibles_funs, accessibles_aggr_funs, features, grouped_featsaggrsnops)
    end

    function ExplicitModalDataset(
        fwd                    :: AbstractFWD{T,WorldType},
        relations              :: AbstractVector{<:Relation},
        init_world_sets_funs   :: AbstractVector{<:initWorldSetFunction},
        accessibles_funs       :: AbstractVector{<:accFunction},
        accessibles_aggr_funs  :: AbstractVector{<:accReprFunction},
        features               :: AbstractVector{<:ModalFeature},
        grouped_featsnops      :: AbstractVector{<:AbstractVector{<:TestOperatorFun}},
        args...;
        kwargs...,
    ) where {T,WorldType<:World}

        grouped_featsaggrsnops = grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops)

        ExplicitModalDataset(fwd, relations, init_world_sets_funs, accessibles_funs, accessibles_aggr_funs, features, grouped_featsaggrsnops, args...; kwargs...)
    end

    # Quite importantly, an fwd can be computed from a dataset in implicit form (domain + ontology + features)
    Base.@propagate_inbounds function ExplicitModalDataset(
        imd                  :: InterpretedModalDataset{T, N, WorldType},
        # FWD                  ::Type{<:AbstractFWD{T,WorldType}} = default_fwd_type(WorldType),
        FWD                  ::Type = default_fwd_type(WorldType),
        args...;
        kwargs...,
    ) where {T, N, WorldType<:World}

        fwd = begin

            # @logmsg DTOverview "InterpretedModalDataset -> ExplicitModalDataset"

            _features = features(imd)

            _n_samples = n_samples(imd)

            @assert goes_with(FWD, WorldType)

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
                @logmsg DTDebug "Instance $(i_sample)/$(_n_samples)"

                # if i_sample == 1 || ((i_sample+1) % (floor(Int, ((_n_samples)/4))+1)) == 0
                #     @logmsg DTOverview "Instance $(i_sample)/$(_n_samples)"
                # end

                # instance = get_instance(imd, i_sample)
                # @logmsg DTDebug "instance" instance

                for w in all_worlds_fun(imd, i_sample)

                    fwd_init_world_slice(fwd, i_sample, w)

                    @logmsg DTDebug "World" w

                    for (i_feature,feature) in enum_features

                        # threshold = computePropositionalThreshold(feature, w, instance)
                        threshold = get_gamma(imd, i_sample, w, feature)

                        @logmsg DTDebug "Feature $(i_feature)" threshold

                        fwd_set(fwd, w, i_sample, i_feature, threshold)

                    end
                end
                # next!(p)
            end
            fwd
        end

        # TODO Think about it, and optimize this: when the underlying DimensionalDataset is an Array,
        #  this is going to be an array of a single function.
        init_world_sets_funs  = [init_world_sets_fun(imd,  i_sample) for i_sample in 1:n_samples(imd)]
        accessibles_funs      = [accessibles_fun(imd,      i_sample) for i_sample in 1:n_samples(imd)]
        accessibles_aggr_funs = [accessibles_aggr_fun(imd, i_sample) for i_sample in 1:n_samples(imd)]

        ExplicitModalDataset(fwd, relations(imd), init_world_sets_funs, accessibles_funs, accessibles_aggr_funs, _features, grouped_featsaggrsnops(imd), args...; kwargs...)
    end

end

Base.getindex(X::ExplicitModalDataset{T,WorldType}, args...) where {T,WorldType} = getindex(X.fwd, args...)
Base.size(X::ExplicitModalDataset)                 where {T,N}          = size(X.fwd) # TODO fix not always defined?
features(X::ExplicitModalDataset)               = X.features
grouped_featsaggrsnops(X::ExplicitModalDataset) = X.grouped_featsaggrsnops
n_features(X::ExplicitModalDataset{T, WorldType})  where {T, WorldType} = length(X.features)
n_relations(X::ExplicitModalDataset{T, WorldType}) where {T, WorldType} = length(X.relations)
n_samples(X::ExplicitModalDataset{T, WorldType})   where {T, WorldType} = n_samples(X.fwd)::Int64
relations(X::ExplicitModalDataset)                                 = X.relations
world_type(X::ExplicitModalDataset{T,WorldType}) where {T,WorldType<:World} = WorldType


init_world_sets_fun(X::ExplicitModalDataset,          i_sample::Integer)  = X.init_world_sets_funs[i_sample]
accessibles_fun(X::ExplicitModalDataset,              i_sample::Integer)  = X.accessibles_funs[i_sample]
all_worlds_fun(X::ExplicitModalDataset{T, WorldType}, i_sample::Integer) where {T, WorldType} = all_worlds(WorldType, accessibles_fun(X, i_sample))
accessibles_aggr_fun(X::ExplicitModalDataset,         i_sample::Integer)  = X.accessibles_aggr_funs[i_sample]


slice_dataset(X::ExplicitModalDataset{T,WorldType}, inds::AbstractVector{<:Integer}, args...; allow_no_instances = false, kwargs...) where {T,WorldType} =
    ExplicitModalDataset{T,WorldType}(
        slice_dataset(X.fwd, inds, args...; allow_no_instances = allow_no_instances, kwargs...),
        X.relations,
        X.init_world_sets_funs[inds],
        X.accessibles_funs[inds],
        X.accessibles_aggr_funs[inds],
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


find_feature_id(X::ExplicitModalDataset{T,WorldType}, feature::ModalFeature) where {T,WorldType} =
    findall(x->x==feature, features(X))[1]
find_relation_id(X::ExplicitModalDataset{T,WorldType}, relation::Relation) where {T,WorldType} =
    findall(x->x==relation, relations(X))[1]

hasnans(emd::ExplicitModalDataset) = begin
    # @show hasnans(emd.fwd)
    hasnans(emd.fwd)
end

Base.@propagate_inbounds @inline get_gamma(
        X::ExplicitModalDataset{T,WorldType},
        i_sample::Integer,
        w::WorldType,
        feature::ModalFeature) where {WorldType<:World, T} = begin
    i_feature = find_feature_id(X, feature)
    X[i_sample, w, i_feature]
end

isminifiable(::ExplicitModalDataset) = true

function minify(X::ExplicitModalDataset)
    new_fwd, backmap = minify(X.fwd)
    X = ExplicitModalDataset(
        new_fwd,
        X.relations,
        X.init_world_sets_funs,
        X.accessibles_funs,
        X.accessibles_aggr_funs,
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
abstract type ExplicitModalDatasetWithSupport{T,WorldType} <: ActiveModalDataset{T, WorldType} end
# And an abstract type for support lookup tables
abstract type AbstractSupport{T, WorldType} end
#
# In general, one can use lookup (with or without memoization) for any decision, even the
#  more complex ones, for example:
#  ⟨G⟩ (minimum(A2) ≥ 10 ∧ (⟨O⟩ (maximum(A3) > 2) ∨ (minimum(A1) < 0)))
#
# In practice, decision trees only ask about simple decisions such as ⟨L⟩ (minimum(A2) ≥ 10),
#  or ⟨G⟩ (maximum(A2) ≤ 50). Because the global operator G behaves differently from other
#  relations, it is natural to differentiate between global and relational support tables:
#
abstract type AbstractRelationalSupport{T, WorldType} <: AbstractSupport{T, WorldType}     end
abstract type AbstractGlobalSupport{T}                <: AbstractSupport{T, World} end
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

struct GenericRelationalSupport{T, WorldType} <: AbstractRelationalSupport{T, WorldType}
    d :: AbstractArray{Dict{WorldType,T}, 3}
end

goes_with(::Type{GenericRelationalSupport}, ::Type{<:World}) = true
# default_fwd_rs_type(::Type{<:World}) = GenericRelationalSupport # TODO implement similar pattern used for fwd

hasnans(emds::GenericRelationalSupport) = begin
    # @show any(map(d->(any(_isnan.(collect(values(d))))), emds.d))
    any(map(d->(any(_isnan.(collect(values(d))))), emds.d))
end

n_samples(emds::GenericRelationalSupport)     = size(emds, 1)
n_featsnaggrs(emds::GenericRelationalSupport) = size(emds, 2)
n_relations(emds::GenericRelationalSupport)   = size(emds, 3)
Base.getindex(
    emds         :: GenericRelationalSupport{T, WorldType},
    i_sample     :: Integer,
    w            :: WorldType,
    i_featsnaggr :: Integer,
    i_relation   :: Integer) where {T, WorldType<:World} = emds.d[i_sample, i_featsnaggr, i_relation][w]
Base.size(emds::GenericRelationalSupport, args...) = size(emds.d, args...)

fwd_rs_init(emd::ExplicitModalDataset{T, WorldType}, n_featsnaggrs::Integer, n_relations::Integer; perform_initialization = false) where {T, WorldType} = begin
    if perform_initialization
        _fwd_rs = fill!(Array{Dict{WorldType,Union{T,Nothing}}, 3}(undef, n_samples(emd), n_featsnaggrs, n_relations), nothing)
        GenericRelationalSupport{Union{T,Nothing}, WorldType}(_fwd_rs)
    else
        _fwd_rs = Array{Dict{WorldType,T}, 3}(undef, n_samples(emd), n_featsnaggrs, n_relations)
        GenericRelationalSupport{T, WorldType}(_fwd_rs)
    end
end
fwd_rs_init_world_slice(emds::GenericRelationalSupport{T, WorldType}, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) where {T, WorldType} =
    emds.d[i_sample, i_featsnaggr, i_relation] = Dict{WorldType,T}()
fwd_rs_set(emds::GenericRelationalSupport{T, WorldType}, i_sample::Integer, w::World, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T, WorldType} =
    emds.d[i_sample, i_featsnaggr, i_relation][w] = threshold
function slice_dataset(emds::GenericRelationalSupport{T, WorldType}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T, WorldType}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    GenericRelationalSupport{T, WorldType}(if return_view @view emds.d[inds,:,:] else emds.d[inds,:,:] end)
end

############################################################################################

# Note: the global support is world-agnostic
struct GenericGlobalSupport{T} <: AbstractGlobalSupport{T}
    d :: AbstractArray{T, 2}
end

goes_with(::Type{AbstractGlobalSupport}, ::Type{<:World}) = true
# default_fwd_gs_type(::Type{<:World}) = GenericGlobalSupport # TODO implement similar pattern used for fwd

hasnans(emds::GenericGlobalSupport) = begin
    # @show any(_isnan.(emds.d))
    any(_isnan.(emds.d))
end

n_samples(emds::GenericGlobalSupport{T}) where {T}  = size(emds, 1)
n_featsnaggrs(emds::GenericGlobalSupport{T}) where {T} = size(emds, 2)
Base.getindex(
    emds         :: GenericGlobalSupport{T},
    i_sample     :: Integer,
    i_featsnaggr  :: Integer) where {T} = emds.d[i_sample, i_featsnaggr]
Base.size(emds::GenericGlobalSupport{T}, args...) where {T} = size(emds.d, args...)

fwd_gs_init(emd::ExplicitModalDataset{T}, n_featsnaggrs::Integer) where {T} =
    GenericGlobalSupport{T}(Array{T, 2}(undef, n_samples(emd), n_featsnaggrs))
fwd_gs_set(emds::GenericGlobalSupport{T}, i_sample::Integer, i_featsnaggr::Integer, threshold::T) where {T} =
    emds.d[i_sample, i_featsnaggr] = threshold
function slice_dataset(emds::GenericGlobalSupport{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    GenericGlobalSupport{T}(if return_view @view emds.d[inds,:] else emds.d[inds,:] end)
end

# A function that computes fwd_rs and fwd_gs from an explicit modal dataset
Base.@propagate_inbounds function compute_fwd_supports(
        emd                 :: ExplicitModalDataset{T, WorldType},
        grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}};
        compute_relation_glob = false,
        simply_init_modal = false,
    ) where {T, N, WorldType<:World}

    # @logmsg DTOverview "ExplicitModalDataset -> ExplicitModalDatasetS "

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

    _n_samples = n_samples(emd)
    n_relations = length(_relations)
    n_featsnaggrs = sum(length.(grouped_featsnaggrs))

    # println(_n_samples)
    # println(n_relations)
    # println(n_featsnaggrs)
    # println(grouped_featsnaggrs)

    # Prepare fwd_rs
    fwd_rs = fwd_rs_init(emd, n_featsnaggrs, n_relations; perform_initialization = simply_init_modal)

    # Prepare fwd_gs
    fwd_gs = begin
        if compute_fwd_gs
            fwd_gs_init(emd, n_featsnaggrs)
        else
            nothing
        end
    end

    # p = Progress(_n_samples, 1, "Computing EMD supports...")
    Threads.@threads for i_sample in 1:_n_samples
        @logmsg DTDebug "Instance $(i_sample)/$(_n_samples)"

        # if i_sample == 1 || ((i_sample+1) % (floor(Int, ((_n_samples)/4))+1)) == 0
        #     @logmsg DTOverview "Instance $(i_sample)/$(_n_samples)"
        # end

        for (i_feature,aggregators) in enumerate(grouped_featsnaggrs)

            @logmsg DTDebug "Feature $(i_feature)"

            cur_fwd_slice = fwd_get_channel(fwd, i_sample, i_feature)

            @logmsg DTDebug cur_fwd_slice

            # Global relation (independent of the current world)
            if compute_fwd_gs
                @logmsg DTDebug "RelationGlob"

                # TODO optimize: all aggregators are likely reading the same raw values.
                for (i_featsnaggr,aggregator) in aggregators
                # Threads.@threads for (i_featsnaggr,aggregator) in aggregators

                    # accessible_worlds = all_worlds_fun(emd, i_sample)
                    # TODO reintroduce the improvements for some operators: e.g. later. Actually, these can be simplified by using a set of representatives, as in some enum_acc_repr!
                    accessible_worlds = ModalLogic.all_worlds_aggr(WorldType, accessibles_aggr_fun(emd, i_sample), _features[i_feature], aggregator)

                    threshold = compute_modal_gamma(cur_fwd_slice, accessible_worlds, aggregator)

                    @logmsg DTDebug "Aggregator[$(i_featsnaggr)]=$(aggregator)  -->  $(threshold)"

                    # @logmsg DTDebug "Aggregator" aggregator threshold

                    fwd_gs_set(fwd_gs, i_sample, i_featsnaggr, threshold)
                end
            end
            # readline()

            if !simply_init_modal
                # Other relations
                for (i_relation,relation) in enumerate(_relations)

                    @logmsg DTDebug "Relation $(i_relation)/$(n_relations)"

                    for (i_featsnaggr,aggregator) in aggregators
                        fwd_rs_init_world_slice(fwd_rs, i_sample, i_featsnaggr, i_relation)
                    end

                    for w in all_worlds_fun(emd, i_sample)

                        @logmsg DTDebug "World" w

                        # TODO optimize: all aggregators are likely reading the same raw values.
                        for (i_featsnaggr,aggregator) in aggregators

                            # accessible_worlds = accessibles_fun(emd, i_sample)(w, relation)
                            # TODO reintroduce the improvements for some operators: e.g. later. Actually, these can be simplified by using a set of representatives, as in some enum_acc_repr!
                            accessible_worlds = accessibles_aggr_fun(emd, i_sample)(_features[i_feature], aggregator, w, relation)

                            threshold = compute_modal_gamma(cur_fwd_slice, accessible_worlds, aggregator)

                            # @logmsg DTDebug "Aggregator" aggregator threshold

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

struct ExplicitModalDatasetS{T<:Number, WorldType<:World} <: ExplicitModalDatasetWithSupport{T, WorldType}

    # Core dataset
    emd                 :: ExplicitModalDataset{T, WorldType}

    # Relational and global support
    fwd_rs              :: AbstractRelationalSupport{T, WorldType}
    fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing}

    # Features and Aggregators
    featsnaggrs         :: AbstractVector{Tuple{<:ModalFeature,<:Aggregator}}
    grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}}

    function ExplicitModalDatasetS{T, WorldType}(
        emd                 :: ExplicitModalDataset{T, WorldType},
        fwd_rs              :: AbstractRelationalSupport{T, WorldType},
        fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing},
        featsnaggrs         :: AbstractVector{Tuple{<:ModalFeature,<:Aggregator}},
        grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
    ) where {T,WorldType<:World}
        @assert n_samples(emd) == n_samples(fwd_rs)                               "Can't instantiate ExplicitModalDatasetS{$(T), $(WorldType)} with unmatching n_samples for emd and fwd_rs support: $(n_samples(emd)) and $(n_samples(fwd_rs))"
        @assert n_relations(emd) == n_relations(fwd_rs)                           "Can't instantiate ExplicitModalDatasetS{$(T), $(WorldType)} with unmatching n_relations for emd and fwd_rs support: $(n_relations(emd)) and $(n_relations(fwd_rs))"
        @assert sum(length.(grouped_featsnaggrs)) == length(featsnaggrs)          "Can't instantiate ExplicitModalDatasetS{$(T), $(WorldType)} with unmatching n_featsnaggrs (grouped vs flattened structure): $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == length(featsnaggrs)   "Can't instantiate ExplicitModalDatasetS{$(T), $(WorldType)} with unmatching n_featsnaggrs for emd and provided featsnaggrs: $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == n_featsnaggrs(fwd_rs) "Can't instantiate ExplicitModalDatasetS{$(T), $(WorldType)} with unmatching n_featsnaggrs for emd and fwd_rs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(n_featsnaggrs(fwd_rs))"

        if fwd_gs != nothing
            @assert n_samples(emd) == n_samples(fwd_gs) "Can't instantiate ExplicitModalDatasetS{$(T), $(WorldType)} with unmatching n_samples for emd and fwd_gs support: $(n_samples(emd)) and $(n_samples(fwd_gs))"
            # @assert somethinglike(emd) == n_featsnaggrs(fwd_gs) "Can't instantiate ExplicitModalDatasetS{$(T), $(WorldType)} with unmatching somethinglike for emd and fwd_gs support: $(somethinglike(emd)) and $(n_featsnaggrs(fwd_gs))"
            @assert sum(length.(emd.grouped_featsaggrsnops)) == n_featsnaggrs(fwd_gs) "Can't instantiate ExplicitModalDatasetS{$(T), $(WorldType)} with unmatching n_featsnaggrs for emd and fwd_gs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(n_featsnaggrs(fwd_gs))"
        end

        new{T, WorldType}(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs)
    end

    function ExplicitModalDatasetS(
        emd                 :: ExplicitModalDataset{T, WorldType};
        compute_relation_glob :: Bool = true,
    ) where {T,WorldType<:World}
        ExplicitModalDatasetS{T, WorldType}(emd, compute_relation_glob = compute_relation_glob)
    end

    function ExplicitModalDatasetS{T, WorldType}(
        emd                   :: ExplicitModalDataset{T, WorldType};
        compute_relation_glob :: Bool = true,
    ) where {T,WorldType<:World}

        featsnaggrs = Tuple{<:ModalFeature,<:Aggregator}[]
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

        ExplicitModalDatasetS{T, WorldType}(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs)
    end

    function ExplicitModalDatasetS(
        X                   :: InterpretedModalDataset{T, N, WorldType};
        compute_relation_glob :: Bool = true,
    ) where {T, N, WorldType<:World}
        ExplicitModalDatasetS{T, WorldType}(X, compute_relation_glob = compute_relation_glob)
    end

    function ExplicitModalDatasetS{T, WorldType}(
        X                   :: InterpretedModalDataset{T, N, WorldType};
        compute_relation_glob :: Bool = true,
    ) where {T, N, WorldType<:World}

        # Compute modal dataset propositions
        emd = ExplicitModalDataset(X);

        ExplicitModalDatasetS{T, WorldType}(emd, compute_relation_glob = compute_relation_glob)
    end
end

mutable struct ExplicitModalDatasetSMemo{T<:Number, WorldType<:World} <: ExplicitModalDatasetWithSupport{T, WorldType}

    # Core dataset
    emd                 :: ExplicitModalDataset{T, WorldType}

    # Relational and global support
    fwd_rs              :: AbstractRelationalSupport{<:Union{T,Nothing}, WorldType}
    fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing} # TODO maybe nothing is not needed here

    # Features and Aggregators
    featsnaggrs         :: AbstractVector{Tuple{<:ModalFeature,<:Aggregator}}
    grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}}

    function ExplicitModalDatasetSMemo{T, WorldType}(
        emd                 :: ExplicitModalDataset{T, WorldType},
        fwd_rs              :: AbstractRelationalSupport{<:Union{T,Nothing}, WorldType},
        fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing},
        featsnaggrs         :: AbstractVector{Tuple{<:ModalFeature,<:Aggregator}},
        grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
    ) where {T,WorldType<:World}
        @assert n_samples(emd) == n_samples(fwd_rs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(WorldType)} with unmatching n_samples for emd and fwd_rs support: $(n_samples(emd)) and $(n_samples(fwd_rs))"
        @assert n_relations(emd) == n_relations(fwd_rs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(WorldType)} with unmatching n_relations for emd and fwd_rs support: $(n_relations(emd)) and $(n_relations(fwd_rs))"
        @assert sum(length.(grouped_featsnaggrs)) == length(featsnaggrs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(WorldType)} with unmatching n_featsnaggrs (grouped vs flattened structure): $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == length(featsnaggrs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(WorldType)} with unmatching n_featsnaggrs for emd and provided featsnaggrs: $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == n_featsnaggrs(fwd_rs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(WorldType)} with unmatching n_featsnaggrs for emd and fwd_rs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(n_featsnaggrs(fwd_rs))"

        if fwd_gs != nothing
            @assert n_samples(emd) == n_samples(fwd_gs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(WorldType)} with unmatching n_samples for emd and fwd_gs support: $(n_samples(emd)) and $(n_samples(fwd_gs))"
            # @assert somethinglike(emd) == n_featsnaggrs(fwd_gs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(WorldType)} with unmatching somethinglike for emd and fwd_gs support: $(somethinglike(emd)) and $(n_featsnaggrs(fwd_gs))"
            @assert sum(length.(emd.grouped_featsaggrsnops)) == n_featsnaggrs(fwd_gs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(WorldType)} with unmatching n_featsnaggrs for emd and fwd_gs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(n_featsnaggrs(fwd_gs))"
        end

        new{T, WorldType}(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs)
    end

    function ExplicitModalDatasetSMemo(
        emd                 :: ExplicitModalDataset{T, WorldType};
        compute_relation_glob :: Bool = true,
    ) where {T,WorldType<:World}
        ExplicitModalDatasetSMemo{T, WorldType}(emd, compute_relation_glob = compute_relation_glob)
    end

    function ExplicitModalDatasetSMemo{T, WorldType}(
        emd                   :: ExplicitModalDataset{T, WorldType};
        compute_relation_glob :: Bool = true,
    ) where {T,WorldType<:World}

        featsnaggrs = Tuple{<:ModalFeature,<:Aggregator}[]
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

        ExplicitModalDatasetSMemo{T, WorldType}(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs)
    end

    function ExplicitModalDatasetSMemo(
        X                   :: InterpretedModalDataset{T, N, WorldType};
        compute_relation_glob :: Bool = true,
    ) where {T, N, WorldType<:World}
        ExplicitModalDatasetSMemo{T, WorldType}(X, compute_relation_glob = compute_relation_glob)
    end

    function ExplicitModalDatasetSMemo{T, WorldType}(
        X                   :: InterpretedModalDataset{T, N, WorldType};
        compute_relation_glob :: Bool = true,
    ) where {T, N, WorldType<:World}

        # Compute modal dataset propositions
        emd = ExplicitModalDataset(X);

        ExplicitModalDatasetSMemo{T, WorldType}(emd, compute_relation_glob = compute_relation_glob)
    end
end

# getindex(X::ExplicitModalDatasetWithSupport{T,WorldType}, args...) where {T,WorldType} = getindex(X.emd, args...)
Base.size(X::ExplicitModalDatasetWithSupport) where {T,N}                          =  (size(X.emd), size(X.fwd_rs), (isnothing(X.fwd_gs) ? nothing : size(X.fwd_gs)))
featsnaggrs(X::ExplicitModalDatasetWithSupport)                                    = X.featsnaggrs
features(X::ExplicitModalDatasetWithSupport)                                       = features(X.emd)
grouped_featsaggrsnops(X::ExplicitModalDatasetWithSupport)                         = grouped_featsaggrsnops(X.emd)
grouped_featsnaggrs(X::ExplicitModalDatasetWithSupport)                            = X.grouped_featsnaggrs
n_features(X::ExplicitModalDatasetWithSupport{T, WorldType}) where {T, WorldType}  = n_features(X.emd)
n_relations(X::ExplicitModalDatasetWithSupport{T, WorldType}) where {T, WorldType} = n_relations(X.emd)
n_samples(X::ExplicitModalDatasetWithSupport{T, WorldType}) where {T, WorldType}   = n_samples(X.emd)::Int64
relations(X::ExplicitModalDatasetWithSupport)                                      = relations(X.emd)
world_type(X::ExplicitModalDatasetWithSupport{T,WorldType}) where {T,WorldType}    = WorldType

init_world_sets_fun(X::ExplicitModalDatasetWithSupport,  i_sample::Integer, ::Type{WorldType}) where {WorldType<:World} = init_world_sets_fun(X.emd, i_sample)
accessibles_fun(X::ExplicitModalDatasetWithSupport,     args...) = accessibles_fun(X.emd, args...)
all_worlds_fun(X::ExplicitModalDatasetWithSupport,  args...) = all_worlds_fun(X.emd, args...)
accessibles_aggr_fun(X::ExplicitModalDatasetWithSupport, args...) = accessibles_aggr_fun(X.emd, args...)

function slice_dataset(X::ExplicitModalDatasetWithSupport, inds::AbstractVector{<:Integer}, args...; kwargs...)
    typeof(X)(
        slice_dataset(X.emd, inds, args...; kwargs...),
        slice_dataset(X.fwd_rs, inds, args...; kwargs...),
        (isnothing(X.fwd_gs) ? nothing : slice_dataset(X.fwd_gs, inds, args...; kwargs...)),
        X.featsnaggrs,
        X.grouped_featsnaggrs)
end

find_feature_id(X::ExplicitModalDatasetWithSupport, feature::ModalFeature) =
    findall(x->x==feature, features(X))[1]
find_relation_id(X::ExplicitModalDatasetWithSupport, relation::Relation) =
    findall(x->x==relation, relations(X))[1]
find_featsnaggr_id(X::ExplicitModalDatasetWithSupport, feature::ModalFeature, aggregator::Aggregator) =
    findall(x->x==(feature, aggregator), featsnaggrs(X))[1]

hasnans(X::ExplicitModalDatasetWithSupport) = begin
    # @show hasnans(X.emd)
    # @show hasnans(X.fwd_rs)
    # @show (!isnothing(X.fwd_gs) && hasnans(X.fwd_gs))
    hasnans(X.emd) || hasnans(X.fwd_rs) || (!isnothing(X.fwd_gs) && hasnans(X.fwd_gs))
end

Base.@propagate_inbounds @inline get_gamma(
        X::ExplicitModalDatasetWithSupport{T,WorldType},
        i_sample::Integer,
        w::WorldType,
        feature::ModalFeature) where {WorldType<:World, T} = get_gamma(X.emd, i_sample, w, feature)

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
        X::ExplicitModalDatasetWithSupport{T,WorldType},
        i_sample::Integer,
        feature::ModalFeature,
        test_operator::TestOperatorFun) where {WorldType<:World, T} = begin
            # @assert !isnothing(X.fwd_gs) "Error. ExplicitModalDatasetWithSupport must be built with compute_relation_glob = true for it to be ready to test global decisions."
            i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
            X.fwd_gs[i_sample, i_featsnaggr]
end

get_modal_gamma(
        X::ExplicitModalDatasetS{T,WorldType},
        i_sample::Integer,
        w::WorldType,
        relation::Relation,
        feature::ModalFeature,
        test_operator::TestOperatorFun) where {WorldType<:World, T} = begin
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
        X::ExplicitModalDatasetWithSupport{T,WorldType},
        i_sample::Integer,
        w::WorldType,
        decision::Decision) where {T, WorldType<:World} = begin
    if is_propositional_decision(decision)
        test_decision(X, i_sample, w, decision.feature, decision.test_operator, decision.threshold)
    else
        gamma = begin
            if decision.relation isa ModalLogic._RelationGlob
                get_global_gamma(X, i_sample, decision.feature, decision.test_operator)
            else
                get_modal_gamma(X, i_sample, w, decision.relation, decision.feature, decision.test_operator)
            end
        end
        evaluate_thresh_decision(decision.test_operator, gamma, decision.threshold)
    end
end


Base.@propagate_inbounds @resumable function generate_propositional_feasible_decisions(
        emd::Union{ExplicitModalDataset{T,WorldType},InterpretedModalDataset{T,WorldType}},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:World}
    relation = RelationId
    _n_samples = length(instances_inds)

    # For each feature
    @inbounds for i_feature in features_inds
        feature = features(emd)[i_feature]
        @logmsg DTDebug "Feature $(i_feature): $(feature)"

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
            @logmsg DTDetail " Instance $(instance_idx)/$(_n_samples)"
            worlds = Sf[instance_idx]

            # TODO also try this instead
            # values = [X.emd[i_sample, w, i_feature] for w in worlds]
            # thresholds[:,instance_idx] = map(aggr->aggr(values), aggregators)

            for w in worlds
                gamma = begin
                    if emd isa ExplicitModalDataset{T,WorldType}
                        fwd_get(emd.fwd, i_sample, w, i_feature) # faster but equivalent to get_gamma(emd, i_sample, w, feature)
                    elseif emd isa InterpretedModalDataset{T,WorldType}
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

        # @logmsg DTDebug "thresholds: " thresholds
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
                @logmsg DTDetail " Test operator $(test_operator)"
                # Look for the best threshold 'a', as in propositions like "feature >= a"
                for threshold in aggr_domain
                    decision = Decision(relation, feature, test_operator, threshold)
                    @logmsg DTDebug " Testing decision: $(display_decision(decision))"
                    @yield decision, aggr_thresholds
                end # for threshold
                # push!(tested_test_operator, test_operator)
            end # for test_operator
        end # for aggregator
    end # for feature
end

Base.@propagate_inbounds @resumable function generate_propositional_feasible_decisions(
        X::ExplicitModalDatasetWithSupport{T,WorldType},
        args...
        ) where {T, WorldType<:World}
        for decision in generate_propositional_feasible_decisions(X.emd, args...)
            @yield decision
        end
end

Base.@propagate_inbounds @resumable function generate_global_feasible_decisions(
        X::ExplicitModalDatasetWithSupport{T,WorldType},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:World}
    relation = RelationGlob
    _n_samples = length(instances_inds)

    @assert !isnothing(X.fwd_gs) "Error. ExplicitModalDatasetWithSupport must be built with compute_relation_glob = true for it to be ready to generate global decisions."

    # For each feature
    for i_feature in features_inds
        feature = features(X)[i_feature]
        @logmsg DTDebug "Feature $(i_feature): $(feature)"

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
            @logmsg DTDetail " Instance $(instance_id)/$(_n_samples)"
            for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
                gamma = X.fwd_gs[i_sample, i_featsnaggr]
                thresholds[i_aggr,instance_id] = aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,instance_id])
                # println(gamma)
                # println(thresholds[i_aggr,instance_id])
            end
        end

        # println(thresholds)
        @logmsg DTDebug "thresholds: " thresholds

        # For each aggregator
        for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)

            # println(aggr)

            aggr_thresholds = thresholds[i_aggr,:]
            aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))

            for (i_test_operator,test_operator) in enumerate(aggrsnops[aggr])
                @logmsg DTDetail " Test operator $(test_operator)"

                # Look for the best threshold 'a', as in propositions like "feature >= a"
                for threshold in aggr_domain
                    decision = Decision(relation, feature, test_operator, threshold)
                    @logmsg DTDebug " Testing decision: $(display_decision(decision))"
                    @yield decision, aggr_thresholds
                end # for threshold
            end # for test_operator
        end # for aggregator
    end # for feature
end


Base.@propagate_inbounds @resumable function generate_modal_feasible_decisions(
        X::ExplicitModalDatasetS{T,WorldType},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        modal_relations_inds::AbstractVector{<:Integer},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:World}
    _n_samples = length(instances_inds)

    # For each relational operator
    for i_relation in modal_relations_inds
        relation = relations(X)[i_relation]
        @logmsg DTDebug "Relation $(relation)..."

        # For each feature
        for i_feature in features_inds
            feature = features(X)[i_feature]
            @logmsg DTDebug "Feature $(i_feature): $(feature)"

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
                @logmsg DTDetail " Instance $(i_sample)/$(_n_samples)"
                worlds = Sf[i_sample] # TODO could also use accessibles_aggr_funs here?

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

            @logmsg DTDebug "thresholds: " thresholds

            # For each aggregator
            for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)

                aggr_thresholds = thresholds[i_aggr,:]
                aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))

                for (i_test_operator,test_operator) in enumerate(aggrsnops[aggr])
                    @logmsg DTDetail " Test operator $(test_operator)"

                    # Look for the best threshold 'a', as in propositions like "feature >= a"
                    for threshold in aggr_domain
                        decision = Decision(relation, feature, test_operator, threshold)
                        @logmsg DTDebug " Testing decision: $(display_decision(decision))"
                        @yield decision, aggr_thresholds
                    end # for threshold
                end # for test_operator
            end # for aggregator
        end # for feature
    end # for relation
end

# get_global_gamma(
#       X::ExplicitModalDatasetSMemo{T,WorldType},
#       i_sample::Integer,
#       feature::ModalFeature,
#       test_operator::TestOperatorFun) where {WorldType<:World, T} = begin
#   @assert !isnothing(X.fwd_gs) "Error. ExplicitModalDatasetSMemo must be built with compute_relation_glob = true for it to be ready to test global decisions."
#   i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
#   # if !isnothing(X.fwd_gs[i_sample, i_featsnaggr])
#   X.fwd_gs[i_sample, i_featsnaggr]
#   # else
#   #   i_feature = find_feature_id(X, feature)
#   #   aggregator = existential_aggregator(test_operator)
#   #   fwd_feature_slice = fwd_get_channel(X.emd.fwd, i_sample, i_feature)
#   #   accessible_worlds = all_worlds_aggr(WorldType, accessibles_aggr_fun(X.emd, i_sample), feature, aggregator)
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
        X::ExplicitModalDatasetSMemo{T,WorldType},
        i_sample::Integer,
        w::WorldType,
        relation::Relation,
        feature::ModalFeature,
        test_operator::TestOperatorFun) where {WorldType<:World, T} = begin
    i_relation = find_relation_id(X, relation)
    aggregator = existential_aggregator(test_operator)
    i_featsnaggr = find_featsnaggr_id(X, feature, aggregator)
    # if coin_flip_no_look_ExplicitModalDatasetSWithMemoization() ||
    if false ||
            isnothing(X.fwd_rs[i_sample, w, i_featsnaggr, i_relation])
        i_feature = find_feature_id(X, feature)
        fwd_feature_slice = fwd_get_channel(X.emd.fwd, i_sample, i_feature)
        accessible_worlds = accessibles_aggr_fun(X.emd, i_sample)(feature, aggregator, w, relation)
        gamma = compute_modal_gamma(fwd_feature_slice, accessible_worlds, aggregator)
        fwd_rs_set(X.fwd_rs, i_sample, w, i_featsnaggr, i_relation, gamma)
    else
        X.fwd_rs[i_sample, w, i_featsnaggr, i_relation]
    end
end


Base.@propagate_inbounds @resumable function generate_modal_feasible_decisions(
        X::ExplicitModalDatasetSMemo{T,WorldType},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        modal_relations_inds::AbstractVector{<:Integer},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:World}
    _n_samples = length(instances_inds)

    # For each relational operator
    for i_relation in modal_relations_inds
        relation = relations(X)[i_relation]
        @logmsg DTDebug "Relation $(relation)..."

        # For each feature
        for i_feature in features_inds
            feature = features(X)[i_feature]
            @logmsg DTDebug "Feature $(i_feature): $(feature)"

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
                @logmsg DTDetail " Instance $(instance_id)/$(_n_samples)"
                worlds = Sf[instance_id] # TODO could also use accessibles_aggr_funs here?

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
                                accessible_worlds = accessibles_aggr_fun(X.emd, i_sample)(feature, aggregator, w, relation)
                                gamma = compute_modal_gamma(fwd_feature_slice, accessible_worlds, aggregator)
                                fwd_rs_set(X.fwd_rs, i_sample, w, i_featsnaggr, i_relation, gamma)
                            else
                                X.fwd_rs[i_sample, w, i_featsnaggr, i_relation]
                            end
                        thresholds[i_aggr,instance_id] = aggregator_to_binary(aggregator)(gamma, thresholds[i_aggr,instance_id])
                    end
                end
            end

            @logmsg DTDebug "thresholds: " thresholds

            # For each aggregator
            for (i_aggr,(_,aggregator)) in enumerate(aggregators_with_ids)

                aggr_thresholds = thresholds[i_aggr,:]
                aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))

                for (i_test_operator,test_operator) in enumerate(aggrsnops[aggregator])
                    @logmsg DTDetail " Test operator $(test_operator)"

                    # Look for the best threshold 'a', as in propositions like "feature >= a"
                    for threshold in aggr_domain
                        decision = Decision(relation, feature, test_operator, threshold)
                        @logmsg DTDebug " Testing decision: $(display_decision(decision))"
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
        X::Union{ActiveModalDataset{T,WorldType},InterpretedModalDataset{T,N,WorldType}},
        i_sample::Integer,
        worlds::WorldSetType,
        decision::Decision{T},
        returns_survivors::Union{Val{true},Val{false}} = Val(false)
    ) where {T, N, WorldType<:World, WorldSetType<:AbstractWorldSet{WorldType}}
    @logmsg DTDetail "modal_step" worlds display_decision(decision)

    satisfied = false

    # TODO space for optimization here: with some relations (e.g. IA_A, IA_L) can be made smaller

    if returns_survivors isa Val{true}
        worlds_map = Dict{WorldType,AbstractWorldSet{WorldType}}()
    end
    if length(worlds) == 0
        # If there are no neighboring worlds, then the modal decision is not met
        @logmsg DTDetail "   No accessible world"
    else
        # Otherwise, check whether at least one of the accessible worlds witnesses truth of the decision.
        # TODO rewrite with new_worlds = map(...acc_worlds)
        # Initialize new worldset
        new_worlds = WorldSetType()

        # List all accessible worlds
        acc_worlds =
            if returns_survivors isa Val{true}
                l = Threads.Condition()
                Threads.@threads for curr_w in worlds
                    acc = accessibles_fun(X, i_sample)(curr_w, decision.relation) |> collect
                    lock(l)
                    worlds_map[curr_w] = acc
                    unlock(l)
                end
                unique(cat([ worlds_map[k] for k in keys(worlds_map) ]...; dims = 1))
            else
                accessibles_fun(X, i_sample)(worlds, decision.relation)
            end

        for w in acc_worlds
            if test_decision(X, i_sample, w, decision.feature, decision.test_operator, decision.threshold)
                # @logmsg DTDetail " Found world " w ch_readWorld ... ch_readWorld(w, channel)
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
        @logmsg DTDetail "   YES" worlds
    else
        @logmsg DTDetail "   NO"
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
        w::World,
        feature::ModalFeature,
        test_operator::TestOperatorFun,
        threshold::T) where {T} = begin
    gamma = get_gamma(X, i_sample, w, feature)
    evaluate_thresh_decision(test_operator, gamma, threshold)
end

test_decision(
        X::ModalDataset{T},
        i_sample::Integer,
        w::World,
        decision::Decision{T}) where {T} = begin
    instance = get_instance(X, i_sample)

    aggregator = existential_aggregator(decision.test_operator)

    worlds = accessibles_aggr(decision.feature, aggregator, w, decision.relation, instance_channel_size(instance)...)
    gamma = if length(worlds |> collect) == 0
        aggregator_bottom(aggregator, T)
    else
        aggregator((w)->get_gamma(X, i_sample, w, decision.feature), worlds)
    end

    evaluate_thresh_decision(decision.test_operator, gamma, decision.threshold)
end


export generate_feasible_decisions
                # ,
                # generate_propositional_feasible_decisions,
                # generate_global_feasible_decisions,
                # generate_modal_feasible_decisions

Base.@propagate_inbounds @resumable function generate_feasible_decisions(
        X::ActiveModalDataset{T,WorldType},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        allow_propositional_decisions::Bool,
        allow_modal_decisions::Bool,
        allow_global_decisions::Bool,
        modal_relations_inds::AbstractVector{<:Integer},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:World}
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
