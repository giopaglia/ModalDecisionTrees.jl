using .ModalDecisionTrees: _CanonicalFeatureGeq, _CanonicalFeatureGeqSoft, _CanonicalFeatureLeq, _CanonicalFeatureLeqSoft

export n_samples, n_attributes, n_features, n_relations,
       max_channel_size,
       n_frames, frames, get_frame, push_frame!,
       display_structure,
       get_gamma, test_decision,
       # 
       relations,
       init_world_sets_fun,
       accessibles_fun,
       accessibles_aggr_funs,
       features,
       grouped_featsaggrsnops,
       featsnaggrs,
       grouped_featsnaggrs,
       # 
       SingleFrameGenericDataset,
       GenericDataset,
       MultiFrameModalDataset,
       ModalDataset,
       InterpretedModalDataset, 
       AbstractFeaturedWorldDataset,
       FeatModalDataset,
       StumpFeatModalDataset,
       StumpFeatModalDatasetWithMemoization,
       DimensionalInstance,
       DimensionalDataset,
       DimensionalChannel

const initWorldSetFunction = Function
const accFunction = Function
const accReprFunction = Function

abstract type ModalDataset{T<:Number,WorldType<:AbstractWorld} end

############################################################################################
# Interpreted modal dataset
############################################################################################
# 
# A modal dataset can be instantiated in *implicit* form, from a dimensional domain, and a few
#  objects inducing an interpretation on the domain; mainly, an ontology (determining worlds and
#  relations), and structures for interpreting modal features onto the domain.
# 
############################################################################################

@computed struct InterpretedModalDataset{T<:Number, N, WorldType<:AbstractWorld} <: ModalDataset{T, WorldType}
    
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
    ) where {T, N, D, WorldType<:AbstractWorld}
        InterpretedModalDataset{T}(domain, ontology, mixed_features)
    end

    function InterpretedModalDataset{T}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        mixed_features::AbstractVector{<:MixedFeature},
    ) where {T, N, D, WorldType<:AbstractWorld}
        InterpretedModalDataset{T, D-1-1}(domain, ontology, mixed_features)
    end

    function InterpretedModalDataset{T, N}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        mixed_features::AbstractVector{<:MixedFeature},
    ) where {T, N, D, WorldType<:AbstractWorld}
        InterpretedModalDataset{T, N, WorldType}(domain, ontology, mixed_features)
    end
    
    function InterpretedModalDataset{T, N, WorldType}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        mixed_features::AbstractVector{<:MixedFeature},
    ) where {T, N, D, WorldType<:AbstractWorld}
        features, featsnops = begin
            features = ModalFeature[]
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
                push!(features, cf)
                push!(featsnops, test_ops)
            end

            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic._CanonicalFeatureGeq) = ([≥],ModalLogic.SingleAttributeMin(i_attr))
            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic._CanonicalFeatureLeq) = ([≤],ModalLogic.SingleAttributeMax(i_attr))
            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic._CanonicalFeatureGeqSoft) = ([≥],ModalLogic.SingleAttributeSoftMin(i_attr, cf.alpha))
            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic._CanonicalFeatureLeqSoft) = ([≤],ModalLogic.SingleAttributeSoftMax(i_attr, cf.alpha))
            single_attr_feats_n_featsnops(i_attr,(test_ops,cf)::Tuple{<:AbstractVector{<:TestOperatorFun},Function}) = (test_ops,SingleAttributeFeature(i_attr, cf))
            single_attr_feats_n_featsnops(i_attr,::Any) = throw_n_log("Unknown mixed_feature type: $(cf), $(typeof(cf))")

            for i_attr in 1:n_attributes(domain)
                for (test_ops,cf) in map((cf)->single_attr_feats_n_featsnops(i_attr,cf),attribute_specific_cfs)
                    push!(featsnops, test_ops)
                    push!(features, cf)
                end
            end
            features, featsnops
        end
        ontology = getIntervalOntologyOfDim(Val(D-1-1))
        InterpretedModalDataset{T, N, world_type(ontology)}(domain, ontology, features, featsnops)
    end

    function InterpretedModalDataset(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops_or_featsnops; # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        kwargs...,
    ) where {T, D, WorldType<:AbstractWorld}
        InterpretedModalDataset{T}(domain, ontology, features, grouped_featsaggrsnops_or_featsnops; kwargs...)
    end

    function InterpretedModalDataset{T}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops_or_featsnops; # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        kwargs...,
    ) where {T, D, WorldType<:AbstractWorld}
        InterpretedModalDataset{T, D-1-1}(domain, ontology, features, grouped_featsaggrsnops_or_featsnops; kwargs...)
    end

    function InterpretedModalDataset{T, N}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops_or_featsnops; # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        kwargs...,
    ) where {T, N, D, WorldType<:AbstractWorld}
        InterpretedModalDataset{T, N, WorldType}(domain, ontology, features, grouped_featsaggrsnops_or_featsnops; kwargs...)
    end
    
    function InterpretedModalDataset{T, N, WorldType}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsnops  :: AbstractVector{<:AbstractVector{<:TestOperatorFun}};
        kwargs...,
    ) where {T, N, D, WorldType<:AbstractWorld}

        grouped_featsaggrsnops = grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops)
        
        InterpretedModalDataset{T, N, WorldType}(domain, ontology, features, grouped_featsaggrsnops; kwargs...)
    end
    function InterpretedModalDataset{T, N, WorldType}(
        domain::DimensionalDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}};
        allow_no_instances = false,
    ) where {T, N, D, WorldType<:AbstractWorld}

        @assert allow_no_instances || n_samples(domain) > 0 "Can't instantiate InterpretedModalDataset{$(T), $(N), $(WorldType)} with no instance. (domain's type $(typeof(domain)))"
        @assert N == dimensionality(WorldType) "ERROR! Dimensionality mismatch: can't interpret WorldType $(WorldType) (dimensionality = $(dimensionality(WorldType)) on DimensionalDataset of dimensionality = $(N))"
        @assert D == (N+1+1) "ERROR! Dimensionality mismatch: can't instantiate InterpretedModalDataset{$(T), $(N)} with DimensionalDataset{$(T),$(D)}"
        @assert length(features) == length(grouped_featsaggrsnops) "Can't instantiate InterpretedModalDataset{$(T), $(N), $(WorldType)} with mismatching length(features) == length(grouped_featsaggrsnops): $(length(features)) != $(length(grouped_featsaggrsnops))"
        # @assert length(grouped_featsaggrsnops) > 0 && sum(length.(grouped_featsaggrsnops)) > 0 && sum(vcat([[length(test_ops) for test_ops in aggrs] for aggrs in grouped_featsaggrsnops]...)) > 0 "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with no test operator: $(grouped_featsaggrsnops)"

        # if prod(max_channel_size(domain)) == 1
        #   TODO throw warning
        # end
        
        new{T, N, WorldType}(domain, ontology, features, grouped_featsaggrsnops)
    end
end

relations(imd::InterpretedModalDataset)        = relations(imd.ontology)
Base.size(imd::InterpretedModalDataset)        = size(imd.domain)
n_samples(imd::InterpretedModalDataset)        = n_samples(imd.domain)::Int64
n_attributes(imd::InterpretedModalDataset)     = n_attributes(imd.domain)::Int64
n_relations(imd::InterpretedModalDataset)      = length(relations(imd))::Int64
world_type(imd::InterpretedModalDataset{T,N,WT}) where {T,N,WT} = WT

features(imd::InterpretedModalDataset)               = imd.features
grouped_featsaggrsnops(imd::InterpretedModalDataset) = imd.grouped_featsaggrsnops
n_features(imd::InterpretedModalDataset)             = length(imd.features)::Int64

init_world_sets_fun(imd::InterpretedModalDataset{T, N, WorldType},  i_instance) where {T, N, WorldType} =
    (iC)->initWorldSet(iC, WorldType, instance_channel_size(get_instance(imd, i_instance)))
accessibles_fun(imd::InterpretedModalDataset, i_instance) = (w,R)->accessibles(w,R, instance_channel_size(get_instance(imd, i_instance))...)
all_worlds_fun(imd::InterpretedModalDataset{T, N, WorldType}, i_instance) where {T, N, WorldType} = all_worlds(WorldType, accessibles_fun(imd, i_instance))
accessibles_aggr_fun(imd::InterpretedModalDataset, i_instance)  = (f,a,w,R)->accessibles_aggr(f,a,w,R,instance_channel_size(get_instance(imd, i_instance))...)

# Note: Can't define Base.length(::DimensionalDataset) & Base.iterate(::DimensionalDataset)
Base.length(imd::InterpretedModalDataset)                = n_samples(imd)
Base.iterate(imd::InterpretedModalDataset, state=1) = state > length(imd) ? nothing : (get_instance(imd, state), state+1) # Base.iterate(imd.domain, state=state)
max_channel_size(imd::InterpretedModalDataset)          = max_channel_size(imd.domain)

get_instance(imd::InterpretedModalDataset, args...)     = get_instance(imd.domain, args...)

slice_dataset(imd::InterpretedModalDataset, inds::AbstractVector{<:Integer}, args...; allow_no_instances = false, kwargs...)    =
    InterpretedModalDataset(slice_dataset(imd.domain, inds, args...; allow_no_instances = allow_no_instances, kwargs...), imd.ontology, imd.features, imd.grouped_featsaggrsnops; allow_no_instances = allow_no_instances)


display_structure(imd::InterpretedModalDataset; indent_str = "") = begin
    out = "$(typeof(imd))\t$(Base.summarysize(imd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ domain shape\t$(Base.size(imd.domain))\n"
    out *= indent_str * "├ $(length(relations(imd))) relations\t$(relations(imd))\n"
    out *= indent_str * "└ max_channel_size\t$(max_channel_size(imd))"
    out
end

get_gamma(imd::InterpretedModalDataset, args...) = get_gamma(imd.domain, args...)

############################################################################################
# Featured modal dataset
############################################################################################
# 
# In the most general case, the representation of a modal dataset is based on a
#  multi-dimensional lookup table, referred to as *propositional lookup table*,
#  or *featured world dataset* (abbreviated into fwd).
# 
# This structure, is such that the value at fwd[i_instance, world, feature]
# answers the question  whether a proposition (e.g., minimum(A1) ≥ 10) holds onto a given world;
# however, an fwd table can be implemented in many ways, mainly depending on the world type.
# 
############################################################################################

abstract type AbstractFeaturedWorldDataset{T<:Number,WorldType<:AbstractWorld} end

# Any implementation for a fwd must indicate their compatible world types via `goes_with`.
# Fallback:
goes_with(::Type{<:AbstractFeaturedWorldDataset}, ::Type{<:AbstractWorld}) = false

# Any world type must also specify their default fwd constructor, which must accept a type
#  parameter for the data type {T}, via:
# default_fwd_type(::Type{<:AbstractWorld})

# 
# Actually, the interface for AbstractFeaturedWorldDataset's is a bit tricky; the most straightforward
#  way of learning it is by considering the fallback fwd structure defined as follows.
# TODO oh, but the implementation is broken due to a strange error (see https://discourse.julialang.org/t/tricky-too-many-parameters-for-type-error/25182 )

# # The most generic fwd structure is a matrix of dictionaries of size (n_samples × n_features)
# struct GenericFeaturedWorldDataset{T, WorldType} <: AbstractFeaturedWorldDataset{T, WorldType}
#   d :: AbstractVector{<:AbstractDict{WorldType,AbstractVector{T,1}},1}
#   n_features :: Integer
# end

# # It goes for any world type
# goes_with(::Type{<:GenericFeaturedWorldDataset}, ::Type{<:AbstractWorld}) = true

# # And it is the default fwd structure for an world type
# default_fwd_type(::Type{<:AbstractWorld}) = GenericFeaturedWorldDataset

# n_samples(fwd::GenericFeaturedWorldDataset{T}) where {T}  = size(fwd, 1)
# n_features(fwd::GenericFeaturedWorldDataset{T}) where {T} = fwd.d
# Base.size(fwd::GenericFeaturedWorldDataset{T}, args...) where {T} = size(fwd.d, args...)

# # The matrix is initialized with #undef values
# function fwd_init(::Type{GenericFeaturedWorldDataset}, imd::InterpretedModalDataset{T}) where {T}
#     d = Array{Dict{WorldType,T}, 2}(undef, n_samples(imd))
#     for i in 1:n_samples
#         d[i] = Dict{WorldType,Array{T,1}}()
#     end
#     GenericFeaturedWorldDataset{T}(d, n_features(imd))
# end

# # A function for initializing individual world slices
# function fwd_init_world_slice(fwd::GenericFeaturedWorldDataset{T}, i_instance::Integer, w::AbstractWorld) where {T}
#     fwd.d[i_instance][w] = Array{T,1}(undef, fwd.n_features)
# end

# # A function for getting a threshold value from the lookup table
# fwd_get(
#     fwd         :: GenericFeaturedWorldDataset{T},
#     i_instance  :: Integer,
#     w           :: AbstractWorld,
#     i_feature   :: Integer) where {T} = fwd.d[i_instance][w][i_feature]

# # A function for setting a threshold value in the lookup table
# function fwd_set(fwd::GenericFeaturedWorldDataset{T}, w::AbstractWorld, i_instance::Integer, i_feature::Integer, threshold::T) where {T}
#     fwd.d[i_instance][w][i_feature] = threshold
# end

# # A function for setting threshold values for a single feature (from a feature slice, experimental)
# function fwd_set_feature(fwd::GenericFeaturedWorldDataset{T}, i_feature::Integer, feature_slice::Any) where {T}
#     throw_n_log("Warning! fwd_set_feature with GenericFeaturedWorldDataset is not yet implemented!")
#     for ((i_instance,w),threshold::T) in read_fwd_feature_slice(feature_slice)
#         fwd.d[i_instance][w][i_feature] = threshold
#     end
# end

# # A function for slicing the dataset
# function slice_dataset(fwd::GenericFeaturedWorldDataset{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
#     @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
#     GenericFeaturedWorldDataset{T}(if return_view @view fwd.d[inds] else fwd.d[inds] end, fwd.n_features)
# end

# Others...
# fwd_get_channel(fwd::GenericFeaturedWorldDataset{T}, i_instance::Integer, i_feature::Integer) where {T} = TODO
# const GenericFeaturedChannel{T} = TODO
# fwd_channel_interpret_world(fwc::GenericFeaturedChannel{T}, w::AbstractWorld) where {T} = TODO

############################################################################################
# Featured modal dataset
# 
# A featured modal dataset is the basic explicit form of a modal dataset, and consists of
#  a wrapper around an fwd lookup table. The information it adds are the relation set,
#  a few functions for enumerating worlds (`accessibles`, `accessibles_aggr`),
#  and a world set initialization function representing initial conditions (initializing world sets).
# 
############################################################################################

struct FeatModalDataset{T<:Number, WorldType<:AbstractWorld} <: ModalDataset{T, WorldType}
    
    # Core data (fwd lookup table)
    fwd                :: AbstractFeaturedWorldDataset{T,WorldType}
    
    ## Modal frame:
    # Accessibility relations
    relations          :: AbstractVector{<:AbstractRelation}
    
    # Worldset initialization functions (one per instance)
    #  with signature (initCondition) -> vs::AbstractWorldSet{WorldType}
    init_world_sets_funs   :: AbstractVector{<:initWorldSetFunction}
    # Accessibility functions (one per instance)
    #  with signature (w::WorldType/AbstractWorldSet{WorldType}, r::AbstractRelation) -> vs::AbstractVector{WorldType}
    accessibles_funs      :: AbstractVector{<:accFunction}
    # Representative accessibility functions (one per instance)
    #  with signature (feature::ModalFeature, aggregator::Aggregator, w::WorldType/AbstractWorldSet{WorldType}, r::AbstractRelation) -> vs::AbstractVector{WorldType}
    accessibles_aggr_funs  :: AbstractVector{<:accReprFunction}
    
    # Features
    features           :: AbstractVector{<:ModalFeature}

    # Test operators associated with each feature, grouped by their respective aggregator
    grouped_featsaggrsnops  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}

    FeatModalDataset(
        fwd                :: AbstractFeaturedWorldDataset{T,WorldType},
        relations          :: AbstractVector{<:AbstractRelation},
        init_world_sets_funs   :: AbstractVector{<:initWorldSetFunction},
        accessibles_funs      :: AbstractVector{<:accFunction},
        accessibles_aggr_funs  :: AbstractVector{<:accReprFunction},
        features           :: AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops_or_featsnops, # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        args...;
        kwargs...,
    ) where {T,WorldType} = begin FeatModalDataset{T, WorldType}(fwd, relations, init_world_sets_funs, accessibles_funs, accessibles_aggr_funs, features, grouped_featsaggrsnops_or_featsnops, args...; kwargs...) end

    function FeatModalDataset{T, WorldType}(
        fwd                :: AbstractFeaturedWorldDataset{T,WorldType},
        relations          :: AbstractVector{<:AbstractRelation},
        init_world_sets_funs   :: AbstractVector{<:initWorldSetFunction},
        accessibles_funs      :: AbstractVector{<:accFunction},
        accessibles_aggr_funs  :: AbstractVector{<:accReprFunction},
        features           :: AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}};
        allow_no_instances = false,
    ) where {T,WorldType<:AbstractWorld}
        @assert allow_no_instances || n_samples(fwd) > 0     "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with no instance. (fwd's type $(typeof(fwd)))"
        @assert length(grouped_featsaggrsnops) > 0 && sum(length.(grouped_featsaggrsnops)) > 0 && sum(vcat([[length(test_ops) for test_ops in aggrs] for aggrs in grouped_featsaggrsnops]...)) > 0 "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with no test operator: grouped_featsaggrsnops"
        @assert  n_samples(fwd) == length(init_world_sets_funs)  "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of init_world_sets_funs $(length(init_world_sets_funs))."
        @assert  n_samples(fwd) == length(accessibles_funs)     "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of accessibles_funs $(length(accessibles_funs))."
        @assert  n_samples(fwd) == length(accessibles_aggr_funs) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of accessibles_aggr_funs $(length(accessibles_aggr_funs))."
        @assert n_features(fwd) == length(features)          "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of features $(length(features))."
        new{T, WorldType}(fwd, relations, init_world_sets_funs, accessibles_funs, accessibles_aggr_funs, features, grouped_featsaggrsnops)
    end

    function FeatModalDataset(
        fwd                :: AbstractFeaturedWorldDataset{T,WorldType},
        relations          :: AbstractVector{<:AbstractRelation},
        init_world_sets_funs   :: AbstractVector{<:initWorldSetFunction},
        accessibles_funs      :: AbstractVector{<:accFunction},
        accessibles_aggr_funs  :: AbstractVector{<:accReprFunction},
        features           :: AbstractVector{<:ModalFeature},
        grouped_featsnops  :: AbstractVector{<:AbstractVector{<:TestOperatorFun}},
        args...;
        kwargs...,
    ) where {T,WorldType<:AbstractWorld}

        grouped_featsaggrsnops = grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops)
 
        FeatModalDataset(fwd, relations, init_world_sets_funs, accessibles_funs, accessibles_aggr_funs, features, grouped_featsaggrsnops, args...; kwargs...)
    end

    # Quite importantly, an fwd can be computed from a dataset in implicit form (domain + ontology + features)
    Base.@propagate_inbounds function FeatModalDataset(
        imd                  :: InterpretedModalDataset{T, N, WorldType},
        FWD::Type{<:AbstractFeaturedWorldDataset} = default_fwd_type(WorldType),
        args...;
        kwargs...,
    ) where {T, N, WorldType<:AbstractWorld}

        fwd = begin

            @logmsg DTOverview "InterpretedModalDataset -> FeatModalDataset"

            features = imd.features
            
            n_instances = n_samples(imd)
            n_features = length(features)
            
            @assert goes_with(FWD, WorldType)

            # Initialize the fwd structure
            fwd = fwd_init(FWD, imd)

            # Load any (possible) external features
            if any(isa.(features, ExternalFWDFeature))
                i_external_features = first.(filter(((i_feature,is_external_fwd),)->(is_external_fwd), collect(enumerate(isa.(features, ExternalFWDFeature)))))
                for i_feature in i_external_features
                    feature = features[i_feature]
                    fwd_set_feature_slice(fwd, i_feature, feature.fwd)
                end
            end

            # Load any internal features
            i_features = first.(filter(((i_feature,is_external_fwd),)->!(is_external_fwd), collect(enumerate(isa.(features, ExternalFWDFeature)))))
            enum_features = zip(i_features, features[i_features])

            # Compute features
            @inbounds Threads.@threads for i_instance in 1:n_instances
                @logmsg DTDebug "Instance $(i_instance)/$(n_instances)"
                
                if i_instance == 1 || ((i_instance+1) % (floor(Int, ((n_instances)/4))+1)) == 0
                    @logmsg DTOverview "Instance $(i_instance)/$(n_instances)"
                end

                # instance = get_instance(imd, i_instance)
                # @logmsg DTDebug "instance" instance

                for w in all_worlds_fun(imd, i_instance)
                    
                    fwd_init_world_slice(fwd, i_instance, w)

                    @logmsg DTDebug "World" w

                    for (i_feature,feature) in enum_features

                        # threshold = computePropositionalThreshold(feature, w, instance)
                        threshold = get_gamma(imd, i_instance, w, feature)

                        @logmsg DTDebug "Feature $(i_feature)" threshold
                    
                        fwd_set(fwd, w, i_instance, i_feature, threshold)

                    end
                end
            end
        end

        # TODO Think about it, and optimize this: when the underlying DimensionalDataset is an Array,
        #  this is going to be an array of a single function.
        init_world_sets_funs  = [init_world_sets_fun(imd,  i_instance) for i_instance in 1:n_samples(imd)]
        accessibles_funs     = [accessibles_fun(imd,     i_instance) for i_instance in 1:n_samples(imd)]
        accessibles_aggr_funs = [accessibles_aggr_fun(imd, i_instance) for i_instance in 1:n_samples(imd)]

        FeatModalDataset(fwd, relations(imd), init_world_sets_funs, accessibles_funs, accessibles_aggr_funs, features(imd), grouped_featsaggrsnops(imd), args...; kwargs...)
    end

end

relations(X::FeatModalDataset)         = X.relations
init_world_sets_fun(X::FeatModalDataset,  i_instance)  = X.init_world_sets_funs[i_instance]
accessibles_fun(X::FeatModalDataset,     i_instance)  = X.accessibles_funs[i_instance]
all_worlds_fun(X::FeatModalDataset{T, WorldType}, i_instance) where {T, WorldType} = all_worlds(WorldType, accessibles_fun(X, i_instance))
accessibles_aggr_fun(X::FeatModalDataset, i_instance)  = X.accessibles_aggr_funs[i_instance]
features(X::FeatModalDataset)          = X.features
grouped_featsaggrsnops(X::FeatModalDataset) = X.grouped_featsaggrsnops

Base.size(X::FeatModalDataset)             where {T,N} =  size(X.fwd)
n_samples(X::FeatModalDataset{T, WorldType}) where {T, WorldType}   = n_samples(X.fwd)::Int64
n_features(X::FeatModalDataset{T, WorldType}) where {T, WorldType}  = length(X.features)
n_relations(X::FeatModalDataset{T, WorldType}) where {T, WorldType} = length(X.relations)
# length(X::FeatModalDataset{T,WorldType})        where {T,WorldType} = n_samples(X)
# Base.iterate(X::FeatModalDataset{T,WorldType}, state=1) where {T, WorldType} = state > length(X) ? nothing : (get_instance(X, state), state+1)
Base.getindex(X::FeatModalDataset{T,WorldType}, args...) where {T,WorldType} = getindex(X.fwd, args...)
world_type(X::FeatModalDataset{T,WorldType}) where {T,WorldType<:AbstractWorld} = WorldType


slice_dataset(X::FeatModalDataset{T,WorldType}, inds::AbstractVector{<:Integer}, args...; allow_no_instances = false, kwargs...) where {T,WorldType} =
    FeatModalDataset{T,WorldType}(
        slice_dataset(X.fwd, inds, args...; allow_no_instances = allow_no_instances, kwargs...),
        X.relations,
        X.init_world_sets_funs[inds],
        X.accessibles_funs[inds],
        X.accessibles_aggr_funs[inds],
        X.features,
        X.grouped_featsaggrsnops;
        allow_no_instances = allow_no_instances
    )


function grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops::AbstractVector{<:AbstractVector{<:TestOperatorFun}})::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
    grouped_featsaggrsnops = Dict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}[]
    for (i_feature, test_operators) in enumerate(grouped_featsnops)
        aggrsnops = Dict{Aggregator,AbstractVector{<:TestOperatorFun}}()
        for test_operator in test_operators
            aggregator = ModalLogic.existential_aggregator(test_operator)
            if (!haskey(aggrsnops, aggregator))
                aggrsnops[aggregator] = TestOperatorFun[]
            end
            push!(aggrsnops[aggregator], test_operator)
        end
        push!(grouped_featsaggrsnops, aggrsnops)
    end
    grouped_featsaggrsnops
end


find_feature_id(X::FeatModalDataset{T,WorldType}, feature::ModalFeature) where {T,WorldType} =
    findall(x->x==feature, features(X))[1]
find_relation_id(X::FeatModalDataset{T,WorldType}, relation::AbstractRelation) where {T,WorldType} =
    findall(x->x==relation, relations(X))[1]

get_gamma(
        X::FeatModalDataset{T,WorldType},
        i_instance::Integer,
        w::WorldType,
        feature::ModalFeature) where {WorldType<:AbstractWorld, T} = begin
    i_feature = find_feature_id(X, feature)
    X[i_instance, w, i_feature]
end


Base.@propagate_inbounds @resumable function generate_propositional_feasible_decisions(
        X::FeatModalDataset{T,WorldType},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:AbstractWorld}
    relation = RelationId
    n_instances = length(instances_inds)

    # For each feature
    @inbounds for i_feature in features_inds
        feature = features(X)[i_feature]
        @logmsg DTDebug "Feature $(i_feature): $(feature)"

        # operators for each aggregator
        aggrsnops = grouped_featsaggrsnops(X)[i_feature]
        # Vector of aggregators
        aggregators = keys(aggrsnops) # Note: order-variant, but that's ok here
        
        # dict->vector
        # aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

        # Initialize thresholds with the bottoms
        thresholds = Array{T,2}(undef, length(aggregators), n_instances)
        for (i_aggr,aggr) in enumerate(aggregators)
            thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
        end

        # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
        for (i_instance,instance_id) in enumerate(instances_inds)
            @logmsg DTDetail " Instance $(i_instance)/$(n_instances)"
            worlds = Sf[i_instance]

            # TODO also try this instead
            # values = [X.fmd[instance_id, w, i_feature] for w in worlds]
            # thresholds[:,i_instance] = map(aggr->aggr(values), aggregators)
            
            for w in worlds
                gamma = fwd_get(X.fwd, instance_id, w, i_feature)
                for (i_aggr,aggr) in enumerate(aggregators)
                    thresholds[i_aggr,i_instance] = ModalLogic.aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,i_instance])
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

abstract type AbstractFMDStumpSupport{T, WorldType} end
abstract type AbstractFMDStumpGlobalSupport{T} end

#  Stump support provides a structure for thresholds.
#   A threshold is the unique value γ for which w ⊨ <R> f ⋈ γ and:
#   if polarity(⋈) == true:      ∀ a > γ:    w ⊭ <R> f ⋈ a
#   if polarity(⋈) == false:     ∀ a < γ:    w ⊭ <R> f ⋈ a
#   for a given feature f, world w, relation R and feature f and test operator ⋈,

struct StumpFeatModalDataset{T<:Number, WorldType<:AbstractWorld} <: ModalDataset{T, WorldType}
    
    # Core data
    fmd                :: FeatModalDataset{T, WorldType}

    # Stump support
    fmd_m              :: AbstractFMDStumpSupport{T, WorldType}
    fmd_g              :: Union{AbstractFMDStumpGlobalSupport{T},Nothing}

    # Features and Aggregators
    featsnaggrs         :: AbstractVector{Tuple{<:ModalFeature,<:Aggregator}}
    grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}}

    function StumpFeatModalDataset{T, WorldType}(
        fmd                :: FeatModalDataset{T, WorldType},
        fmd_m              :: AbstractFMDStumpSupport{T, WorldType},
        fmd_g              :: Union{AbstractFMDStumpGlobalSupport{T},Nothing},
        featsnaggrs         :: AbstractVector{Tuple{<:ModalFeature,<:Aggregator}},
        grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
    ) where {T,WorldType<:AbstractWorld}
        @assert n_samples(fmd) == n_samples(fmd_m) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_samples for fmd and fmd_m support: $(n_samples(fmd)) and $(n_samples(fmd_m))"
        @assert n_relations(fmd) == n_relations(fmd_m) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_relations for fmd and fmd_m support: $(n_relations(fmd)) and $(n_relations(fmd_m))"
        @assert world_type(fmd) == world_type(fmd_m) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching world_type for fmd and fmd_m support: $(world_type(fmd)) and $(world_type(fmd_m))"
        @assert sum(length.(grouped_featsnaggrs)) == length(featsnaggrs) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_featsnaggrs (grouped vs flattened structure): $(sum(length.(fmd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(fmd.grouped_featsaggrsnops)) == length(featsnaggrs) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and provided featsnaggrs: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(fmd.grouped_featsaggrsnops)) == n_featsnaggrs(fmd_m) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and fmd_m support: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(n_featsnaggrs(fmd_m))"

        if fmd_g != nothing
            @assert n_samples(fmd) == n_samples(fmd_g) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_samples for fmd and fmd_g support: $(n_samples(fmd)) and $(n_samples(fmd_g))"
            # @assert somethinglike(fmd) == n_featsnaggrs(fmd_g) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching somethinglike for fmd and fmd_g support: $(somethinglike(fmd)) and $(n_featsnaggrs(fmd_g))"
            # @assert world_type(fmd) == world_type(fmd_g) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching world_type for fmd and fmd_g support: $(world_type(fmd)) and $(world_type(fmd_g))"
            @assert sum(length.(fmd.grouped_featsaggrsnops)) == n_featsnaggrs(fmd_g) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and fmd_g support: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(n_featsnaggrs(fmd_g))"
        end

        new{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
    end

    function StumpFeatModalDataset(
        fmd                 :: FeatModalDataset{T, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T,WorldType<:AbstractWorld}
        StumpFeatModalDataset{T, WorldType}(fmd, computeRelationGlob = computeRelationGlob)
    end

    function StumpFeatModalDataset{T, WorldType}(
        fmd                 :: FeatModalDataset{T, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T,WorldType<:AbstractWorld}
        
        featsnaggrs = Tuple{<:ModalFeature,<:Aggregator}[]
        grouped_featsnaggrs = AbstractVector{Tuple{<:Integer,<:Aggregator}}[]

        i_featsnaggr = 1
        for (feat,aggrsnops) in zip(fmd.features,fmd.grouped_featsaggrsnops)
            aggrs = []
            for aggr in keys(aggrsnops)
                push!(featsnaggrs, (feat,aggr))
                push!(aggrs, (i_featsnaggr,aggr))
                i_featsnaggr += 1
            end
            push!(grouped_featsnaggrs, aggrs)
        end

        # Compute modal dataset propositions and 1-modal decisions
        fmd_m, fmd_g = computeModalDatasetStumpSupport(fmd, grouped_featsnaggrs, computeRelationGlob = computeRelationGlob);

        StumpFeatModalDataset{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
    end

    function StumpFeatModalDataset(
        X                   :: InterpretedModalDataset{T, N, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T, N, WorldType<:AbstractWorld}
        StumpFeatModalDataset{T, WorldType}(X, computeRelationGlob = computeRelationGlob)
    end

    function StumpFeatModalDataset{T, WorldType}(
        X                   :: InterpretedModalDataset{T, N, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T, N, WorldType<:AbstractWorld}

        # Compute modal dataset propositions
        fmd = FeatModalDataset(X);

        StumpFeatModalDataset{T, WorldType}(fmd, computeRelationGlob = computeRelationGlob)

        # TODO bring back ModalDatasetStumpSupport computation from X. 

        # fmd_m, fmd_g = computeModalDatasetStumpSupport(X, relations, fmd.grouped_featsaggrsnops??, fmd, features, computeRelationGlob = computeRelationGlob);

        # new{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
    end
end

mutable struct StumpFeatModalDatasetWithMemoization{T<:Number, WorldType<:AbstractWorld} <: ModalDataset{T, WorldType}
    
    # Core data
    fmd                :: FeatModalDataset{T, WorldType}

    # Stump support
    fmd_m              :: AbstractFMDStumpSupport{<:Union{T,Nothing}, WorldType}
    fmd_g              :: Union{AbstractFMDStumpGlobalSupport{T},Nothing}

    # Features and Aggregators
    featsnaggrs         :: AbstractVector{Tuple{<:ModalFeature,<:Aggregator}}
    grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}}

    function StumpFeatModalDatasetWithMemoization{T, WorldType}(
        fmd                :: FeatModalDataset{T, WorldType},
        fmd_m              :: AbstractFMDStumpSupport{<:Union{T,Nothing}, WorldType},
        fmd_g              :: Union{AbstractFMDStumpGlobalSupport{T},Nothing},
        featsnaggrs         :: AbstractVector{Tuple{<:ModalFeature,<:Aggregator}},
        grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
    ) where {T,WorldType<:AbstractWorld}
        @assert n_samples(fmd) == n_samples(fmd_m) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching n_samples for fmd and fmd_m support: $(n_samples(fmd)) and $(n_samples(fmd_m))"
        @assert n_relations(fmd) == n_relations(fmd_m) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching n_relations for fmd and fmd_m support: $(n_relations(fmd)) and $(n_relations(fmd_m))"
        @assert world_type(fmd) == world_type(fmd_m) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching world_type for fmd and fmd_m support: $(world_type(fmd)) and $(world_type(fmd_m))"
        @assert sum(length.(grouped_featsnaggrs)) == length(featsnaggrs) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching n_featsnaggrs (grouped vs flattened structure): $(sum(length.(fmd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(fmd.grouped_featsaggrsnops)) == length(featsnaggrs) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and provided featsnaggrs: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(fmd.grouped_featsaggrsnops)) == n_featsnaggrs(fmd_m) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and fmd_m support: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(n_featsnaggrs(fmd_m))"

        if fmd_g != nothing
            @assert n_samples(fmd) == n_samples(fmd_g) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching n_samples for fmd and fmd_g support: $(n_samples(fmd)) and $(n_samples(fmd_g))"
            # @assert somethinglike(fmd) == n_featsnaggrs(fmd_g) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching somethinglike for fmd and fmd_g support: $(somethinglike(fmd)) and $(n_featsnaggrs(fmd_g))"
            # @assert world_type(fmd) == world_type(fmd_g) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching world_type for fmd and fmd_g support: $(world_type(fmd)) and $(world_type(fmd_g))"
            @assert sum(length.(fmd.grouped_featsaggrsnops)) == n_featsnaggrs(fmd_g) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and fmd_g support: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(n_featsnaggrs(fmd_g))"
        end

        new{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
    end

    function StumpFeatModalDatasetWithMemoization(
        fmd                 :: FeatModalDataset{T, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T,WorldType<:AbstractWorld}
        StumpFeatModalDatasetWithMemoization{T, WorldType}(fmd, computeRelationGlob = computeRelationGlob)
    end

    function StumpFeatModalDatasetWithMemoization{T, WorldType}(
        fmd                 :: FeatModalDataset{T, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T,WorldType<:AbstractWorld}
        
        featsnaggrs = Tuple{<:ModalFeature,<:Aggregator}[]
        grouped_featsnaggrs = AbstractVector{Tuple{<:Integer,<:Aggregator}}[]

        i_featsnaggr = 1
        for (feat,aggrsnops) in zip(fmd.features,fmd.grouped_featsaggrsnops)
            aggrs = []
            for aggr in keys(aggrsnops)
                push!(featsnaggrs, (feat,aggr))
                push!(aggrs, (i_featsnaggr,aggr))
                i_featsnaggr += 1
            end
            push!(grouped_featsnaggrs, aggrs)
        end

        # Compute modal dataset propositions and 1-modal decisions
        fmd_m, fmd_g = computeModalDatasetStumpSupport(fmd, grouped_featsnaggrs, computeRelationGlob = computeRelationGlob, simply_init_modal = true);

        StumpFeatModalDatasetWithMemoization{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
    end

    function StumpFeatModalDatasetWithMemoization(
        X                   :: InterpretedModalDataset{T, N, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T, N, WorldType<:AbstractWorld}
        StumpFeatModalDatasetWithMemoization{T, WorldType}(X, computeRelationGlob = computeRelationGlob)
    end

    function StumpFeatModalDatasetWithMemoization{T, WorldType}(
        X                   :: InterpretedModalDataset{T, N, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T, N, WorldType<:AbstractWorld}

        # Compute modal dataset propositions
        fmd = FeatModalDataset(X);

        StumpFeatModalDatasetWithMemoization{T, WorldType}(fmd, computeRelationGlob = computeRelationGlob)

        # TODO bring back ModalDatasetStumpSupport computation from X. 

        # fmd_m, fmd_g = computeModalDatasetStumpSupport(X, relations, fmd.grouped_featsaggrsnops??, fmd, features, computeRelationGlob = computeRelationGlob);

        # new{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
    end
end

StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType} = Union{StumpFeatModalDataset{T,WorldType},StumpFeatModalDatasetWithMemoization{T,WorldType}}


featsnaggrs(X::StumpFeatModalDatasetWithOrWithoutMemoization)         = X.featsnaggrs
grouped_featsnaggrs(X::StumpFeatModalDatasetWithOrWithoutMemoization) = X.grouped_featsnaggrs
relations(X::StumpFeatModalDatasetWithOrWithoutMemoization)           = relations(X.fmd)
init_world_sets_fun(X::StumpFeatModalDatasetWithOrWithoutMemoization,  args...) = init_world_sets_fun(X.fmd, args...)
accessibles_fun(X::StumpFeatModalDatasetWithOrWithoutMemoization,     args...) = accessibles_fun(X.fmd, args...)
all_worlds_fun(X::StumpFeatModalDatasetWithOrWithoutMemoization,  args...) = all_worlds_fun(X.fmd, args...)
accessibles_aggr_fun(X::StumpFeatModalDatasetWithOrWithoutMemoization, args...) = accessibles_aggr_fun(X.fmd, args...)
features(X::StumpFeatModalDatasetWithOrWithoutMemoization)           = features(X.fmd)
grouped_featsaggrsnops(X::StumpFeatModalDatasetWithOrWithoutMemoization)  = grouped_featsaggrsnops(X.fmd)


Base.size(X::StumpFeatModalDatasetWithOrWithoutMemoization)             where {T,N} =  (size(X.fmd), size(X.fmd_m), (isnothing(X.fmd_g) ? nothing : size(X.fmd_g)))
n_samples(X::StumpFeatModalDatasetWithOrWithoutMemoization{T, WorldType}) where {T, WorldType}   = n_samples(X.fmd)::Int64
n_features(X::StumpFeatModalDatasetWithOrWithoutMemoization{T, WorldType}) where {T, WorldType}  = n_features(X.fmd)
n_relations(X::StumpFeatModalDatasetWithOrWithoutMemoization{T, WorldType}) where {T, WorldType} = n_relations(X.fmd)
# getindex(X::StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType}, args...) where {T,WorldType} = getindex(X.fmd, args...)
world_type(X::StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType}) where {T,WorldType<:AbstractWorld} = WorldType

slice_dataset(X::StumpFeatModalDatasetWithOrWithoutMemoization, inds::AbstractVector{<:Integer}, args...; kwargs...) =
    typeof(X)(
        slice_dataset(X.fmd, inds, args...; kwargs...),
        slice_dataset(X.fmd_m, inds, args...; kwargs...),
        (isnothing(X.fmd_g) ? nothing : slice_dataset(X.fmd_g, inds, args...; kwargs...)),
        X.featsnaggrs,
        X.grouped_featsnaggrs)

display_structure(X::StumpFeatModalDataset; indent_str = "") = begin
    out = "$(typeof(X))\t$((Base.summarysize(X.fmd) + Base.summarysize(X.fmd_m) + Base.summarysize(X.fmd_g)) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ fmd\t$(Base.summarysize(X.fmd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t(shape $(Base.size(X.fmd)))\n"
    out *= indent_str * "├ fmd_m\t$(Base.summarysize(X.fmd_m) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t(shape $(Base.size(X.fmd_m)))\n"
    out *= indent_str * "└ fmd_g\t$(Base.summarysize(X.fmd_g) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t"
    if !isnothing(X.fmd_g)
        out *= "\t(shape $(Base.size(X.fmd_g)))"
    else
        out *= "\t−"
    end
    out
end

find_feature_id(X::StumpFeatModalDatasetWithOrWithoutMemoization, feature::ModalFeature) =
    findall(x->x==feature, features(X))[1]
find_relation_id(X::StumpFeatModalDatasetWithOrWithoutMemoization, relation::AbstractRelation) =
    findall(x->x==relation, relations(X))[1]
find_featsnaggr_id(X::StumpFeatModalDatasetWithOrWithoutMemoization, feature::ModalFeature, aggregator::Aggregator) =
    findall(x->x==(feature, aggregator), featsnaggrs(X))[1]

get_gamma(
        X::StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType},
        i_instance::Integer,
        w::WorldType,
        feature::ModalFeature) where {WorldType<:AbstractWorld, T} = get_gamma(X.fmd, i_instance, w, feature)

get_global_gamma(
        X::StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType},
        i_instance::Integer,
        feature::ModalFeature,
        test_operator::TestOperatorFun) where {WorldType<:AbstractWorld, T} = begin
            # @assert !isnothing(X.fmd_g) "Error. StumpFeatModalDatasetWithOrWithoutMemoization must be built with computeRelationGlob = true for it to be ready to test global decisions."
            i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
            X.fmd_g[i_instance, i_featsnaggr]
end

get_modal_gamma(
        X::StumpFeatModalDataset{T,WorldType},
        i_instance::Integer,
        w::WorldType,
        relation::AbstractRelation,
        feature::ModalFeature,
        test_operator::TestOperatorFun) where {WorldType<:AbstractWorld, T} = begin
            i_relation = find_relation_id(X, relation)
            i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
            X.fmd_m[i_instance, w, i_featsnaggr, i_relation]
end

test_decision(
        X::StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType},
        i_instance::Integer,
        w::WorldType,
        decision::Decision) where {T, WorldType<:AbstractWorld} = begin
    if is_propositional_decision(decision)
        test_decision(X, i_instance, w, decision.feature, decision.test_operator, decision.threshold)
    else
        gamma = begin
            if decision.relation isa ModalLogic._RelationGlob
                get_global_gamma(X, i_instance, decision.feature, decision.test_operator)
            else
                get_modal_gamma(X, i_instance, w, decision.relation, decision.feature, decision.test_operator)
            end
        end
        evaluate_thresh_decision(decision.test_operator, gamma, decision.threshold)
    end
end

Base.@propagate_inbounds @resumable function generate_propositional_feasible_decisions(
        X::StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType},
        args...
        ) where {T, WorldType<:AbstractWorld}
        for decision in generate_propositional_feasible_decisions(X.fmd, args...)
            @yield decision
        end
end

Base.@propagate_inbounds @resumable function generate_global_feasible_decisions(
        X::StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:AbstractWorld}
    relation = RelationGlob
    n_instances = length(instances_inds)
    
    @assert !isnothing(X.fmd_g) "Error. StumpFeatModalDatasetWithOrWithoutMemoization must be built with computeRelationGlob = true for it to be ready to generate global decisions."

    # For each feature
    @inbounds for i_feature in features_inds
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
        #   thresholds can in fact be directly given by slicing fmd_g and permuting the two dimensions
        # aggregators_ids = fst.(aggregators_with_ids)
        # thresholds = transpose(X.fmd_g[instances_inds, aggregators_ids])

        # Initialize thresholds with the bottoms
        thresholds = Array{T,2}(undef, length(aggregators_with_ids), n_instances)
        for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)
            thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
        end
        
        # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
        for (instance_id,i_instance) in enumerate(instances_inds)
            @logmsg DTDetail " Instance $(instance_id)/$(n_instances)"
            for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
                gamma = X.fmd_g[i_instance, i_featsnaggr]
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
        X::StumpFeatModalDataset{T,WorldType},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        modal_relations_inds::AbstractVector{<:Integer},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:AbstractWorld}
    n_instances = length(instances_inds)

    # For each relational operator
    @inbounds for i_relation in modal_relations_inds
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
            thresholds = Array{T,2}(undef, length(aggregators_with_ids), n_instances)
            for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)
                thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
            end

            # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
                for (i_instance,instance_id) in enumerate(instances_inds)
                @logmsg DTDetail " Instance $(i_instance)/$(n_instances)"
                worlds = Sf[i_instance] # TODO could also use accessibles_aggr_funs here?

                # TODO also try this instead (TODO fix first)
                # values = [X.fmd_m[instance_id, w, i_feature] for w in worlds]
                # thresholds[:,i_instance] = map((_,aggr)->aggr(values), aggregators_with_ids)
                    
                for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
                    for w in worlds
                        gamma = X.fmd_m[instance_id, w, i_featsnaggr, i_relation]
                        thresholds[i_aggr,i_instance] = ModalLogic.aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,i_instance])
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


display_structure(X::StumpFeatModalDatasetWithMemoization; indent_str = "") = begin
    out = "$(typeof(X))\t$((Base.summarysize(X.fmd) + Base.summarysize(X.fmd_m) + Base.summarysize(X.fmd_g)) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ fmd\t$(Base.summarysize(X.fmd) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
        out *= "\t(shape $(Base.size(X.fmd.fwd)))\n"
        # out *= "\t(shape $(Base.size(X.fmd.fwd)), $(n_nothing) nothings, $((1-(n_nothing    / size(X.fmd)))*100)%)\n"
    # TODO n_nothing_m = count_nothings(X.fmd_m)
    n_nothing_m = count(isnothing, X.fmd_m.d)
    out *= indent_str * "├ fmd_m\t$(Base.summarysize(X.fmd_m) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
        out *= "\t(shape $(Base.size(X.fmd_m)), $(n_nothing_m) nothings)\n" # , $((1-(n_nothing_m  / n_v(X.fmd_m)))*100)%)\n"
    out *= indent_str * "└ fmd_g\t$(Base.summarysize(X.fmd_g) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t"
    if !isnothing(X.fmd_g)
        # n_nothing_g = count(isnothing, X.fmd_g)
        # out *= "\t(shape $(Base.size(X.fmd_g)), $(n_nothing_g) nothings, $((1-(n_nothing_g  / size(X.fmd_g)))*100)%)"
        out *= "\t(shape $(Base.size(X.fmd_g)))"
    else
        out *= "\t−"
    end
    out
end

# get_global_gamma(
#       X::StumpFeatModalDatasetWithMemoization{T,WorldType},
#       i_instance::Integer,
#       feature::ModalFeature,
#       test_operator::TestOperatorFun) where {WorldType<:AbstractWorld, T} = begin
#   @assert !isnothing(X.fmd_g) "Error. StumpFeatModalDatasetWithMemoization must be built with computeRelationGlob = true for it to be ready to test global decisions."
#   i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
#   # if !isnothing(X.fmd_g[i_instance, i_featsnaggr])
#   X.fmd_g[i_instance, i_featsnaggr]
#   # else
#   #   i_feature = find_feature_id(X, feature)
#   #   aggregator = existential_aggregator(test_operator)
#   #   cur_fwd_slice = fwd_get_channel(X.fmd.fwd, i_instance, i_feature)
#   #   accessible_worlds = all_worlds_aggr(WorldType, accessibles_aggr_fun(X.fmd, i_instance), feature, aggregator)
#   #   gamma = compute_modal_gamma(cur_fwd_slice, accessible_worlds, aggregator)
#   #   FMDStumpGlobalSupportSet(X.fmd_g, i_instance, i_featsnaggr, gamma)
#   # end
# end

# TODO scan this value for an example problem and different number of threads

using Random
coin_flip_memoiz_rng = Random.default_rng()

cfnls_max = 0.8
# cfnls_k = 5.9
cfnls_k = 30
coin_flip_no_look_StumpFeatModalDatasetWithMemoization_value = cfnls_max*cfnls_k/((Threads.nthreads())-1+cfnls_k)
coin_flip_no_look_StumpFeatModalDatasetWithMemoization() = (rand(coin_flip_memoiz_rng) >= coin_flip_no_look_StumpFeatModalDatasetWithMemoization_value)
# coin_flip_no_look_StumpFeatModalDatasetWithMemoization() = false

get_modal_gamma(
        X::StumpFeatModalDatasetWithMemoization{T,WorldType},
        i_instance::Integer,
        w::WorldType,
        relation::AbstractRelation,
        feature::ModalFeature,
        test_operator::TestOperatorFun) where {WorldType<:AbstractWorld, T} = begin
    i_relation = find_relation_id(X, relation)
    aggregator = existential_aggregator(test_operator)
    i_featsnaggr = find_featsnaggr_id(X, feature, aggregator)
    # if coin_flip_no_look_StumpFeatModalDatasetWithMemoization() || 
    if false || 
            isnothing(X.fmd_m[i_instance, w, i_featsnaggr, i_relation])
        i_feature = find_feature_id(X, feature)
        cur_fwd_slice = fwd_get_channel(X.fmd.fwd, i_instance, i_feature)
        accessible_worlds = accessibles_aggr_fun(X.fmd, i_instance)(feature, aggregator, w, relation)
        gamma = compute_modal_gamma(cur_fwd_slice, accessible_worlds, aggregator)
        FMDStumpSupportSet(X.fmd_m, w, i_instance, i_featsnaggr, i_relation, gamma)
    else
        X.fmd_m[i_instance, w, i_featsnaggr, i_relation]
    end
end


Base.@propagate_inbounds @resumable function generate_modal_feasible_decisions(
        X::StumpFeatModalDatasetWithMemoization{T,WorldType},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        modal_relations_inds::AbstractVector{<:Integer},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:AbstractWorld}
    n_instances = length(instances_inds)

    # For each relational operator
    @inbounds for i_relation in modal_relations_inds
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
            thresholds = Array{T,2}(undef, length(aggregators_with_ids), n_instances)
            for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)
                thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
            end

            # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
            for (instance_id,i_instance) in enumerate(instances_inds)
                @logmsg DTDetail " Instance $(instance_id)/$(n_instances)"
                worlds = Sf[instance_id] # TODO could also use accessibles_aggr_funs here?

                # TODO also try this instead (TODO fix first)
                # values = [X.fmd_m[i_instance, w, i_feature] for w in worlds]
                # thresholds[:,instance_id] = map((_,aggr)->aggr(values), aggregators_with_ids)
                    
                for (i_aggr,(i_featsnaggr,aggregator)) in enumerate(aggregators_with_ids)
                    for w in worlds
                        gamma = 
                            # if coin_flip_no_look_StumpFeatModalDatasetWithMemoization() || 
                            if false || 
                                isnothing(X.fmd_m[i_instance, w, i_featsnaggr, i_relation])
                                cur_fwd_slice = fwd_get_channel(X.fmd.fwd, i_instance, i_feature)
                                accessible_worlds = accessibles_aggr_fun(X.fmd, i_instance)(feature, aggregator, w, relation)
                                gamma = compute_modal_gamma(cur_fwd_slice, accessible_worlds, aggregator)
                                FMDStumpSupportSet(X.fmd_m, w, i_instance, i_featsnaggr, i_relation, gamma)
                            else
                                X.fmd_m[i_instance, w, i_featsnaggr, i_relation]
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


const SingleFrameGenericDataset{T} = Union{DimensionalDataset{T},InterpretedModalDataset{T},ModalDataset{T}}

struct MultiFrameModalDataset
    frames  :: AbstractVector{<:SingleFrameGenericDataset}
    # function MultiFrameModalDataset(Xs::AbstractVector{<:SingleFrameGenericDataset{<:T, <:AbstractWorld}}) where {T}
    # function MultiFrameModalDataset(Xs::AbstractVector{<:SingleFrameGenericDataset{T, <:AbstractWorld}}) where {T}
    function MultiFrameModalDataset(Xs::AbstractVector{<:SingleFrameGenericDataset})
        @assert length(Xs) > 0 && length(unique(n_samples.(Xs))) == 1 "Can't create an empty MultiFrameModalDataset or with mismatching number of samples (n_frames: $(length(Xs)), frame_sizes: $(n_samples.(Xs)))."
        new(Xs)
    end
    function MultiFrameModalDataset(X::SingleFrameGenericDataset)
        new([X])
    end
    function MultiFrameModalDataset()
        new(SingleFrameGenericDataset[])
    end
end

# TODO: test all these methods
Base.size(X::MultiFrameModalDataset) = map(size, X.frames)
get_frame(X::MultiFrameModalDataset, i) = X.frames[i]
push_frame!(X::MultiFrameModalDataset, f::ModalDataset) = push!(X.frames, f)
n_frames(X::MultiFrameModalDataset)             = length(X.frames)
n_samples(X::MultiFrameModalDataset)            = n_samples(X.frames[1])::Int64 # n_frames(X) > 0 ? n_samples(X.frames[1]) : 0
Base.length(X::MultiFrameModalDataset)               = n_samples(X)
frames(X::MultiFrameModalDataset) = X.frames
Base.iterate(X::MultiFrameModalDataset, state=1) = state > length(X) ? nothing : (get_instance(X, state), state+1)

max_channel_size(X::MultiFrameModalDataset) = map(max_channel_size, X.frames)
# get total number of features (TODO: figure if this is useless or not)
n_features(X::MultiFrameModalDataset) = map(n_features, X.frames)
# get number of features in a single frame
n_features(X::MultiFrameModalDataset, i_frame::Integer) = n_features(X.frames[i_frame])
# TODO: Note: channel_size doesn't make sense at this point. Only the accessibles_funs[i] functions.
n_relations(X::MultiFrameModalDataset) = map(n_relations, X.frames)
n_relations(X::MultiFrameModalDataset, i_frame::Integer) = n_relations(X.frames[i_frame])
world_type(X::MultiFrameModalDataset, i_frame::Integer) = world_type(X.frames[i_frame])
world_types(X::MultiFrameModalDataset) = Vector{Type{<:AbstractWorld}}(world_type.(X.frames))

get_instance(X::MultiFrameModalDataset,  i_frame::Integer, idx_i::Integer, args...)  = get_instance(X.frames[i_frame], idx_i, args...)
# slice_dataset(X::MultiFrameModalDataset, i_frame::Integer, inds::AbstractVector{<:Integer}, args...)  = slice_dataset(X.frames[i_frame], inds, args...; kwargs...)

# get_instance(X::MultiFrameModalDataset, idx_i::Integer, args...)  = get_instance(X.frames[i], idx_i, args...) # TODO should slice across the frames!
slice_dataset(X::MultiFrameModalDataset, inds::AbstractVector{<:Integer}, args...; kwargs...) =
    MultiFrameModalDataset(Vector{ModalDataset}(map(frame->slice_dataset(frame, inds, args...; kwargs...), X.frames)))


const GenericDataset = Union{SingleFrameGenericDataset,MultiFrameModalDataset}

display_structure(Xs::MultiFrameModalDataset; indent_str = "") = begin
    out = "$(typeof(Xs))" # * "\t\t\t$(Base.summarysize(Xs) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
    for (i_frame, X) in enumerate(frames(Xs))
        if i_frame == n_frames(Xs)
            out *= "\n$(indent_str)└ "
        else
            out *= "\n$(indent_str)├ "
        end
        out *= "[$(i_frame)] "
        # \t\t\t$(Base.summarysize(X) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t(world_type: $(world_type(X)))"
        out *= display_structure(X; indent_str = indent_str * (i_frame == n_frames(Xs) ? "   " : "│  ")) * "\n"
    end
    out
end


################################################################################
################################################################################

# Perform the modal step, that is, evaluate a modal formula
#  on a domain, and eventually compute the new world set.
function modal_step(
        X::Union{ModalDataset{T,WorldType},InterpretedModalDataset{T,N,WorldType}},
        i_instance::Integer,
        worlds::WorldSetType,
        decision::Decision{T},
        returns_survivors::Union{Val{true},Val{false}} = Val(false)
    ) where {T, N, WorldType<:AbstractWorld, WorldSetType<:AbstractWorldSet{WorldType}}
    @logmsg DTDetail "modal_step" worlds display_decision(decision)

    satisfied = false
    
    # TODO space for optimization here: with some relations (e.g. IA_A, IA_L) can be made smaller

    if returns_survivors isa Val{true}
        worlds_map = Dict{AbstractWorld,AbstractWorldSet{WorldType}}()
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
                Threads.@threads for curr_w in worlds
                    worlds_map[curr_w] = accessibles_fun(X, i_instance)(curr_w, decision.relation)
                end
                unique(cat([ worlds_map[k] for k in keys(worlds_map) ]...; dims = 1))
            else
                accessibles_fun(X, i_instance)(worlds, decision.relation)
            end

        for w in acc_worlds
            if test_decision(X, i_instance, w, decision.feature, decision.test_operator, decision.threshold)
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
        X::SingleFrameGenericDataset{T},
        i_instance::Integer,
        w::AbstractWorld,
        feature::ModalFeature,
        test_operator::TestOperatorFun,
        threshold::T) where {T} = begin
    gamma = get_gamma(X, i_instance, w, feature)
    evaluate_thresh_decision(test_operator, gamma, threshold)
end

test_decision(
        X::SingleFrameGenericDataset{T},
        i_instance::Integer,
        w::AbstractWorld,
        decision::Decision{T}) where {T} = begin
    instance = get_instance(X, i_instance)

    aggregator = existential_aggregator(decision.test_operator)
    
    worlds = accessibles_aggr(decision.feature, aggregator, w, decision.relation, instance_channel_size(instance)...)
    gamma = if length(worlds |> collect) == 0
        ModalLogic.aggregator_bottom(aggregator, T)
    else
        aggregator((w)->get_gamma(X, i_instance, w, decision.feature), worlds)
    end

    evaluate_thresh_decision(decision.test_operator, gamma, decision.threshold)
end


export generate_feasible_decisions
                # ,
                # generate_propositional_feasible_decisions,
                # generate_global_feasible_decisions,
                # generate_modal_feasible_decisions

Base.@propagate_inbounds @resumable function generate_feasible_decisions(
        X::ModalDataset{T,WorldType},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        allow_propositional_decisions::Bool,
        allow_modal_decisions::Bool,
        allow_global_decisions::Bool,
        modal_relations_inds::AbstractVector{<:Integer},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:AbstractWorld}
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
