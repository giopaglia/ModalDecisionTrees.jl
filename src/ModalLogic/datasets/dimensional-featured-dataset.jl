
############################################################################################
# Interpreted modal dataset
############################################################################################
# 
# A modal dataset can be instantiated in *implicit* form, from a dimensional domain, and a few
#  objects inducing an interpretation on the domain; mainly, an ontology (determining worlds and
#  relations), and structures for interpreting features onto the domain.
# 
############################################################################################

struct DimensionalFeaturedDataset{
    V<:Number,
    N,
    W<:AbstractWorld,
    D<:PassiveDimensionalDataset{N,W},
    FT<:AbstractFeature{V},
    G1<:AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}},
    G2<:AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
} <: ActiveFeaturedDataset{V,W,FullDimensionalFrame{N,W,Bool},FT}

    # Core data (a dimensional domain)
    domain                  :: D
    
    # Worlds & Relations
    ontology                :: Ontology{W} # Union{Nothing,}
    
    # Features
    features                :: Vector{FT}

    # Test operators associated with each feature, grouped by their respective aggregator
    # Note: currently, cannot specify the full type (probably due to @computed)
    grouped_featsaggrsnops  :: G1

    # Features and Aggregators
    grouped_featsnaggrs     :: G2

    ########################################################################################
    
    function DimensionalFeaturedDataset{V,N,W}(
        domain::PassiveDimensionalDataset{N},
        ontology::Ontology{W},
        features::AbstractVector{<:AbstractFeature},
        grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}};
        allow_no_instances = false,
    ) where {V,N,W<:AbstractWorld}
        ty = "DimensionalFeaturedDataset{$(V),$(N),$(W)}"
        features = collect(features)
        FT = Union{typeof.(features)...}
        features = Vector{FT}(features)
        @assert allow_no_instances || nsamples(domain) > 0 "" *
            "Can't instantiate $(ty) with no instance. (domain's type $(typeof(domain)))"
        @assert length(features) == length(grouped_featsaggrsnops) "" *
            "Can't instantiate $(ty) with mismatching length(features) and" *
            " length(grouped_featsaggrsnops):" *
            " $(length(features)) != $(length(grouped_featsaggrsnops))"
        @assert length(grouped_featsaggrsnops) > 0 &&
            sum(length.(grouped_featsaggrsnops)) > 0 &&
            sum(vcat([[length(test_ops) for test_ops in aggrs] for aggrs in grouped_featsaggrsnops]...)) > 0 "" *
            "Can't instantiate $(ty) with no test operator: $(grouped_featsaggrsnops)"
        grouped_featsnaggrs = features_grouped_featsaggrsnops2grouped_featsnaggrs(features, grouped_featsaggrsnops)
        new{V,N,W,typeof(domain),FT,typeof(grouped_featsaggrsnops),typeof(grouped_featsnaggrs)}(domain, ontology, features, grouped_featsaggrsnops, grouped_featsnaggrs)
    end

    ########################################################################################

    function DimensionalFeaturedDataset{V,N,W}(
        domain             :: Union{PassiveDimensionalDataset{N,W},AbstractDimensionalDataset},
        ontology           :: Ontology{W},
        features           :: AbstractVector{<:AbstractFeature},
        grouped_featsnops  :: AbstractVector;
        kwargs...,
    ) where {V,N,W<:AbstractWorld}
        domain = (domain isa AbstractDimensionalDataset ? PassiveDimensionalDataset{N,W}(domain) : domain)
        grouped_featsaggrsnops = grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops)
        DimensionalFeaturedDataset{V,N,W}(domain, ontology, features, grouped_featsaggrsnops; kwargs...)
    end

    function DimensionalFeaturedDataset{V,N,W}(
        domain           :: Union{PassiveDimensionalDataset{N,W},AbstractDimensionalDataset},
        ontology         :: Ontology{W},
        mixed_features   :: AbstractVector;
        kwargs...,
    ) where {V,N,W<:AbstractWorld}
        domain = (domain isa AbstractDimensionalDataset ? PassiveDimensionalDataset{N,W}(domain) : domain)
        mixed_features = Vector{MixedFeature}(mixed_features)
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

            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic.CanonicalFeatureGeq) = ([≥],ModalDecisionTrees.SingleAttributeMin{V}(i_attr))
            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic.CanonicalFeatureLeq) = ([≤],ModalDecisionTrees.SingleAttributeMax{V}(i_attr))
            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic.CanonicalFeatureGeqSoft) = ([≥],ModalDecisionTrees.SingleAttributeSoftMin{V}(i_attr, cf.alpha))
            single_attr_feats_n_featsnops(i_attr,cf::ModalLogic.CanonicalFeatureLeqSoft) = ([≤],ModalDecisionTrees.SingleAttributeSoftMax{V}(i_attr, cf.alpha))
            single_attr_feats_n_featsnops(i_attr,(test_ops,cf)::Tuple{<:AbstractVector{<:TestOperatorFun},typeof(minimum)}) = (test_ops,SingleAttributeMin{V}(i_attr))
            single_attr_feats_n_featsnops(i_attr,(test_ops,cf)::Tuple{<:AbstractVector{<:TestOperatorFun},typeof(maximum)}) = (test_ops,SingleAttributeMax{V}(i_attr))
            single_attr_feats_n_featsnops(i_attr,(test_ops,cf)::Tuple{<:AbstractVector{<:TestOperatorFun},Function})        = (test_ops,SingleAttributeGenericFeature{V}(i_attr, (x)->(V(cf(x)))))
            single_attr_feats_n_featsnops(i_attr,::Any) = throw_n_log("Unknown mixed_feature type: $(cf), $(typeof(cf))")

            for i_attr in 1:nattributes(domain)
                for (test_ops,cf) in map((cf)->single_attr_feats_n_featsnops(i_attr,cf),attribute_specific_cfs)
                    push!(featsnops, test_ops)
                    push!(_features, cf)
                end
            end
            _features, featsnops
        end
        DimensionalFeaturedDataset{V,N,worldtype(ontology)}(domain, ontology, _features, featsnops; kwargs...)
    end

    ########################################################################################

    function DimensionalFeaturedDataset{V,N}(
        domain             :: Union{PassiveDimensionalDataset{N,W},AbstractDimensionalDataset},
        ontology           :: Ontology{W},
        args...;
        kwargs...,
    ) where {V,N,W<:AbstractWorld}
        domain = (domain isa AbstractDimensionalDataset ? PassiveDimensionalDataset{N,W}(domain) : domain)
        DimensionalFeaturedDataset{V,N,W}(domain, ontology, args...; kwargs...)
    end

    ########################################################################################

    function DimensionalFeaturedDataset{V}(
        domain             :: Union{PassiveDimensionalDataset,AbstractDimensionalDataset},
        args...;
        kwargs...,
    ) where {V}
        DimensionalFeaturedDataset{V,dimensionality(domain)}(domain, args...; kwargs...)
    end

    ########################################################################################

    function DimensionalFeaturedDataset(
        domain             :: Union{PassiveDimensionalDataset{N,W},AbstractDimensionalDataset},
        ontology           :: Ontology{W},
        features           :: AbstractVector{<:AbstractFeature},
        args...;
        kwargs...,
    ) where {N,W<:AbstractWorld}
        V = Union{featvaltype.(features)...}
        DimensionalFeaturedDataset{V}(domain, ontology, features, args...; kwargs...)
    end

    function DimensionalFeaturedDataset(
        domain           :: Union{PassiveDimensionalDataset{N,W},AbstractDimensionalDataset},
        ontology         :: Ontology{W},
        mixed_features   :: AbstractVector;
        kwargs...,
    ) where {N,W<:AbstractWorld}
        domain = (domain isa AbstractDimensionalDataset ? PassiveDimensionalDataset{N,W}(domain) : domain)
        @assert all((f)->(f isa CanonicalFeature && SoleModels.preserves_type(f)), mixed_features) "Please, specify the feature output type V upon construction, as in: DimensionalFeaturedDataset{V}(...)." # TODO highlight and improve
        V = eltype(domain)
        DimensionalFeaturedDataset{V}(domain, ontology, mixed_features; kwargs...)
    end

end

domain(X::DimensionalFeaturedDataset)                 = X.domain
ontology(X::DimensionalFeaturedDataset)               = X.ontology
features(X::DimensionalFeaturedDataset)               = X.features
grouped_featsaggrsnops(X::DimensionalFeaturedDataset) = X.grouped_featsaggrsnops
grouped_featsnaggrs(X::DimensionalFeaturedDataset)    = X.grouped_featsnaggrs

function Base.getindex(X::DimensionalFeaturedDataset, args...)
    domain(X)[args...]::featvaltype(X)
end

Base.size(X::DimensionalFeaturedDataset)              = Base.size(domain(X))

dimensionality(X::DimensionalFeaturedDataset{V,N,W}) where {V,N,W} = N
worldtype(X::DimensionalFeaturedDataset{V,N,W}) where {V,N,W} = W

nsamples(X::DimensionalFeaturedDataset)               = nsamples(domain(X))
nattributes(X::DimensionalFeaturedDataset)            = nattributes(domain(X))

relations(X::DimensionalFeaturedDataset)              = relations(ontology(X))
nrelations(X::DimensionalFeaturedDataset)             = length(relations(X))
nfeatures(X::DimensionalFeaturedDataset)              = length(features(X))

channel_size(X::DimensionalFeaturedDataset, args...)     = channel_size(domain(X), args...)
max_channel_size(X::DimensionalFeaturedDataset)          = max_channel_size(domain(X))

get_instance(X::DimensionalFeaturedDataset, args...)     = get_instance(domain(X), args...)

_slice_dataset(X::DimensionalFeaturedDataset, inds::AbstractVector{<:Integer}, args...; kwargs...)    =
    DimensionalFeaturedDataset(_slice_dataset(domain(X), inds, args...; kwargs...), ontology(X), features(X), X.grouped_featsaggrsnops)

frame(X::DimensionalFeaturedDataset, i_sample) = frame(domain(X), i_sample)

function display_structure(X::DimensionalFeaturedDataset; indent_str = "")
    out = "$(typeof(X))\t$(Base.summarysize(X) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(X))))\t$(relations(X))\n"
    out *= indent_str * "├ domain shape\t$(Base.size(domain(X)))\n"
    out *= indent_str * "└ max_channel_size\t$(max_channel_size(X))"
    out
end

hasnans(X::DimensionalFeaturedDataset) = hasnans(domain(X))

