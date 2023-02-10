
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

end

domain(imd::DimensionalFeaturedDataset)                 = imd.domain
ontology(imd::DimensionalFeaturedDataset)               = imd.ontology
features(imd::DimensionalFeaturedDataset)               = imd.features
grouped_featsaggrsnops(imd::DimensionalFeaturedDataset) = imd.grouped_featsaggrsnops
grouped_featsnaggrs(imd::DimensionalFeaturedDataset)    = imd.grouped_featsnaggrs

function Base.getindex(X::DimensionalFeaturedDataset, args...)
    domain(X)[args...]::featvaltype(X)
end

Base.size(imd::DimensionalFeaturedDataset)              = Base.size(domain(imd))

# find_feature_id(X::DimensionalFeaturedDataset{V,W}, feature::AbstractFeature) where {V,W} =
#     findall(x->x==feature, features(X))[1]

dimensionality(imd::DimensionalFeaturedDataset{V,N,W}) where {V,N,W} = N
worldtype(imd::DimensionalFeaturedDataset{V,N,W}) where {V,N,W} = W

nsamples(imd::DimensionalFeaturedDataset)               = nsamples(domain(imd))
nattributes(imd::DimensionalFeaturedDataset)            = nattributes(domain(imd))

relations(imd::DimensionalFeaturedDataset)              = relations(ontology(imd))
nrelations(imd::DimensionalFeaturedDataset)             = length(relations(imd))
nfeatures(imd::DimensionalFeaturedDataset)              = length(features(imd))

channel_size(imd::DimensionalFeaturedDataset, args...)     = channel_size(domain(imd), args...)
max_channel_size(imd::DimensionalFeaturedDataset)          = max_channel_size(domain(imd))

get_instance(imd::DimensionalFeaturedDataset, args...)     = get_instance(domain(imd), args...)

_slice_dataset(imd::DimensionalFeaturedDataset, inds::AbstractVector{<:Integer}, args...; kwargs...)    =
    DimensionalFeaturedDataset(_slice_dataset(domain(imd), inds, args...; kwargs...), ontology(imd), features(imd), imd.grouped_featsaggrsnops)

initialworldset(imd::DimensionalFeaturedDataset, args...) = initialworldset(domain(imd), args...)
accessibles(imd::DimensionalFeaturedDataset, args...) = accessibles(domain(imd), args...)
representatives(imd::DimensionalFeaturedDataset, args...) = representatives(domain(imd), args...)
allworlds(imd::DimensionalFeaturedDataset, args...) = allworlds(domain(imd), args...)

function display_structure(imd::DimensionalFeaturedDataset; indent_str = "")
    out = "$(typeof(imd))\t$(Base.summarysize(imd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(imd))))\t$(relations(imd))\n"
    out *= indent_str * "├ domain shape\t$(Base.size(domain(imd)))\n"
    out *= indent_str * "└ max_channel_size\t$(max_channel_size(imd))"
    out
end

hasnans(imd::DimensionalFeaturedDataset) = hasnans(domain(imd))
