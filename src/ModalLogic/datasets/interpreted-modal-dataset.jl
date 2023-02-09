
############################################################################################
# Interpreted modal dataset
############################################################################################
# 
# A modal dataset can be instantiated in *implicit* form, from a dimensional domain, and a few
#  objects inducing an interpretation on the domain; mainly, an ontology (determining worlds and
#  relations), and structures for interpreting features onto the domain.
# 
############################################################################################

struct InterpretedModalDataset{
    T<:Number,
    N,
    W<:AbstractWorld,
    D<:PassiveDimensionalDataset{T,N,W},
    U,
    FT<:AbstractFeature{U},
    G1<:AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}},
    G2<:AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
} <: ActiveModalDataset{T,W,FullDimensionalFrame{N,W,Bool},U,FT}

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
    
    function InterpretedModalDataset{T,N,W}(
        domain::PassiveDimensionalDataset{T,N},
        ontology::Ontology{W},
        features::AbstractVector{<:AbstractFeature},
        grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}};
        allow_no_instances = false,
    ) where {T,N,W<:AbstractWorld}
        ty = "InterpretedModalDataset{$(T),$(N),$(W)}"
        features = collect(features)
        U = Union{featvaltype.(features)...}
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
        new{T,N,W,typeof(domain),U,FT,typeof(grouped_featsaggrsnops),typeof(grouped_featsnaggrs)}(domain, ontology, features, grouped_featsaggrsnops, grouped_featsnaggrs)
    end

    ########################################################################################
    
    function InterpretedModalDataset{T,N,W}(
        domain           :: Union{PassiveDimensionalDataset{T,N,W},AbstractDimensionalDataset{T}},
        ontology         :: Ontology{W},
        mixed_features   :: AbstractVector,
    ) where {T,N,W<:AbstractWorld}
        domain = PassiveDimensionalDataset{T,N,W}(domain)
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
        InterpretedModalDataset{T,N,worldtype(ontology)}(domain, ontology, _features, featsnops)
    end

    ########################################################################################
    
    function InterpretedModalDataset{T,N,W}(
        domain             :: PassiveDimensionalDataset{T,N,W},
        ontology           :: Ontology{W},
        features           :: AbstractVector{<:AbstractFeature},
        grouped_featsnops  :: AbstractVector;
        kwargs...,
    ) where {T,N,W<:AbstractWorld}
        grouped_featsaggrsnops = grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops)
        InterpretedModalDataset{T,N,W}(domain, ontology, features, grouped_featsaggrsnops; kwargs...)
    end

    function InterpretedModalDataset{T,N,W}(
        domain             :: AbstractDimensionalDataset{T},
        ontology           :: Ontology{W},
        features           :: AbstractVector{<:AbstractFeature},
        grouped_featsnops  :: AbstractVector;
        kwargs...,
    ) where {T,N,W<:AbstractWorld}
        domain = PassiveDimensionalDataset{T,N,W}(domain)
        InterpretedModalDataset{T,N,W}(domain, ontology, features, grouped_featsnops; kwargs...)
    end

    ########################################################################################

    function InterpretedModalDataset(
        domain::PassiveDimensionalDataset{T,N,W},
        args...;
        kwargs...,
    ) where {T,N,W<:AbstractWorld}
        InterpretedModalDataset{T,N,W}(domain, args...; kwargs...)
    end

    function InterpretedModalDataset(
        domain::AbstractDimensionalDataset{T,D},
        ontology::Ontology{W},
        args...;
        kwargs...,
    ) where {T,D,W<:AbstractWorld}
        InterpretedModalDataset{T,D-1-1,W}(domain, ontology, args...; kwargs...)
    end

    function InterpretedModalDataset(
        domain::AbstractDimensionalDataset{T,D},
        worldtype::Type{W},
        args...;
        kwargs...,
    ) where {T,D,W<:AbstractWorld}
        InterpretedModalDataset{T,D-1-1,W}(domain, args...; kwargs...)
    end

end

domain(imd::InterpretedModalDataset)                 = imd.domain
ontology(imd::InterpretedModalDataset)               = imd.ontology
features(imd::InterpretedModalDataset)               = imd.features
grouped_featsaggrsnops(imd::InterpretedModalDataset) = imd.grouped_featsaggrsnops
grouped_featsnaggrs(imd::InterpretedModalDataset)    = imd.grouped_featsnaggrs

function Base.getindex(X::InterpretedModalDataset, args...)
    domain(X)[args...]::featvaltype(X)
end

Base.size(imd::InterpretedModalDataset)              = Base.size(domain(imd))

# find_feature_id(X::InterpretedModalDataset{T,W}, feature::AbstractFeature) where {T,W} =
#     findall(x->x==feature, features(X))[1]

dimensionality(imd::InterpretedModalDataset{T,N,W}) where {T,N,W} = N
worldtype(imd::InterpretedModalDataset{T,N,W}) where {T,N,W} = W

nsamples(imd::InterpretedModalDataset)               = nsamples(domain(imd))
nattributes(imd::InterpretedModalDataset)            = nattributes(domain(imd))

relations(imd::InterpretedModalDataset)              = relations(ontology(imd))
nrelations(imd::InterpretedModalDataset)             = length(relations(imd))
nfeatures(imd::InterpretedModalDataset)              = length(features(imd))

channel_size(imd::InterpretedModalDataset, args...)     = channel_size(domain(imd), args...)
max_channel_size(imd::InterpretedModalDataset)          = max_channel_size(domain(imd))

get_instance(imd::InterpretedModalDataset, args...)     = get_instance(domain(imd), args...)

_slice_dataset(imd::InterpretedModalDataset, inds::AbstractVector{<:Integer}, args...; kwargs...)    =
    InterpretedModalDataset(_slice_dataset(domain(imd), inds, args...; kwargs...), ontology(imd), features(imd), imd.grouped_featsaggrsnops)

initialworldset(imd::InterpretedModalDataset, args...) = initialworldset(domain(imd), args...)
accessibles(imd::InterpretedModalDataset, args...) = accessibles(domain(imd), args...)
representatives(imd::InterpretedModalDataset, args...) = representatives(domain(imd), args...)
allworlds(imd::InterpretedModalDataset, args...) = allworlds(domain(imd), args...)

function display_structure(imd::InterpretedModalDataset; indent_str = "")
    out = "$(typeof(imd))\t$(Base.summarysize(imd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(imd))))\t$(relations(imd))\n"
    out *= indent_str * "├ domain shape\t$(Base.size(domain(imd)))\n"
    out *= indent_str * "└ max_channel_size\t$(max_channel_size(imd))"
    out
end

hasnans(imd::InterpretedModalDataset) = hasnans(domain(imd))
