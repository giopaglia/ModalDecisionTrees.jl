
############################################################################################
# Interpreted modal dataset
############################################################################################
# 
# A modal dataset can be instantiated in *implicit* form, from a dimensional domain, and a few
#  objects inducing an interpretation on the domain; mainly, an ontology (determining worlds and
#  relations), and structures for interpreting features onto the domain.
# 
############################################################################################

@computed struct InterpretedModalDataset{T<:Number,N,W<:AbstractWorld} <: ActiveModalDataset{T,W,FullDimensionalFrame{N,W,Bool}}

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
