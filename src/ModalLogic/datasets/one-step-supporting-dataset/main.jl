
# Compute modal dataset propositions and 1-modal decisions
struct OneStepSupportingDataset{
    V<:Number,
    W<:AbstractWorld,
    FR<:AbstractFrame{W,Bool},
    VV<:Union{V,Nothing},
    FWDRS<:AbstractRelationalSupport{VV,W,FR},
    FWDGS<:Union{AbstractGlobalSupport{V},Nothing},
    G<:AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}},
} <: SupportingModalDataset{V,W,FR}

    # Relational support
    fwd_rs              :: FWDRS

    # Global support
    fwd_gs              :: FWDGS

    # Features and Aggregators
    featsnaggrs         :: G

    function OneStepSupportingDataset(
        fwd_rs::FWDRS,
        fwd_gs::FWDGS,
        featsnaggrs::G,
    ) where {
        V<:Number,
        W<:AbstractWorld,
        FR<:AbstractFrame{W,Bool},
        VV<:Union{V,Nothing},
        FWDRS<:AbstractRelationalSupport{VV,W,FR},
        FWDGS<:Union{AbstractGlobalSupport{V},Nothing},
        G<:AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}},
    }
        @assert nfeatsnaggrs(fwd_rs) == length(featsnaggrs)       "Can't instantiate $(ty) with unmatching nfeatsnaggrs for fwd_rs and provided featsnaggrs: $(nfeatsnaggrs(fwd_rs)) and $(length(featsnaggrs))"
        if fwd_gs != nothing
            @assert nfeatsnaggrs(fwd_gs) == length(featsnaggrs)   "Can't instantiate $(ty) with unmatching nfeatsnaggrs for fwd_gs and provided featsnaggrs: $(nfeatsnaggrs(fwd_gs)) and $(length(featsnaggrs))"
            @assert nsamples(fwd_gs) == nsamples(fwd_rs)          "Can't instantiate $(ty) with unmatching nsamples for fwd_gs and fwd_rs support: $(nsamples(fwd_gs)) and $(nsamples(fwd_rs))"
        end
        new{V,W,FR,VV,FWDRS,FWDGS,G}(fwd_rs, fwd_gs, featsnaggrs)
    end

    _default_rs_type(::Type{<:AbstractWorld}) = GenericRelationalSupport
    _default_rs_type(::Type{<:Union{OneWorld,Interval,Interval2D}}) = UniformFullDimensionalRelationalSupport

    # A function that computes the support from an explicit modal dataset
    Base.@propagate_inbounds function OneStepSupportingDataset(
        emd                 :: FeaturedDataset{V,W},
        relational_support_type :: Type{<:AbstractRelationalSupport} = _default_rs_type(W);
        compute_relation_glob = false,
        use_memoization = false,
    ) where {V,W<:AbstractWorld}

        # @logmsg LogOverview "FeaturedDataset -> SupportedFeaturedDataset "

        _fwd = fwd(emd)
        _features = features(emd)
        _relations = relations(emd)
        _grouped_featsnaggrs =  grouped_featsnaggrs(emd)
        featsnaggrs = features_grouped_featsaggrsnops2featsnaggrs(features(emd), grouped_featsaggrsnops(emd))
    
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
        nfeatsnaggrs = sum(length.(_grouped_featsnaggrs))

        # Prepare fwd_rs
        fwd_rs = relational_support_type(emd, use_memoization)

        # Prepare fwd_gs
        fwd_gs = begin
            if compute_fwd_gs
                GenericGlobalSupport(emd)
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

            for (i_feature,aggregators) in enumerate(_grouped_featsnaggrs)
                feature = _features[i_feature]
                @logmsg LogDebug "Feature $(i_feature)"

                cur_fwd_slice = fwdread_channel(_fwd, i_sample, i_feature)

                # @logmsg LogDebug cur_fwd_slice

                # Global relation (independent of the current world)
                if compute_fwd_gs
                    @logmsg LogDebug "RelationGlob"

                    # TODO optimize: all aggregators are likely reading the same raw values.
                    for (i_featsnaggr,aggr) in aggregators
                    # Threads.@threads for (i_featsnaggr,aggr) in aggregators
                        
                        threshold = fwd_slice_compute_global_gamma(emd, i_sample, cur_fwd_slice, feature, aggr)

                        @logmsg LogDebug "Aggregator[$(i_featsnaggr)]=$(aggr)  -->  $(threshold)"

                        # @logmsg LogDebug "Aggregator" aggr threshold

                        fwd_gs_set(fwd_gs, i_sample, i_featsnaggr, threshold)
                    end
                end

                if !use_memoization
                    # Other relations
                    for (i_relation,relation) in enumerate(_relations)

                        @logmsg LogDebug "Relation $(i_relation)/$(nrelations)"

                        for (i_featsnaggr,aggr) in aggregators
                            fwd_rs_init_world_slice(fwd_rs, i_sample, i_featsnaggr, i_relation)
                        end

                        for w in allworlds(emd, i_sample)

                            @logmsg LogDebug "World" w

                            # TODO optimize: all aggregators are likely reading the same raw values.
                            for (i_featsnaggr,aggr) in aggregators
                                
                                threshold = fwd_slice_compute_modal_gamma(emd, i_sample, cur_fwd_slice, w, relation, feature, aggr)

                                # @logmsg LogDebug "Aggregator" aggr threshold

                                fwd_rs_set(fwd_rs, i_sample, w, i_featsnaggr, i_relation, threshold)
                            end
                        end
                    end
                end
            end
            # next!(p)
        end
        OneStepSupportingDataset(fwd_rs, fwd_gs, featsnaggrs)
    end
end

fwd_rs(X::OneStepSupportingDataset) = X.fwd_rs
fwd_gs(X::OneStepSupportingDataset) = X.fwd_gs
featsnaggrs(X::OneStepSupportingDataset) = X.featsnaggrs

nsamples(X::OneStepSupportingDataset) = nsamples(fwd_rs(X))
# nfeatsnaggrs(X::OneStepSupportingDataset) = nfeatsnaggrs(fwd_rs(X))

# TODO delegate to the two components...
function checksupportconsistency(
    emd::FeaturedDataset{V,W},
    X::OneStepSupportingDataset{V,W},
) where {V,W<:AbstractWorld}
    @assert nsamples(emd) == nsamples(X)                "Consistency check failed! Unmatching nsamples for emd and support: $(nsamples(emd)) and $(nsamples(X))"
    # @assert nrelations(emd) == (nrelations(fwd_rs(X)) + (isnothing(fwd_gs(X)) ? 0 : 1))            "Consistency check failed! Unmatching nrelations for emd and support: $(nrelations(emd)) and $(nrelations(fwd_rs(X)))+$((isnothing(fwd_gs(X)) ? 0 : 1))"
    @assert nrelations(emd) >= nrelations(fwd_rs(X))            "Consistency check failed! Inconsistent nrelations for emd and support: $(nrelations(emd)) < $(nrelations(fwd_rs(X)))"
    _nfeatsnaggrs = nfeatsnaggrs(emd)
    @assert _nfeatsnaggrs == length(featsnaggrs(X))  "Consistency check failed! Unmatching featsnaggrs for emd and support: $(featsnaggrs(emd)) and $(featsnaggrs(X))"
    return true
end

usesmemo(X::OneStepSupportingDataset) = usesglobalmemo(X) || usesmodalmemo(X)
usesglobalmemo(X::OneStepSupportingDataset) = false
usesmodalmemo(X::OneStepSupportingDataset) = usesmemo(fwd_rs(X))

worldtype(X::OneStepSupportingDataset{V,W}) where {V,W}    = W

Base.size(X::OneStepSupportingDataset) = (size(fwd_rs(X)), (isnothing(fwd_gs(X)) ? () : size(fwd_gs(X))))

find_featsnaggr_id(X::OneStepSupportingDataset, feature::AbstractFeature, aggregator::Aggregator) = findall(x->x==(feature, aggregator), featsnaggrs(X))[1]

function _slice_dataset(X::OneStepSupportingDataset, inds::AbstractVector{<:Integer}, args...; kwargs...)
    OneStepSupportingDataset(
        _slice_dataset(fwd_rs(X), inds, args...; kwargs...),
        (isnothing(fwd_gs(X)) ? nothing : _slice_dataset(fwd_gs(X), inds, args...; kwargs...)),
        featsnaggrs(X)
    )
end


function hasnans(X::OneStepSupportingDataset)
    hasnans(fwd_rs(X)) || (!isnothing(fwd_gs(X)) && hasnans(fwd_gs(X)))
end

isminifiable(X::OneStepSupportingDataset) = isminifiable(fwd_rs(X)) && (isnothing(fwd_gs(X)) || isminifiable(fwd_gs(X)))

function minify(X::OSSD) where {OSSD<:OneStepSupportingDataset}
    (new_fwd_rs, new_fwd_gs), backmap =
        minify([
            fwd_rs(X),
            fwd_gs(X),
        ])

    X = OSSD(
        new_fwd_rs,
        new_fwd_gs,
        featsnaggrs(X),
    )
    X, backmap
end

function display_structure(X::OneStepSupportingDataset; indent_str = "")
    out = "$(typeof(X))\t$((Base.summarysize(fwd_rs(X)) + Base.summarysize(fwd_gs(X))) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ fwd_rs\t$(Base.summarysize(fwd_rs(X)) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t"
    if usesmodalmemo(X)
        out *= "(shape $(Base.size(fwd_rs(X))), $(round(nonnothingshare(fwd_rs(X))*100, digits=2))% memoized)\n"
    else
        out *= "(shape $(Base.size(fwd_rs(X))))\n"
    end
    out *= indent_str * "└ fwd_gs\t"
    if !isnothing(fwd_gs(X))
        out *= "$(Base.summarysize(fwd_gs(X)) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t"
        if usesglobalmemo(X)
            out *= "(shape $(Base.size(fwd_gs(X))), $(round(nonnothingshare(fwd_gs(X))*100, digits=2))% memoized)\n"
        else
            out *= "(shape $(Base.size(fwd_gs(X))))\n"
        end
    else
        out *= "−"
    end
    out
end


############################################################################################

function compute_global_gamma(
    X::OneStepSupportingDataset{V,W},
    emd::FeaturedDataset{V,W},
    i_sample::Integer,
    feature::AbstractFeature,
    aggregator::Aggregator,
    i_featsnaggr::Integer = find_featsnaggr_id(X, feature, aggregator),
) where {V,W<:AbstractWorld}
    _fwd_gs = fwd_gs(X)
    # @assert !isnothing(_fwd_gs) "Error. SupportedFeaturedDataset must be built with compute_relation_glob = true for it to be ready to test global decisions."
    if usesglobalmemo(X) && (false || isnothing(_fwd_gs[i_sample, i_featsnaggr]))
        error("TODO finish this: memoization on the global table")
        # gamma = TODO...
        # i_feature = find_feature_id(emd, feature)
        # fwd_feature_slice = fwdread_channel(fwd(emd), i_sample, i_feature)
        # fwd_gs_set(_fwd_gs, i_sample, i_featsnaggr, gamma)
    end
    _fwd_gs[i_sample, i_featsnaggr]
end

function compute_modal_gamma(
    X::OneStepSupportingDataset{V,W},
    emd::FeaturedDataset{V,W},
    i_sample::Integer,
    w::W,
    relation::AbstractRelation,
    feature::AbstractFeature,
    aggregator::Aggregator,
    i_relation::Integer,
) where {V,W<:AbstractWorld}
    i_featsnaggr = find_featsnaggr_id(X, feature, aggregator)
    _compute_modal_gamma(X, emd, i_sample, w, relation, feature, aggregator, i_featsnaggr, i_relation)

end

function _compute_modal_gamma(
    X::OneStepSupportingDataset{V,W},
    emd::FeaturedDataset{V,W},
    i_sample::Integer,
    w::W,
    relation::AbstractRelation,
    feature::AbstractFeature,
    aggregator::Aggregator,
    i_featsnaggr,
    i_relation,
)::V where {V,W<:AbstractWorld}
    _fwd_rs = fwd_rs(X)
    if usesmodalmemo(X) && (false ||  isnothing(_fwd_rs[i_sample, w, i_featsnaggr, i_relation]))
        i_feature = find_feature_id(emd, feature)
        fwd_feature_slice = fwdread_channel(fwd(emd), i_sample, i_feature)
        gamma = fwd_slice_compute_modal_gamma(emd, i_sample, fwd_feature_slice, w, relation, feature, aggregator)
        fwd_rs_set(_fwd_rs, i_sample, w, i_featsnaggr, i_relation, gamma)
    end
    _fwd_rs[i_sample, w, i_featsnaggr, i_relation]
end

include("generic-supports.jl")
include("dimensional-supports.jl")
