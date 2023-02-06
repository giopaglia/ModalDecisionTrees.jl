
# Compute modal dataset propositions and 1-modal decisions
struct OneStepSupportingDataset{
        T<:Number,
        W<:AbstractWorld,
        FR<:AbstractFrame{W,Bool},
        TT<:Union{T,Nothing},
        FWDRS<:AbstractRelationalSupport{TT,W,FR},
        FWDGS<:Union{AbstractGlobalSupport{T},Nothing},
        G<:AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}},
    } <: SupportingModalDataset{T,W,FR}

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
        T<:Number,
        W<:AbstractWorld,
        FR<:AbstractFrame{W,Bool},
        TT<:Union{T,Nothing},
        FWDRS<:AbstractRelationalSupport{TT,W,FR},
        FWDGS<:Union{AbstractGlobalSupport{T},Nothing},
        G<:AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}},
    }
        @assert nfeatsnaggrs(fwd_rs) == length(featsnaggrs)       "Can't instantiate $(ty) with unmatching nfeatsnaggrs for fwd_rs and provided featsnaggrs: $(nfeatsnaggrs(fwd_rs)) and $(length(featsnaggrs))"
        if fwd_gs != nothing
            @assert nfeatsnaggrs(fwd_gs) == length(featsnaggrs)   "Can't instantiate $(ty) with unmatching nfeatsnaggrs for fwd_gs and provided featsnaggrs: $(nfeatsnaggrs(fwd_gs)) and $(length(featsnaggrs))"
            @assert nsamples(fwd_gs) == nsamples(fwd_rs)              "Can't instantiate $(ty) with unmatching nsamples for fwd_gs and fwd_rs support: $(nsamples(fwd_gs)) and $(nsamples(fwd_rs))"
        end
        new{T,W,FR,TT,FWDRS,FWDGS,G}(fwd_rs, fwd_gs, featsnaggrs)
    end
    
    # A function that computes the support from an explicit modal dataset
    Base.@propagate_inbounds function OneStepSupportingDataset(
            emd                 :: ExplicitModalDataset{T,W};
            compute_relation_glob = false,
            use_memoization = false,
        ) where {T,W<:AbstractWorld}

        # @logmsg LogOverview "ExplicitModalDataset -> ExplicitModalDatasetS "

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

        # println(_n_samples)
        # println(nrelations)
        # println(nfeatsnaggrs)
        # println(_grouped_featsnaggrs)

        # Prepare fwd_rs
        fwd_rs = fwd_rs_init(emd, use_memoization)

        # Prepare fwd_gs
        fwd_gs = begin
            if compute_fwd_gs
                fwd_gs_init(emd)
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

                cur_fwd_slice = fwd_get_channel(_fwd, i_sample, i_feature)

                @logmsg LogDebug cur_fwd_slice

                # Global relation (independent of the current world)
                if compute_fwd_gs
                    @logmsg LogDebug "RelationGlob"

                    # TODO optimize: all aggregators are likely reading the same raw values.
                    for (i_featsnaggr,aggr) in aggregators
                    # Threads.@threads for (i_featsnaggr,aggr) in aggregators
                        
                        threshold = _compute_global_gamma(emd, i_sample, cur_fwd_slice, feature, aggr)

                        @logmsg LogDebug "Aggregator[$(i_featsnaggr)]=$(aggr)  -->  $(threshold)"

                        # @logmsg LogDebug "Aggregator" aggr threshold

                        fwd_gs_set(fwd_gs, i_sample, i_featsnaggr, threshold)
                    end
                end
                # readline()

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
                                
                                threshold = _compute_modal_gamma(emd, i_sample, cur_fwd_slice, w, relation, feature, aggr)

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

# TODO delegate to the two components
function checksupportconsistency(
    emd::ExplicitModalDataset{T,W},
    X::OneStepSupportingDataset{T,W},
) where {T,W<:AbstractWorld}
    @assert nsamples(emd) == nsamples(X)                "Consistency check failed! Unmatching nsamples for emd and support: $(nsamples(emd)) and $(nsamples(X))"
    # @assert nrelations(emd) == (nrelations(fwd_rs(X)) + (isnothing(fwd_gs(X)) ? 0 : 1))            "Consistency check failed! Unmatching nrelations for emd and support: $(nrelations(emd)) and $(nrelations(fwd_rs(X)))+$((isnothing(fwd_gs(X)) ? 0 : 1))"
    @assert nrelations(emd) >= nrelations(fwd_rs(X))            "Consistency check failed! Inconsistent nrelations for emd and support: $(nrelations(emd)) < $(nrelations(fwd_rs(X)))"
    emd_nfeatsnaggrs = sum(length.(grouped_featsnaggrs(emd)))
    @assert emd_nfeatsnaggrs == length(featsnaggrs(X))  "Consistency check failed! Unmatching featsnaggrs for emd and support: $(featsnaggrs(emd)) and $(featsnaggrs(X))"
    return true
end

usesmemo(X::OneStepSupportingDataset) = usesglobalmemo(X) || usesmodalmemo(X)
usesglobalmemo(X::OneStepSupportingDataset) = false
usesmodalmemo(X::OneStepSupportingDataset) = usesmemo(fwd_rs(X))

world_type(X::OneStepSupportingDataset{T,W}) where {T,W}    = W

Base.size(X::OneStepSupportingDataset) = (size(fwd_rs(X)), (isnothing(fwd_gs(X)) ? () : size(fwd_gs(X))))

find_featsnaggr_id(X::OneStepSupportingDataset, feature::AbstractFeature, aggregator::Aggregator) = findall(x->x==(feature, aggregator), featsnaggrs(X))[1]

function slice_dataset(X::OneStepSupportingDataset, inds::AbstractVector{<:Integer}, args...; kwargs...)
    OneStepSupportingDataset(
        slice_dataset(fwd_rs(X), inds, args...; kwargs...),
        (isnothing(fwd_gs(X)) ? nothing : slice_dataset(fwd_gs(X), inds, args...; kwargs...)),
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
    X::OneStepSupportingDataset{T,W},
    emd::ExplicitModalDataset{T,W},
    i_sample::Integer,
    feature::AbstractFeature,
    aggregator::Aggregator,
) where {T,W<:AbstractWorld}
    i_featsnaggr = find_featsnaggr_id(X, feature, aggregator)
    _compute_global_gamma(X, emd, i_sample, feature, aggregator, i_featsnaggr, cur_fwd_slice)
end

function _compute_global_gamma(
    X::OneStepSupportingDataset{T,W},
    emd::ExplicitModalDataset{T,W},
    i_sample::Integer,
    feature::AbstractFeature,
    aggregator::Aggregator,
    i_featsnaggr::Integer,
) where {T,W<:AbstractWorld}
    _fwd_gs = fwd_gs(X)
    # @assert !isnothing(_fwd_gs) "Error. ExplicitModalDatasetS must be built with compute_relation_glob = true for it to be ready to test global decisions."
    if usesglobalmemo(X) && (false || isnothing(_fwd_gs[i_sample, i_featsnaggr]))
        error("TODO finish this: memoization on the global table")
        # gamma = TODO...
        # i_feature = find_feature_id(emd, feature)
        # fwd_feature_slice = fwd_get_channel(fwd(emd), i_sample, i_feature)
        # fwd_gs_set(_fwd_gs, i_sample, i_featsnaggr, gamma)
    end
    _fwd_gs[i_sample, i_featsnaggr]
end

function compute_modal_gamma(
    X::OneStepSupportingDataset{T,W},
    emd::ExplicitModalDataset{T,W},
    i_sample::Integer,
    w::W,
    i_relation::Integer,
    relation::AbstractRelation,
    feature::AbstractFeature,
    aggregator::Aggregator,
) where {T,W<:AbstractWorld}
    i_featsnaggr = find_featsnaggr_id(X, feature, aggregator)
    _compute_modal_gamma(X, emd, i_sample, w, i_featsnaggr, i_relation, relation, feature, aggregator)

end

function _compute_modal_gamma(
    X::OneStepSupportingDataset{T,W},
    emd::ExplicitModalDataset{T,W},
    i_sample::Integer,
    w::W,
    i_featsnaggr,
    i_relation,
    relation::AbstractRelation,
    feature::AbstractFeature,
    aggregator::Aggregator
)::T where {T,W<:AbstractWorld}
    _fwd_rs = fwd_rs(X)
    if usesmodalmemo(X) && (false ||  isnothing(_fwd_rs[i_sample, w, i_featsnaggr, i_relation]))
        i_feature = find_feature_id(emd, feature)
        fwd_feature_slice = fwd_get_channel(fwd(emd), i_sample, i_feature)
        gamma = _compute_modal_gamma(emd, i_sample, fwd_feature_slice, w, relation, feature, aggregator)
        fwd_rs_set(_fwd_rs, i_sample, w, i_featsnaggr, i_relation, gamma)
    end
    _fwd_rs[i_sample, w, i_featsnaggr, i_relation]
end
