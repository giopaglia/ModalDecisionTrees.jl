struct ExplicitModalDatasetS{
    T<:Number,
    W<:AbstractWorld,
    FR<:AbstractFrame{W,Bool},
    MEMO<:Union{Val{true},Val{false}},
    FWDRS<:AbstractRelationalSupport{<:Union{T,Nothing}, W},
    G1<:AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}},
    G2<:AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
} <: ActiveModalDataset{T,W,FR}

    # Core dataset
    emd                 :: ExplicitModalDataset{T,W,FR}

    # Relational and global support
    fwd_rs              :: FWDRS
    fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing}

    # Features and Aggregators
    featsnaggrs         :: G1
    grouped_featsnaggrs :: G2

    ########################################################################################
    
    function ExplicitModalDatasetS{T,W,FR,MEMO}(
        emd                 :: ExplicitModalDataset{T,W,FR},
        fwd_rs              :: FWDRS,
        fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing},
        featsnaggrs         :: AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}},
        grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},MEMO<:Union{Val{true},Val{false}},FWDRS<:AbstractRelationalSupport{<:Union{T,Nothing}, W}}
        ty = "ExplicitModalDatasetS{$(T), $(W), $(FR), $(MEMO), $(FWDRS)}"
        @assert nsamples(emd) == nsamples(fwd_rs)                                    "Can't instantiate $(ty) with unmatching nsamples for emd and fwd_rs support: $(nsamples(emd)) and $(nsamples(fwd_rs))"
        @assert nrelations(emd) == nrelations(fwd_rs)                                "Can't instantiate $(ty) with unmatching nrelations for emd and fwd_rs support: $(nrelations(emd)) and $(nrelations(fwd_rs))"
        @assert sum(length.(grouped_featsnaggrs)) == length(featsnaggrs)             "Can't instantiate $(ty) with unmatching nfeatsnaggrs (grouped vs flattened structure): $(sum(length.(grouped_featsaggrsnops(emd)))) and $(length(featsnaggrs))"
        @assert sum(length.(grouped_featsaggrsnops(emd))) == length(featsnaggrs)      "Can't instantiate $(ty) with unmatching nfeatsnaggrs for emd and provided featsnaggrs: $(sum(length.(grouped_featsaggrsnops(emd)))) and $(length(featsnaggrs))"
        @assert sum(length.(grouped_featsaggrsnops(emd))) == nfeatsnaggrs(fwd_rs)     "Can't instantiate $(ty) with unmatching nfeatsnaggrs for emd and fwd_rs support: $(sum(length.(grouped_featsaggrsnops(emd)))) and $(nfeatsnaggrs(fwd_rs))"

        if fwd_gs != nothing
            @assert nsamples(emd) == nsamples(fwd_gs)                                "Can't instantiate $(ty) with unmatching nsamples for emd and fwd_gs support: $(nsamples(emd)) and $(nsamples(fwd_gs))"
            # @assert somethinglike(emd) == nfeatsnaggrs(fwd_gs)                     "Can't instantiate $(ty) with unmatching somethinglike for emd and fwd_gs support: $(somethinglike(emd)) and $(nfeatsnaggrs(fwd_gs))"
            @assert sum(length.(grouped_featsaggrsnops(emd))) == nfeatsnaggrs(fwd_gs) "Can't instantiate $(ty) with unmatching nfeatsnaggrs for emd and fwd_gs support: $(sum(length.(grouped_featsaggrsnops(emd)))) and $(nfeatsnaggrs(fwd_gs))"
        end

        if MEMO == Val{false}
            @assert FWDRS<:AbstractRelationalSupport{T, W} "Can't instantiate $(ty) with FWDRS = $(FWDRS). FWDRS<:AbstractRelationalSupport{$(T), $(W)} should hold."
        else
            @assert FWDRS<:Union{
                AbstractRelationalSupport{Union{T,Nothing}, W},
                AbstractRelationalSupport{T, W}
            } "Can't instantiate $(ty) with FWDRS = $(FWDRS). FWDRS<:Union{" *
                "AbstractRelationalSupport{Union{$(T),Nothing}, $(W)}," *
                "AbstractRelationalSupport{$(T), $(W)}} should hold."
        end

        new{T,W,FR,MEMO,FWDRS,typeof(featsnaggrs),typeof(grouped_featsnaggrs)}(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs)
    end

    function ExplicitModalDatasetS{T,W,FR,MEMO}(
        emd                 :: ExplicitModalDataset{T,W,FR},
        fwd_rs              :: FWDRS,
        args...;
        kwargs...
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},MEMO<:Union{Val{true},Val{false}},FWDRS<:AbstractRelationalSupport{<:Union{T,Nothing}, W}}
        ExplicitModalDatasetS{T,W,FR,MEMO,FWDRS}(emd, fwd_rs, args...; kwargs...)
    end

    function ExplicitModalDatasetS{T,W,FR}(emd::ExplicitModalDataset{T,W}, args...; use_memoization = true, kwargs...) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        MEMO = Val{use_memoization}
        ExplicitModalDatasetS{T,W,FR,MEMO}(emd, args...; kwargs...)
    end

    function ExplicitModalDatasetS{T,W}(emd::ExplicitModalDataset{T,W,FR}, args...; kwargs...) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        ExplicitModalDatasetS{T,W,FR}(emd, args...; kwargs...)
    end

    function ExplicitModalDatasetS{T}(emd::ExplicitModalDataset{T,W,FR}, args...; kwargs...) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        ExplicitModalDatasetS{T,W}(emd, args...; kwargs...)
    end

    function ExplicitModalDatasetS(emd::ExplicitModalDataset{T,W,FR}, args...; kwargs...) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        ExplicitModalDatasetS{T}(emd, args...; kwargs...)
    end
    
    ########################################################################################
    
    function ExplicitModalDatasetS(
        emd                   :: ExplicitModalDataset{T,W,FR};
        compute_relation_glob :: Bool = true,
        use_memoization       :: Bool = true,
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        
        # A function that computes fwd_rs and fwd_gs from an explicit modal dataset
        Base.@propagate_inbounds function compute_fwd_supports(
                emd                 :: ExplicitModalDataset{T,W},
                grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}};
                compute_relation_glob = false,
                simply_init_rs = false,
            ) where {T,W<:AbstractWorld}

            # @logmsg LogOverview "ExplicitModalDataset -> ExplicitModalDatasetS "

            _fwd = fwd(emd)
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
            fwd_rs = fwd_rs_init(emd, nfeatsnaggrs, nrelations; perform_initialization = simply_init_rs)

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

                    if !simply_init_rs
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
            fwd_rs, fwd_gs
        end
        
        # TODO use emd.grouped_featsnaggrs instead of recomputing it!!
        featsnaggrs, grouped_featsnaggrs = features_grouped_featsaggrsnops2featsnaggrs_grouped_featsnaggrs(emd.features, grouped_featsaggrsnops(emd))
        
        # Compute modal dataset propositions and 1-modal decisions
        fwd_rs, fwd_gs = compute_fwd_supports(emd, grouped_featsnaggrs, compute_relation_glob = compute_relation_glob, simply_init_rs = use_memoization);

        ExplicitModalDatasetS(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs; use_memoization = use_memoization)
    end

    ########################################################################################
    
    function ExplicitModalDatasetS(
        X                   :: InterpretedModalDataset{T,N,W};
        kwargs...,
    ) where {T,N,W<:AbstractWorld}
        emd = ExplicitModalDataset(X);
        ExplicitModalDatasetS(emd; kwargs...)
    end

end

emd(X::ExplicitModalDatasetS) = X.emd

Base.size(X::ExplicitModalDatasetS)                                      =  (size(emd(X)), size(X.fwd_rs), (isnothing(X.fwd_gs) ? nothing : size(X.fwd_gs)))
featsnaggrs(X::ExplicitModalDatasetS)                                    = X.featsnaggrs
fwd_rs(X::ExplicitModalDatasetS)                                         = X.fwd_rs
fwd_gs(X::ExplicitModalDatasetS)                                         = X.fwd_gs
features(X::ExplicitModalDatasetS)                                       = features(emd(X))
grouped_featsaggrsnops(X::ExplicitModalDatasetS)                         = grouped_featsaggrsnops(emd(X))
grouped_featsnaggrs(X::ExplicitModalDatasetS)                            = X.grouped_featsnaggrs
nfeatures(X::ExplicitModalDatasetS)                                      = nfeatures(emd(X))
nrelations(X::ExplicitModalDatasetS)                                     = nrelations(emd(X))
nsamples(X::ExplicitModalDatasetS)                                       = nsamples(emd(X))
relations(X::ExplicitModalDatasetS)                                      = relations(emd(X))
world_type(X::ExplicitModalDatasetS{T,W}) where {T,W}    = W
fwd(X::ExplicitModalDatasetS)                                            = fwd(emd(X))

usesmemo(X::ExplicitModalDatasetS{T,W,FR,MEMO}) where {T,W,FR,MEMO} = (MEMO == Val{true})

initialworldset(X::ExplicitModalDatasetS,  args...) = initialworldset(emd(X), args...)
accessibles(X::ExplicitModalDatasetS,     args...) = accessibles(emd(X), args...)
representatives(X::ExplicitModalDatasetS, args...) = representatives(emd(X), args...)
allworlds(X::ExplicitModalDatasetS,  args...) = allworlds(emd(X), args...)

function slice_dataset(X::ExplicitModalDatasetS, inds::AbstractVector{<:Integer}, args...; kwargs...)
    ExplicitModalDatasetS(
        slice_dataset(emd(X), inds, args...; kwargs...),
        slice_dataset(X.fwd_rs, inds, args...; kwargs...),
        (isnothing(X.fwd_gs) ? nothing : slice_dataset(X.fwd_gs, inds, args...; kwargs...)),
        X.featsnaggrs,
        X.grouped_featsnaggrs)
end

find_feature_id(X::ExplicitModalDatasetS, feature::AbstractFeature) =
    findall(x->x==feature, features(X))[1]
find_relation_id(X::ExplicitModalDatasetS, relation::AbstractRelation) =
    findall(x->x==relation, relations(X))[1]
find_featsnaggr_id(X::ExplicitModalDatasetS, feature::AbstractFeature, aggregator::Aggregator) =
    findall(x->x==(feature, aggregator), featsnaggrs(X))[1]

hasnans(X::ExplicitModalDatasetS) = begin
    # @show hasnans(emd(X))
    # @show hasnans(X.fwd_rs)
    # @show (!isnothing(X.fwd_gs) && hasnans(X.fwd_gs))
    hasnans(emd(X)) || hasnans(X.fwd_rs) || (!isnothing(X.fwd_gs) && hasnans(X.fwd_gs))
end

isminifiable(::ExplicitModalDatasetS) = true

function minify(X::EMD) where {EMD<:ExplicitModalDatasetS}
    (new_emd, new_fwd_rs, new_fwd_gs), backmap =
        minify([
            emd(X),
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

function display_structure(X::ExplicitModalDatasetS; indent_str = "")
    out = "$(typeof(X))\t$((Base.summarysize(emd(X)) + Base.summarysize(X.fwd_rs) + Base.summarysize(X.fwd_gs)) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(emd(X)))))\t$(relations(emd(X)))\n"
    out *= indent_str * "├ emd\t$(Base.summarysize(emd(X)) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
        out *= "\t(shape $(Base.size(emd(X))))\n"
    out *= indent_str * "├ fwd_rs\t$(Base.summarysize(X.fwd_rs) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
    if usesmemo(X)
        out *= "\t(shape $(Base.size(X.fwd_rs)), $(round(nonnothingshare(X.fwd_rs)*100, digits=2))% memoized)\n"
    else
        out *= "\t(shape $(Base.size(X.fwd_rs)))\n"
    end
    out *= indent_str * "└ fwd_gs\t$(Base.summarysize(X.fwd_gs) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t"
    if !isnothing(X.fwd_gs)
        out *= "\t(shape $(Base.size(X.fwd_gs)))"
    else
        out *= "\t−"
    end
    out
end

############################################################################################
############################################################################################
############################################################################################

Base.@propagate_inbounds @inline get_gamma(
        X::ExplicitModalDatasetS{T,W},
        i_sample::Integer,
        w::W,
        feature::AbstractFeature) where {T,W<:AbstractWorld} = get_gamma(emd(X), i_sample, w, feature)

function get_global_gamma(
    X::ExplicitModalDatasetS{T,W},
    i_sample::Integer,
    feature::AbstractFeature,
    test_operator::TestOperatorFun
) where {T,W<:AbstractWorld}
    # @assert !isnothing(X.fwd_gs) "Error. ExplicitModalDatasetS must be built with compute_relation_glob = true for it to be ready to test global decisions."
    i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
    X.fwd_gs[i_sample, i_featsnaggr]
end

function get_modal_gamma(
    X::ExplicitModalDatasetS{T,W},
    i_sample::Integer,
    w::W,
    relation::AbstractRelation,
    feature::AbstractFeature,
    test_operator::TestOperatorFun
)::T where {T,W<:AbstractWorld}
    i_relation = find_relation_id(X, relation)
    aggregator = existential_aggregator(test_operator)
    i_featsnaggr = find_featsnaggr_id(X, feature, aggregator)
    _get_modal_gamma(X, i_sample, w, i_featsnaggr, i_relation, relation, feature, aggregator)
end

function _get_modal_gamma(
    X::ExplicitModalDatasetS{T,W},
    i_sample::Integer,
    w::W,
    i_featsnaggr,
    i_relation,
    relation::AbstractRelation,
    feature::AbstractFeature,
    aggregator::Aggregator
)::T where {T,W<:AbstractWorld}
    if usesmemo(X) && (false ||  isnothing(X.fwd_rs[i_sample, w, i_featsnaggr, i_relation]))
        i_feature = find_feature_id(X, feature)
        fwd_feature_slice = fwd_get_channel(emd(X).fwd, i_sample, i_feature)
        gamma = _compute_modal_gamma(emd(X), i_sample, fwd_feature_slice, w, relation, feature, aggregator)
        fwd_rs_set(X.fwd_rs, i_sample, w, i_featsnaggr, i_relation, gamma)
    end
    X.fwd_rs[i_sample, w, i_featsnaggr, i_relation]
end

function test_decision(
    X::ExplicitModalDatasetS{T,W},
    i_sample::Integer,
    w::W,
    decision::ExistentialDimensionalDecision
) where {T,W<:AbstractWorld}
    
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

