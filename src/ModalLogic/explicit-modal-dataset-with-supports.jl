struct ExplicitModalDatasetS{
    T<:Number,
    W<:AbstractWorld,
    FR<:AbstractFrame{W,Bool},
    MEMO<:Union{Val{true},Val{false}},
    FWDRS<:AbstractRelationalSupport{<:Union{T,Nothing}, W}
} <: ActiveModalDataset{T,W,FR}

    # Core dataset
    emd                 :: ExplicitModalDataset{T,W,FR}

    # Relational and global support
    fwd_rs              :: FWDRS
    fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing}

    # Features and Aggregators
    featsnaggrs         :: AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}}
    grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}}

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
        @assert sum(length.(grouped_featsnaggrs)) == length(featsnaggrs)             "Can't instantiate $(ty) with unmatching nfeatsnaggrs (grouped vs flattened structure): $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == length(featsnaggrs)      "Can't instantiate $(ty) with unmatching nfeatsnaggrs for emd and provided featsnaggrs: $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == nfeatsnaggrs(fwd_rs)     "Can't instantiate $(ty) with unmatching nfeatsnaggrs for emd and fwd_rs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(nfeatsnaggrs(fwd_rs))"

        if fwd_gs != nothing
            @assert nsamples(emd) == nsamples(fwd_gs)                                "Can't instantiate $(ty) with unmatching nsamples for emd and fwd_gs support: $(nsamples(emd)) and $(nsamples(fwd_gs))"
            # @assert somethinglike(emd) == nfeatsnaggrs(fwd_gs)                     "Can't instantiate $(ty) with unmatching somethinglike for emd and fwd_gs support: $(somethinglike(emd)) and $(nfeatsnaggrs(fwd_gs))"
            @assert sum(length.(emd.grouped_featsaggrsnops)) == nfeatsnaggrs(fwd_gs) "Can't instantiate $(ty) with unmatching nfeatsnaggrs for emd and fwd_gs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(nfeatsnaggrs(fwd_gs))"
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

        new{T,W,FR,MEMO,FWDRS}(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs)
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
features(X::ExplicitModalDatasetS)                                       = features(emd(X))
grouped_featsaggrsnops(X::ExplicitModalDatasetS)                         = grouped_featsaggrsnops(emd(X))
grouped_featsnaggrs(X::ExplicitModalDatasetS)                            = X.grouped_featsnaggrs
nfeatures(X::ExplicitModalDatasetS)                                      = nfeatures(emd(X))
nrelations(X::ExplicitModalDatasetS)                                     = nrelations(emd(X))
nsamples(X::ExplicitModalDatasetS)                                       = nsamples(emd(X))
relations(X::ExplicitModalDatasetS)                                      = relations(emd(X))
world_type(X::ExplicitModalDatasetS{T,W}) where {T,W}    = W

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

Base.@propagate_inbounds @inline get_gamma(
        X::ExplicitModalDatasetS{T,W},
        i_sample::Integer,
        w::W,
        feature::AbstractFeature) where {T,W<:AbstractWorld} = get_gamma(emd(X), i_sample, w, feature)

isminifiable(::ExplicitModalDatasetS) = true

function minify(X::EMD) where {EMD<:ExplicitModalDatasetS}
    (new_emd, new_fwd_rs, new_fwd_gs), backmap =
        util.minify([
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
        out *= "\t(shape $(Base.size(X.fwd_rs)), $(nonnothingshare(X.fwd_rs)*100)% memoized)\n"
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
        accessible_worlds = representatives(emd(X), i_sample, w, relation, feature, aggregator)
        gamma = compute_modal_gamma(fwd_feature_slice, accessible_worlds, aggregator)
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
            # values = [emd(X)[i_sample, w, i_feature] for w in worlds]
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
        X::ExplicitModalDatasetS{T,W},
        args...
        ) where {T,W<:AbstractWorld}
        for decision in generate_propositional_feasible_decisions(emd(X), args...)
            @yield decision
        end
end

Base.@propagate_inbounds @resumable function generate_global_feasible_decisions(
        X::ExplicitModalDatasetS{T,W},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{W}},
        features_inds::AbstractVector{<:Integer},
        ) where {T,W<:AbstractWorld}
    relation = RelationGlob
    _n_samples = length(instances_inds)

    
    @assert !isnothing(X.fwd_gs) "Error. ExplicitModalDatasetS must be built with compute_relation_glob = true for it to be ready to generate global decisions."

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
            for (i_sample,instance_ind) in enumerate(instances_inds)
                @logmsg LogDetail " Instance $(i_sample)/$(_n_samples)"
                worlds = Sf[i_sample] # TODO could also use representativess here?

                # TODO also try this instead (TODO fix first)
                # values = [X.fwd_rs[instance_ind, w, i_feature] for w in worlds]
                # thresholds[:,i_sample] = map((_,aggr)->aggr(values), aggregators_with_ids)
                    
                for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
                    for w in worlds
                        gamma = _get_modal_gamma(X, instance_ind, w, i_featsnaggr, i_relation, relation, feature, aggr)
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
