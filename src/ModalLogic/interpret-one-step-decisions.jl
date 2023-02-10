
using ..ModalDecisionTrees: is_propositional_decision, display_decision

export generate_feasible_decisions

# function test_decision(
#     X::SupportedFeaturedDataset{V,W},
#     i_sample::Integer,
#     w::W,
#     decision::ExistentialDimensionalDecision{U}
# ) where {V,W<:AbstractWorld,U}
    
#     if is_propositional_decision(decision)
#         _test_decision(X, i_sample, w, feature(decision), test_operator(decision), threshold(decision))
#     else SimpleDecision
#         gamma = onestep_accessible_aggregation(...X, i_sample, w, relation(decision), feature(decision), test_operator(decision))
#         evaluate_thresh_decision(test_operator(decision), gamma, threshold(decision))
#     end
# end

function _test_decision(
    X::AbstractConditionalDataset,
    i_sample::Integer,
    w::AbstractWorld,
    feature::AbstractFeature{V},
    test_operator::TestOperatorFun,
    threshold::U
) where {V,U}
    gamma = X[i_sample, w, feature]::V
    evaluate_thresh_decision(test_operator, gamma, threshold)
end


# Perform the modal step, that is, evaluate a modal formula
#  on a domain, and eventually compute the new world set.
function modal_step(
    X::ActiveFeaturedDataset{V,W},
    i_sample::Integer,
    worlds::WorldSetType,
    decision::ExistentialDimensionalDecision{U},
    returns_survivors::Union{Val{true},Val{false}} = Val(false)
) where {V,W<:AbstractWorld,WorldSetType<:AbstractWorldSet{W},U}
    @logmsg LogDetail "modal_step" worlds display_decision(decision)

    satisfied = false
    
    # TODO space for optimization here: with some relations (e.g. IA_A, IA_L) can be made smaller

    if returns_survivors isa Val{true}
        worlds_map = Dict{W,AbstractWorldSet{W}}()
    end
    if length(worlds) == 0
        # If there are no neighboring worlds, then the modal decision is not met
        @logmsg LogDetail "   No accessible world"
    else
        # Otherwise, check whether at least one of the accessible worlds witnesses truth of the decision.
        # TODO rewrite with new_worlds = map(...acc_worlds)
        # Initialize new worldset
        new_worlds = WorldSetType()

        # List all accessible worlds
        acc_worlds = 
            if returns_survivors isa Val{true}
                Threads.@threads for curr_w in worlds
                    l = ReentrantLock()
                    acc = accessibles(X, i_sample, curr_w, relation(decision)) |> collect
                    lock(l)
                    worlds_map[curr_w] = acc
                    unlock(l)
                end
                unique(cat([ worlds_map[k] for k in keys(worlds_map) ]...; dims = 1))
            else
                accessibles(X, i_sample, worlds, relation(decision))
            end

        for w in acc_worlds
            if _test_decision(X, i_sample, w, feature(decision), test_operator(decision), threshold(decision))
                # @logmsg LogDetail " Found world " w ch_readWorld ... ch_readWorld(w, channel)
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
        @logmsg LogDetail "   YES" worlds
    else
        @logmsg LogDetail "   NO"
    end
    if returns_survivors isa Val{true}
        return (satisfied, worlds, worlds_map)
    else
        return (satisfied, worlds)
    end
end

############################################################################################
############################################################################################
############################################################################################


Base.@propagate_inbounds @resumable function generate_feasible_decisions(
    X::ActiveFeaturedDataset{V,W},
    i_samples::AbstractVector{<:Integer},
    Sf::AbstractVector{<:AbstractWorldSet{W}},
    allow_propositional_decisions::Bool,
    allow_modal_decisions::Bool,
    allow_global_decisions::Bool,
    modal_relations_inds::AbstractVector{<:Integer},
    features_inds::AbstractVector{<:Integer},
) where {V,W<:AbstractWorld}
    # Propositional splits
    if allow_propositional_decisions
        for decision in generate_propositional_feasible_decisions(X, i_samples, Sf, features_inds)
            @yield decision
        end
    end
    # Global splits
    if allow_global_decisions
        for decision in generate_global_feasible_decisions(X, i_samples, Sf, features_inds)
            @yield decision
        end
    end
    # Modal splits
    if allow_modal_decisions
        for decision in generate_modal_feasible_decisions(X, i_samples, Sf, modal_relations_inds, features_inds)
            @yield decision
        end
    end
end

############################################################################################

Base.@propagate_inbounds @resumable function generate_propositional_feasible_decisions(
    X::ActiveFeaturedDataset{V,W,FR},
    i_samples::AbstractVector{<:Integer},
    Sf::AbstractVector{<:AbstractWorldSet{W}},
    features_inds::AbstractVector{<:Integer},
) where {V,W<:AbstractWorld,N,FR<:FullDimensionalFrame{N,W,Bool}}
    relation = RelationId
    _n_samples = length(i_samples)

    _features = features(X)
    _grouped_featsaggrsnops = grouped_featsaggrsnops(X)

    # For each feature
    @inbounds for i_feature in features_inds
        feature = _features[i_feature]
        @logmsg LogDebug "Feature $(i_feature): $(feature)"

        # operators for each aggregator
        aggrsnops = _grouped_featsaggrsnops[i_feature]
        # Vector of aggregators
        aggregators = keys(aggrsnops) # Note: order-variant, but that's ok here
        
        # dict->vector
        # aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

        # Initialize thresholds with the bottoms
        thresholds = Array{V,2}(undef, length(aggregators), _n_samples)
        for (i_aggr,aggr) in enumerate(aggregators)
            thresholds[i_aggr,:] .= aggregator_bottom(aggr, V)
        end

        # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
        for (instance_idx,i_sample) in enumerate(i_samples)
            # @logmsg LogDetail " Instance $(instance_idx)/$(_n_samples)"
            worlds = Sf[instance_idx]

            # TODO also try this instead
            # values = [X[i_sample, w, i_feature] for w in worlds]
            # thresholds[:,instance_idx] = map(aggr->aggr(values), aggregators)
            
            for w in worlds
                gamma = X[i_sample, w, feature, i_feature]
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
            aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(V), typemax(V)]))
            for (i_test_operator,test_operator) in enumerate(aggrsnops[aggr])
                # TODO figure out a solution to this issue: ≥ and ≤ in a propositional condition can find more or less the same optimum, so no need to check both; but which one of them should be the one on the left child, the one that makes the modal step?
                # if dual_test_operator(test_operator) in tested_test_operator
                #   throw_n_log("Double-check this part of the code: there's a foundational issue here to settle!")
                #   println("Found $(test_operator)'s dual $(dual_test_operator(test_operator)) in tested_test_operator = $(tested_test_operator)")
                #   continue
                # end
                # @logmsg LogDetail " Test operator $(test_operator)"
                # Look for the best threshold 'a', as in propositions like "feature >= a"
                for threshold in aggr_domain
                    decision = ExistentialDimensionalDecision(relation, feature, test_operator, threshold)
                    # @logmsg LogDebug " Testing decision: $(display_decision(decision))"
                    @yield decision, aggr_thresholds
                end # for threshold
                # push!(tested_test_operator, test_operator)
            end # for test_operator
        end # for aggregator
    end # for feature
end

############################################################################################

Base.@propagate_inbounds @resumable function generate_modal_feasible_decisions(
    X::ActiveFeaturedDataset{V,W,FR},
    i_samples::AbstractVector{<:Integer},
    Sf::AbstractVector{<:AbstractWorldSet{W}},
    modal_relations_inds::AbstractVector{<:Integer},
    features_inds::AbstractVector{<:Integer},
) where {V,W<:AbstractWorld,N,FR<:FullDimensionalFrame{N,W,Bool}}
    _n_samples = length(i_samples)

    _relations = relations(X)
    _features = features(X)
    _grouped_featsaggrsnops = grouped_featsaggrsnops(X)
    _grouped_featsnaggrs = grouped_featsnaggrs(X)
    
    # For each relational operator
    for i_relation in modal_relations_inds
        relation = _relations[i_relation]
        @logmsg LogDebug "Relation $(relation)..."

        # For each feature
        for i_feature in features_inds
            feature = _features[i_feature]
            # @logmsg LogDebug "Feature $(i_feature): $(feature)"

            # operators for each aggregator
            aggrsnops = _grouped_featsaggrsnops[i_feature]
            # Vector of aggregators
            aggregators_with_ids = _grouped_featsnaggrs[i_feature]

            # dict->vector?
            # aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

            # Initialize thresholds with the bottoms
            thresholds = Array{V,2}(undef, length(aggregators_with_ids), _n_samples)
            for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)
                thresholds[i_aggr,:] .= aggregator_bottom(aggr, V)
            end

            # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
            for (instance_id,i_sample) in enumerate(i_samples)
                # @logmsg LogDetail " Instance $(instance_id)/$(_n_samples)"
                worlds = Sf[instance_id]
                if X isa Union{FeaturedDataset,SupportedFeaturedDataset}
                    fwdslice = fwdread_channel(fwd(X), i_sample, i_feature)
                end
                for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
                    for w in worlds
                        gamma = begin
                            if X isa Union{FeaturedDataset,SupportedFeaturedDataset}
                                # fwdslice = fwdread_channel(fwd(X), i_sample, i_feature)
                                fwdslice_onestep_accessible_aggregation(X, fwdslice, i_sample, w, relation, feature, aggr, i_featsnaggr, i_relation)
                                # onestep_accessible_aggregation(X, i_sample, w, relation, feature, aggr, i_featsnaggr, i_relation)
                            elseif X isa DimensionalFeaturedDataset
                                 onestep_accessible_aggregation(X, i_sample, w, relation, feature, aggr, i_featsnaggr, i_relation)
                            else
                                error("generate_global_feasible_decisions is broken.")
                            end
                        end
                        thresholds[i_aggr,instance_id] = aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,instance_id])
                    end
                end
                
                # for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
                #     gammas = [onestep_accessible_aggregation(X, i_sample, w, relation, feature, aggr, i_featsnaggr, i_relation) for w in worlds]
                #     thresholds[i_aggr,instance_id] = aggr(gammas)
                # end
            end

            # @logmsg LogDebug "thresholds: " thresholds

            # For each aggregator
            for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)

                aggr_thresholds = thresholds[i_aggr,:]
                aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(V), typemax(V)]))

                for (i_test_operator,test_operator) in enumerate(aggrsnops[aggr])
                    # @logmsg LogDetail " Test operator $(test_operator)"

                    # Look for the best threshold 'a', as in propositions like "feature >= a"
                    for threshold in aggr_domain
                        decision = ExistentialDimensionalDecision(relation, feature, test_operator, threshold)
                        # @logmsg LogDebug " Testing decision: $(display_decision(decision))"
                        @yield decision, aggr_thresholds
                    end # for threshold
                end # for test_operator
            end # for aggregator
        end # for feature
    end # for relation
end

############################################################################################

Base.@propagate_inbounds @resumable function generate_global_feasible_decisions(
    X::ActiveFeaturedDataset{V,W,FR},
    i_samples::AbstractVector{<:Integer},
    Sf::AbstractVector{<:AbstractWorldSet{W}},
    features_inds::AbstractVector{<:Integer},
) where {V,W<:AbstractWorld,N,FR<:FullDimensionalFrame{N,W,Bool}}
    relation = RelationGlob
    _n_samples = length(i_samples)

    _features = features(X)
    _grouped_featsaggrsnops = grouped_featsaggrsnops(X)
    _grouped_featsnaggrs = grouped_featsnaggrs(X)

    @assert !(X isa SupportedFeaturedDataset && isnothing(fwd_gs(support(X)))) "Error. SupportedFeaturedDataset must be built with compute_relation_glob = true for it to be ready to generate global decisions."

    # For each feature
    for i_feature in features_inds
        feature = _features[i_feature]
        @logmsg LogDebug "Feature $(i_feature): $(feature)"

        # operators for each aggregator
        aggrsnops = _grouped_featsaggrsnops[i_feature]
        # println(aggrsnops)
        # Vector of aggregators
        aggregators_with_ids = _grouped_featsnaggrs[i_feature]
        # println(aggregators_with_ids)

        # dict->vector
        # aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

        # # TODO use this optimized version for SupportedFeaturedDataset:
        #   thresholds can in fact be directly given by slicing fwd_gs and permuting the two dimensions
        # aggregators_ids = fst.(aggregators_with_ids)
        # thresholds = transpose(fwd_gs(X)[i_samples, aggregators_ids])

        # Initialize thresholds with the bottoms
        thresholds = Array{V,2}(undef, length(aggregators_with_ids), _n_samples)
        # for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)
        #     thresholds[i_aggr,:] .= aggregator_bottom(aggr, V)
        # end
        
        # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
        for (instance_id,i_sample) in enumerate(i_samples)
            # @logmsg LogDetail " Instance $(instance_id)/$(_n_samples)"
            if X isa Union{FeaturedDataset,SupportedFeaturedDataset}
                fwdslice = fwdread_channel(fwd(X), i_sample, i_feature)
            end
            for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
                # TODO delegate this job to different flavors of `get_global_gamma`. Test whether the fwdslice assignment outside is faster!
                gamma = begin
                    if X isa Union{FeaturedDataset,SupportedFeaturedDataset}
                        # fwdslice = fwdread_channel(fwd(X), i_sample, i_feature)
                        fwdslice_onestep_accessible_aggregation(X, fwdslice, i_sample, RelationGlob, feature, aggr, i_featsnaggr)
                        # onestep_accessible_aggregation(X, i_sample, RelationGlob, feature, aggr, i_featsnaggr)
                    elseif X isa DimensionalFeaturedDataset
                        onestep_accessible_aggregation(X, i_sample, RelationGlob, feature, aggr, i_featsnaggr)
                    else
                        error("generate_global_feasible_decisions is broken.")
                    end
                end

                thresholds[i_aggr,instance_id] = gamma
                # thresholds[i_aggr,instance_id] = aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,instance_id])
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
            aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(V), typemax(V)]))

            for (i_test_operator,test_operator) in enumerate(aggrsnops[aggr])
                # @logmsg LogDetail " Test operator $(test_operator)"

                # Look for the best threshold 'a', as in propositions like "feature >= a"
                for threshold in aggr_domain
                    decision = ExistentialDimensionalDecision(relation, feature, test_operator, threshold)
                    # @logmsg LogDebug " Testing decision: $(display_decision(decision))"
                    @yield decision, aggr_thresholds
                end # for threshold
            end # for test_operator
        end # for aggregator
    end # for feature
end
