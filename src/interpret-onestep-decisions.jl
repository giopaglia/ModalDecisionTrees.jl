
using ResumableFunctions
using SoleLogics: AbstractFrame
using SoleModels: AbstractWorld, AbstractWorldSet, AbstractFeature
using Logging: @logmsg
using SoleModels: AbstractLogiset, SupportedLogiset

using SoleModels: base, globmemoset
using SoleModels: featchannel,
                    featchannel_onestep_aggregation,
                    onestep_aggregation

using SoleModels: SupportedLogiset, ScalarOneStepMemoset, AbstractFullMemoset
using SoleModels.DimensionalDatasets: UniformFullDimensionalLogiset

import SoleModels: relations, nrelations, metaconditions, nmetaconditions
import SoleModels: supports
import SoleModels.DimensionalDatasets: nfeatures, features

using SoleModels: Aggregator, TestOperator, ScalarMetaCondition
using SoleModels: ScalarExistentialFormula

"""
Logical datasets with scalar features.
"""
const AbstractScalarLogiset{
    W<:AbstractWorld,
    U<:Number,
    FT<:AbstractFeature,
    FR<:AbstractFrame{W}
} = AbstractLogiset{W,U,FT,FR}

nrelations(X::SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset}}) where {W,U,FT,FR,L,N} = nrelations(supports(X)[1])
nrelations(X::SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset,<:AbstractFullMemoset}}) where {W,U,FT,FR,L,N} = nrelations(supports(X)[1])
relations(X::SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset}}) where {W,U,FT,FR,L,N} = relations(supports(X)[1])
relations(X::SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset,<:AbstractFullMemoset}}) where {W,U,FT,FR,L,N} = relations(supports(X)[1])
nmetaconditions(X::SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset}}) where {W,U,FT,FR,L,N} = nmetaconditions(supports(X)[1])
nmetaconditions(X::SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset,<:AbstractFullMemoset}}) where {W,U,FT,FR,L,N} = nmetaconditions(supports(X)[1])
metaconditions(X::SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset}}) where {W,U,FT,FR,L,N} = metaconditions(supports(X)[1])
metaconditions(X::SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset,<:AbstractFullMemoset}}) where {W,U,FT,FR,L,N} = metaconditions(supports(X)[1])

"""
Perform the modal step, that is, evaluate an existential formula
 on a set of worlds, eventually computing the new world set.
"""
function modalstep(
    X, # ::AbstractScalarLogiset{W},
    i_instance::Integer,
    worlds::AbstractVector{W},
    decision::SimpleDecision{<:ScalarExistentialFormula},
    return_worldmap::Union{Val{true},Val{false}} = Val(false)
) where {W<:AbstractWorld}
    @logmsg LogDetail "modalstep" worlds displaydecision(decision)

    # W = worldtype(frame(X, i_instance))

    φ = formula(decision)
    satisfied = false
    
    # TODO the's room for optimization here: with some relations (e.g. IA_A, IA_L) can be made smaller

    if return_worldmap isa Val{true}
        worlds_map = ThreadSafeDict{W,AbstractVector{W}}()
    end
    if length(worlds) == 0
        # If there are no neighboring worlds, then the modal decision is not met
        @logmsg LogDetail "   Empty worldset"
    else
        # Otherwise, check whether at least one of the accessible worlds witnesses truth of the decision.
        # TODO rewrite with new_worlds = map(...acc_worlds)
        # Initialize new worldset
        new_worlds = Vector{W}()

        # List all accessible worlds
        acc_worlds = begin
            if return_worldmap isa Val{true}
                Threads.@threads for curr_w in worlds
                    acc = accessibles(frame(X, i_instance), curr_w, relation(φ)) |> collect
                    worlds_map[curr_w] = acc
                end
                unique(cat([ worlds_map[k] for k in keys(worlds_map) ]...; dims = 1))
            else
                accessibles(frame(X, i_instance), worlds, relation(φ))
            end
        end

        for w in acc_worlds
            if checkcondition(atom(proposition(φ)), X, i_instance, w)
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
    if return_worldmap isa Val{true}
        return (satisfied, worlds, worlds_map)
    else
        return (satisfied, worlds)
    end
end

############################################################################################
############################################################################################
############################################################################################


Base.@propagate_inbounds @resumable function generate_feasible_decisions(
    X::AbstractScalarLogiset{W,U},
    i_instances::AbstractVector{<:Integer},
    Sf::AbstractVector{<:AbstractWorldSet{W}},
    allow_propositional_decisions::Bool,
    allow_modal_decisions::Bool,
    allow_global_decisions::Bool,
    modal_relations_inds::AbstractVector{<:Integer},
    features_inds::AbstractVector{<:Integer},
    grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:ScalarMetaCondition}}},
    grouped_featsnaggrs::AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
) where {W<:AbstractWorld,U}
    # Propositional splits
    if allow_propositional_decisions
        for decision in generate_propositional_feasible_decisions(X, i_instances, Sf, features_inds, grouped_featsaggrsnops, grouped_featsnaggrs)
            # @logmsg LogDebug " Testing decision: $(displaydecision(decision))"
            @yield decision
        end
    end
    # Global splits
    if allow_global_decisions
        for decision in generate_global_feasible_decisions(X, i_instances, Sf, features_inds, grouped_featsaggrsnops, grouped_featsnaggrs)
            # @logmsg LogDebug " Testing decision: $(displaydecision(decision))"
            @yield decision
        end
    end
    # Modal splits
    if allow_modal_decisions
        for decision in generate_modal_feasible_decisions(X, i_instances, Sf, modal_relations_inds, features_inds, grouped_featsaggrsnops, grouped_featsnaggrs)
            # @logmsg LogDebug " Testing decision: $(displaydecision(decision))"
            @yield decision
        end
    end
end

############################################################################################

Base.@propagate_inbounds @resumable function generate_propositional_feasible_decisions(
    X::AbstractScalarLogiset{W,U,FT,FR},
    i_instances::AbstractVector{<:Integer},
    Sf::AbstractVector{<:AbstractWorldSet{W}},
    features_inds::AbstractVector{<:Integer},
    grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:ScalarMetaCondition}}},
    grouped_featsnaggrs::AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
) where {W<:AbstractWorld,U,FT<:AbstractFeature,N,FR<:FullDimensionalFrame{N,W}}
    relation = identityrel
    _ninstances = length(i_instances)

    _features = features(X)

    # For each feature
    @inbounds for i_feature in features_inds
        feature = _features[i_feature]
        @logmsg LogDebug "Feature $(i_feature): $(feature)"

        # operators for each aggregator
        aggrsnops = grouped_featsaggrsnops[i_feature]
        # Vector of aggregators
        aggregators = keys(aggrsnops) # Note: order-variant, but that's ok here
        
        # dict->vector
        # aggrsnops = [aggrsnops[i_aggregator] for i_aggregator in aggregators]

        # Initialize thresholds with the bottoms
        thresholds = Array{U,2}(undef, length(aggregators), _ninstances)
        for (i_aggregator,aggregator) in enumerate(aggregators)
            thresholds[i_aggregator,:] .= aggregator_bottom(aggregator, U)
        end

        # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
        for (instance_idx,i_instance) in enumerate(i_instances)
            # @logmsg LogDetail " Instance $(instance_idx)/$(_ninstances)"
            worlds = Sf[instance_idx]

            # TODO also try this instead
            # values = [X[i_instance, w, i_feature] for w in worlds]
            # thresholds[:,instance_idx] = map(aggregator->aggregator(values), aggregators)
            
            for w in worlds
                # gamma = featvalue(X[i_instance, w, feature) # TODO in general!
                gamma = featvalue(X, i_instance, w, feature, i_feature)
                for (i_aggregator,aggregator) in enumerate(aggregators)
                    thresholds[i_aggregator,instance_idx] = SoleModels.aggregator_to_binary(aggregator)(gamma, thresholds[i_aggregator,instance_idx])
                end
            end
        end
        
        # tested_metacondition = TestOperator[]

        # @logmsg LogDebug "thresholds: " thresholds
        # For each aggregator
        for (i_aggregator,aggregator) in enumerate(aggregators)
            aggr_thresholds = thresholds[i_aggregator,:]
            aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(U), typemax(U)]))
            for metacondition in aggrsnops[aggregator]
                # TODO figure out a solution to this issue: ≥ and ≤ in a propositional condition can find more or less the same optimum, so no need to check both; but which one of them should be the one on the left child, the one that makes the modal step?
                # if dual_metacondition(metacondition) in tested_metacondition
                #   error("Double-check this part of the code: there's a foundational issue here to settle!")
                #   println("Found $(metacondition)'s dual $(dual_metacondition(metacondition)) in tested_metacondition = $(tested_metacondition)")
                #   continue
                # end
                # @logmsg LogDetail " Test operator $(metacondition)"
                # Look for the best threshold 'a', as in propositions like "feature >= a"
                for threshold in aggr_domain
                    decision = SimpleDecision(ScalarExistentialFormula(relation, ScalarCondition(metacondition, threshold)))
                    # @logmsg LogDebug " Testing decision: $(displaydecision(decision))"
                    @yield decision, aggr_thresholds
                end # for threshold
                # push!(tested_metacondition, metacondition)
            end # for metacondition
        end # for aggregator
    end # for feature
end

############################################################################################

Base.@propagate_inbounds @resumable function generate_modal_feasible_decisions(
    X::AbstractScalarLogiset{W,U,FT,FR},
    i_instances::AbstractVector{<:Integer},
    Sf::AbstractVector{<:AbstractWorldSet{W}},
    modal_relations_inds::AbstractVector{<:Integer},
    features_inds::AbstractVector{<:Integer},
    grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:ScalarMetaCondition}}},
    grouped_featsnaggrs::AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
) where {W<:AbstractWorld,U,FT<:AbstractFeature,N,FR<:FullDimensionalFrame{N,W}}
    _ninstances = length(i_instances)

    _relations = relations(X)
    _features = features(X)
    
    # For each relational operator
    for i_relation in modal_relations_inds
        relation = _relations[i_relation]
        @logmsg LogDebug "Relation $(relation)..."

        # For each feature
        for i_feature in features_inds
            feature = _features[i_feature]
            # @logmsg LogDebug "Feature $(i_feature): $(feature)"

            # operators for each aggregator
            aggrsnops = grouped_featsaggrsnops[i_feature]
            # Vector of aggregators
            aggregators_with_ids = grouped_featsnaggrs[i_feature]

            # dict->vector?
            # aggrsnops = [aggrsnops[i_aggregator] for i_aggregator in aggregators]

            # Initialize thresholds with the bottoms
            thresholds = Array{U,2}(undef, length(aggregators_with_ids), _ninstances)
            for (i_aggregator,(_,aggregator)) in enumerate(aggregators_with_ids)
                thresholds[i_aggregator,:] .= aggregator_bottom(aggregator, U)
            end

            # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
            for (instance_id,i_instance) in enumerate(i_instances)
                # @logmsg LogDetail " Instance $(instance_id)/$(_ninstances)"
                worlds = Sf[instance_id]
                _featchannel = featchannel(base(X), i_instance, i_feature)
                for (i_aggregator,(i_metacond,aggregator)) in enumerate(aggregators_with_ids)
                    metacondition = metaconditions(X)[i_metacond]
                    for w in worlds
                        gamma = begin
                            if true
                                # _featchannel = featchannel(base(X), i_instance, i_feature)
                                # featchannel_onestep_aggregation(X, _featchannel, i_instance, w, relation, feature(metacondition), aggregator)
                                featchannel_onestep_aggregation(X, _featchannel, i_instance, w, relation, metacondition, i_metacond, i_relation)
                                # onestep_aggregation(X, i_instance, w, relation, feature, aggregator, i_metacond, i_relation)
                            # elseif X isa UniformFullDimensionalLogiset
                            #      onestep_aggregation(X, i_instance, w, relation, feature, aggregator, i_metacond, i_relation)
                            else
                                error("generate_global_feasible_decisions is broken.")
                            end
                        end
                        thresholds[i_aggregator,instance_id] = SoleModels.aggregator_to_binary(aggregator)(gamma, thresholds[i_aggregator,instance_id])
                    end
                end
                
                # for (i_aggregator,(i_metacond,aggregator)) in enumerate(aggregators_with_ids)
                #     gammas = [onestep_aggregation(X, i_instance, w, relation, feature, aggregator, i_metacond, i_relation) for w in worlds]
                #     thresholds[i_aggregator,instance_id] = aggregator(gammas)
                # end
            end

            # @logmsg LogDebug "thresholds: " thresholds

            # For each aggregator
            for (i_aggregator,(_,aggregator)) in enumerate(aggregators_with_ids)

                aggr_thresholds = thresholds[i_aggregator,:]
                aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(U), typemax(U)]))

                for metacondition in aggrsnops[aggregator]
                    # @logmsg LogDetail " Test operator $(metacondition)"

                    # Look for the best threshold 'a', as in propositions like "feature >= a"
                    for threshold in aggr_domain
                        decision = SimpleDecision(ScalarExistentialFormula(relation, ScalarCondition(metacondition, threshold)))
                        # @logmsg LogDebug " Testing decision: $(displaydecision(decision))"
                        @yield decision, aggr_thresholds
                    end # for threshold
                end # for metacondition
            end # for aggregator
        end # for feature
    end # for relation
end

############################################################################################

Base.@propagate_inbounds @resumable function generate_global_feasible_decisions(
    X::AbstractScalarLogiset{W,U,FT,FR},
    i_instances::AbstractVector{<:Integer},
    Sf::AbstractVector{<:AbstractWorldSet{W}},
    features_inds::AbstractVector{<:Integer},
    grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:ScalarMetaCondition}}},
    grouped_featsnaggrs::AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
) where {W<:AbstractWorld,U,FT<:AbstractFeature,N,FR<:FullDimensionalFrame{N,W}}
    relation = globalrel
    _ninstances = length(i_instances)

    _features = features(X)

    # For each feature
    for i_feature in features_inds
        feature = _features[i_feature]
        @logmsg LogDebug "Feature $(i_feature): $(feature)"

        # operators for each aggregator
        aggrsnops = grouped_featsaggrsnops[i_feature]
        # println(aggrsnops)
        # Vector of aggregators
        aggregators_with_ids = grouped_featsnaggrs[i_feature]
        # println(aggregators_with_ids)

        # dict->vector
        # aggrsnops = [aggrsnops[i_aggregator] for i_aggregator in aggregators]

        # # TODO use this optimized version for SupportedLogiset:
        #   thresholds can in fact be directly given by slicing globmemoset and permuting the two dimensions
        # aggregators_ids = fst.(aggregators_with_ids)
        # thresholds = transpose(globmemoset(X)[i_instances, aggregators_ids])

        # Initialize thresholds with the bottoms
        thresholds = Array{U,2}(undef, length(aggregators_with_ids), _ninstances)
        # for (i_aggregator,(_,aggregator)) in enumerate(aggregators_with_ids)
        #     thresholds[i_aggregator,:] .= aggregator_bottom(aggregator, U)
        # end
        
        # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
        for (instance_id,i_instance) in enumerate(i_instances)
            # @logmsg LogDetail " Instance $(instance_id)/$(_ninstances)"
            _featchannel = featchannel(base(X), i_instance, i_feature)
            for (i_aggregator,(i_metacond,aggregator)) in enumerate(aggregators_with_ids)
                # TODO delegate this job to different flavors of `get_global_gamma`. Test whether the _featchannel assignment outside is faster!
                metacondition = metaconditions(X)[i_metacond]
                gamma = begin
                    if true
                        # _featchannel = featchannel(base(X), i_instance, i_feature)
                        featchannel_onestep_aggregation(X, _featchannel, i_instance, SoleLogics.emptyworld(frame(X, i_instance)), relation, metacondition, i_metacond)
                        # onestep_aggregation(X, i_instance, dummyworldTODO, relation, feature, aggregator, i_metacond)
                    # elseif X isa UniformFullDimensionalLogiset
                    #     onestep_aggregation(X, i_instance, dummyworldTODO, relation, feature, aggregator, i_metacond)
                    else
                        error("generate_global_feasible_decisions is broken.")
                    end
                end

                thresholds[i_aggregator,instance_id] = gamma
                # thresholds[i_aggregator,instance_id] = SoleModels.aggregator_to_binary(aggregator)(gamma, thresholds[i_aggregator,instance_id])
                # println(gamma)
                # println(thresholds[i_aggregator,instance_id])
            end
        end

        # println(thresholds)
        @logmsg LogDetail "thresholds: " thresholds

        # For each aggregator
        for (i_aggregator,(_,aggregator)) in enumerate(aggregators_with_ids)

            # println(aggregator)

            aggr_thresholds = thresholds[i_aggregator,:]
            aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(U), typemax(U)]))

            for metacondition in aggrsnops[aggregator]
                # @logmsg LogDetail " Test operator $(metacondition)"

                # Look for the best threshold 'a', as in propositions like "feature >= a"
                for threshold in aggr_domain
                    decision = SimpleDecision(ScalarExistentialFormula(relation, ScalarCondition(metacondition, threshold)))
                    # @logmsg LogDebug " Testing decision: $(displaydecision(decision))"
                    @yield decision, aggr_thresholds
                end # for threshold
            end # for metacondition
        end # for aggregator
    end # for feature
end
