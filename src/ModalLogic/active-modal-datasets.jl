using ProgressMeter

using SoleModels: CanonicalFeatureGeq, CanonicalFeatureGeqSoft, CanonicalFeatureLeq, CanonicalFeatureLeqSoft
using SoleModels: evaluate_thresh_decision, existential_aggregator, aggregator_bottom, aggregator_to_binary

import SoleData: get_instance, instance, max_channel_size, channel_size, nattributes, nsamples, slice_dataset

using SoleLogics: goeswith_dim

# decision.jl
using ..ModalDecisionTrees: is_propositional_decision, display_decision

############################################################################################

# Convenience function
function grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops::AbstractVector{<:AbstractVector{<:TestOperatorFun}})::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
    grouped_featsaggrsnops = Dict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}[]
    for (i_feature, test_operators) in enumerate(grouped_featsnops)
        aggrsnops = Dict{Aggregator,AbstractVector{<:TestOperatorFun}}()
        for test_operator in test_operators
            aggregator = existential_aggregator(test_operator)
            if (!haskey(aggrsnops, aggregator))
                aggrsnops[aggregator] = TestOperatorFun[]
            end
            push!(aggrsnops[aggregator], test_operator)
        end
        push!(grouped_featsaggrsnops, aggrsnops)
    end
    grouped_featsaggrsnops
end

include("interpreted-modal-dataset.jl")

############################################################################################
# Featured world dataset
############################################################################################
# 
# In the most general case, the representation of a modal dataset is based on a
#  multi-dimensional lookup table, referred to as *propositional lookup table*,
#  or *featured world dataset* (abbreviated into fwd).
# 
# This structure, is such that the value at fwd[i, w, f], referred to as *gamma*,
#  is the value of feature f on world w on the i-th instance, and can be used to answer the
#  question whether a proposition (e.g., minimum(A1) ≥ 10) holds onto a given world and instance;
#  however, an fwd table can be implemented in many ways, mainly depending on the world type.
# 
# Note that this structure does not constitute a ActiveModalDataset (see ExplicitModalDataset a few lines below)
# 
############################################################################################

include("featured-world-dataset.jl")

include("explicit-modal-dataset.jl")

############################################################################################
# Explicit modal dataset with support
###########################################################################################

# The lookup table (fwd) in a featured modal dataset provides a quick answer on the truth of
#  propositional decisions; as for answering modal decisions (e.g., ⟨L⟩ (minimum(A2) ≥ 10) )
#  with an fwd, one must enumerate the accessible worlds, compute the truth on each world,
#  and aggregate the answer (by means of all/any). This process is costly; instead, it is
#  sometimes more convenient to initially spend more time computing the truth of any decision,
#  and store this information in a *support* lookup table. Similarly, one can decide to deploy
#  memoization on this table (instead of computing everything at the beginning, compute it on
#  the fly and store it for later calls).
# 
# We define an abstract type for explicit modal dataset with support lookup tables
abstract type ExplicitModalDatasetWithSupport{T,W,FR} <: ActiveModalDataset{T,W,FR} end
# And an abstract type for support lookup tables
abstract type AbstractSupport{T,W} end
# 
# In general, one can use lookup (with or without memoization) for any decision, even the
#  more complex ones, for example:
#  ⟨G⟩ (minimum(A2) ≥ 10 ∧ (⟨O⟩ (maximum(A3) > 2) ∨ (minimum(A1) < 0)))
# 
# In practice, decision trees only ask about simple decisions such as ⟨L⟩ (minimum(A2) ≥ 10),
#  or ⟨G⟩ (maximum(A2) ≤ 50). Because the global operator G behaves differently from other
#  relations, it is natural to differentiate between global and relational support tables:
# 
abstract type AbstractRelationalSupport{T,W} <: AbstractSupport{T,W}     end
abstract type AbstractGlobalSupport{T}       <: AbstractSupport{T,W where W<:AbstractWorld} end
#
# Be an *fwd_rs* an fwd relational support, and a *fwd_gs* an fwd global support,
#  for simple support tables like these, it is convenient to store, again, modal *gamma* values.
# Similarly to fwd, gammas are basically values on the verge of truth, that can straightforwardly
#  anser simple modal questions.
# Consider the decision (w ⊨ <R> f ⋈ a) on the i-th instance, for a given feature f,
#  world w, relation R and test operator ⋈, and let gamma (γ) be:
#  - fwd_rs[i, f, a, R, w] if R is a regular relation, or
#  - fwd_gs[i, f, a]       if R is the global relation G,
#  where a = aggregator(⋈). In this context, γ is the unique value for which w ⊨ <R> f ⋈ γ holds and:
#  - if aggregator(⋈) = minimum:     ∀ a > γ:   (w ⊨ <R> f ⋈ a) does not hold
#  - if aggregator(⋈) = maximum:     ∀ a < γ:   (w ⊨ <R> f ⋈ a) does not hold
# 
# Let us define the world type-agnostic implementations for fwd_rs and fwd_gs (note that any fwd_gs
#  is actually inherently world agnostic); world type-specific implementations can be defined
#  in a similar way.

############################################################################################
############################################################################################

isminifiable(::Union{AbstractFWD,AbstractRelationalSupport,AbstractGlobalSupport}) = true

function minify(fwd_or_support::Union{AbstractFWD,AbstractRelationalSupport,AbstractGlobalSupport})
    util.minify(fwd_or_support.d)
end

############################################################################################

include("supports.jl")

############################################################################################
# Finally, let us define two implementations for explicit modal dataset with support, one
#  without memoization and one with memoization
# TODO avoid code duplication
############################################################################################

include("explicit-modal-dataset-with-supports.jl")

############################################################################################
############################################################################################
############################################################################################

test_decision(
        X::ExplicitModalDatasetWithSupport{T,W},
        i_sample::Integer,
        w::W,
        decision::ExistentialDimensionalDecision) where {T,W<:AbstractWorld} = begin
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
            # values = [X.emd[i_sample, w, i_feature] for w in worlds]
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
        X::ExplicitModalDatasetWithSupport{T,W},
        args...
        ) where {T,W<:AbstractWorld}
        for decision in generate_propositional_feasible_decisions(X.emd, args...)
            @yield decision
        end
end

Base.@propagate_inbounds @resumable function generate_global_feasible_decisions(
        X::ExplicitModalDatasetWithSupport{T,W},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{W}},
        features_inds::AbstractVector{<:Integer},
        ) where {T,W<:AbstractWorld}
    relation = RelationGlob
    _n_samples = length(instances_inds)

    
    @assert !isnothing(X.fwd_gs) "Error. ExplicitModalDatasetWithSupport must be built with compute_relation_glob = true for it to be ready to generate global decisions."

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
                for (i_sample,instance_id) in enumerate(instances_inds)
                @logmsg LogDetail " Instance $(i_sample)/$(_n_samples)"
                worlds = Sf[i_sample] # TODO could also use representativess here?

                # TODO also try this instead (TODO fix first)
                # values = [X.fwd_rs[instance_id, w, i_feature] for w in worlds]
                # thresholds[:,i_sample] = map((_,aggr)->aggr(values), aggregators_with_ids)
                    
                for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
                    for w in worlds
                        gamma = X.fwd_rs[instance_id, w, i_featsnaggr, i_relation]
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

# Memoization for global gammas
# get_global_gamma(
#       X::ExplicitModalDatasetSMemo{T,W},
#       i_sample::Integer,
#       feature::AbstractFeature,
#       test_operator::TestOperatorFun) where {T,W<:AbstractWorld} = begin
#   @assert !isnothing(X.fwd_gs) "Error. ExplicitModalDatasetSMemo must be built with compute_relation_glob = true for it to be ready to test global decisions."
#   i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
#   # if !isnothing(X.fwd_gs[i_sample, i_featsnaggr])
#   X.fwd_gs[i_sample, i_featsnaggr]
#   # else
#   #   i_feature = find_feature_id(X, feature)
#   #   aggregator = existential_aggregator(test_operator)
#   #   fwd_feature_slice = fwd_get_channel(X.emd.fwd, i_sample, i_feature)
#   #   accessible_worlds = allworlds_aggr(X.emd, i_sample, feature, aggregator)
#   #   gamma = compute_modal_gamma(fwd_feature_slice, accessible_worlds, aggregator)
#   #   fwd_gs_set(X.fwd_gs, i_sample, i_featsnaggr, gamma)
#   # end
# end

# TODO scan this value for an example problem and different number of threads

# using Random
# coin_flip_memoiz_rng = Random.default_rng()

# cfnls_max = 0.8
# # cfnls_k = 5.9
# cfnls_k = 30
# coin_flip_no_look_ExplicitModalDatasetSWithMemoization_value = cfnls_max*cfnls_k/((Threads.nthreads())-1+cfnls_k)
# coin_flip_no_look_ExplicitModalDatasetSWithMemoization() = (rand(coin_flip_memoiz_rng) >= coin_flip_no_look_ExplicitModalDatasetSWithMemoization_value)
# coin_flip_no_look_ExplicitModalDatasetSWithMemoization() = false

get_modal_gamma(
        X::ExplicitModalDatasetSMemo{T,W},
        i_sample::Integer,
        w::W,
        relation::AbstractRelation,
        feature::AbstractFeature,
        test_operator::TestOperatorFun) where {T,W<:AbstractWorld} = begin
    i_relation = find_relation_id(X, relation)
    aggregator = existential_aggregator(test_operator)
    i_featsnaggr = find_featsnaggr_id(X, feature, aggregator)
    # if coin_flip_no_look_ExplicitModalDatasetSWithMemoization() || 
    if false || 
            isnothing(X.fwd_rs[i_sample, w, i_featsnaggr, i_relation])
        i_feature = find_feature_id(X, feature)
        fwd_feature_slice = fwd_get_channel(X.emd.fwd, i_sample, i_feature)
        accessible_worlds = representatives(X.emd, i_sample, w, relation, feature, aggregator)
        gamma = compute_modal_gamma(fwd_feature_slice, accessible_worlds, aggregator)
        fwd_rs_set(X.fwd_rs, i_sample, w, i_featsnaggr, i_relation, gamma)
    else
        X.fwd_rs[i_sample, w, i_featsnaggr, i_relation]
    end
end


Base.@propagate_inbounds @resumable function generate_modal_feasible_decisions(
        X::ExplicitModalDatasetSMemo{T,W},
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
            for (instance_id,i_sample) in enumerate(instances_inds)
                @logmsg LogDetail " Instance $(instance_id)/$(_n_samples)"
                worlds = Sf[instance_id] # TODO could also use representativess here?

                # TODO also try this instead (TODO fix first)
                # values = [X.fwd_rs[i_sample, w, i_feature] for w in worlds]
                # thresholds[:,instance_id] = map((_,aggr)->aggr(values), aggregators_with_ids)
                    
                for (i_aggr,(i_featsnaggr,aggregator)) in enumerate(aggregators_with_ids)
                    for w in worlds
                        gamma = 
                            # if coin_flip_no_look_ExplicitModalDatasetSWithMemoization() || 
                            if false || 
                                isnothing(X.fwd_rs[i_sample, w, i_featsnaggr, i_relation])
                                fwd_feature_slice = fwd_get_channel(X.emd.fwd, i_sample, i_feature)
                                accessible_worlds = representatives(X.emd, i_sample, w, relation, feature, aggregator)
                                gamma = compute_modal_gamma(fwd_feature_slice, accessible_worlds, aggregator)
                                fwd_rs_set(X.fwd_rs, i_sample, w, i_featsnaggr, i_relation, gamma)
                            else
                                X.fwd_rs[i_sample, w, i_featsnaggr, i_relation]
                            end
                        thresholds[i_aggr,instance_id] = aggregator_to_binary(aggregator)(gamma, thresholds[i_aggr,instance_id])
                    end
                end
            end

            @logmsg LogDebug "thresholds: " thresholds

            # For each aggregator
            for (i_aggr,(_,aggregator)) in enumerate(aggregators_with_ids)

                aggr_thresholds = thresholds[i_aggr,:]
                aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))

                for (i_test_operator,test_operator) in enumerate(aggrsnops[aggregator])
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

############################################################################################
############################################################################################
############################################################################################

# Perform the modal step, that is, evaluate a modal formula
#  on a domain, and eventually compute the new world set.
function modal_step(
        X::ActiveModalDataset{T,W},
        i_sample::Integer,
        worlds::WorldSetType,
        decision::ExistentialDimensionalDecision{T},
        returns_survivors::Union{Val{true},Val{false}} = Val(false)
    ) where {T,W<:AbstractWorld,WorldSetType<:AbstractWorldSet{W}}
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
            if test_decision(X, i_sample, w, feature(decision), test_operator(decision), threshold(decision))
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

test_decision(
        X::ModalDataset{T},
        i_sample::Integer,
        w::AbstractWorld,
        feature::AbstractFeature,
        test_operator::TestOperatorFun,
        threshold::T) where {T} = begin
    gamma = get_gamma(X, i_sample, w, feature)
    evaluate_thresh_decision(test_operator, gamma, threshold)
end

test_decision(
        X::ModalDataset{T},
        i_sample::Integer,
        w::AbstractWorld,
        decision::ExistentialDimensionalDecision{T}) where {T} = begin
    instance = get_instance(X, i_sample)

    aggregator = existential_aggregator(test_operator(decision))
    
    worlds = representatives(FullDimensionalFrame(instance_channel_size(instance)), w, relation(decision), feature(decision), aggregator)
    gamma = if length(worlds |> collect) == 0
        aggregator_bottom(aggregator, T)
    else
        aggregator((w)->get_gamma(X, i_sample, w, feature(decision)), worlds)
    end

    evaluate_thresh_decision(test_operator(decision), gamma, threshold(decision))
end


export generate_feasible_decisions
                # ,
                # generate_propositional_feasible_decisions,
                # generate_global_feasible_decisions,
                # generate_modal_feasible_decisions

Base.@propagate_inbounds @resumable function generate_feasible_decisions(
        X::ActiveModalDataset{T,W},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{W}},
        allow_propositional_decisions::Bool,
        allow_modal_decisions::Bool,
        allow_global_decisions::Bool,
        modal_relations_inds::AbstractVector{<:Integer},
        features_inds::AbstractVector{<:Integer},
        ) where {T,W<:AbstractWorld}
    # Propositional splits
    if allow_propositional_decisions
        for decision in generate_propositional_feasible_decisions(X, instances_inds, Sf, features_inds)
            @yield decision
        end
    end
    # Global splits
    if allow_global_decisions
        for decision in generate_global_feasible_decisions(X, instances_inds, Sf, features_inds)
            @yield decision
        end
    end
    # Modal splits
    if allow_modal_decisions
        for decision in generate_modal_feasible_decisions(X, instances_inds, Sf, modal_relations_inds, features_inds)
            @yield decision
        end
    end
end


# function slice_dataset(x::Any, dataset_slice::AbstractVector{<:Integer}; allow_no_instances = false, kwargs...)
#     @assert (allow_no_instances || length(dataset_slice) > 0) "Can't apply empty slice to dataset."
#     slice_dataset(x, dataset_slice; kwargs...)
# end
