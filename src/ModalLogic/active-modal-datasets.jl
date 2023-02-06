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

function features_grouped_featsaggrsnops2featsnaggrs_grouped_featsnaggrs(features, grouped_featsaggrsnops)
    featsnaggrs = Tuple{<:AbstractFeature,<:Aggregator}[]
    grouped_featsnaggrs = AbstractVector{Tuple{<:Integer,<:Aggregator}}[]
    i_featsnaggr = 1
    for (feat,aggrsnops) in zip(features, grouped_featsaggrsnops)
        aggrs = []
        for aggr in keys(aggrsnops)
            push!(featsnaggrs, (feat,aggr))
            push!(aggrs, (i_featsnaggr,aggr))
            i_featsnaggr += 1
        end
        push!(grouped_featsnaggrs, aggrs)
    end
    featsnaggrs, grouped_featsnaggrs
end

function features_grouped_featsaggrsnops2featsnaggrs(features, grouped_featsaggrsnops)
    featsnaggrs = Tuple{<:AbstractFeature,<:Aggregator}[]
    i_featsnaggr = 1
    for (feat,aggrsnops) in zip(features, grouped_featsaggrsnops)
        for aggr in keys(aggrsnops)
            push!(featsnaggrs, (feat,aggr))
            i_featsnaggr += 1
        end
    end
    featsnaggrs
end

function features_grouped_featsaggrsnops2grouped_featsnaggrs(features, grouped_featsaggrsnops)
    grouped_featsnaggrs = AbstractVector{Tuple{<:Integer,<:Aggregator}}[]
    i_featsnaggr = 1
    for (feat,aggrsnops) in zip(features, grouped_featsaggrsnops)
        aggrs = []
        for aggr in keys(aggrsnops)
            push!(aggrs, (i_featsnaggr,aggr))
            i_featsnaggr += 1
        end
        push!(grouped_featsnaggrs, aggrs)
    end
    grouped_featsnaggrs
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
# remove: abstract type ExplicitModalDatasetWithSupport{T,W,FR} <: ActiveModalDataset{T,W,FR} end
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
abstract type AbstractRelationalSupport{T,W,FR<:AbstractFrame} <: AbstractSupport{T,W}     end
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
    minify(fwd_or_support.d)
end

############################################################################################

include("modal-datasets.jl")

############################################################################################

include("supports.jl")

############################################################################################
# Finally, let us define two implementations for explicit modal dataset with support, one
#  without memoization and one with memoization
# TODO avoid code duplication
############################################################################################

include("explicit-modal-dataset-with-supports.jl")

############################################################################################

export generate_feasible_decisions

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

############################################################################################

Base.@propagate_inbounds @resumable function generate_propositional_feasible_decisions(
        X::Union{InterpretedModalDataset{T,N,W},ExplicitModalDataset{T,W},ExplicitModalDatasetS{T,W}} where {N},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{W}},
        features_inds::AbstractVector{<:Integer},
        ) where {T,W<:AbstractWorld}
    relation = RelationId
    _n_samples = length(instances_inds)

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
        thresholds = Array{T,2}(undef, length(aggregators), _n_samples)
        for (i_aggr,aggr) in enumerate(aggregators)
            thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
        end

        # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
        for (instance_idx,i_sample) in enumerate(instances_inds)
            @logmsg LogDetail " Instance $(instance_idx)/$(_n_samples)"
            worlds = Sf[instance_idx]

            # TODO also try this instead
            # values = [X[i_sample, w, i_feature] for w in worlds]
            # thresholds[:,instance_idx] = map(aggr->aggr(values), aggregators)
            
            for w in worlds
                gamma = begin # TODO delegate
                    if X isa Union{ExplicitModalDataset,ExplicitModalDatasetS}
                        # Note: TODO figure this out faster but equivalent to get_gamma(X, i_sample, w, feature)
                        fwd_get(fwd(X), i_sample, w, i_feature)
                    elseif X isa InterpretedModalDataset
                        get_gamma(X, i_sample, w, feature)
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

############################################################################################

Base.@propagate_inbounds @resumable function generate_modal_feasible_decisions(
        X::Union{InterpretedModalDataset{T,N,W},ExplicitModalDataset{T,W},ExplicitModalDatasetS{T,W}} where {N},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{W}},
        modal_relations_inds::AbstractVector{<:Integer},
        features_inds::AbstractVector{<:Integer},
        ) where {T,W<:AbstractWorld}
    _n_samples = length(instances_inds)

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
            @logmsg LogDebug "Feature $(i_feature): $(feature)"

            # operators for each aggregator
            aggrsnops = _grouped_featsaggrsnops[i_feature]
            # Vector of aggregators
            aggregators_with_ids = _grouped_featsnaggrs[i_feature]

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
                # values = [fwd_rs(X)[i_sample, w, i_feature] for w in worlds]
                # thresholds[:,instance_id] = map((_,aggr)->aggr(values), aggregators_with_ids)
                    
                for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
                    for w in worlds
                        gamma = begin # TODO delegate this job to different flavors of `get_global_gamma`
                            if X isa ExplicitModalDatasetS
                                __get_modal_gamma(X, i_sample, w, i_featsnaggr, i_relation, relation, feature, aggr)
                            elseif X isa ExplicitModalDataset
                               _get_modal_gamma(X, i_sample, w, relation, feature, aggr)
                            elseif X isa InterpretedModalDataset
                               _get_modal_gamma(X, i_sample, w, relation, feature, aggr)
                            else
                                error("generate_modal_feasible_decisions is broken.")
                            end
                        end
                        thresholds[i_aggr,instance_id] = aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,instance_id])
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

############################################################################################

Base.@propagate_inbounds @resumable function generate_global_feasible_decisions(
        X::Union{InterpretedModalDataset{T,N,W},ExplicitModalDataset{T,W},ExplicitModalDatasetS{T,W}} where {N},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{W}},
        features_inds::AbstractVector{<:Integer},
        ) where {T,W<:AbstractWorld}
    relation = RelationGlob
    _n_samples = length(instances_inds)

    _features = features(X)
    _grouped_featsaggrsnops = grouped_featsaggrsnops(X)
    _grouped_featsnaggrs = grouped_featsnaggrs(X)

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

        # # TODO use this optimized version for ExplicitModalDatasetS:
        #   thresholds can in fact be directly given by slicing fwd_gs and permuting the two dimensions
        # aggregators_ids = fst.(aggregators_with_ids)
        # thresholds = transpose(fwd_gs(X)[instances_inds, aggregators_ids])

        # Initialize thresholds with the bottoms
        thresholds = Array{T,2}(undef, length(aggregators_with_ids), _n_samples)
        for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)
            thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
        end
        
        # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
        for (instance_id,i_sample) in enumerate(instances_inds)
            @logmsg LogDetail " Instance $(instance_id)/$(_n_samples)"
            if X isa ExplicitModalDataset
                # Note: refer to ExplicitModalDatasetS
                cur_fwd_slice = fwd_get_channel(fwd(X), i_sample, i_feature)
            end
            for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
                gamma = begin # TODO delegate this job to different flavors of `get_global_gamma`. Test whether the cur_fwd_slice assignment outside is faster!
                    if X isa ExplicitModalDatasetS
                        fwd_gs(support(X))[i_sample, i_featsnaggr]
                        # TODO? _compute_global_gamma(X, emd, i_sample, feature, aggregator, cur_fwd_slice TODO, i_featsnaggr)
                    elseif X isa ExplicitModalDataset
                        # cur_fwd_slice = fwd_get_channel(fwd(X), i_sample, i_feature)
                        _compute_global_gamma(X, i_sample, cur_fwd_slice, feature, aggr)
                    elseif X isa InterpretedModalDataset
                        _get_global_gamma(X, i_sample, feature, aggr)
                    else
                        error("generate_global_feasible_decisions is broken.")
                    end
                end

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
