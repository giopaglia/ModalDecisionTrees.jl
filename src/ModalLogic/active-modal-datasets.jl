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

include("modal-datasets.jl")

############################################################################################

include("supports.jl")

############################################################################################
# Finally, let us define two implementations for explicit modal dataset with support, one
#  without memoization and one with memoization
# TODO avoid code duplication
############################################################################################

include("explicit-modal-dataset-with-supports.jl")
