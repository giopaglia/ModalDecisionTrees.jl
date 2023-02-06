
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

isminifiable(::Union{AbstractRelationalSupport,AbstractGlobalSupport}) = true

function minify(support::Union{AbstractRelationalSupport,AbstractGlobalSupport})
    minify(support.d) #TODO improper
end

############################################################################################
############################################################################################
############################################################################################
############################################################################################

abstract type SupportingModalDataset{T<:Number,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}} end

isminifiable(X::SupportingModalDataset) = false

############################################################################################
# Finally, let us define the implementation for explicit modal dataset with support
############################################################################################


struct ExplicitModalDatasetS{
    T<:Number,
    W<:AbstractWorld,
    FR<:AbstractFrame{W,Bool},
    S<:SupportingModalDataset{T,W,FR},
} <: ActiveModalDataset{T,W,FR}

    # Core dataset
    emd                 :: ExplicitModalDataset{T,W,FR}

    # Support structure
    support             :: S
    
    ########################################################################################
    
    function ExplicitModalDatasetS{T,W,FR,S}(
        emd                 :: ExplicitModalDataset{T,W,FR},
        support             :: S;
        allow_no_instances = false,
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},S<:SupportingModalDataset{T,W,FR}}
        ty = "ExplicitModalDatasetS{$(T), $(W), $(FR), $(S)}"
        @assert allow_no_instances || nsamples(emd) > 0  "Can't instantiate $(ty) with no instance."
        @assert checksupportconsistency(emd, support)    "Can't instantiate $(ty) with an inconsistent support:\n\nemd:\n$(display_structure(emd))\n\nsupport:\n$(display_structure(support))"
        new{T,W,FR,S}(emd, support)
    end

    function ExplicitModalDatasetS{T,W,FR}(emd::ExplicitModalDataset{T,W}, support::S, args...; kwargs...) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},S<:SupportingModalDataset{T,W,FR}}
        ExplicitModalDatasetS{T,W,FR,S}(emd, support, args...; kwargs...)
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
        
        support = OneStepSupportingDataset(
            emd,
            compute_relation_glob = compute_relation_glob,
            use_memoization = use_memoization
        );

        ExplicitModalDatasetS(emd, support)
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

emd(X::ExplicitModalDatasetS)                                            = X.emd
support(X::ExplicitModalDatasetS)                                        = X.support

Base.size(X::ExplicitModalDatasetS)                                      = (size(emd(X)), size(support(X)))
features(X::ExplicitModalDatasetS)                                       = features(emd(X))
grouped_featsaggrsnops(X::ExplicitModalDatasetS)                         = grouped_featsaggrsnops(emd(X))
grouped_featsnaggrs(X::ExplicitModalDatasetS)                            = grouped_featsnaggrs(emd(X))
nfeatures(X::ExplicitModalDatasetS)                                      = nfeatures(emd(X))
nrelations(X::ExplicitModalDatasetS)                                     = nrelations(emd(X))
nsamples(X::ExplicitModalDatasetS)                                       = nsamples(emd(X))
relations(X::ExplicitModalDatasetS)                                      = relations(emd(X))
fwd(X::ExplicitModalDatasetS)                                            = fwd(emd(X))
worldtype(X::ExplicitModalDatasetS{T,W}) where {T,W}    = W

usesmemo(X::ExplicitModalDatasetS) = usesmemo(support(X))

initialworldset(X::ExplicitModalDatasetS,  args...) = initialworldset(emd(X), args...)
accessibles(X::ExplicitModalDatasetS,     args...) = accessibles(emd(X), args...)
representatives(X::ExplicitModalDatasetS, args...) = representatives(emd(X), args...)
allworlds(X::ExplicitModalDatasetS,  args...) = allworlds(emd(X), args...)

function _slice_dataset(X::ExplicitModalDatasetS, inds::AbstractVector{<:Integer}, args...; kwargs...)
    ExplicitModalDatasetS(
        _slice_dataset(emd(X), inds, args...; kwargs...),
        _slice_dataset(support(X), inds, args...; kwargs...),
    )
end

find_feature_id(X::ExplicitModalDatasetS, feature::AbstractFeature) = findall(x->x==feature, features(X))[1]
find_relation_id(X::ExplicitModalDatasetS, relation::AbstractRelation) = findall(x->x==relation, relations(X))[1]

hasnans(X::ExplicitModalDatasetS) = hasnans(emd(X)) || hasnans(support(X))

isminifiable(X::ExplicitModalDatasetS) = isminifiable(emd(X)) && isminifiable(emd(X))

function minify(X::EMD) where {EMD<:ExplicitModalDatasetS}
    (new_emd, new_support), backmap =
        minify([
            emd(X),
            support(X),
        ])

    X = EMD(
        new_emd,
        new_support,
    )
    X, backmap
end

function display_structure(X::ExplicitModalDatasetS; indent_str = "")
    out = "$(typeof(X))\t$((Base.summarysize(emd(X)) + Base.summarysize(support(X))) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(emd(X)))))\t$(relations(emd(X)))\n"
    out *= indent_str * "├ emd\t$(Base.summarysize(emd(X)) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
        out *= "\t(shape $(Base.size(emd(X))))\n"
    out *= indent_str * "└ support: $(display_structure(support(X); indent_str = "  "))"
    out
end

############################################################################################
############################################################################################
############################################################################################

get_gamma(X::ExplicitModalDatasetS, args...) = get_gamma(emd(X), args...)
_get_gamma(X::ExplicitModalDatasetS, args...) = _get_gamma(emd(X), args...)

function _get_global_gamma(
    X::ExplicitModalDatasetS{T,W},
    i_sample::Integer,
    feature::AbstractFeature,
    aggregator::Aggregator
) where {T,W<:AbstractWorld}
    compute_global_gamma(support(X), emd(X), i_sample, feature, aggregator)
end

function _get_modal_gamma(
    X::ExplicitModalDatasetS{T,W},
    i_sample::Integer,
    w::W,
    relation::AbstractRelation,
    feature::AbstractFeature,
    aggregator::Aggregator,
    i_featsnaggr::Union{Nothing,Integer} = nothing,
    i_relation::Integer = find_relation_id(X, relation),
) where {T,W<:AbstractWorld}
    if isnothing(i_featsnaggr)
        compute_modal_gamma(support(X), emd(X), i_sample, w, relation, feature, aggregator, i_relation)
    else
        _compute_modal_gamma(support(X), emd(X), i_sample, w, relation, feature, aggregator, i_featsnaggr, i_relation)
    end
end

############################################################################################

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

