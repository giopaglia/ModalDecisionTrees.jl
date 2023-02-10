
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
# remove: abstract type ExplicitModalDatasetWithSupport{V,W,FR} <: ActiveFeaturedDataset{V,W,FR,FT} end
# And an abstract type for support lookup tables
abstract type AbstractSupport{V,W} end
# 
# In general, one can use lookup (with or without memoization) for any decision, even the
#  more complex ones, for example:
#  ⟨G⟩ (minimum(A2) ≥ 10 ∧ (⟨O⟩ (maximum(A3) > 2) ∨ (minimum(A1) < 0)))
# 
# In practice, decision trees only ask about simple decisions such as ⟨L⟩ (minimum(A2) ≥ 10),
#  or ⟨G⟩ (maximum(A2) ≤ 50). Because the global operator G behaves differently from other
#  relations, it is natural to differentiate between global and relational support tables:
# 
abstract type AbstractRelationalSupport{V,W,FR<:AbstractFrame} <: AbstractSupport{V,W}     end
abstract type AbstractGlobalSupport{V}       <: AbstractSupport{V,W where W<:AbstractWorld} end
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

abstract type SupportingModalDataset{V<:Number,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}} end

isminifiable(X::SupportingModalDataset) = false

############################################################################################
# Finally, let us define the implementation for explicit modal dataset with support
############################################################################################


struct SupportedFeaturedDataset{
    V<:Number,
    W<:AbstractWorld,
    FR<:AbstractFrame{W,Bool},
    FT<:AbstractFeature{V},
    S<:SupportingModalDataset{V,W,FR},
} <: ActiveFeaturedDataset{V,W,FR,FT}

    # Core dataset
    emd                 :: FeaturedDataset{V,W,FR,FT}

    # Support structure
    support             :: S
    
    ########################################################################################
    
    function SupportedFeaturedDataset{V,W,FR,FT,S}(
        emd                 :: FeaturedDataset{V,W,FR,FT},
        support             :: S;
        allow_no_instances = false,
    ) where {V,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},FT<:AbstractFeature{V},S<:SupportingModalDataset{V,W,FR}}
        ty = "SupportedFeaturedDataset{$(V),$(W),$(FR),$(FT),$(S)}"
        @assert allow_no_instances || nsamples(emd) > 0  "Can't instantiate $(ty) with no instance."
        @assert checksupportconsistency(emd, support)    "Can't instantiate $(ty) with an inconsistent support:\n\nemd:\n$(display_structure(emd))\n\nsupport:\n$(display_structure(support))"
        new{V,W,FR,FT,S}(emd, support)
    end

    function SupportedFeaturedDataset{V,W,FR,FT}(emd::FeaturedDataset{V,W}, support::S, args...; kwargs...) where {V,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},FT<:AbstractFeature{V},S<:SupportingModalDataset{V,W,FR}}
        SupportedFeaturedDataset{V,W,FR,FT,S}(emd, support, args...; kwargs...)
    end

    function SupportedFeaturedDataset{V,W,FR}(emd::FeaturedDataset{V,W,FR,FT}, args...; kwargs...) where {V,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},FT<:AbstractFeature{V}}
        SupportedFeaturedDataset{V,W,FR,FT}(emd, args...; kwargs...)
    end

    function SupportedFeaturedDataset{V,W}(emd::FeaturedDataset{V,W,FR}, args...; kwargs...) where {V,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        SupportedFeaturedDataset{V,W,FR}(emd, args...; kwargs...)
    end

    function SupportedFeaturedDataset{V}(emd::FeaturedDataset{V,W,FR}, args...; kwargs...) where {V,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        SupportedFeaturedDataset{V,W}(emd, args...; kwargs...)
    end

    function SupportedFeaturedDataset(emd::FeaturedDataset{V,W,FR}, args...; kwargs...) where {V,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        SupportedFeaturedDataset{V}(emd, args...; kwargs...)
    end
    
    ########################################################################################
    
    function SupportedFeaturedDataset(
        emd                   :: FeaturedDataset{V,W,FR};
        compute_relation_glob :: Bool = true,
        use_memoization       :: Bool = true,
    ) where {V,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        
        support = OneStepSupportingDataset(
            emd,
            compute_relation_glob = compute_relation_glob,
            use_memoization = use_memoization
        );

        SupportedFeaturedDataset(emd, support)
    end

    ########################################################################################
    
    function SupportedFeaturedDataset(
        X                   :: DimensionalFeaturedDataset{V,N,W};
        kwargs...,
    ) where {V,N,W<:AbstractWorld}
        emd = FeaturedDataset(X);
        SupportedFeaturedDataset(emd; kwargs...)
    end

end

emd(X::SupportedFeaturedDataset)                        = X.emd
support(X::SupportedFeaturedDataset)                    = X.support

Base.getindex(X::SupportedFeaturedDataset, args...)     = Base.getindex(emd(X), args...)::featvaltype(X)
Base.size(X::SupportedFeaturedDataset)                  = (size(emd(X)), size(support(X)))
features(X::SupportedFeaturedDataset)                   = features(emd(X))
grouped_featsaggrsnops(X::SupportedFeaturedDataset)     = grouped_featsaggrsnops(emd(X))
grouped_featsnaggrs(X::SupportedFeaturedDataset)        = grouped_featsnaggrs(emd(X))
nfeatures(X::SupportedFeaturedDataset)                  = nfeatures(emd(X))
nrelations(X::SupportedFeaturedDataset)                 = nrelations(emd(X))
nsamples(X::SupportedFeaturedDataset)                   = nsamples(emd(X))
relations(X::SupportedFeaturedDataset)                  = relations(emd(X))
fwd(X::SupportedFeaturedDataset)                        = fwd(emd(X))
worldtype(X::SupportedFeaturedDataset{V,W}) where {V,W} = W

usesmemo(X::SupportedFeaturedDataset) = usesmemo(support(X))

initialworldset(X::SupportedFeaturedDataset,  args...) = initialworldset(emd(X), args...)
accessibles(X::SupportedFeaturedDataset,     args...) = accessibles(emd(X), args...)
representatives(X::SupportedFeaturedDataset, args...) = representatives(emd(X), args...)
allworlds(X::SupportedFeaturedDataset,  args...) = allworlds(emd(X), args...)

function _slice_dataset(X::SupportedFeaturedDataset, inds::AbstractVector{<:Integer}, args...; kwargs...)
    SupportedFeaturedDataset(
        _slice_dataset(emd(X), inds, args...; kwargs...),
        _slice_dataset(support(X), inds, args...; kwargs...),
    )
end

find_feature_id(X::SupportedFeaturedDataset, feature::AbstractFeature) = findall(x->x==feature, features(X))[1]
find_relation_id(X::SupportedFeaturedDataset, relation::AbstractRelation) = findall(x->x==relation, relations(X))[1]

hasnans(X::SupportedFeaturedDataset) = hasnans(emd(X)) || hasnans(support(X))

isminifiable(X::SupportedFeaturedDataset) = isminifiable(emd(X)) && isminifiable(emd(X))

function minify(X::EMD) where {EMD<:SupportedFeaturedDataset}
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

function display_structure(X::SupportedFeaturedDataset; indent_str = "")
    out = "$(typeof(X))\t$((Base.summarysize(emd(X)) + Base.summarysize(support(X))) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(emd(X)))))\t$(relations(emd(X)))\n"
    out *= indent_str * "├ emd\t$(Base.summarysize(emd(X)) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
        out *= "\t(shape $(Base.size(emd(X))))\n"
    out *= indent_str * "└ support: $(display_structure(support(X); indent_str = "  "))"
    out
end
