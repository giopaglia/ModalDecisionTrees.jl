module ModalLogic

using ..ModalDecisionTrees
using ..ModalDecisionTrees: util, get_interpretation_function, alpha, display_feature_test_operator_pair

using BenchmarkTools
using ComputedFieldTypes
using DataStructures
using IterTools
using Logging: @logmsg
using ResumableFunctions

import Base: size, show, getindex, iterate, length

export AbstractWorld, AbstractRelation,
       Ontology,
       AbstractWorldSet, WorldSet,
       RelationGlob, RelationNone, RelationId,
       world_type, world_types

# Fix (not needed from Julia 1.7, see https://github.com/JuliaLang/julia/issues/34674 )
if length(methods(Base.keys, (Base.Generator,))) == 0
    Base.keys(g::Base.Generator) = g.iter
end

################################################################################
# Worlds
################################################################################

# Abstract types for worlds
abstract type AbstractWorld end

# These constants is used for specifying different initial world conditions for each world type
#  (e.g. Interval(::_emptyWorld) = Interval(-1,0))
struct _firstWorld end;
struct _emptyWorld end;
struct _centeredWorld end;

# World enumerators generate array/set-like structures
const AbstractWorldSet{W} = Union{AbstractVector{W},AbstractSet{W}} where {W<:AbstractWorld}
const WorldSet{W} = Vector{W} where {W<:AbstractWorld}
WorldSet{W}(S::WorldSet{W}) where {W<:AbstractWorld} = S

################################################################################
# Relations
################################################################################

# Abstract types for relations
abstract type AbstractRelation end

################################################################################
# Singletons representing natural relations

# Relations are defined via methods that return iterators the accessible worlds.
# Each relation R<:AbstractRelation must provide a method for `accessibles`, which returns an iterator
#  to the worlds that are accessible from a given world w:
# accessibles(w::WorldType,           ::R, args...) where {WorldType<:AbstractWorld}

# Alternatively, one can provide a *bare* definition, that is, method `_accessibles`,
#  returning an iterator of *tuples* which is then fed to a constructor of the same world type, as in:
# _accessibles(w::WorldType,           ::R, args...) where {WorldType<:AbstractWorld}

# The following fallback ensures that the two definitions are equivalent
accessibles(w::WorldType, r::AbstractRelation, args...) where {T,WorldType<:AbstractWorld} = begin
    IterTools.imap(WorldType, _accessibles(w, r, args...))
end

# 

# It is convenient to define methods for `accessibles` that take a world set instead of a
#  single world. Generally, this falls back to calling `_accessibles` on each world in
#  the set, and returning a constructor of wolds from the union; however, one may provide
#  improved implementations for special cases (e.g. ⟨L⟩ of a world set in interval algebra).
accessibles(S::AbstractWorldSet{WorldType}, r::AbstractRelation, args...) where {T,WorldType<:AbstractWorld} = begin
    IterTools.imap(WorldType,
        IterTools.distinct(
            Iterators.flatten(
                (_accessibles(w, r, args...) for w in S)
            )
        )
    )
end

# 

# It is also convenient to deploy some optimizations when you have intel about the decision
#  to test. For example, when you need to test a decision ⟨L⟩ (minimum(A2) ≥ 10) on a world w,
#  instead of computing minimum(A2) on all worlds, computing it on a single world is enough
#  to decide the truth. A few cases arise depending on the relation, the feature and the aggregator induced by the test
#  operator, thus one can provide optimized methods that return iterators to a few *representative*
#  worlds.
# accessibles_aggr(f::ModalFeature, a::Aggregator, S::AbstractWorldSet{W}, ::R, args...) where {W<:AbstractWorld}
# Of course, the fallback is enumerating all accessible worlds via `accessibles`
accessibles_aggr(::ModalFeature, ::Aggregator, w::WorldType, r::AbstractRelation, args...) where {WorldType<:AbstractWorld} = accessibles(w, r, args...)

################################################################################

# No relation
struct _RelationNone  <: AbstractRelation end; const RelationNone = _RelationNone();

################################################################################

# Identity relation: any world -> itself
struct _RelationId    <: AbstractRelation end; const RelationId   = _RelationId();

Base.show(io::IO, ::_RelationId) = print(io, "=")

accessibles(w::WorldType,           ::_RelationId, args...) where {WorldType<:AbstractWorld} = [w] # TODO try IterTools.imap(identity, [w])
accessibles(S::AbstractWorldSet{W}, ::_RelationId, args...) where {W<:AbstractWorld} = S # TODO try IterTools.imap(identity, S)

accessibles_aggr(::ModalFeature, ::Aggregator, w::WorldType, r::_RelationId,      args...) where {WorldType<:AbstractWorld} = accessibles(w, r, args...)

################################################################################

# Global relation:  any world -> all worlds
struct _RelationGlob   <: AbstractRelation end; const RelationGlob  = _RelationGlob();

Base.show(io::IO, ::_RelationGlob) = print(io, "G")

# Note: these methods must be defined for any newly defined world type WT:
# accessibles(w::WT,           ::_RelationGlob, args...)
# accessibles(S::AbstractWorldSet{WT}, ::_RelationGlob, args...)
# accessibles_aggr(f::ModalFeature, a::Aggregator, w::AbstractWorld, ::_RelationGlob, args...)
# accessibles_aggr(f::ModalFeature, a::Aggregator, S::AbstractWorldSet{WT}, ::_RelationGlob, args...) TODO maybe this as well ?

################################################################################

# Shortcuts using global relation for enumerating all worlds
all_worlds(::Type{WorldType}, args...) where {WorldType<:AbstractWorld} = accessibles(WorldType[], RelationGlob, args...)
all_worlds(::Type{WorldType}, enumAccFun::Function) where {WorldType<:AbstractWorld} = enumAccFun(WorldType[], RelationGlob)
all_worlds_aggr(::Type{WorldType}, enumReprFun::Function, f::ModalFeature, a::Aggregator) where {WorldType<:AbstractWorld} = enumReprFun(f, a, WorldType[], RelationGlob)

################################################################################

# Concrete type for ontologies
# An ontology is a pair `world type` + `set of relations`, and represents the kind of
#  modal frame that underlies a certain logic
struct Ontology{WorldType<:AbstractWorld}

    relations :: AbstractVector{<:AbstractRelation}

    function Ontology{WorldType}(relations::AbstractVector) where {WorldType<:AbstractWorld}
        relations = collect(unique(relations))
        for relation in relations
            @assert goesWith(WorldType, relation) "Can't instantiate Ontology{$(WorldType)} with relation $(relation)!"
        end
        if WorldType == OneWorld && length(relations) > 0
          relations = similar(relations, 0)
          @warn "Instantiating Ontology{$(WorldType)} with empty set of relations!"
        end
        new{WorldType}(relations)
    end

    Ontology(worldType::Type{<:AbstractWorld}, relations) = Ontology{worldType}(relations)
end

world_type(::Ontology{WT}) where {WT<:AbstractWorld} = WT
relations(o::Ontology) = o.relations

Base.show(io::IO, o::Ontology{WT}) where {WT<:AbstractWorld} = begin
    if o == OneWorldOntology
        print(io, "OneWorldOntology")
    else
        print(io, "Ontology{")
        show(io, WT)
        print(io, "}(")
        if issetequal(relations(o), IARelations)
            print(io, "IA")
        elseif issetequal(relations(o), IARelations_extended)
            print(io, "IA_extended")
        elseif issetequal(relations(o), IA2DRelations)
            print(io, "IA²")
        elseif issetequal(relations(o), IA2DRelations_U)
            print(io, "IA²_U")
        elseif issetequal(relations(o), IA2DRelations_extended)
            print(io, "IA²_extended")
        elseif issetequal(relations(o), RCC8Relations)
            print(io, "RCC8")
        elseif issetequal(relations(o), RCC5Relations)
            print(io, "RCC5")
        else
            show(io, relations(o))
        end
        print(io, ")")
    end
end

################################################################################
# Decision
################################################################################

export Decision,
       # 
       relation, feature, test_operator, threshold,
       is_modal_decision,
       # 
       display_decision, display_decision_inverse

# A decision inducing a branching/split (e.g., ⟨L⟩ (minimum(A2) ≥ 10) )
struct Decision{T}
    
    # Relation, interpreted as an existential modal operator
    #  Note: e.g. RelationId for the propositional case
    relation      :: AbstractRelation
    
    # Modal feature (a scalar function that can be computed on a world)
    feature       :: ModalFeature
    
    # Test operator (e.g. ≥)
    test_operator :: TestOperatorFun

    # Threshold value
    threshold     :: T
end

relation(d::Decision) = d.relation
feature(d::Decision) = d.feature
test_operator(d::Decision) = d.test_operator
threshold(d::Decision) = d.threshold

is_modal_decision(d::Decision) = !(relation(d) isa ModalLogic._RelationId)

function Base.show(io::IO, decision::Decision)
    println(io, display_decision(decision))
end

function display_decision(decision::Decision; threshold_display_method::Function = x -> x, universal = false)
    display_propositional_decision(feature::ModalFeature, test_operator::TestOperatorFun, threshold::Number; threshold_display_method::Function = x -> x) =
        "$(display_feature_test_operator_pair(feature, test_operator)) $(threshold_display_method(threshold))"
    prop_decision_str = display_propositional_decision(feature(decision), test_operator(decision), threshold(decision); threshold_display_method = threshold_display_method)
    if is_modal_decision(decision)
        "$((universal ? display_universal : display_existential)(relation(decision))) ($prop_decision_str)"
    else
        "$prop_decision_str"
    end
end

display_existential(rel::AbstractRelation) = "⟨$(rel)⟩"
display_universal(rel::AbstractRelation)   = "[$(rel)]"

################################################################################

function display_decision(i_frame::Integer, decision::Decision; threshold_display_method::Function = x -> x, universal = false)
    "{$i_frame} $(display_decision(decision; threshold_display_method = threshold_display_method, universal = universal))"
end

function display_decision_inverse(i_frame::Integer, decision::Decision; threshold_display_method::Function = x -> x)
    inv_decision = Decision{T}(relation(decision), feature(decision), test_operator_inverse(test_operator), threshold(decision))
    display_decision(i_frame, inv_decision; threshold_display_method = threshold_display_method, universal = true)
end

################################################################################
# Dataset structures
################################################################################

include("modal-datasets.jl")

################################################################################
################################################################################

export OneWorldOntology,
        # genericIntervalOntology,
        IntervalOntology,
        Interval2DOntology,
        getIntervalOntologyOfDim,
        # genericIntervalRCC8Ontology,
        IntervalRCC8Ontology,
        Interval2DRCC8Ontology,
        getIntervalRCC8OntologyOfDim,
        getIntervalRCC5OntologyOfDim,
        IntervalRCC5Ontology,
        Interval2DRCC5Ontology

minExtrema(extr::Union{NTuple{N,NTuple{2,T}},AbstractVector{NTuple{2,T}}}) where {T<:Real,N} = reduce(((fst,snd),(f,s))->(min(fst,f),max(snd,s)), extr; init=(typemax(T),typemin(T)))
maxExtrema(extr::Union{NTuple{N,NTuple{2,T}},AbstractVector{NTuple{2,T}}}) where {T<:Real,N} = reduce(((fst,snd),(f,s))->(max(fst,f),min(snd,s)), extr; init=(typemin(T),typemax(T)))
minExtrema(extr::Vararg{NTuple{2,T}}) where {T<:Real} = minExtrema(extr)
maxExtrema(extr::Vararg{NTuple{2,T}}) where {T<:Real} = maxExtrema(extr)

include("OneWorld.jl")
# include("Point.jl")

include("Interval.jl")
include("IARelations.jl")
include("TopoRelations.jl")

include("Interval2D.jl")
include("IA2DRelations.jl")
include("Topo2DRelations.jl")

# abstract type OntologyType end

const OneWorldOntology   = Ontology{OneWorld}(AbstractRelation[])

# struct _genericIntervalOntology  <: OntologyType end; const genericIntervalOntology  = _genericIntervalOntology();
const IntervalOntology   = Ontology{Interval}(IARelations)
const Interval2DOntology = Ontology{Interval2D}(IA2DRelations)

# struct _genericIntervalRCC8Ontology  <: OntologyType end; const genericIntervalRCC8Ontology  = _genericIntervalRCC8Ontology();
const IntervalRCC8Ontology   = Ontology{Interval}(RCC8Relations)
const Interval2DRCC8Ontology = Ontology{Interval2D}(RCC8Relations)
const IntervalRCC5Ontology   = Ontology{Interval}(RCC5Relations)
const Interval2DRCC5Ontology = Ontology{Interval2D}(RCC5Relations)

getIntervalOntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalOntologyOfDim(Val(D-2))
getIntervalOntologyOfDim(::Val{1}) = IntervalOntology
getIntervalOntologyOfDim(::Val{2}) = Interval2DOntology

getIntervalRCC8OntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalRCC8OntologyOfDim(Val(D-2))
getIntervalRCC8OntologyOfDim(::Val{1}) = IntervalRCC8Ontology
getIntervalRCC8OntologyOfDim(::Val{2}) = Interval2DRCC8Ontology

getIntervalRCC5OntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalRCC5OntologyOfDim(Val(D-2))
getIntervalRCC5OntologyOfDim(::Val{1}) = IntervalRCC5Ontology
getIntervalRCC5OntologyOfDim(::Val{2}) = Interval2DRCC5Ontology


################################################################################
################################################################################

include("BuildStumpSupport.jl")

function computePropositionalThreshold(feature::ModalFeature, w::AbstractWorld, instance::MatricialInstance{T,N}) where {T,N}
    get_interpretation_function(feature)(inst_readWorld(w, instance)::MatricialChannel{T,N-1})::T
end

# TODO add AbstractWorldSet type
computeModalThreshold(fwd_propositional_slice::FeaturedWorldDatasetSlice{T}, worlds::Any, aggregator::Agg) where {T, Agg<:Aggregator} = begin
    
    # TODO try reduce(aggregator, worlds; init=ModalLogic.bottom(aggregator, T))
    # TODO remove this aggregator_to_binary...
    
    if length(worlds |> collect) == 0
        ModalLogic.aggregator_bottom(aggregator, T)
    else
        aggregator((w)->modalDatasetChannelSliceGet(fwd_propositional_slice, w), worlds)
    end

    # opt = aggregator_to_binary(aggregator)
    # threshold = ModalLogic.bottom(aggregator, T)
    # for w in worlds
    #   e = modalDatasetChannelSliceGet(fwd_propositional_slice, w)
    #   threshold = opt(threshold,e)
    # end
    # threshold
end

################################################################################
################################################################################

# TODO A relation can be defined as a union of other relations.
# In this case, thresholds can be computed by maximization/minimization of the
#  thresholds referred to the relations involved.
# struct _UnionOfRelations{T<:NTuple{N,<:AbstractRelation} where N} <: AbstractRelation end;

# computeModalThresholdDual(test_operator::TestOperatorFun, w::WorldType, relation::R where R<:_UnionOfRelations{relsTuple}, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
#   computePropositionalThresholdDual(test_operator, w, channel)
#   fieldtypes(relsTuple)
# computeModalThreshold(test_operator::TestOperatorFun, w::WorldType, relation::R where R<:_UnionOfRelations{relsTuple}, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
#   computePropositionalThreshold(test_operator, w, channel)
#   fieldtypes(relsTuple)

################################################################################
################################################################################

end # module
