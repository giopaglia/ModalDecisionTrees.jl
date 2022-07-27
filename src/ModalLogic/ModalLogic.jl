module ModalLogic

using ..ModalDecisionTrees
using ..ModalDecisionTrees: util, interpret_feature, alpha, display_feature_test_operator_pair

using ..ModalDecisionTrees: DimensionalDataset, AbstractDimensionalChannel, AbstractDimensionalInstance, UniformDimensionalDataset, DimensionalChannel, DimensionalInstance

using ..ModalDecisionTrees: DTOverview, DTDebug, DTDetail

using ..ModalDecisionTrees: test_operator_inverse

using BenchmarkTools
using ComputedFieldTypes
using DataStructures
using IterTools
using Logging: @logmsg
using ResumableFunctions

import Base: size, show, getindex, iterate, length, push!

export AbstractWorld, AbstractRelation,
       Ontology,
       AbstractWorldSet, WorldSet,
       RelationGlob, RelationNone, RelationId,
       world_type, world_types

# Fix (not needed from Julia 1.7, see https://github.com/JuliaLang/julia/issues/34674 )
if length(methods(Base.keys, (Base.Generator,))) == 0
    Base.keys(g::Base.Generator) = g.iter
end

############################################################################################
# Worlds
############################################################################################

# Abstract types for worlds
abstract type AbstractWorld end

# These constants is used for specifying different initial world conditions for each world type
#  (e.g. Interval(::_emptyWorld) = Interval(-1,0))
struct _emptyWorld end;
struct _centeredWorld end;

# More specifically, any world type W must provide constructors for:
# `W(::_emptyWorld)` # A dummy world (= no world in particular)
# `W(::_centeredWorld, args...)` # A world that is *central* to the modal frame

# Any world type W must also provide an `interpret_world` method for interpreting a world
#  onto a modal instance:
# interpret_world(::W, modal_instance)
# Note: for dimensional world types: modal_instance::DimensionalInstance

# Dimensional world types are interpreted on DimensionalInstances, and must also provide
#  a `dimensionality` method indicating the number of dimensions of a modal channel size
# dimensionality(::Type{W})
# For example, dimensionality(Interval) = 1.

# For convenience, each world type can be instantiated with a tuple of values, one for each field.
(W::Type{<:AbstractWorld})(args::Tuple) = W(args...)

# World enumerators generate array/set-like structures
const AbstractWorldSet{W} = Union{AbstractVector{W},AbstractSet{W}} where {W<:AbstractWorld}
const WorldSet{W} = Vector{W} where {W<:AbstractWorld}
WorldSet{W}(S::WorldSet{W}) where {W<:AbstractWorld} = S

############################################################################################
# Relations
############################################################################################

# Abstract types for relations
abstract type AbstractRelation end

# Relations must indicate their compatible world types via `goes_with`.
#  For example, if world type W is compatible with relation R
# goes_with(::Type{W}, ::R) = true
# Here's the fallback:
goes_with(::Type{W}, ::AbstractRelation) where {W<:AbstractWorld} = false

# Relations are defined via methods that return iterators the accessible worlds.
# Each relation R<:AbstractRelation must provide a method for `accessibles`, which returns an iterator
#  to the worlds that are accessible from a given world w:
# `accessibles(w::W,           ::R, args...)`

# Alternatively, one can provide a *bare* definition, that is, method `_accessibles`,
#  returning an iterator of *tuples* which is then fed to a constructor of the same world type, as in:
# `_accessibles(w::W,           ::R, args...)`

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
# accessibles_aggr(f::ModalFeature, a::Aggregator, S::AbstractWorldSet{W}, ::R, args...)
# Of course, the fallback is enumerating all accessible worlds via `accessibles`
accessibles_aggr(::ModalFeature, ::Aggregator, w::WorldType, r::AbstractRelation, args...) where {WorldType<:AbstractWorld} = accessibles(w, r, args...)

############################################################################################
# Singletons representing natural relations
############################################################################################

# Dummy relation (= no relation)
struct _RelationNone  <: AbstractRelation end; const RelationNone = _RelationNone();

############################################################################################

# Identity relation: any world -> itself
struct _RelationId    <: AbstractRelation end; const RelationId   = _RelationId();

Base.show(io::IO, ::_RelationId) = print(io, "=")

accessibles(w::WorldType,           ::_RelationId, args...) where {WorldType<:AbstractWorld} = [w] # TODO try IterTools.imap(identity, [w])
accessibles(S::AbstractWorldSet{W}, ::_RelationId, args...) where {W<:AbstractWorld} = S # TODO try IterTools.imap(identity, S)

accessibles_aggr(::ModalFeature, ::Aggregator, w::WorldType, r::_RelationId,      args...) where {WorldType<:AbstractWorld} = accessibles(w, r, args...)

############################################################################################

# Global relation:  any world -> all worlds
struct _RelationGlob   <: AbstractRelation end; const RelationGlob  = _RelationGlob();

Base.show(io::IO, ::_RelationGlob) = print(io, "G")

# Note: these methods must be defined for any newly defined world type WT:
# `accessibles(w::WT,           ::_RelationGlob, args...)`
# `accessibles(S::AbstractWorldSet{WT}, ::_RelationGlob, args...)`
# `accessibles_aggr(f::ModalFeature, a::Aggregator, S::AbstractWorldSet{WT}, ::_RelationGlob, args...)`

############################################################################################

# Shortcuts using global relation for enumerating all worlds
all_worlds(::Type{WorldType}, args...) where {WorldType<:AbstractWorld} = accessibles(WorldType[], RelationGlob, args...)
all_worlds(::Type{WorldType}, enum_acc_fun::Function) where {WorldType<:AbstractWorld} = enum_acc_fun(WorldType[], RelationGlob)
all_worlds_aggr(::Type{WorldType}, enum_repr_fun::Function, f::ModalFeature, a::Aggregator) where {WorldType<:AbstractWorld} = enum_repr_fun(f, a, WorldType[], RelationGlob)

############################################################################################

# Concrete type for ontologies
# An ontology is a pair `world type` + `set of relations`, and represents the kind of
#  modal frame that underlies a certain logic
struct Ontology{WorldType<:AbstractWorld}

    relations :: AbstractVector{<:AbstractRelation}

    function Ontology{WorldType}(_relations::AbstractVector) where {WorldType<:AbstractWorld}
        _relations = collect(unique(_relations))
        for relation in _relations
            @assert goes_with(WorldType, relation) "Can't instantiate Ontology{$(WorldType)} with relation $(relation)!"
        end
        if WorldType == OneWorld && length(_relations) > 0
          _relations = similar(_relations, 0)
          @warn "Instantiating Ontology{$(WorldType)} with empty set of relations!"
        end
        new{WorldType}(_relations)
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
        elseif issetequal(relations(o), IA2D_URelations)
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

############################################################################################
# Decision
############################################################################################

export Decision,
       #
       relation, feature, test_operator, threshold,
       is_propositional_decision,
       is_global_decision,
       #
       display_decision, display_decision_inverse

# A decision inducing a branching/split (e.g., ⟨L⟩ (minimum(A2) ≥ 10) )
struct Decision{T}

    # Relation, interpreted as an existential modal operator
    #  Note: RelationId for propositional decisions
    relation      :: AbstractRelation

    # Modal feature (a scalar function that can be computed on a world)
    feature       :: ModalFeature

    # Test operator (e.g. ≥)
    test_operator :: TestOperatorFun

    # Threshold value
    threshold     :: T
end

is_propositional_decision(d::Decision) = (d.relation isa ModalLogic._RelationId)
is_global_decision(d::Decision) = (d.relation isa ModalLogic._RelationGlob)

function Base.show(io::IO, decision::Decision)
    println(io, display_decision(decision))
end

function display_decision(decision::Decision; threshold_display_method::Function = x -> x, universal = false)
    display_propositional_decision(feature::ModalFeature, test_operator::TestOperatorFun, threshold::Number; threshold_display_method::Function = x -> x) =
        "$(display_feature_test_operator_pair(feature, test_operator)) $(threshold_display_method(threshold))"
    prop_decision_str = display_propositional_decision(decision.feature, decision.test_operator, decision.threshold; threshold_display_method = threshold_display_method)
    if !is_propositional_decision(decision)
        "$((universal ? display_universal : display_existential)(decision.relation)) ($prop_decision_str)"
    else
        "$prop_decision_str"
    end
end

display_existential(rel::AbstractRelation) = "⟨$(rel)⟩"
display_universal(rel::AbstractRelation)   = "[$(rel)]"

############################################################################################

function display_decision(i_frame::Integer, decision::Decision; threshold_display_method::Function = x -> x, universal = false)
    "{$i_frame} $(display_decision(decision; threshold_display_method = threshold_display_method, universal = universal))"
end

function display_decision_inverse(i_frame::Integer, decision::Decision{T}; threshold_display_method::Function = x -> x) where T
    inv_decision = Decision{T}(decision.relation, decision.feature, test_operator_inverse(decision.test_operator), decision.threshold)
    display_decision(i_frame, inv_decision; threshold_display_method = threshold_display_method, universal = true)
end

############################################################################################
# Dataset structures
############################################################################################
# TODO sort these
import ..ModalDecisionTrees: slice_dataset, concat_datasets,
       n_samples, n_attributes, max_channel_size, get_instance,
       instance_channel_size, get_instance_attribute


export n_features, n_relations,
       n_frames, # TODO remove
       nframes, frames, get_frame,
       display_structure,
       get_gamma, test_decision,
       #
       relations,
       init_world_sets_fun,
       #
       ModalDataset,
       GenericModalDataset,
       ActiveMultiFrameModalDataset,
       MultiFrameModalDataset,
       ActiveModalDataset,
       InterpretedModalDataset,
       ExplicitModalDataset,
       ExplicitModalDatasetS,
       ExplicitModalDatasetSMemo
       #
#
# A modal dataset can be *active* or *passive*.
#
# A passive modal dataset is one that you can interpret decisions on, but cannot necessarily
#  enumerate decisions for, as it doesn't have objects for storing the logic (relations, features, etc.).
# Dimensional datasets are passive.
include("dimensional-dataset-bindings.jl")
#
const PassiveModalDataset{T} = Union{DimensionalDataset{T}}
#
# Active datasets comprehend structures for representing relation sets, features, enumerating worlds,
#  etc. While learning a model can be done only with active modal datasets, testing a model
#  can be done with both active and passive modal datasets.
#
abstract type ActiveModalDataset{T<:Number,WorldType<:AbstractWorld} end
#
# Active modal datasets hold the WorldType, and thus can initialize world sets with a lighter interface
#
init_world_sets_fun(imd::ActiveModalDataset{T, WorldType},  i_sample::Integer, ::Type{WorldType}) where {T, WorldType} =
    init_world_sets_fun(imd, i_sample)
#
const ModalDataset{T} = Union{PassiveModalDataset{T},ActiveModalDataset{T}}
#
include("active-modal-datasets.jl")
#
# Define the multi-modal version of modal datasets (basically, a vector of datasets with the
#  same number of instances)
#
include("multi-frame-dataset.jl")
#
# TODO figure out which convert function works best: convert(::Type{<:MultiFrameModalDataset{T}}, X::MD) where {T,MD<:ModalDataset{T}} = MultiFrameModalDataset{MD}([X])
# convert(::Type{<:MultiFrameModalDataset}, X::ModalDataset) = MultiFrameModalDataset([X])
#
const ActiveMultiFrameModalDataset{T} = MultiFrameModalDataset{<:ActiveModalDataset{<:T}}
#
const GenericModalDataset = Union{ModalDataset,MultiFrameModalDataset}
#
#
############################################################################################
# Ontologies
############################################################################################

# Directional relations
abstract type DirectionalRelation <: AbstractRelation end

# Topological relations
abstract type TopologicalRelation <: AbstractRelation end

# Here are the definitions for world types and relations for known modal logics
#

export OneWorldOntology,
       #
       IntervalOntology,
       Interval2DOntology,
       getIntervalOntologyOfDim,
       #
       IntervalRCC8Ontology,
       Interval2DRCC8Ontology,
       getIntervalRCC8OntologyOfDim,
       #
       IntervalRCC5Ontology,
       Interval2DRCC5Ontology,
       getIntervalRCC5OntologyOfDim

############################################################################################
# Dimensionality: 0

# World type definitions for the propositional case, where there exist only one world,
#  and `Decision`s only allow the RelationId.
include("worlds/OneWorld.jl")                 # <- OneWorld world type

const OneWorldOntology   = Ontology{OneWorld}(AbstractRelation[])

############################################################################################
# Dimensionality: 1

# World type definitions for punctual logics, where worlds are points
# include("worlds/PointWorld.jl")

# World type definitions for interval logics, where worlds are intervals
include("worlds/Interval.jl")                 # <- Interval world type
include("relations/IA+Interval.jl")       # <- Allen relations
include("relations/RCC+Interval.jl")    # <- RCC relations

const IntervalOntology       = Ontology{Interval}(IARelations)
const IntervalRCC8Ontology   = Ontology{Interval}(RCC8Relations)
const IntervalRCC5Ontology   = Ontology{Interval}(RCC5Relations)

############################################################################################
# Dimensionality: 2

# World type definitions for 2D iterval logics, where worlds are rectangles
#  parallel to a frame of reference.
include("worlds/Interval2D.jl")               # <- Interval2D world type
include("relations/IA2+Interval2D.jl")     # <- Allen relations
include("relations/RCC+Interval2D.jl")  # <- RCC relations

const Interval2DOntology     = Ontology{Interval2D}(IA2DRelations)
const Interval2DRCC8Ontology = Ontology{Interval2D}(RCC8Relations)
const Interval2DRCC5Ontology = Ontology{Interval2D}(RCC5Relations)

############################################################################################


getIntervalOntologyOfDim(N::Integer) = getIntervalOntologyOfDim(Val(N))
getIntervalOntologyOfDim(::DimensionalDataset{T,D}) where {T,D} = getIntervalOntologyOfDim(Val(D-2))
getIntervalOntologyOfDim(::Val{0}) = OneWorldOntology
getIntervalOntologyOfDim(::Val{1}) = IntervalOntology
getIntervalOntologyOfDim(::Val{2}) = Interval2DOntology

getIntervalRCC8OntologyOfDim(N::Integer) = getIntervalRCC8OntologyOfDim(Val(N))
getIntervalRCC8OntologyOfDim(::DimensionalDataset{T,D}) where {T,D} = getIntervalRCC8OntologyOfDim(Val(D-2))
getIntervalRCC8OntologyOfDim(::Val{1}) = IntervalRCC8Ontology
getIntervalRCC8OntologyOfDim(::Val{2}) = Interval2DRCC8Ontology

getIntervalRCC5OntologyOfDim(N::Integer) = getIntervalRCC5OntologyOfDim(Val(N))
getIntervalRCC5OntologyOfDim(::DimensionalDataset{T,D}) where {T,D} = getIntervalRCC5OntologyOfDim(Val(D-2))
getIntervalRCC5OntologyOfDim(::Val{1}) = IntervalRCC5Ontology
getIntervalRCC5OntologyOfDim(::Val{2}) = Interval2DRCC5Ontology

const WorldType0D = Union{OneWorld}
const WorldType1D = Union{Interval}
const WorldType2D = Union{Interval2D}

############################################################################################
# World-specific featured world datasets and supports
############################################################################################

include("world-specific-fwds.jl")

############################################################################################
############################################################################################

end # module
