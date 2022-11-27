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

using SoleLogics.Relations
using SoleLogics.Worlds

import Base: size, show, getindex, iterate, length, push!

# This is a reexport from SoleLogics.Relations and SoleLogics.Worlds
export World, Relation
export AbstractWorldSet, WorldSet
export RelationGlob, RelationId

export Ontology, world_type, world_types

# Fix (not needed from Julia 1.7, see https://github.com/JuliaLang/julia/issues/34674 )
if length(methods(Base.keys, (Base.Generator,))) == 0
    Base.keys(g::Base.Generator) = g.iter
end

############################################################################################

# Concrete type for ontologies
# An ontology is a pair `world type` + `set of relations`, and represents the kind of
#  modal frame that underlies a certain logic
struct Ontology{WorldType<:World}

    relations :: AbstractVector{<:Relation}

    function Ontology{WorldType}(_relations::AbstractVector) where {WorldType<:World}
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

    Ontology(worldType::Type{<:World}, relations) = Ontology{worldType}(relations)
end

world_type(::Ontology{WT}) where {WT<:World} = WT
relations(o::Ontology) = o.relations

Base.show(io::IO, o::Ontology{WT}) where {WT<:World} = begin
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

############################################################################################
# Worlds
############################################################################################

# Any world type W must provide an `interpret_world` method for interpreting a world
#  onto a modal instance:
# interpret_world(::W, modal_instance)
# Note: for dimensional world types: modal_instance::DimensionalInstance

############################################################################################
# Relations
############################################################################################

# Relations are defined via methods that return iterators to the accessible worlds.
# Each relation R<:Relation must provide a method for `accessibles`, which returns an iterator
#  to the worlds that are accessible from a given world w:
# `accessibles(w::W,           ::R, args...)`

# Alternatively, one can provide a *bare* definition, that is, method `_accessibles`,
#  returning an iterator of *tuples* which is then fed to a constructor of the same world type, as in:
# `_accessibles(w::W,           ::R, args...)`

# The following fallback ensures that the two definitions are equivalent
accessibles(w::WorldType, r::Relation, args...) where {T,WorldType<:World} = begin
    IterTools.imap(WorldType, _accessibles(w, r, args...))
end

#

# It is convenient to define methods for `accessibles` that take a world set instead of a
#  single world. Generally, this falls back to calling `_accessibles` on each world in
#  the set, and returning a constructor of wolds from the union; however, one may provide
#  improved implementations for special cases (e.g. ⟨L⟩ of a world set in interval algebra).
accessibles(S::AbstractWorldSet{WorldType}, r::Relation, args...) where {T,WorldType<:World} = begin
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
accessibles_aggr(::ModalFeature, ::Aggregator, w::WorldType, r::Relation, args...) where {WorldType<:World} = accessibles(w, r, args...)

############################################################################################
# Singletons representing natural relations
############################################################################################

accessibles(w::WorldType,           ::_RelationId, args...) where {WorldType<:World} = [w] # TODO try IterTools.imap(identity, [w])
accessibles(S::AbstractWorldSet{W}, ::_RelationId, args...) where {W<:World} = S # TODO try IterTools.imap(identity, S)

accessibles_aggr(::ModalFeature, ::Aggregator, w::WorldType, r::_RelationId,      args...) where {WorldType<:World} = accessibles(w, r, args...)

############################################################################################

# Note: these methods must be defined for any newly defined world type WT:
# `accessibles(w::WT,           ::_RelationGlob, args...)`
# `accessibles(S::AbstractWorldSet{WT}, ::_RelationGlob, args...)`
# `accessibles_aggr(f::ModalFeature, a::Aggregator, S::AbstractWorldSet{WT}, ::_RelationGlob, args...)`

############################################################################################

# Shortcuts using global relation for enumerating all worlds
all_worlds(::Type{WorldType}, args...) where {WorldType<:World} = accessibles(WorldType[], RelationGlob, args...)
all_worlds(::Type{WorldType}, enum_acc_fun::Function) where {WorldType<:World} = enum_acc_fun(WorldType[], RelationGlob)
all_worlds_aggr(::Type{WorldType}, enum_repr_fun::Function, f::ModalFeature, a::Aggregator) where {WorldType<:World} = enum_repr_fun(f, a, WorldType[], RelationGlob)

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
    relation      :: Relation

    # Modal feature (a scalar function that can be computed on a world)
    feature       :: ModalFeature

    # Test operator (e.g. ≥)
    test_operator :: TestOperatorFun

    # Threshold value
    threshold     :: T

    function Decision{T}() where {T}
        new{T}()
    end

    function Decision{T}(
        relation      :: Relation,
        feature       :: ModalFeature,
        test_operator :: TestOperatorFun,
        threshold     :: T
    ) where {T}
        new{T}(relation, feature, test_operator, threshold)
    end

    function Decision(
        relation      :: Relation,
        feature       :: ModalFeature,
        test_operator :: TestOperatorFun,
        threshold     :: T
    ) where {T}
        Decision{T}(relation, feature, test_operator, threshold)
    end

    function Decision(
        decision      :: Decision{T},
        threshold_f   :: Function
    ) where {T}
        Decision{T}(decision.relation, decision.feature, decision.test_operator, threshold_f(decision.threshold))
    end
end

is_propositional_decision(d::Decision) = (d.relation isa ModalLogic._RelationId)
is_global_decision(d::Decision) = (d.relation isa ModalLogic._RelationGlob)

function Base.show(io::IO, decision::Decision)
    println(io, display_decision(decision))
end

function display_decision(
        decision::Decision;
        threshold_display_method::Function = x -> x,
        universal = false,
        attribute_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing,
        use_feature_abbreviations::Bool = false,
    )
    prop_decision_str = "$(
        display_feature_test_operator_pair(
            decision.feature,
            decision.test_operator;
            attribute_names_map = attribute_names_map,
            use_feature_abbreviations = use_feature_abbreviations,
        )
    ) $(threshold_display_method(decision.threshold))"
    if !is_propositional_decision(decision)
        "$((universal ? display_universal : display_existential)(decision.relation)) ($prop_decision_str)"
    else
        "$prop_decision_str"
    end
end

display_existential(rel::Relation) = "⟨$(rel)⟩"
display_universal(rel::Relation)   = "[$(rel)]"

############################################################################################

function display_decision(
        i_frame::Integer,
        decision::Decision;
        attribute_names_map::Union{Nothing,AbstractVector{<:AbstractVector},AbstractVector{<:AbstractDict}} = nothing,
        kwargs...)
    _attribute_names_map = isnothing(attribute_names_map) ? nothing : attribute_names_map[i_frame]
    "{$i_frame} $(display_decision(decision; attribute_names_map = _attribute_names_map, kwargs...))"
end

function display_decision_inverse(i_frame::Integer, decision::Decision{T}; args...) where {T}
    inv_decision = Decision{T}(decision.relation, decision.feature, test_operator_inverse(decision.test_operator), decision.threshold)
    display_decision(i_frame, inv_decision; universal = true, args...)
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

_isnan(n::Number) = isnan(n)
_isnan(n::Nothing) = false
hasnans(n::Number) = _isnan(n)
hasnans(n::AbstractArray{<:Union{Nothing, Number}}) = any(_isnan.(n))

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
abstract type ActiveModalDataset{T<:Number,WorldType<:World} end
#
# Active modal datasets hold the WorldType, and thus can initialize world sets with a lighter interface
#
init_world_sets_fun(imd::ActiveModalDataset{T, WorldType},  i_sample::Integer, ::Type{WorldType}) where {T, WorldType} =
    init_world_sets_fun(imd, i_sample)
#
# By default an active modal dataset cannot be miniaturized
isminifiable(::ActiveModalDataset) = false
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

# Here are the definitions for world types and relations for known modal logics
#
export get_ontology,
       get_interval_ontology

get_ontology(N::Integer, args...) = get_ontology(Val(N), args...)
get_ontology(::Val{0}, args...) = OneWorldOntology
function get_ontology(::Val{1}, world = :interval, relations::Union{Symbol,AbstractVector{<:Relation}} = :IA)
    world_possible_values = [:point, :interval, :rectangle, :hyperrectangle]
    relations_possible_values = [:IA, :IA3, :IA7, :RCC5, :RCC8]
    @assert world in world_possible_values "Unexpected value encountered for `world`: $(world). Legal values are in $(world_possible_values)"
    @assert (relations isa AbstractVector{<:Relation}) || relations in relations_possible_values "Unexpected value encountered for `relations`: $(relations). Legal values are in $(relations_possible_values)"

    if world in [:point]
        error("TODO point-based ontologies not implemented yet")
    elseif world in [:interval, :rectangle, :hyperrectangle]
        if relations isa AbstractVector{<:Relation}
            Ontology{Interval}(relations)
        elseif relations == :IA   IntervalOntology
        elseif relations == :IA3  Interval3Ontology
        elseif relations == :IA7  Interval7Ontology
        elseif relations == :RCC8 IntervalRCC8Ontology
        elseif relations == :RCC5 IntervalRCC5Ontology
        else
            error("Unexpected value encountered for `relations`: $(relations). Legal values are in $(relations_possible_values)")
        end
    else
        error("Unexpected value encountered for `world`: $(world). Legal values are in $(possible_values)")
    end
end

function get_ontology(::Val{2}, world = :interval, relations::Union{Symbol,AbstractVector{<:Relation}} = :IA)
    world_possible_values = [:point, :interval, :rectangle, :hyperrectangle]
    relations_possible_values = [:IA, :RCC5, :RCC8]
    @assert world in world_possible_values "Unexpected value encountered for `world`: $(world). Legal values are in $(world_possible_values)"
    @assert (relations isa AbstractVector{<:Relation}) || relations in relations_possible_values "Unexpected value encountered for `relations`: $(relations). Legal values are in $(relations_possible_values)"

    if world in [:point]
        error("TODO point-based ontologies not implemented yet")
    elseif world in [:interval, :rectangle, :hyperrectangle]
        if relations isa AbstractVector{<:Relation}
            Ontology{Interval2D}(relations)
        elseif relations == :IA   Interval2DOntology
        elseif relations == :RCC8 Interval2DRCC8Ontology
        elseif relations == :RCC5 Interval2DRCC5Ontology
        else
            error("Unexpected value encountered for `relations`: $(relations). Legal values are in $(relations_possible_values)")
        end
    else
        error("Unexpected value encountered for `world`: $(world). Legal values are in $(possible_values)")
    end
end

############################################################################################

get_interval_ontology(N::Integer, args...) = get_interval_ontology(Val(N), args...)
get_interval_ontology(N::Val, relations::Union{Symbol,AbstractVector{<:Relation}} = :IA) = get_ontology(N, :interval, relations)

############################################################################################
# Dimensionality: 0

export OneWorld

# World type definitions for the propositional case, where there exist only one world,
#  and `Decision`s only allow the RelationId.
include("worlds/OneWorld.jl")                 # <- OneWorld world type

const OneWorldOntology   = Ontology{OneWorld}(Relation[])

############################################################################################
# Dimensionality: 1

# World type definitions for punctual logics, where worlds are points
# include("worlds/PointWorld.jl")

# World type definitions for interval logics, where worlds are intervals
include("worlds/Interval.jl")                 # <- Interval world type
include("bindings/IA+Interval.jl")       # <- Allen relations
include("bindings/RCC+Interval.jl")    # <- RCC relations

const IntervalOntology       = Ontology{Interval}(IARelations)
const Interval3Ontology      = Ontology{ModalLogic.Interval}(ModalLogic.IA7Relations)
const Interval7Ontology      = Ontology{ModalLogic.Interval}(ModalLogic.IA3Relations)

const IntervalRCC8Ontology   = Ontology{Interval}(RCC8Relations)
const IntervalRCC5Ontology   = Ontology{Interval}(RCC5Relations)

############################################################################################
# Dimensionality: 2

# World type definitions for 2D iterval logics, where worlds are rectangles
#  parallel to a frame of reference.
include("worlds/Interval2D.jl")        # <- Interval2D world type
include("bindings/IA2+Interval2D.jl")  # <- Allen relations
include("bindings/RCC+Interval2D.jl")  # <- RCC relations

const Interval2DOntology     = Ontology{Interval2D}(IA2DRelations)
const Interval2DRCC8Ontology = Ontology{Interval2D}(RCC8Relations)
const Interval2DRCC5Ontology = Ontology{Interval2D}(RCC5Relations)

############################################################################################

# Default
const WorldType0D = Union{OneWorld}
const WorldType1D = Union{Interval}
const WorldType2D = Union{Interval2D}

get_ontology(::DimensionalDataset{T,D}, args...) where {T,D} = get_ontology(Val(D-2), args...)
get_interval_ontology(::DimensionalDataset{T,D}, args...) where {T,D} = get_interval_ontology(Val(D-2), args...)

############################################################################################
# World-specific featured world datasets and supports
############################################################################################

include("world-specific-fwds.jl")

############################################################################################
############################################################################################

end # module
