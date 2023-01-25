############################################################################################
# Ontologies
############################################################################################

# Here are the definitions for world types and relations for known modal logics
#
export get_ontology,
       get_interval_ontology

get_ontology(N::Integer, args...) = get_ontology(Val(N), args...)
get_ontology(::Val{0}, args...) = OneWorldOntology
function get_ontology(::Val{1}, world = :interval, relations::Union{Symbol,AbstractVector{<:AbstractRelation}} = :IA)
    world_possible_values = [:point, :interval, :rectangle, :hyperrectangle]
    relations_possible_values = [:IA, :IA3, :IA7, :RCC5, :RCC8]
    @assert world in world_possible_values "Unexpected value encountered for `world`: $(world). Legal values are in $(world_possible_values)"
    @assert (relations isa AbstractVector{<:AbstractRelation}) || relations in relations_possible_values "Unexpected value encountered for `relations`: $(relations). Legal values are in $(relations_possible_values)"

    if world in [:point]
        error("TODO point-based ontologies not implemented yet")
    elseif world in [:interval, :rectangle, :hyperrectangle]
        if relations isa AbstractVector{<:AbstractRelation}
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

function get_ontology(::Val{2}, world = :interval, relations::Union{Symbol,AbstractVector{<:AbstractRelation}} = :IA)
    world_possible_values = [:point, :interval, :rectangle, :hyperrectangle]
    relations_possible_values = [:IA, :RCC5, :RCC8]
    @assert world in world_possible_values "Unexpected value encountered for `world`: $(world). Legal values are in $(world_possible_values)"
    @assert (relations isa AbstractVector{<:AbstractRelation}) || relations in relations_possible_values "Unexpected value encountered for `relations`: $(relations). Legal values are in $(relations_possible_values)"

    if world in [:point]
        error("TODO point-based ontologies not implemented yet")
    elseif world in [:interval, :rectangle, :hyperrectangle]
        if relations isa AbstractVector{<:AbstractRelation}
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
get_interval_ontology(N::Val, relations::Union{Symbol,AbstractVector{<:AbstractRelation}} = :IA) = get_ontology(N, :interval, relations)

############################################################################################
# Dimensionality: 0

export OneWorld

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
