############################################################################################

# Abstract types for worlds
abstract type World end

include("logics/dimensional-worlds.jl")

############################################################################################

# Abstract types for relations
abstract type Relation end

# Relations must indicate their compatible world types via `goes_with`.
#  For example, if world type W is compatible with relation R
# goes_with(::Type{W}, ::R) = true
# Here's the fallback:
goes_with(::Type{W}, ::Relation) where {W<:World} = false

# Relations can be symmetric, reflexive and/or transitive.
# By default, none of this cases holds:
is_symmetric(r::Relation) = false
is_reflexive(r::Relation) = false
is_transitive(r::Relation) = false

# TODO add are_inverse_relation trait

############################################################################################
# Singletons representing natural relations
############################################################################################

# Identity relation: any world -> itself
struct _RelationId    <: Relation end; const RelationId   = _RelationId();

Base.show(io::IO, ::_RelationId) = print(io, "=")

is_symmetric(r::_RelationId) = true
is_reflexive(r::_RelationId) = true
is_transitive(r::_RelationId) = true

############################################################################################

# Global relation:  any world -> all worlds
struct _RelationGlob   <: Relation end; const RelationGlob  = _RelationGlob();

Base.show(io::IO, ::_RelationGlob) = print(io, "G")

is_symmetric(r::_RelationGlob) = true
is_reflexive(r::_RelationGlob) = true
is_transitive(r::_RelationGlob) = true

############################################################################################

# Abstract type for relations with a geometrical interpretation
abstract type GeometricalRelation <: Relation end

# Geometrical relations can have geometrical properties such as being topological (i.e.,
#  invariant under homeomorphisms. # see https://en.m.wikipedia.org/wiki/Topological_property
# By default, this does not hold:
is_topological(r::GeometricalRelation) = false

# 1D Allen relations
include("logics/IA.jl")

# 2D Allen relations
include("logics/IA2.jl")

# RCC relations
include("logics/RCC.jl")
############################################################################################

# TODO kripke frames represented by graphs of "named" worlds with labelled, "named" relations

# # Named-world type
# struct NamedWorld <: World end
#     name::Symbol
# end

# # Named-relation type
# struct NamedRelation <: Relation end
#     name::Symbol
#     adjacency matrix between NamedWorld's
# end
