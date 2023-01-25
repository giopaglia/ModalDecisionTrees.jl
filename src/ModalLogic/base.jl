
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
# Each relation R<:AbstractRelation must provide a method for `accessibles`, which returns an iterator
#  to the worlds that are accessible from a given world w:
# `accessibles(w::W,           ::R, args...)`

# Alternatively, one can provide a *bare* definition, that is, method `_accessibles`,
#  returning an iterator of *tuples* which is then fed to a constructor of the same world type, as in:
# `_accessibles(w::W,           ::R, args...)`

# The following fallback ensures that the two definitions are equivalent
accessibles(w::WorldType, r::AbstractRelation, args...) where {WorldType<:AbstractWorld} = begin
    IterTools.imap(WorldType, _accessibles(w, r, args...))
end

#

# It is convenient to define methods for `accessibles` that take a world set instead of a
#  single world. Generally, this falls back to calling `_accessibles` on each world in
#  the set, and returning a constructor of wolds from the union; however, one may provide
#  improved implementations for special cases (e.g. ⟨L⟩ of a world set in interval algebra).
accessibles(S::AbstractWorldSet{WorldType}, r::AbstractRelation, args...) where {WorldType<:AbstractWorld} = begin
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
# accessibles_aggr(f::AbstractFeature, a::Aggregator, S::AbstractWorldSet{W}, ::R, args...)
# Of course, the fallback is enumerating all accessible worlds via `accessibles`
accessibles_aggr(::AbstractFeature, ::Aggregator, w::WorldType, r::AbstractRelation, args...) where {WorldType<:AbstractWorld} = accessibles(w, r, args...)

############################################################################################
# Singletons representing natural relations
############################################################################################

accessibles(w::WorldType,           ::_RelationId, args...) where {WorldType<:AbstractWorld} = [w] # TODO try IterTools.imap(identity, [w])
accessibles(S::AbstractWorldSet{W}, ::_RelationId, args...) where {W<:AbstractWorld} = S # TODO try IterTools.imap(identity, S)

accessibles_aggr(::AbstractFeature, ::Aggregator, w::WorldType, r::_RelationId,      args...) where {WorldType<:AbstractWorld} = accessibles(w, r, args...)

############################################################################################

# Note: these methods must be defined for any newly defined world type WT:
# `accessibles(w::WT,           ::_RelationGlob, args...)`
# `accessibles(S::AbstractWorldSet{WT}, ::_RelationGlob, args...)`
# `accessibles_aggr(f::AbstractFeature, a::Aggregator, S::AbstractWorldSet{WT}, ::_RelationGlob, args...)`

############################################################################################

# Shortcuts using global relation for enumerating all worlds
all_worlds(::Type{WorldType}, args...) where {WorldType<:AbstractWorld} = accessibles(WorldType[], RelationGlob, args...)
all_worlds(::Type{WorldType}, enum_acc_fun::Function) where {WorldType<:AbstractWorld} = enum_acc_fun(WorldType[], RelationGlob)
all_worlds_aggr(::Type{WorldType}, enum_repr_fun::Function, f::AbstractFeature, a::Aggregator) where {WorldType<:AbstractWorld} = enum_repr_fun(f, a, WorldType[], RelationGlob)
