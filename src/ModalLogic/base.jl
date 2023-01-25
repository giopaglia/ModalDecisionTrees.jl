using SoleLogics: AbstractMultiModalFrame, TruthValue

abstract type DimensionalFrame{N,W<:AbstractWorld,T<:TruthValue,NR,Rs<:NTuple{NR,R where R<:AbstractRelation}} <: AbstractMultiModalFrame{W,T,NR,Rs} end

struct FullDimensionalFrame{N,W<:AbstractWorld,T<:TruthValue,NR,Rs<:NTuple{NR,R where R<:AbstractRelation}} <: DimensionalFrame{N,W,T,NR,Rs}
    dims::NTuple{N,Int}
    function FullDimensionalFrame{N,W,T,NR,Rs}(dims::NTuple{N,Int}) where
            {N,W<:AbstractWorld,T<:TruthValue,NR,Rs<:NTuple{NR,R where R<:AbstractRelation}}
        new{N,W,T,NR,Rs}(dims)
    end
    function FullDimensionalFrame{N,W,T}(dims::NTuple{N,Int}) where
            {N,W<:AbstractWorld,T<:TruthValue}
        FullDimensionalFrame{N,W,T,0,Tuple{}}(dims)
    end
    FullDimensionalFrame(dims::Tuple{}) = FullDimensionalFrame{0,OneWorld,Bool}(dims)
    FullDimensionalFrame(dims::Tuple{Int}) = FullDimensionalFrame{1,Interval,Bool}(dims)
    FullDimensionalFrame(dims::Tuple{Int,Int}) = FullDimensionalFrame{2,Interval2D,Bool}(dims)
end

Base.getindex(fr::FullDimensionalFrame, i::Int) = fr.dims[i]

# Shorthands
X(fr::FullDimensionalFrame) = fr[1]
Y(fr::FullDimensionalFrame) = fr[2]
Z(fr::FullDimensionalFrame) = fr[3]

nworlds(fr::FullDimensionalFrame{1}) = div(X(fr)*(X(fr)+1),2)
nworlds(fr::FullDimensionalFrame{2}) = div(X(fr)*(X(fr)+1),2) * div(Y(fr)*(Y(fr)+1),2)
nworlds(fr::FullDimensionalFrame{3}) = div(X(fr)*(X(fr)+1),2) * div(Y(fr)*(Y(fr)+1),2) * div(Z(fr)*(Z(fr)+1),2)

const Full0DFrame = FullDimensionalFrame{0,OneWorld,Bool}
const Full1DFrame = FullDimensionalFrame{1,Interval,Bool}
const Full2DFrame = FullDimensionalFrame{2,Interval2D,Bool}

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
# `accessibles(fr::AbstractMultiModalFrame{W}, w::W,           r::R)`

# Alternatively, one can provide a *bare* definition, that is, method `_accessibles`,
#  returning an iterator of *tuples* which is then fed to a constructor of the same world type, as in:
# `_accessibles(fr::AbstractMultiModalFrame{W}, w::W,           r::R)`

# The following fallback ensures that the two definitions are equivalent
accessibles(fr::AbstractMultiModalFrame{W}, w::W, r::AbstractRelation) where {W<:AbstractWorld} = begin
    IterTools.imap(W, _accessibles(fr, w, r))
end

#

# It is convenient to define methods for `accessibles` that take a world set instead of a
#  single world. Generally, this falls back to calling `_accessibles` on each world in
#  the set, and returning a constructor of wolds from the union; however, one may provide
#  improved implementations for special cases (e.g. ⟨L⟩ of a world set in interval algebra).
accessibles(fr::AbstractMultiModalFrame{W}, S::AbstractWorldSet{W}, r::AbstractRelation) where {W<:AbstractWorld} = begin
    IterTools.imap(W,
        IterTools.distinct(
            Iterators.flatten(
                (_accessibles(fr, w, r) for w in S)
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
# accessibles_aggr(fr::AbstractMultiModalFrame{W}, f::AbstractFeature, a::Aggregator, S::AbstractWorldSet{W}, ::R)
# Of course, the fallback is enumerating all accessible worlds via `accessibles`
accessibles_aggr(fr::AbstractMultiModalFrame{W}, ::AbstractFeature, ::Aggregator, w::W, r::AbstractRelation) where {W<:AbstractWorld} = accessibles(fr, w, r)

############################################################################################
# Singletons representing natural relations
############################################################################################

accessibles(fr::AbstractMultiModalFrame{W}, w::W,           ::_RelationId) where {W<:AbstractWorld} = [w] # TODO try IterTools.imap(identity, [w])
accessibles(fr::AbstractMultiModalFrame{W}, S::AbstractWorldSet{W}, ::_RelationId) where {W<:AbstractWorld} = S # TODO try IterTools.imap(identity, S)

accessibles_aggr(fr::AbstractMultiModalFrame{W}, ::AbstractFeature, ::Aggregator, w::W, r::_RelationId) where {W<:AbstractWorld} =
    accessibles(fr, w, r)

############################################################################################

# Note: these methods must be defined for any newly defined world type WT:
# `accessibles(fr::AbstractMultiModalFrame{W}, w::WT,           ::_RelationGlob)`
# `accessibles(fr::AbstractMultiModalFrame{W}, S::AbstractWorldSet{WT}, ::_RelationGlob)`
# `accessibles_aggr(fr::AbstractMultiModalFrame{W}, f::AbstractFeature, a::Aggregator, S::AbstractWorldSet{WT}, ::_RelationGlob)`

############################################################################################

# Shortcuts using global relation for enumerating all worlds
all_worlds(fr::AbstractMultiModalFrame{W}) where {W<:AbstractWorld} = accessibles(fr, W[], RelationGlob)
all_worlds(::Type{W}, accessible_fun::Function) where {W<:AbstractWorld} = accessible_fun(W[], RelationGlob)
all_worlds_aggr(::Type{W}, accessibles_aggr_fun::Function, f::AbstractFeature, a::Aggregator) where {W<:AbstractWorld} = accessibles_aggr_fun(f, a, W[], RelationGlob)
