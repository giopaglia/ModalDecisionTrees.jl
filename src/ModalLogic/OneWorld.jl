export OneWorld

# One unique world (propositional case)
struct OneWorld    <: AbstractWorld
    OneWorld() = new()
    OneWorld(w::_emptyWorld) = new()
    OneWorld(w::_firstWorld) = new()
    OneWorld(w::_centeredWorld) = new()
end;

Base.show(io::IO, w::OneWorld) = begin
    print(io, "−")
end

worldTypeDimensionality(::Type{OneWorld}) = 0
print_world(::OneWorld) = println("−")

inst_readWorld(::OneWorld, instance::MatricialInstance{T,1}) where {T} = instance

_accessibles(::OneWorld, ::AbstractRelation, XYZ::Vararg{Integer}) = throw_n_log("Can't access any world via any relation other than RelationId from a OneWorld")
_accessibles(::OneWorld, ::_RelationId, XYZ::Vararg{Integer}) = [OneWorld()]
accessibles(::OneWorld, ::_RelationGlob, XYZ::Vararg{Integer}) = [OneWorld()]

# TODO remove these:
all_worlds(::Type{OneWorld}, args::Vararg) = [OneWorld()]
all_worlds(::Type{OneWorld}, enumAccFun::Function) = [OneWorld()]
all_worlds_aggr(::Type{OneWorld}, enumReprFun::Function, f::ModalFeature, a::Aggregator) = [OneWorld()]


accessibles_aggr(f::ModalFeature, a::Aggregator, ::Vector{OneWorld}, ::ModalLogic._RelationGlob) = [OneWorld()]
