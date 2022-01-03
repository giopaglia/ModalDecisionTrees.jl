export OneWorld

# One unique world (propositional case)
struct OneWorld    <: AbstractWorld
    OneWorld() = new()
    OneWorld(w::_emptyWorld) = new()
    OneWorld(w::_firstWorld) = new()
    OneWorld(w::_centeredWorld) = new()
end;

show(io::IO, w::OneWorld) = begin
    print(io, "−")
end

worldTypeDimensionality(::Type{OneWorld}) = 0
print_world(::OneWorld) = println("−")

inst_readWorld(::OneWorld, instance::MatricialInstance{T,1}) where {T} = instance

enumAccBare(::OneWorld, ::AbstractRelation, XYZ::Vararg{Integer}) = throw_n_log("Can't access any world via any relation other than RelationId from a OneWorld")
enumAccBare(::OneWorld, ::_RelationId, XYZ::Vararg{Integer}) = [OneWorld()]
enumAccessibles(::OneWorld, ::_RelationGlob, XYZ::Vararg{Integer}) = [OneWorld()]

enumAll(::Type{OneWorld}, args::Vararg) = [OneWorld()]
enumAll(::Type{OneWorld}, enumAccFun::Function) = [OneWorld()]
enumReprAll(::Type{OneWorld}, enumReprFun::Function, f::FeatureTypeFun, a::Aggregator) = [OneWorld()]


enumAccReprAggr(f::FeatureTypeFun, a::Aggregator, ::Vector{OneWorld}, ::DecisionTree.ModalLogic._RelationGlob) = [OneWorld()]
