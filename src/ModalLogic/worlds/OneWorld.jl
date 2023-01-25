# Dimensional world type: it can be interpreted on dimensional instances.
interpret_world(::OneWorld, instance::DimensionalInstance{T,1}) where {T} = instance

accessibles(fr::Full0DFrame, ::Union{OneWorld,AbstractWorldSet{OneWorld}}, ::_RelationGlob) = [OneWorld()]

accessibles_aggr(fr::Full0DFrame, f::AbstractFeature, a::Aggregator, ::AbstractWorldSet{OneWorld}, ::ModalLogic._RelationGlob) = [OneWorld()]

# Perhaps these help the compiler? TODO figure out if these are needed
all_worlds(fr::Full0DFrame) = [OneWorld()]
all_worlds(fr::Full0DFrame, enumAccFun::Function) = [OneWorld()]
all_worlds_aggr(fr::Full0DFrame, enumReprFun::Function, f::AbstractFeature, a::Aggregator) = [OneWorld()]

# _accessibles(fr::Full0DFrame, ::OneWorld, ::_RelationId) = [OneWorld()]
