# Dimensional world type: it can be interpreted on dimensional instances.
interpret_world(::OneWorld, instance::DimensionalInstance{T,1}) where {T} = instance

accessibles(::Union{OneWorld,AbstractWorldSet{OneWorld}}, ::_RelationGlob, args...) = [OneWorld()]

accessibles_aggr(f::AbstractFeature, a::Aggregator, ::AbstractWorldSet{OneWorld}, ::ModalLogic._RelationGlob) = [OneWorld()]

# Perhaps these help the compiler? TODO figure out if these are needed
all_worlds(::Type{OneWorld}, args::Vararg) = [OneWorld()]
all_worlds(::Type{OneWorld}, enumAccFun::Function) = [OneWorld()]
all_worlds_aggr(::Type{OneWorld}, enumReprFun::Function, f::AbstractFeature, a::Aggregator) = [OneWorld()]

# _accessibles(::OneWorld, ::_RelationId, args...) = [OneWorld()]
