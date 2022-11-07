# NOTE: removed
#=
export OneWorld

# One unique world (propositional case)
struct OneWorld    <: World
    OneWorld() = new()
    #
    OneWorld(w::EmptyWorld) = new()
    OneWorld(w::CenteredWorld, args...) = new()
end;

Base.show(io::IO, w::OneWorld) = begin
    print(io, "−")
end

dimensionality(::Type{OneWorld}) = 0
=#

# Dimensional world type: it can be interpreted on dimensional instances.
interpret_world(::OneWorld, instance::DimensionalInstance{T,1}) where {T} = instance

accessibles(::Union{OneWorld,AbstractWorldSet{OneWorld}}, ::_RelationGlob, args...) = [OneWorld()]

accessibles_aggr(f::ModalFeature, a::Aggregator, ::AbstractWorldSet{OneWorld}, ::ModalLogic._RelationGlob) = [OneWorld()]

# Perhaps these help the compiler? TODO figure out if these are needed
all_worlds(::Type{OneWorld}, args::Vararg) = [OneWorld()]
all_worlds(::Type{OneWorld}, enumAccFun::Function) = [OneWorld()]
all_worlds_aggr(::Type{OneWorld}, enumReprFun::Function, f::ModalFeature, a::Aggregator) = [OneWorld()]

# _accessibles(::OneWorld, ::_RelationId, args...) = [OneWorld()]
