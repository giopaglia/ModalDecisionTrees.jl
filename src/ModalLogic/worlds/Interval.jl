# n_worlds(::Type{Interval}, X::Integer) = div(X*(X+1),2)

# Dimensional world type: it can be interpreted on dimensional instances.
interpret_world(w::Interval, instance::DimensionalInstance{T,2}) where {T} = instance[w.x:w.y-1,:]

# Convenience functions: enumerate all & 1-length intervals in a given range
_intervals_in(a::Integer, b::Integer) = Iterators.filter(((x,y),)->x<y, Iterators.product(a:b-1, a+1:b))
intervals_in(a::Integer, b::Integer) = IterTools.imap(Interval, _intervals_in(a, b))
short_intervals_in(a::Integer, b::Integer) = IterTools.imap((x)->Interval(x,x+1), a:b-1)

accessibles(::Union{Interval,AbstractWorldSet{Interval}}, r::_RelationGlob, X::Integer) = intervals_in(1, X+1)

accessibles_aggr(f::ModalFeature, a::TestOperatorFun, ::AbstractWorldSet{Interval}, r::_RelationGlob,  X::Integer) = intervals_in(1, X+1)

accessibles_aggr(f::Union{SingleAttributeMin,SingleAttributeMax}, a::Union{typeof(minimum),typeof(maximum)}, ::AbstractWorldSet{Interval}, r::_RelationGlob,  X::Integer) = short_intervals_in(1, X+1)
accessibles_aggr(f::Union{SingleAttributeMax}, a::typeof(maximum), ::AbstractWorldSet{Interval}, r::_RelationGlob,  X::Integer) = Interval[Interval(1, X+1)  ]
accessibles_aggr(f::Union{SingleAttributeMin}, a::typeof(minimum), ::AbstractWorldSet{Interval}, r::_RelationGlob,  X::Integer) = Interval[Interval(1, X+1)  ]

accessibles_aggr(f::Union{SingleAttributeSoftMin,SingleAttributeSoftMax}, a::Union{typeof(minimum),typeof(maximum)}, ::AbstractWorldSet{Interval}, r::_RelationGlob,  X::Integer) = short_intervals_in(1, X+1)
accessibles_aggr(f::Union{SingleAttributeSoftMax}, a::typeof(maximum), ::AbstractWorldSet{Interval}, r::_RelationGlob,  X::Integer) = Interval[Interval(1, X+1)  ]
accessibles_aggr(f::Union{SingleAttributeSoftMin}, a::typeof(minimum), ::AbstractWorldSet{Interval}, r::_RelationGlob,  X::Integer) = Interval[Interval(1, X+1)  ]

# TODO remove:
# Note: only needed for a smooth definition of IA2DRelations
# _accessibles(w::Interval, ::_RelationId, args...) = [(w.x, w.y)]
