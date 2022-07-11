
# 2-dimensional Interval counterpart: combination of two orthogonal Intervals
struct Interval2D <: World
    x :: Interval
    y :: Interval
    # 
    Interval2D(x::Interval,y::Interval) = new(x,y)
    Interval2D(w::Interval2D) = Interval2D(w.x,w.y)
    Interval2D(x::Tuple{Integer,Integer}, y::Tuple{Integer,Integer}) = Interval2D(Interval(x),Interval(y))
    # 
    Interval2D(w::EmptyWorld) = Interval2D(Interval(w),Interval(w))
    Interval2D(w::CenteredWorld, X::Integer, Y::Integer) = Interval2D(Interval(w,X),Interval(w,Y))
end

Base.show(io::IO, w::Interval2D) = begin
    print(io, "(")
    print(io, w.x)
    print(io, "×")
    print(io, w.y)
    print(io, ")")
end

dimensionality(::Type{Interval2D}) = 2
# n_worlds(::Type{Interval2D}, X::Integer, Y::Integer) = n_worlds(Interval, X) * n_worlds(Interval, Y)

# Dimensional world type: it can be interpreted on dimensional instances.
interpret_world(w::Interval2D, instance::DimensionalInstance{T,3}) where {T} = instance[w.x.x:w.x.y-1,w.y.x:w.y.y-1,:]

# Convenience function: enumerate all interval2Ds in a given range
intervals2D_in(a1::Integer, a2::Integer, b1::Integer, b2::Integer) = IterTools.imap(Interval2D, Iterators.product(_intervals_in(a1, a2), _intervals_in(b1, b2)))

accessibles(::Union{Interval2D,AbstractWorldSet{Interval2D}}, r::_RelationGlob, X::Integer, Y::Integer) =
    intervals2D_in(1,X+1,1,Y+1)

accessibles_aggr(f::Union{SingleAttributeMin,SingleAttributeMax}, a::Union{typeof(minimum),typeof(maximum)}, ::AbstractWorldSet{Interval2D}, r::_RelationGlob,  X::Integer,  Y::Integer) = intervals2D_in(1,X+1,1,Y+1)
accessibles_aggr(f::Union{SingleAttributeMax}, a::typeof(maximum), ::AbstractWorldSet{Interval2D}, r::_RelationGlob,  X::Integer,  Y::Integer) = Interval2D[Interval2D(Interval(1,X+1), Interval(1,Y+1))  ]
accessibles_aggr(f::Union{SingleAttributeMin}, a::typeof(minimum), ::AbstractWorldSet{Interval2D}, r::_RelationGlob,  X::Integer,  Y::Integer) = Interval2D[Interval2D(Interval(1,X+1), Interval(1,Y+1))  ]

accessibles_aggr(f::Union{SingleAttributeSoftMin,SingleAttributeSoftMax}, a::Union{typeof(minimum),typeof(maximum)}, ::AbstractWorldSet{Interval2D}, r::_RelationGlob,  X::Integer,  Y::Integer) = intervals2D_in(1,X+1,1,Y+1)
accessibles_aggr(f::Union{SingleAttributeSoftMax}, a::typeof(maximum), ::AbstractWorldSet{Interval2D}, r::_RelationGlob,  X::Integer,  Y::Integer) = Interval2D[Interval2D(Interval(1,X+1), Interval(1,Y+1))  ]
accessibles_aggr(f::Union{SingleAttributeSoftMin}, a::typeof(minimum), ::AbstractWorldSet{Interval2D}, r::_RelationGlob,  X::Integer,  Y::Integer) = Interval2D[Interval2D(Interval(1,X+1), Interval(1,Y+1))  ]

# _accessibles(w::Interval2D, ::_RelationId, args...) = [(w.x, w.y)]
