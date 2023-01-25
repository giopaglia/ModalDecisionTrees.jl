
# Dimensional world type: it can be interpreted on dimensional instances.
interpret_world(w::Interval2D, instance::DimensionalInstance{T,3}) where {T} = instance[w.x.x:w.x.y-1,w.y.x:w.y.y-1,:]

# Convenience function: enumerate all interval2Ds in a given range
intervals2D_in(a1::Integer, a2::Integer, b1::Integer, b2::Integer) = IterTools.imap(Interval2D, Iterators.product(_intervals_in(a1, a2), _intervals_in(b1, b2)))

accessibles(fr::Full2DFrame, ::Union{Interval2D,AbstractWorldSet{Interval2D}}, r::_RelationGlob) =
    intervals2D_in(1,X(fr)+1,1,Y(fr)+1)

accessibles_aggr(fr::Full2DFrame, f::Union{SingleAttributeMin,SingleAttributeMax}, a::Union{typeof(minimum),typeof(maximum)}, ::AbstractWorldSet{Interval2D}, r::_RelationGlob) = intervals2D_in(1,X(fr)+1,1,Y(fr)+1)
accessibles_aggr(fr::Full2DFrame, f::Union{SingleAttributeMax}, a::typeof(maximum), ::AbstractWorldSet{Interval2D}, r::_RelationGlob) = Interval2D[Interval2D(Interval(1,X(fr)+1), Interval(1,Y(fr)+1))  ]
accessibles_aggr(fr::Full2DFrame, f::Union{SingleAttributeMin}, a::typeof(minimum), ::AbstractWorldSet{Interval2D}, r::_RelationGlob) = Interval2D[Interval2D(Interval(1,X(fr)+1), Interval(1,Y(fr)+1))  ]

accessibles_aggr(fr::Full2DFrame, f::Union{SingleAttributeSoftMin,SingleAttributeSoftMax}, a::Union{typeof(minimum),typeof(maximum)}, ::AbstractWorldSet{Interval2D}, r::_RelationGlob) = intervals2D_in(1,X(fr)+1,1,Y(fr)+1)
accessibles_aggr(fr::Full2DFrame, f::Union{SingleAttributeSoftMax}, a::typeof(maximum), ::AbstractWorldSet{Interval2D}, r::_RelationGlob) = Interval2D[Interval2D(Interval(1,X(fr)+1), Interval(1,Y(fr)+1))  ]
accessibles_aggr(fr::Full2DFrame, f::Union{SingleAttributeSoftMin}, a::typeof(minimum), ::AbstractWorldSet{Interval2D}, r::_RelationGlob) = Interval2D[Interval2D(Interval(1,X(fr)+1), Interval(1,Y(fr)+1))  ]
