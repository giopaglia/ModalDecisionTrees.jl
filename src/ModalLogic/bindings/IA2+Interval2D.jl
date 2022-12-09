goeswith(::Type{Interval2D}, ::RectangleRelation) = true

# Convenience function
_accessibles__(w::Interval, r::IntervalRelation, X::Integer) = _accessibles(w,r,X)
_accessibles__(w::Interval, r::_RelationId, args...) = [(w.x, w.y)]
_accessibles__(w::Interval, r::_RelationGlob, X::Integer) = _intervals_in(1, X+1)

# Accessibles are easily coded using methods for one-dimensional interval logic
_accessibles(w::Interval2D, r::RectangleRelation, X::Integer, Y::Integer) =
    Iterators.product(_accessibles__(w.x, r.x, X), _accessibles__(w.y, r.y, Y))

# TODO write More efficient implementations for edge cases
# Example for _IA2D_URelations:
# accessibles(S::AbstractWorldSet{Interval2D}, r::_IA2D_URelations, X::Integer, Y::Integer) = begin
#   IterTools.imap(Interval2D,
#       Iterators.flatten(
#           Iterators.product((accessibles(w, r.x, X) for w in S), accessibles(S, r, Y))
#       )
#   )
# end
