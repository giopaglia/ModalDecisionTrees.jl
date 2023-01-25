goeswith(::Type{Interval2D}, ::RectangleRelation) = true

# Convenience function
_accessibles__(fr::Full1DFrame, w::Interval, r::IntervalRelation) = _accessibles(fr,w,r)
_accessibles__(fr::Full1DFrame, w::Interval, r::_RelationId, args...) = [(w.x, w.y)]
_accessibles__(fr::Full1DFrame, w::Interval, r::_RelationGlob) = _intervals_in(1, X(fr)+1)

# Accessibles are easily coded using methods for one-dimensional interval logic
_accessibles(fr::Full2DFrame, w::Interval2D, r::RectangleRelation) =
    Iterators.product(_accessibles__(fr(X), w.x, r.x), _accessibles__(fr(Y), w.y, r.y))

# TODO write More efficient implementations for edge cases
# Example for _IA2D_URelations:
# accessibles(fr::Full2DFrame, S::AbstractWorldSet{Interval2D}, r::_IA2D_URelations) = begin
#   IterTools.imap(Interval2D,
#       Iterators.flatten(
#           Iterators.product((accessibles(FullDimensionalFrame(X(fr)), w, r.x) for w in S), accessibles(FullDimensionalFrame(Y(fr)), S, r))
#       )
#   )
# end
