# TODO PointWorld

#   PointWorld(::EmptyWorld) = new(0)
#   PointWorld(::CenteredWorld, X::Integer) = new(div(X,2)+1)

# Note: with PointWorlds, < and >= is redundant

# _accessibles(w::PointWorld, ::_RelationId, XYZ::Vararg{Integer,N}) where N = [(w.x,)]
# accessibles(S::AbstractWorldSet{PointWorld}, r::_RelationGlob, X::Integer) =
#   IterTools.imap(PointWorld, 1:X)

# @inline ch_readWorld(w::PointWorld, channel::DimensionalChannel{T,1}) where {T} = channel[w.x]
