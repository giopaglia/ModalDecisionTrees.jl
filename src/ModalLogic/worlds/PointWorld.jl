# TODO PointWorld

# Note: with PointWorlds, < and >= is redundant

# accessibles(fr::..., S::AbstractWorldSet{PointWorld}, r::_RelationGlob) =
#   IterTools.imap(PointWorld, 1:X(fr))

# @inline ch_readWorld(w::PointWorld, channel::DimensionalChannel{T,1}) where {T} = channel[w.x]
