# TODO PointWorld
# ################################################################################
# # BEGIN PointWorld
# ################################################################################

# struct PointWorld <: AbstractWorld
    # PointWorld(w::PointWorld) = new(w.x,w.y)
#   x :: Integer
#   # TODO check x<=N but only in debug mode
#   # PointWorld(x) = x<=N ... ? new(x) : throw_n_log("Can't instantiate PointWorld(x={$x})")
#   PointWorld(x::Integer) = new(x)
#   PointWorld(::EmptyWorld) = new(0)
#   PointWorld(::CenteredWorld, X::Integer) = new(div(X,2)+1)
# end

# Note: with PointWorlds, < and >= is redundant

# show(io::IO, r::Interval) = print(io, "($(x)×$(y))")

# _accessibles(w::PointWorld, ::_RelationId, XYZ::Vararg{Integer,N}) where N = [(w.x,)]
# accessibles(S::AbstractWorldSet{PointWorld}, r::_RelationGlob, X::Integer) =
#   IterTools.imap(PointWorld, 1:X)

# @inline ch_readWorld(w::PointWorld, channel::DimensionalChannel{T,1}) where {T} = channel[w.x]

# ################################################################################
# # END PointWorld
# ################################################################################
