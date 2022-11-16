############################################################################################
# Dimensonal Worlds
############################################################################################

# Abstract type for dimensional worlds
abstract type DimensionalWorld <: World end

# Dimensional worlds can be interpreted on dimensional data of given sizes.
# The size is referred to as `dimensionality`, and must be specified for each newly defined 
#  world type via the following trait:
goes_with_dimensionality(W::Type{<:DimensionalWorld}, d::Integer) = goes_with_dimensionality(W, Val(d))
goes_with_dimensionality(::Type{<:DimensionalWorld}, ::Val) = false

############################################################################################

# One unique world (propositional case)
struct OneWorld    <: DimensionalWorld
    OneWorld() = new()
    # 
end;

Base.show(io::IO, w::OneWorld) = begin
    print(io, "−")
end

goes_with_dimensionality(::Type{OneWorld}, ::Val{0}) = true

############################################################################################

# struct PointWorld <: DimensionalWorld
    # PointWorld(w::PointWorld) = new(w.x,w.y)
#   x :: Integer
#   # TODO check x<=N but only in debug mode
#   # PointWorld(x) = x<=N ... ? new(x) : throw_n_log("Can't instantiate PointWorld(x={$x})")
#   PointWorld(x::Integer) = new(x)
# end

# show(io::IO, r::PointWorld) = print(io, "($(x)×$(y))")

# goes_with_dimensionality(::Type{PointWorld}, ::Val{1}) = true

############################################################################################

# An interval is a pair of natural numbers (x,y) where: i) x > 0; ii) y > 0; iii) x < y.
struct Interval <: DimensionalWorld
    x :: Integer
    y :: Integer
    # 
    Interval(x::Integer,y::Integer) = new(x,y)
    Interval(w::Interval) = Interval(w.x,w.y)
    # TODO: perhaps check x<y (and  x<=N, y<=N ?), but only in debug mode.
    # Interval(x,y) = x>0 && y>0 && x < y ? new(x,y) : throw_n_log("Can't instantiate Interval(x={$x},y={$y})")
    # 
end

Base.show(io::IO, w::Interval) = begin
    print(io, "(")
    print(io, w.x)
    print(io, "−")
    print(io, w.y)
    print(io, ")")
end

goes_with_dimensionality(::Type{Interval}, ::Val{1}) = true

############################################################################################

# 2-dimensional Interval counterpart: combination of two orthogonal Intervals
struct Interval2D <: DimensionalWorld
    x :: Interval
    y :: Interval
    # 
    Interval2D(x::Interval,y::Interval) = new(x,y)
    Interval2D(w::Interval2D) = Interval2D(w.x,w.y)
    Interval2D(x::Tuple{Integer,Integer}, y::Tuple{Integer,Integer}) = Interval2D(Interval(x),Interval(y))
    # 
end

Base.show(io::IO, w::Interval2D) = begin
    print(io, "(")
    print(io, w.x)
    print(io, "×")
    print(io, w.y)
    print(io, ")")
end

goes_with_dimensionality(::Type{Interval2D}, ::Val{2}) = true
