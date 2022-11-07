
############################################################################################
# Allen's Interval Algebra relations + definitions for Interval
############################################################################################

# Interval directional relations
# abstract type IntervalRelation <: DirectionalRelation end # NOTE: removed

goes_with(::Type{Interval}, ::IntervalRelation) = true

############################################################################################
# Interval algebra comprehends 12 relations (plus equality, i.e. RelationId):
#  - the 6 relations After, Later, Begins, Ends, During, Overlaps
#  - their inverses
############################################################################################
# Graphical representation of R((x,y),(z,t)) for R ∈ {After, Later, Begins, Ends, During, Overlaps}:
#
#                       x                   y
#                       |-------------------|
#                       .                   .
#                       .                   z        t            y = z
# After       (A)       .                   |--------|
#                       .                   .
#                       .                   .   z         t       y < z
# Later       (L)       .                   .   |---------|
#                       .                   .
#                       z     t             .                     x = z, t < y
# Begins      (B)       |-----|             .
#                       .                   .
#                       .             z     t                     y = t, x < z
# Ends        (E)       .             |-----|
#                       .                   .
#                       .   z        t      .                     x < z, t < y
# During      (D)       .   |--------|      .
#                       .                   .
#                       .           z       .    t                x < z < y < t
# Overlaps    (O)       .           |------------|
#
############################################################################################

# NOTE: remove begin
#= struct _IA_A  <: IntervalRelation end; const IA_A  = _IA_A();  # After
struct _IA_L  <: IntervalRelation end; const IA_L  = _IA_L();  # Later
struct _IA_B  <: IntervalRelation end; const IA_B  = _IA_B();  # Begins
struct _IA_E  <: IntervalRelation end; const IA_E  = _IA_E();  # Ends
struct _IA_D  <: IntervalRelation end; const IA_D  = _IA_D();  # During
struct _IA_O  <: IntervalRelation end; const IA_O  = _IA_O();  # Overlaps

struct _IA_Ai <: IntervalRelation end; const IA_Ai = _IA_Ai(); # After inverse
struct _IA_Li <: IntervalRelation end; const IA_Li = _IA_Li(); # Later inverse
struct _IA_Bi <: IntervalRelation end; const IA_Bi = _IA_Bi(); # Begins inverse
struct _IA_Ei <: IntervalRelation end; const IA_Ei = _IA_Ei(); # Ends inverse
struct _IA_Di <: IntervalRelation end; const IA_Di = _IA_Di(); # During inverse
struct _IA_Oi <: IntervalRelation end; const IA_Oi = _IA_Oi(); # Overlaps inverse

Base.show(io::IO, ::_IA_A)  = print(io, "A")
Base.show(io::IO, ::_IA_L)  = print(io, "L")
Base.show(io::IO, ::_IA_B)  = print(io, "B")
Base.show(io::IO, ::_IA_E)  = print(io, "E")
Base.show(io::IO, ::_IA_D)  = print(io, "D")
Base.show(io::IO, ::_IA_O)  = print(io, "O")
Base.show(io::IO, ::_IA_Ai) = print(io, "A̅")
Base.show(io::IO, ::_IA_Li) = print(io, "L̅")
Base.show(io::IO, ::_IA_Bi) = print(io, "B̅")
Base.show(io::IO, ::_IA_Ei) = print(io, "E̅")
Base.show(io::IO, ::_IA_Di) = print(io, "D̅")
Base.show(io::IO, ::_IA_Oi) = print(io, "O̅") =#
# NOTE: remove end

############################################################################################

# NOTE: remove begin
# Coarser relations: IA7
#= abstract type _IA7Rel <: IntervalRelation  end
struct _IA_AorO       <: _IA7Rel end; const IA_AorO       = _IA_AorO();       # After ∪ Overlaps
struct _IA_DorBorE    <: _IA7Rel end; const IA_DorBorE    = _IA_DorBorE();    # During ∪ Begins ∪ Ends
struct _IA_AiorOi     <: _IA7Rel end; const IA_AiorOi     = _IA_AiorOi();     # (After ∪ Overlaps) inverse
struct _IA_DiorBiorEi <: _IA7Rel end; const IA_DiorBiorEi = _IA_DiorBiorEi(); # (During ∪ Begins ∪ Ends) inverse

# Even coarser relations: IA3
abstract type _IA3Rel <: IntervalRelation  end
struct _IA_I          <: _IA3Rel end; const IA_I          = _IA_I();   # Intersecting (ABEDO ∪ ABEDO inverse)

IA72IARelations(::_IA_AorO)       = [IA_A,  IA_O]
IA72IARelations(::_IA_AiorOi)     = [IA_Ai, IA_Oi]
IA72IARelations(::_IA_DorBorE)    = [IA_D,  IA_B,  IA_E]
IA72IARelations(::_IA_DiorBiorEi) = [IA_Di, IA_Bi, IA_Ei]
IA72IARelations(::_IA_I)          = [
    IA_A,  IA_O,  IA_D,  IA_B,  IA_E,
    IA_Ai, IA_Oi, IA_Di, IA_Bi, IA_Ei
]

Base.show(io::IO, r::_IA7Rel)       = print(io, join(IA72IARelations(r), "∨"))
Base.show(io::IO, ::_IA_I)          = print(io, "I") =#
# NOTE: remove end

############################################################################################

# NOTE: remove begin
# 12 IA relations
#= const IARelations = [IA_A,  IA_L,  IA_B,  IA_E,  IA_D,  IA_O,
                     IA_Ai, IA_Li, IA_Bi, IA_Ei, IA_Di, IA_Oi]

# 7 IA7 relations
const IA7Relations = [IA_AorO,   IA_L,  IA_DorBorE,
                      IA_AiorOi, IA_Li, IA_DiorBiorEi]

# 3 IA3 relations
const IA3Relations = [IA_I, IA_L, IA_Li]

# 13 Interval Algebra extended with universal
const IARelations_extended = [RelationGlob, IARelations...] =#
# NOTE: remove end

############################################################################################

_accessibles(w::Interval, ::_IA_A,  X::Integer) = zip(Iterators.repeated(w.y), w.y+1:X+1)
_accessibles(w::Interval, ::_IA_Ai, X::Integer) = zip(1:w.x-1, Iterators.repeated(w.x))
_accessibles(w::Interval, ::_IA_L,  X::Integer) = _intervals_in(w.y+1, X+1)
_accessibles(w::Interval, ::_IA_Li, X::Integer) = _intervals_in(1, w.x-1)
_accessibles(w::Interval, ::_IA_B,  X::Integer) = zip(Iterators.repeated(w.x), w.x+1:w.y-1)
_accessibles(w::Interval, ::_IA_Bi, X::Integer) = zip(Iterators.repeated(w.x), w.y+1:X+1)
_accessibles(w::Interval, ::_IA_E,  X::Integer) = zip(w.x+1:w.y-1, Iterators.repeated(w.y))
_accessibles(w::Interval, ::_IA_Ei, X::Integer) = zip(1:w.x-1, Iterators.repeated(w.y))
_accessibles(w::Interval, ::_IA_D,  X::Integer) = _intervals_in(w.x+1, w.y-1)
_accessibles(w::Interval, ::_IA_Di, X::Integer) = Iterators.product(1:w.x-1, w.y+1:X+1)
_accessibles(w::Interval, ::_IA_O,  X::Integer) = Iterators.product(w.x+1:w.y-1, w.y+1:X+1)
_accessibles(w::Interval, ::_IA_Oi, X::Integer) = Iterators.product(1:w.x-1, w.x+1:w.y-1)

_accessibles(w::Interval, ::_IA_AorO,   X::Integer) = Iterators.product(w.x+1:w.y, w.y+1:X+1)
_accessibles(w::Interval, ::_IA_AiorOi, X::Integer) = Iterators.product(1:w.x-1,   w.x:w.y-1)

_accessibles(w::Interval, ::_IA_DorBorE,     X::Integer) = Iterators.flatten((_accessibles(w, IA_B,  X), _accessibles(w, IA_D,  X), _accessibles(w, IA_E,  X)))
_accessibles(w::Interval, ::_IA_DiorBiorEi,  X::Integer) = Iterators.flatten((_accessibles(w, IA_Bi, X), _accessibles(w, IA_Di, X), _accessibles(w, IA_Ei, X)))

_accessibles(w::Interval, ::_IA_I,  X::Integer) = Iterators.flatten((
    # Iterators.product(1:w.x-1, w.y+1:X+1),   # Di
    # Iterators.product(w.x:w.y, w.y+1:X+1),   # A+O+Bi
    Iterators.product(1:w.y, w.y+1:X+1),       # Di+A+O+Bi
    Iterators.product(1:w.x-1, w.x:w.y),       # Ai+Oi+Ei
    zip(Iterators.repeated(w.x), w.x+1:w.y-1), # B
    zip(w.x+1:w.y-1, Iterators.repeated(w.y)), # E
    _intervals_in(w.x+1, w.y-1),               # D
))


# More efficient implementations for edge cases (Later)
accessibles(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) =
    accessibles(nth(S, argmin(map((w)->w.y, S))), IA_L, X)
accessibles(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) =
    accessibles(nth(S, argmax(map((w)->w.x, S))), IA_Li, X)

# More efficient implementations for edge cases (After)
accessibles(S::AbstractWorldSet{Interval}, ::_IA_A, X::Integer) =
    IterTools.imap(Interval,
        Iterators.flatten(
            IterTools.imap((y)->zip(Iterators.repeated(y), y+1:X+1),
                IterTools.distinct(map((w)->w.y, S))
            )
        )
    )
accessibles(S::AbstractWorldSet{Interval}, ::_IA_Ai, X::Integer) =
    IterTools.imap(Interval,
        Iterators.flatten(
            IterTools.imap((x)->zip(1:x-1, Iterators.repeated(x)),
                IterTools.distinct(map((w)->w.x, S))
            )
        )
    )

############################################################################################
# When defining `accessibles_aggr` for minimum & maximum modal features, we find that we can
#  categorized  interval relations according to their behavior.
# Consider the decision ⟨R⟩ (minimum(A1) ≥ 10) evaluated on a world w = (x,y):
#  - With R = RelationId, it requires computing minimum(A1) on w;
#  - With R = RelationGlob, it requires computing maximum(A1) on 1:(X+1) (the largest world);
#  - With R = Begins inverse, it requires computing minimum(A1) on (x,y+1), if such interval exists;
#  - With R = During, it requires computing maximum(A1) on (x+1,y-1), if such interval exists;
#  - With R = After, it requires reading the single value in (y,y+1) (or, alternatively, computing minimum(A1) on it), if such interval exists;
#
# Here is the categorization assuming feature = minimum and test_operator = ≥:
#
#                                    .----------------------.
#                                    |(  Id  minimum)       |
#                                    |IA_Bi  minimum        |
#                                    |IA_Ei  minimum        |
#                                    |IA_Di  minimum        |
#                                    |IA_O   minimum        |
#                                    |IA_Oi  minimum        |
#                                    |----------------------|
#                                    |(Glob  maximum)       |
#                                    |IA_L   maximum        |
#                                    |IA_Li  maximum        |
#                                    |IA_D   maximum        |
#                                    |----------------------|
#                                    |IA_A   single-value   |
#                                    |IA_Ai  single-value   |
#                                    |IA_B   single-value   |
#                                    |IA_E   single-value   |
#                                    '----------------------'
#
# When feature = maximum, the two categories minimum and maximum swap roles.
# Furthermore, if test_operator = ≤, or, more generally, existential_aggregator(test_operator)
#  is minimum instead of maximum, again, the two categories minimum and maximum swap roles.
############################################################################################

# e.g., minimum + ≥
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Bi, X::Integer) = (w.y < X+1)                 ?  Interval[Interval(w.x,   w.y+1)] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Ei, X::Integer) = (1 < w.x)                   ?  Interval[Interval(w.x-1, w.y  )] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ?  Interval[Interval(w.x-1, w.y+1)] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ?  Interval[Interval(w.y-1, w.y+1)] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(w.x-1, w.x+1)] : Interval[]

# e.g., maximum + ≤
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Bi, X::Integer) = (w.y < X+1)                 ?  Interval[Interval(w.x,   w.y+1)] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Ei, X::Integer) = (1 < w.x)                   ?  Interval[Interval(w.x-1, w.y  )] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ?  Interval[Interval(w.x-1, w.y+1)] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ?  Interval[Interval(w.y-1, w.y+1)] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(w.x-1, w.x+1)] : Interval[]

# e.g., minimum + ≥
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Bi, X::Integer) = (w.y < X+1)                 ?  Interval[Interval(w.x,   X+1)]   : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Ei, X::Integer) = (1 < w.x)                   ?  Interval[Interval(1,     w.y  )] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ?  Interval[Interval(1,     X+1  )] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ?  Interval[Interval(w.x+1, X+1  )] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(1,     w.y-1)] : Interval[]

# e.g., maximum + ≤
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Bi, X::Integer) = (w.y < X+1)                 ?  Interval[Interval(w.x,   X+1)]   : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Ei, X::Integer) = (1 < w.x)                   ?  Interval[Interval(1,     w.y  )] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ?  Interval[Interval(1,     X+1  )] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ?  Interval[Interval(w.x+1, X+1  )] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(1,     w.y-1)] : Interval[]

############################################################################################

# e.g., minimum + ≥
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_L,  X::Integer) = (w.y+1 < X+1)   ? short_intervals_in(w.y+1, X+1)   : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Li, X::Integer) = (1 < w.x-1)     ? short_intervals_in(1, w.x-1)     : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_D,  X::Integer) = (w.x+1 < w.y-1) ? short_intervals_in(w.x+1, w.y-1) : Interval[]

# e.g., maximum + ≤
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_L,  X::Integer) = (w.y+1 < X+1)   ? short_intervals_in(w.y+1, X+1)   : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Li, X::Integer) = (1 < w.x-1)     ? short_intervals_in(1, w.x-1)     : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_D,  X::Integer) = (w.x+1 < w.y-1) ? short_intervals_in(w.x+1, w.y-1) : Interval[]

# e.g., minimum + ≥
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_L,  X::Integer) = (w.y+1 < X+1)   ? Interval[Interval(w.y+1, X+1)  ] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Li, X::Integer) = (1 < w.x-1)     ? Interval[Interval(1, w.x-1)    ] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_D,  X::Integer) = (w.x+1 < w.y-1) ? Interval[Interval(w.x+1, w.y-1)] : Interval[]

# e.g., maximum + ≤
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_L,  X::Integer) = (w.y+1 < X+1)   ? Interval[Interval(w.y+1, X+1)  ] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Li, X::Integer) = (1 < w.x-1)     ? Interval[Interval(1, w.x-1)    ] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_D,  X::Integer) = (w.x+1 < w.y-1) ? Interval[Interval(w.x+1, w.y-1)] : Interval[]

############################################################################################

# e.g., minimum + ≥
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)     ?   Interval[Interval(w.y,   w.y+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.y, X+1)]
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)       ?   Interval[Interval(w.x-1, w.x  )] : Interval[] #  _ReprVal(Interval   )# [Interval(1, w.x)]
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.x+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x, w.y-1)]
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)   ?   Interval[Interval(w.y-1, w.y  )] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x+1, w.y)]

# e.g., maximum + ≤
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)     ?   Interval[Interval(w.y,   w.y+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.y, X+1)]
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)       ?   Interval[Interval(w.x-1, w.x  )] : Interval[] #  _ReprVal(Interval   )# [Interval(1, w.x)]
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.x+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x, w.y-1)]
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)   ?   Interval[Interval(w.y-1, w.y  )] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x+1, w.y)]

# e.g., minimum + ≥
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)     ?   Interval[Interval(w.y,   X+1  )] : Interval[] #  _ReprVal(Interval(w.y, w.y+1)   )# [Interval(w.y, X+1)]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)       ?   Interval[Interval(1,     w.x  )] : Interval[] #  _ReprVal(Interval(w.x-1, w.x)   )# [Interval(1, w.x)]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.y-1)] : Interval[] #  _ReprVal(Interval(w.x, w.x+1)   )# [Interval(w.x, w.y-1)]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)   ?   Interval[Interval(w.x+1, w.y  )] : Interval[] #  _ReprVal(Interval(w.y-1, w.y)   )# [Interval(w.x+1, w.y)]

# e.g., maximum + ≤
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)     ?   Interval[Interval(w.y,   X+1  )] : Interval[] #  _ReprVal(Interval(w.y, w.y+1)   )# [Interval(w.y, X+1)]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)       ?   Interval[Interval(1,     w.x  )] : Interval[] #  _ReprVal(Interval(w.x-1, w.x)   )# [Interval(1, w.x)]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.y-1)] : Interval[] #  _ReprVal(Interval(w.x, w.x+1)   )# [Interval(w.x, w.y-1)]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)   ?   Interval[Interval(w.x+1, w.y  )] : Interval[] #  _ReprVal(Interval(w.y-1, w.y)   )# [Interval(w.x+1, w.y)]

############################################################################################
# Similarly, here is the categorization for IA7 & IA3 assuming feature = minimum and test_operator = ≥:
#
#                               .-----------------------------.
#                               |(  Id         minimum)       |
#                               |IA_AorO       minimum        |
#                               |IA_AiorOi     minimum        |
#                               |IA_DiorBiorEi minimum        |
#                               |-----------------------------|
#                               |IA_DorBorE    maximum        |
#                               |-----------------------------|
#                               |IA_I          ?              |
#                               '-----------------------------'
# TODO write the correct `accessibles_aggr` methods, instead of these fallbacks:
accessibles_aggr(f::ModalFeature, a::Aggregator, w::Interval, r::_IA_AorO,       X::Integer) =
    Iterators.flatten([accessibles_aggr(f, a, w, r, X) for r in IA72IARelations(IA_AorO)])
accessibles_aggr(f::ModalFeature, a::Aggregator, w::Interval, r::_IA_AiorOi,     X::Integer) =
    Iterators.flatten([accessibles_aggr(f, a, w, r, X) for r in IA72IARelations(IA_AiorOi)])
accessibles_aggr(f::ModalFeature, a::Aggregator, w::Interval, r::_IA_DorBorE,    X::Integer) =
    Iterators.flatten([accessibles_aggr(f, a, w, r, X) for r in IA72IARelations(IA_DorBorE)])
accessibles_aggr(f::ModalFeature, a::Aggregator, w::Interval, r::_IA_DiorBiorEi, X::Integer) =
    Iterators.flatten([accessibles_aggr(f, a, w, r, X) for r in IA72IARelations(IA_DiorBiorEi)])
accessibles_aggr(f::ModalFeature, a::Aggregator, w::Interval, r::_IA_I,          X::Integer) =
    Iterators.flatten([accessibles_aggr(f, a, w, r, X) for r in IA72IARelations(IA_I)])
############################################################################################
