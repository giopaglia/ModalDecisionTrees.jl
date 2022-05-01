################################################################################
# BEGIN IA relations
################################################################################

# Interval Algebra relations (or Allen relations)
abstract type _IARel <: AbstractRelation end
struct _IA_A  <: _IARel end; const IA_A  = _IA_A();  # After
struct _IA_L  <: _IARel end; const IA_L  = _IA_L();  # Later
struct _IA_B  <: _IARel end; const IA_B  = _IA_B();  # Begins
struct _IA_E  <: _IARel end; const IA_E  = _IA_E();  # Ends
struct _IA_D  <: _IARel end; const IA_D  = _IA_D();  # During
struct _IA_O  <: _IARel end; const IA_O  = _IA_O();  # Overlaps
struct _IA_Ai <: _IARel end; const IA_Ai = _IA_Ai(); # After inverse
struct _IA_Li <: _IARel end; const IA_Li = _IA_Li(); # Later inverse
struct _IA_Bi <: _IARel end; const IA_Bi = _IA_Bi(); # Begins inverse
struct _IA_Ei <: _IARel end; const IA_Ei = _IA_Ei(); # Ends inverse
struct _IA_Di <: _IARel end; const IA_Di = _IA_Di(); # During inverse
struct _IA_Oi <: _IARel end; const IA_Oi = _IA_Oi(); # Overlaps inverse
# Coarser relations for IA7
abstract type _IA7Rel <: _IARel end
struct _IA_AorO   <: _IA7Rel end; const IA_AorO   = _IA_AorO();   # After ∪ Overlaps
struct _IA_DorBorE  <: _IA7Rel end; const IA_DorBorE  = _IA_DorBorE();  # During ∪ Begins ∪ Ends
struct _IA_AiorOi  <: _IA7Rel end; const IA_AiorOi  = _IA_AiorOi();  # (After ∪ Overlaps) inverse
struct _IA_DiorBiorEi <: _IA7Rel end; const IA_DiorBiorEi = _IA_DiorBiorEi(); # (During ∪ Begins ∪ Ends) inverse
# Ever coarser relations for IA3
abstract type _IA3Rel <: _IARel end
struct _IA_I   <: _IA3Rel end; const IA_I   = _IA_I();   # Intersecting (AODBE + AODBE inverse)

IA72IARelations(::_IA_AorO)       = [IA_A,  IA_O]
IA72IARelations(::_IA_AiorOi)     = [IA_Ai, IA_Oi]
IA72IARelations(::_IA_DorBorE)    = [IA_D,  IA_B,  IA_E]
IA72IARelations(::_IA_DiorBiorEi) = [IA_Di, IA_Bi, IA_Ei]
IA72IARelations(::_IA_I)          = [
	IA_A,  IA_O,  IA_D,  IA_B,  IA_E,
	IA_Ai, IA_Oi, IA_Di, IA_Bi, IA_Ei
]

goesWith(::Type{Interval}, ::R where R<:_IARel) = true

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
Base.show(io::IO, ::_IA_Oi) = print(io, "O̅")

Base.show(io::IO, ::_IA_AorO)       = print(io, "A∨O")
Base.show(io::IO, ::_IA_DorBorE)    = print(io, "D∨B∨E")
Base.show(io::IO, ::_IA_AiorOi)     = print(io, "A̅∨O̅")
Base.show(io::IO, ::_IA_DiorBiorEi) = print(io, "D̅∨B̅∨E̅")
Base.show(io::IO, ::_IA_I)          = print(io, "I")

# 12 Interval Algebra relations
const IARelations = [IA_A,  IA_L,  IA_B,  IA_E,  IA_D,  IA_O,
										 IA_Ai, IA_Li, IA_Bi, IA_Ei, IA_Di, IA_Oi]

# 7 IA7 relations
const IA7Relations = [IA_AorO,   IA_L,  IA_DorBorE,
										  IA_AiorOi, IA_Li, IA_DiorBiorEi]

# 3 IA3 relations
const IA3Relations = [IA_I, IA_L, IA_Li]

# 13 Interval Algebra extended with universal
const IARelations_extended = [RelationGlob, IARelations...]

# Enumerate accessible worlds from a single world
_accessibles(w::Interval, ::_IA_A,  X::Integer) = zip(Iterators.repeated(w.y), w.y+1:X+1)
_accessibles(w::Interval, ::_IA_Ai, X::Integer) = zip(1:w.x-1, Iterators.repeated(w.x))
_accessibles(w::Interval, ::_IA_L,  X::Integer) = enumPairsIn(w.y+1, X+1)
_accessibles(w::Interval, ::_IA_Li, X::Integer) = enumPairsIn(1, w.x-1)
_accessibles(w::Interval, ::_IA_B,  X::Integer) = zip(Iterators.repeated(w.x), w.x+1:w.y-1)
_accessibles(w::Interval, ::_IA_Bi, X::Integer) = zip(Iterators.repeated(w.x), w.y+1:X+1)
_accessibles(w::Interval, ::_IA_E,  X::Integer) = zip(w.x+1:w.y-1, Iterators.repeated(w.y))
_accessibles(w::Interval, ::_IA_Ei, X::Integer) = zip(1:w.x-1, Iterators.repeated(w.y))
_accessibles(w::Interval, ::_IA_D,  X::Integer) = enumPairsIn(w.x+1, w.y-1)
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
	enumPairsIn(w.x+1, w.y-1),                 # D
	))


# More efficient implementations for edge cases
accessibles(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) =
	accessibles(nth(S, argmin(map((w)->w.y, S))), IA_L, X)
accessibles(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) =
	accessibles(nth(S, argmax(map((w)->w.x, S))), IA_Li, X)
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

# Other options:
# accessibles2_1_2(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) =
# 	IterTools.imap(Interval, _accessibles(Base.argmin((w.y for w in S)), IA_L, X))
# accessibles2_1_2(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) =
# 	IterTools.imap(Interval, _accessibles(Base.argmax((w.x for w in S)), IA_Li, X))
# accessibles2_2(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) = begin
# 	m = argmin(map((w)->w.y, S))
# 	IterTools.imap(Interval, _accessibles([w for (i,w) in enumerate(S) if i == m][1], IA_L, X))
# end
# accessibles2_2(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) = begin
# 	m = argmax(map((w)->w.x, S))
# 	IterTools.imap(Interval, _accessibles([w for (i,w) in enumerate(S) if i == m][1], IA_Li, X))
# end
# # This makes sense if we have 2-Tuples instead of intervals
# function snd((a,b)::Tuple) b end
# function fst((a,b)::Tuple) a end
# accessibles2_1(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) = 
# 	IterTools.imap(Interval,
# 		_accessibles(S[argmin(map(snd, S))], IA_L, X)
# 	)
# accessibles2_1(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) = 
# 	IterTools.imap(Interval,
# 		_accessibles(S[argmax(map(fst, S))], IA_Li, X)
# 	)



# IA_All max
# IA_Id  min
# -------
# IA_Bi  min
# IA_Ei  min
# IA_Di  min
# IA_O   min
# IA_Oi  min
# -------
# IA_L   max
# IA_Li  max
# IA_D   max
# -------
# IA_A   val
# IA_Ai  val
# IA_B   val
# IA_E   val

#=
# TODO parametrize on the test_operator. These are wrong anyway...
# Note: these conditions are the ones that make a modal_step inexistent
enumAccRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)                 ? _ReprVal(Interval(w.y, w.y+1)   ) : _ReprNone{Interval}() # [Interval(w.y, X+1)]     : Interval[]
enumAccRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)                   ? _ReprVal(Interval(w.x-1, w.x)   ) : _ReprNone{Interval}() # [Interval(1, w.x)]       : Interval[]
enumAccRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)               ? _ReprVal(Interval(w.x, w.x+1)   ) : _ReprNone{Interval}() # [Interval(w.x, w.y-1)]   : Interval[]
enumAccRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)               ? _ReprVal(Interval(w.y-1, w.y)   ) : _ReprNone{Interval}() # [Interval(w.x+1, w.y)]   : Interval[]

enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_L,  X::Integer) = (w.y+1 < X+1)               ? _ReprMax(Interval(w.y+1, X+1)   ) : _ReprNone{Interval}() # [Interval(w.y+1, X+1)]   : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Li, X::Integer) = (1 < w.x-1)                 ? _ReprMax(Interval(1, w.x-1)     ) : _ReprNone{Interval}() # [Interval(1, w.x-1)]     : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_D,  X::Integer) = (w.x+1 < w.y-1)             ? _ReprMax(Interval(w.x+1, w.y-1) ) : _ReprNone{Interval}() # [Interval(w.x+1, w.y-1)] : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_L,  X::Integer) = (w.y+1 < X+1)               ? _ReprMin(Interval(w.y+1, X+1)   ) : _ReprNone{Interval}() # [Interval(w.y+1, X+1)]   : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_Li, X::Integer) = (1 < w.x-1)                 ? _ReprMin(Interval(1, w.x-1)     ) : _ReprNone{Interval}() # [Interval(1, w.x-1)]     : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_D,  X::Integer) = (w.x+1 < w.y-1)             ? _ReprMin(Interval(w.x+1, w.y-1) ) : _ReprNone{Interval}() # [Interval(w.x+1, w.y-1)] : Interval[]

enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Bi, X::Integer) = (w.y < X+1)                 ? _ReprMin(Interval(w.x, w.y+1)   ) : _ReprNone{Interval}() # [Interval(w.x, X+1)]     : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Ei, X::Integer) = (1 < w.x)                   ? _ReprMin(Interval(w.x-1, w.y)   ) : _ReprNone{Interval}() # [Interval(1, w.y)]       : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ? _ReprMin(Interval(w.x-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(1, X+1)]       : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ? _ReprMin(Interval(w.y-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(w.x+1, X+1)]   : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ? _ReprMin(Interval(w.x-1, w.x+1) ) : _ReprNone{Interval}() # [Interval(1, w.y-1)]     : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_Bi, X::Integer) = (w.y < X+1)                 ? _ReprMax(Interval(w.x, w.y+1)   ) : _ReprNone{Interval}() # [Interval(w.x, X+1)]     : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_Ei, X::Integer) = (1 < w.x)                   ? _ReprMax(Interval(w.x-1, w.y)   ) : _ReprNone{Interval}() # [Interval(1, w.y)]       : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ? _ReprMax(Interval(w.x-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(1, X+1)]       : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ? _ReprMax(Interval(w.y-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(w.x+1, X+1)]   : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ? _ReprMax(Interval(w.x-1, w.x+1) ) : _ReprNone{Interval}() # [Interval(1, w.y-1)]     : Interval[]
=#

# TODO maybe better to explicit the cases
accessibles_aggr(f::Union{SingleAttributeMin,SingleAttributeMax}, a::Union{typeof(minimum),typeof(maximum)}, w::Interval, r::_IA_L,  X::Integer) = (w.y+1 < X+1)   ? IterTools.imap(Interval, enumShortPairsIn(w.y+1, X+1))   : Interval[]
accessibles_aggr(f::Union{SingleAttributeMin,SingleAttributeMax}, a::Union{typeof(minimum),typeof(maximum)}, w::Interval, r::_IA_Li, X::Integer) = (1 < w.x-1)     ? IterTools.imap(Interval, enumShortPairsIn(1, w.x-1))     : Interval[]
accessibles_aggr(f::Union{SingleAttributeMin,SingleAttributeMax}, a::Union{typeof(minimum),typeof(maximum)}, w::Interval, r::_IA_D,  X::Integer) = (w.x+1 < w.y-1) ? IterTools.imap(Interval, enumShortPairsIn(w.x+1, w.y-1)) : Interval[]

# accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_L,  X::Integer) = (w.y+1 < X+1)   ? IterTools.imap(Interval, enumShortPairsIn(w.y+1, X+1))   : Interval[]
# accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Li, X::Integer) = (1 < w.x-1)     ? IterTools.imap(Interval, enumShortPairsIn(1, w.x-1))     : Interval[]
# accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_D,  X::Integer) = (w.x+1 < w.y-1) ? IterTools.imap(Interval, enumShortPairsIn(w.x+1, w.y-1)) : Interval[]

# accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_L,  X::Integer) = (w.y+1 < X+1)   ? IterTools.imap(Interval, enumShortPairsIn(w.y+1, X+1))   : Interval[]
# accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Li, X::Integer) = (1 < w.x-1)     ? IterTools.imap(Interval, enumShortPairsIn(1, w.x-1))     : Interval[]
# accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_D,  X::Integer) = (w.x+1 < w.y-1) ? IterTools.imap(Interval, enumShortPairsIn(w.x+1, w.y-1)) : Interval[]


accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_L,  X::Integer) = (w.y+1 < X+1)   ? Interval[Interval(w.y+1, X+1)  ] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Li, X::Integer) = (1 < w.x-1)     ? Interval[Interval(1, w.x-1)    ] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_D,  X::Integer) = (w.x+1 < w.y-1) ? Interval[Interval(w.x+1, w.y-1)] : Interval[]

accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_L,  X::Integer) = (w.y+1 < X+1)   ? Interval[Interval(w.y+1, X+1)  ] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Li, X::Integer) = (1 < w.x-1)     ? Interval[Interval(1, w.x-1)    ] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_D,  X::Integer) = (w.x+1 < w.y-1) ? Interval[Interval(w.x+1, w.y-1)] : Interval[]



accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Bi, X::Integer) = (w.y < X+1)                 ?  Interval[Interval(w.x,   w.y+1)] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Ei, X::Integer) = (1 < w.x)                   ?  Interval[Interval(w.x-1, w.y  )] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ?  Interval[Interval(w.x-1, w.y+1)] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ?  Interval[Interval(w.y-1, w.y+1)] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(w.x-1, w.x+1)] : Interval[]

accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Bi, X::Integer) = (w.y < X+1)                 ?  Interval[Interval(w.x,   w.y+1)] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Ei, X::Integer) = (1 < w.x)                   ?  Interval[Interval(w.x-1, w.y  )] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ?  Interval[Interval(w.x-1, w.y+1)] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ?  Interval[Interval(w.y-1, w.y+1)] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(w.x-1, w.x+1)] : Interval[]


accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Bi, X::Integer) = (w.y < X+1)                 ?  Interval[Interval(w.x,   w.X+1)] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Ei, X::Integer) = (1 < w.x)                   ?  Interval[Interval(1,     w.y  )] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ?  Interval[Interval(1,     X+1  )] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ?  Interval[Interval(w.x+1, X+1  )] : Interval[]
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(1,     w.y-1)] : Interval[]

accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Bi, X::Integer) = (w.y < X+1)                 ?  Interval[Interval(w.x,   w.X+1)] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Ei, X::Integer) = (1 < w.x)                   ?  Interval[Interval(1,     w.y  )] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ?  Interval[Interval(1,     X+1  )] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ?  Interval[Interval(w.x+1, X+1  )] : Interval[]
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(1,     w.y-1)] : Interval[]


accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)     ?   Interval[Interval(w.y,   w.y+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.y, X+1)]   
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)       ?   Interval[Interval(w.x-1, w.x  )] : Interval[] #  _ReprVal(Interval   )# [Interval(1, w.x)]     
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.x+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x, w.y-1)] 
accessibles_aggr(f::SingleAttributeMax, a::typeof(minimum), w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)   ?   Interval[Interval(w.y-1, w.y  )] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x+1, w.y)] 

accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)     ?   Interval[Interval(w.y,   w.y+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.y, X+1)]   
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)       ?   Interval[Interval(w.x-1, w.x  )] : Interval[] #  _ReprVal(Interval   )# [Interval(1, w.x)]     
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.x+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x, w.y-1)] 
accessibles_aggr(f::SingleAttributeMin, a::typeof(maximum), w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)   ?   Interval[Interval(w.y-1, w.y  )] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x+1, w.y)] 

accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)     ?   Interval[Interval(w.y,   X+1  )] : Interval[] #  _ReprVal(Interval(w.y, w.y+1)   )# [Interval(w.y, X+1)]   
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)       ?   Interval[Interval(1,     w.x  )] : Interval[] #  _ReprVal(Interval(w.x-1, w.x)   )# [Interval(1, w.x)]     
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.y-1)] : Interval[] #  _ReprVal(Interval(w.x, w.x+1)   )# [Interval(w.x, w.y-1)] 
accessibles_aggr(f::SingleAttributeMax, a::typeof(maximum), w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)   ?   Interval[Interval(w.x+1, w.y  )] : Interval[] #  _ReprVal(Interval(w.y-1, w.y)   )# [Interval(w.x+1, w.y)] 

accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)     ?   Interval[Interval(w.y,   X+1  )] : Interval[] #  _ReprVal(Interval(w.y, w.y+1)   )# [Interval(w.y, X+1)]   
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)       ?   Interval[Interval(1,     w.x  )] : Interval[] #  _ReprVal(Interval(w.x-1, w.x)   )# [Interval(1, w.x)]     
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.y-1)] : Interval[] #  _ReprVal(Interval(w.x, w.x+1)   )# [Interval(w.x, w.y-1)] 
accessibles_aggr(f::SingleAttributeMin, a::typeof(minimum), w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)   ?   Interval[Interval(w.x+1, w.y  )] : Interval[] #  _ReprVal(Interval(w.y-1, w.y)   )# [Interval(w.x+1, w.y)] 


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







 
# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_A, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? (channel[w.y],channel[w.y]) : (typemax(T),typemin(T))
# computeModalThreshold(test_operator::_TestOpGeq, w::Interval, ::_IA_A, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? channel[w.y] : typemax(T)
# computeModalThreshold(test_operator::_TestOpLeq, w::Interval, ::_IA_A, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? channel[w.y] : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_Ai, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? (channel[w.x-1],channel[w.x-1]) : (typemax(T),typemin(T))
# computeModalThreshold(test_operator::_TestOpGeq, w::Interval, ::_IA_Ai, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? channel[w.x-1] : typemax(T)
# computeModalThreshold(test_operator::_TestOpLeq, w::Interval, ::_IA_Ai, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? channel[w.x-1] : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_L, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y+1 < length(channel)+1) ? reverse(extrema(channel[w.y+1:length(channel)])) : (typemax(T),typemin(T))
# computeModalThreshold(test_operator::_TestOpGeq, w::Interval, ::_IA_L, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y+1 < length(channel)+1) ? maximum(channel[w.y+1:length(channel)]) : typemax(T)
# computeModalThreshold(test_operator::_TestOpLeq, w::Interval, ::_IA_L, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y+1 < length(channel)+1) ? minumum(channel[w.y+1:length(channel)]) : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_Li, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x-1) ? reverse(extrema(channel[1:w.x-2])) : (typemax(T),typemin(T))
# computeModalThreshold(test_operator::_TestOpGeq, w::Interval, ::_IA_Li, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x-1) ? maximum(channel[1:w.x-2]) : typemax(T)
# computeModalThreshold(test_operator::_TestOpLeq, w::Interval, ::_IA_Li, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x-1) ? minumum(channel[1:w.x-2]) : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_B, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x < w.y-1) ? (channel[w.x],channel[w.x]) : (typemax(T),typemin(T))
# computeModalThreshold(test_operator::_TestOpGeq, w::Interval, ::_IA_B, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x < w.y-1) ? channel[w.x] : typemax(T)
# computeModalThreshold(test_operator::_TestOpLeq, w::Interval, ::_IA_B, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x < w.y-1) ? channel[w.x] : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_Bi, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? (minimum(channel[w.x:w.y-1+1]),maximum(channel[w.x:w.y-1+1])) : (typemax(T),typemin(T))
# computeModalThreshold(test_operator::_TestOpGeq, w::Interval, ::_IA_Bi, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? minimum(channel[w.x:w.y-1+1]) : typemax(T)
# computeModalThreshold(test_operator::_TestOpLeq, w::Interval, ::_IA_Bi, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? maximum(channel[w.x:w.y-1+1]) : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_E, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y) ? (channel[w.y-1],channel[w.y-1]) : (typemax(T),typemin(T))
# computeModalThreshold(test_operator::_TestOpGeq, w::Interval, ::_IA_E, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y) ? channel[w.y-1] : typemax(T)
# computeModalThreshold(test_operator::_TestOpLeq, w::Interval, ::_IA_E, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y) ? channel[w.y-1] : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_Ei, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? (minimum(channel[w.x-1:w.y-1]),maximum(channel[w.x-1:w.y-1])) : (typemax(T),typemin(T))
# computeModalThreshold(test_operator::_TestOpGeq, w::Interval, ::_IA_Ei, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? minimum(channel[w.x-1:w.y-1]) : typemax(T)
# computeModalThreshold(test_operator::_TestOpLeq, w::Interval, ::_IA_Ei, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? maximum(channel[w.x-1:w.y-1]) : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_D, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y-1) ? reverse(extrema(channel[w.x+1:w.y-1-1])) : (typemax(T),typemin(T))
# computeModalThreshold(test_operator::_TestOpGeq, w::Interval, ::_IA_D, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y-1) ? maximum(channel[w.x+1:w.y-1-1]) : typemax(T)
# computeModalThreshold(test_operator::_TestOpLeq, w::Interval, ::_IA_D, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y-1) ? minumum(channel[w.x+1:w.y-1-1]) : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_Di, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.y < length(channel)+1) ? (minimum(channel[w.x-1:w.y-1+1]),maximum(channel[w.x-1:w.y-1+1])) : (typemax(T),typemin(T))
# computeModalThreshold(test_operator::_TestOpGeq, w::Interval, ::_IA_Di, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.y < length(channel)+1) ? minimum(channel[w.x-1:w.y-1+1]) : typemax(T)
# computeModalThreshold(test_operator::_TestOpLeq, w::Interval, ::_IA_Di, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.y < length(channel)+1) ? maximum(channel[w.x-1:w.y-1+1]) : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_O, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y && w.y < length(channel)+1) ? (minimum(channel[w.y-1:w.y-1+1]),maximum(channel[w.y-1:w.y-1+1])) : (typemax(T),typemin(T))
# computeModalThreshold(test_operator::_TestOpGeq, w::Interval, ::_IA_O, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y && w.y < length(channel)+1) ? minimum(channel[w.y-1:w.y-1+1]) : typemax(T)
# computeModalThreshold(test_operator::_TestOpLeq, w::Interval, ::_IA_O, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y && w.y < length(channel)+1) ? maximum(channel[w.y-1:w.y-1+1]) : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_Oi, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.x+1 < w.y) ? (minimum(channel[w.x-1:w.x]),maximum(channel[w.x-1:w.x])) : (typemax(T),typemin(T))
# computeModalThreshold(test_operator::_TestOpGeq, w::Interval, ::_IA_Oi, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.x+1 < w.y) ? minimum(channel[w.x-1:w.x]) : typemax(T)
# computeModalThreshold(test_operator::_TestOpLeq, w::Interval, ::_IA_Oi, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.x+1 < w.y) ? maximum(channel[w.x-1:w.x]) : typemin(T)

################################################################################
# END IA relations
################################################################################
