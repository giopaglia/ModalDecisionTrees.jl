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

# TODO change name to display_rel_short
display_rel_short(::_IA_A)  = "A"
display_rel_short(::_IA_L)  = "L"
display_rel_short(::_IA_B)  = "B"
display_rel_short(::_IA_E)  = "E"
display_rel_short(::_IA_D)  = "D"
display_rel_short(::_IA_O)  = "O"
display_rel_short(::_IA_Ai) = "A̅"
display_rel_short(::_IA_Li) = "L̅"
display_rel_short(::_IA_Bi) = "B̅"
display_rel_short(::_IA_Ei) = "E̅"
display_rel_short(::_IA_Di) = "D̅"
display_rel_short(::_IA_Oi) = "O̅"

display_rel_short(::_IA_AorO)       = "A∨O"
display_rel_short(::_IA_DorBorE)    = "D∨B∨E"
display_rel_short(::_IA_AiorOi)     = "A̅∨O̅"
display_rel_short(::_IA_DiorBiorEi) = "D̅∨B̅∨E̅"
display_rel_short(::_IA_I)          = "I"

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
enumAccBare(w::Interval, ::_IA_A,  X::Integer) = zip(Iterators.repeated(w.y), w.y+1:X+1)
enumAccBare(w::Interval, ::_IA_Ai, X::Integer) = zip(1:w.x-1, Iterators.repeated(w.x))
enumAccBare(w::Interval, ::_IA_L,  X::Integer) = enumPairsIn(w.y+1, X+1)
enumAccBare(w::Interval, ::_IA_Li, X::Integer) = enumPairsIn(1, w.x-1)
enumAccBare(w::Interval, ::_IA_B,  X::Integer) = zip(Iterators.repeated(w.x), w.x+1:w.y-1)
enumAccBare(w::Interval, ::_IA_Bi, X::Integer) = zip(Iterators.repeated(w.x), w.y+1:X+1)
enumAccBare(w::Interval, ::_IA_E,  X::Integer) = zip(w.x+1:w.y-1, Iterators.repeated(w.y))
enumAccBare(w::Interval, ::_IA_Ei, X::Integer) = zip(1:w.x-1, Iterators.repeated(w.y))
enumAccBare(w::Interval, ::_IA_D,  X::Integer) = enumPairsIn(w.x+1, w.y-1)
enumAccBare(w::Interval, ::_IA_Di, X::Integer) = Iterators.product(1:w.x-1, w.y+1:X+1)
enumAccBare(w::Interval, ::_IA_O,  X::Integer) = Iterators.product(w.x+1:w.y-1, w.y+1:X+1)
enumAccBare(w::Interval, ::_IA_Oi, X::Integer) = Iterators.product(1:w.x-1, w.x+1:w.y-1)

enumAccBare(w::Interval, ::_IA_AorO,   X::Integer) = Iterators.product(w.x+1:w.y, w.y+1:X+1)
enumAccBare(w::Interval, ::_IA_AiorOi, X::Integer) = Iterators.product(1:w.x-1,   w.x:w.y-1)

enumAccBare(w::Interval, ::_IA_DorBorE,     X::Integer) = Iterators.flatten((enumAccBare(w, IA_B,  X), enumAccBare(w, IA_D,  X), enumAccBare(w, IA_E,  X)))
enumAccBare(w::Interval, ::_IA_DiorBiorEi,  X::Integer) = Iterators.flatten((enumAccBare(w, IA_Bi, X), enumAccBare(w, IA_Di, X), enumAccBare(w, IA_Ei, X)))

enumAccBare(w::Interval, ::_IA_I,  X::Integer) = Iterators.flatten((
	# Iterators.product(1:w.x-1, w.y+1:X+1),   # Di
	# Iterators.product(w.x:w.y, w.y+1:X+1),   # A+O+Bi
	Iterators.product(1:w.y, w.y+1:X+1),       # Di+A+O+Bi
	Iterators.product(1:w.x-1, w.x:w.y),       # Ai+Oi+Ei
	zip(Iterators.repeated(w.x), w.x+1:w.y-1), # B
	zip(w.x+1:w.y-1, Iterators.repeated(w.y)), # E
	enumPairsIn(w.x+1, w.y-1),                 # D
	))


# More efficient implementations for edge cases
enumAccessibles(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) =
	enumAccessibles(nth(S, argmin(map((w)->w.y, S))), IA_L, X)
enumAccessibles(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) =
	enumAccessibles(nth(S, argmax(map((w)->w.x, S))), IA_Li, X)
enumAccessibles(S::AbstractWorldSet{Interval}, ::_IA_A, X::Integer) =
	IterTools.imap(Interval,
		Iterators.flatten(
			IterTools.imap((y)->zip(Iterators.repeated(y), y+1:X+1),
				IterTools.distinct(map((w)->w.y, S))
			)
		)
	)
enumAccessibles(S::AbstractWorldSet{Interval}, ::_IA_Ai, X::Integer) =
	IterTools.imap(Interval,
		Iterators.flatten(
			IterTools.imap((x)->zip(1:x-1, Iterators.repeated(x)),
				IterTools.distinct(map((w)->w.x, S))
			)
		)
	)

# Other options:
# enumAccessibles2_1_2(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) =
# 	IterTools.imap(Interval, enumAccBare(Base.argmin((w.y for w in S)), IA_L, X))
# enumAccessibles2_1_2(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) =
# 	IterTools.imap(Interval, enumAccBare(Base.argmax((w.x for w in S)), IA_Li, X))
# enumAccessibles2_2(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) = begin
# 	m = argmin(map((w)->w.y, S))
# 	IterTools.imap(Interval, enumAccBare([w for (i,w) in enumerate(S) if i == m][1], IA_L, X))
# end
# enumAccessibles2_2(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) = begin
# 	m = argmax(map((w)->w.x, S))
# 	IterTools.imap(Interval, enumAccBare([w for (i,w) in enumerate(S) if i == m][1], IA_Li, X))
# end
# # This makes sense if we have 2-Tuples instead of intervals
# function snd((a,b)::Tuple) b end
# function fst((a,b)::Tuple) a end
# enumAccessibles2_1(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) = 
# 	IterTools.imap(Interval,
# 		enumAccBare(S[argmin(map(snd, S))], IA_L, X)
# 	)
# enumAccessibles2_1(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) = 
# 	IterTools.imap(Interval,
# 		enumAccBare(S[argmax(map(fst, S))], IA_Li, X)
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


enumAccReprAggr(f::Union{AttributeMinimumFeatureType,AttributeMaximumFeatureType}, a::Union{typeof(minimum),typeof(maximum)}, w::Interval, r::_IA_L,  X::Integer) = (w.y+1 < X+1)   ? IterTools.imap(Interval, enumShortPairsIn(w.y+1, X+1))   : Interval[]
enumAccReprAggr(f::Union{AttributeMinimumFeatureType,AttributeMaximumFeatureType}, a::Union{typeof(minimum),typeof(maximum)}, w::Interval, r::_IA_Li, X::Integer) = (1 < w.x-1)     ? IterTools.imap(Interval, enumShortPairsIn(1, w.x-1))     : Interval[]
enumAccReprAggr(f::Union{AttributeMinimumFeatureType,AttributeMaximumFeatureType}, a::Union{typeof(minimum),typeof(maximum)}, w::Interval, r::_IA_D,  X::Integer) = (w.x+1 < w.y-1) ? IterTools.imap(Interval, enumShortPairsIn(w.x+1, w.y-1)) : Interval[]

enumAccReprAggr(f::Union{AttributeMaximumFeatureType}, a::typeof(maximum), w::Interval, r::_IA_L,  X::Integer) = (w.y+1 < X+1)   ? Interval[Interval(w.y+1, X+1)  ] : Interval[]
enumAccReprAggr(f::Union{AttributeMaximumFeatureType}, a::typeof(maximum), w::Interval, r::_IA_Li, X::Integer) = (1 < w.x-1)     ? Interval[Interval(1, w.x-1)    ] : Interval[]
enumAccReprAggr(f::Union{AttributeMaximumFeatureType}, a::typeof(maximum), w::Interval, r::_IA_D,  X::Integer) = (w.x+1 < w.y-1) ? Interval[Interval(w.x+1, w.y-1)] : Interval[]

enumAccReprAggr(f::Union{AttributeMinimumFeatureType}, a::typeof(minimum), w::Interval, r::_IA_L,  X::Integer) = (w.y+1 < X+1)   ? Interval[Interval(w.y+1, X+1)  ] : Interval[]
enumAccReprAggr(f::Union{AttributeMinimumFeatureType}, a::typeof(minimum), w::Interval, r::_IA_Li, X::Integer) = (1 < w.x-1)     ? Interval[Interval(1, w.x-1)    ] : Interval[]
enumAccReprAggr(f::Union{AttributeMinimumFeatureType}, a::typeof(minimum), w::Interval, r::_IA_D,  X::Integer) = (w.x+1 < w.y-1) ? Interval[Interval(w.x+1, w.y-1)] : Interval[]



enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(maximum), w::Interval, r::_IA_Bi, X::Integer) = (w.y < X+1)                 ?  Interval[Interval(w.x,   w.y+1)] : Interval[]
enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(maximum), w::Interval, r::_IA_Ei, X::Integer) = (1 < w.x)                   ?  Interval[Interval(w.x-1, w.y  )] : Interval[]
enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(maximum), w::Interval, r::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ?  Interval[Interval(w.x-1, w.y+1)] : Interval[]
enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(maximum), w::Interval, r::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ?  Interval[Interval(w.y-1, w.y+1)] : Interval[]
enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(maximum), w::Interval, r::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(w.x-1, w.x+1)] : Interval[]

enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(minimum), w::Interval, r::_IA_Bi, X::Integer) = (w.y < X+1)                 ?  Interval[Interval(w.x,   w.y+1)] : Interval[]
enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(minimum), w::Interval, r::_IA_Ei, X::Integer) = (1 < w.x)                   ?  Interval[Interval(w.x-1, w.y  )] : Interval[]
enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(minimum), w::Interval, r::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ?  Interval[Interval(w.x-1, w.y+1)] : Interval[]
enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(minimum), w::Interval, r::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ?  Interval[Interval(w.y-1, w.y+1)] : Interval[]
enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(minimum), w::Interval, r::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(w.x-1, w.x+1)] : Interval[]


enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(minimum), w::Interval, r::_IA_Bi, X::Integer) = (w.y < X+1)                 ?  Interval[Interval(w.x,   w.X+1)] : Interval[]
enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(minimum), w::Interval, r::_IA_Ei, X::Integer) = (1 < w.x)                   ?  Interval[Interval(1,     w.y  )] : Interval[]
enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(minimum), w::Interval, r::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ?  Interval[Interval(1,     X+1  )] : Interval[]
enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(minimum), w::Interval, r::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ?  Interval[Interval(w.x+1, X+1  )] : Interval[]
enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(minimum), w::Interval, r::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(1,     w.y-1)] : Interval[]

enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(maximum), w::Interval, r::_IA_Bi, X::Integer) = (w.y < X+1)                 ?  Interval[Interval(w.x,   w.X+1)] : Interval[]
enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(maximum), w::Interval, r::_IA_Ei, X::Integer) = (1 < w.x)                   ?  Interval[Interval(1,     w.y  )] : Interval[]
enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(maximum), w::Interval, r::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ?  Interval[Interval(1,     X+1  )] : Interval[]
enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(maximum), w::Interval, r::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ?  Interval[Interval(w.x+1, X+1  )] : Interval[]
enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(maximum), w::Interval, r::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(1,     w.y-1)] : Interval[]


enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(minimum), w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)     ?   Interval[Interval(w.y,   w.y+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.y, X+1)]   
enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(minimum), w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)       ?   Interval[Interval(w.x-1, w.x  )] : Interval[] #  _ReprVal(Interval   )# [Interval(1, w.x)]     
enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(minimum), w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.x+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x, w.y-1)] 
enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(minimum), w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)   ?   Interval[Interval(w.y-1, w.y  )] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x+1, w.y)] 

enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(maximum), w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)     ?   Interval[Interval(w.y,   w.y+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.y, X+1)]   
enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(maximum), w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)       ?   Interval[Interval(w.x-1, w.x  )] : Interval[] #  _ReprVal(Interval   )# [Interval(1, w.x)]     
enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(maximum), w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.x+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x, w.y-1)] 
enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(maximum), w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)   ?   Interval[Interval(w.y-1, w.y  )] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x+1, w.y)] 

enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(maximum), w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)     ?   Interval[Interval(w.y,   X+1  )] : Interval[] #  _ReprVal(Interval(w.y, w.y+1)   )# [Interval(w.y, X+1)]   
enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(maximum), w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)       ?   Interval[Interval(1,     w.x  )] : Interval[] #  _ReprVal(Interval(w.x-1, w.x)   )# [Interval(1, w.x)]     
enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(maximum), w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.y-1)] : Interval[] #  _ReprVal(Interval(w.x, w.x+1)   )# [Interval(w.x, w.y-1)] 
enumAccReprAggr(f::AttributeMaximumFeatureType, a::typeof(maximum), w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)   ?   Interval[Interval(w.x+1, w.y  )] : Interval[] #  _ReprVal(Interval(w.y-1, w.y)   )# [Interval(w.x+1, w.y)] 

enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(minimum), w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)     ?   Interval[Interval(w.y,   X+1  )] : Interval[] #  _ReprVal(Interval(w.y, w.y+1)   )# [Interval(w.y, X+1)]   
enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(minimum), w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)       ?   Interval[Interval(1,     w.x  )] : Interval[] #  _ReprVal(Interval(w.x-1, w.x)   )# [Interval(1, w.x)]     
enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(minimum), w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.y-1)] : Interval[] #  _ReprVal(Interval(w.x, w.x+1)   )# [Interval(w.x, w.y-1)] 
enumAccReprAggr(f::AttributeMinimumFeatureType, a::typeof(minimum), w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)   ?   Interval[Interval(w.x+1, w.y  )] : Interval[] #  _ReprVal(Interval(w.y-1, w.y)   )# [Interval(w.x+1, w.y)] 


enumAccReprAggr(f::FeatureTypeFun, a::Function, w::Interval, r::_IA_AorO,       X::Integer) = 
	Iterators.flatten([enumAccReprAggr(f, a, w, r, X) for r in IA72IARelations(IA_AorO)])
enumAccReprAggr(f::FeatureTypeFun, a::Function, w::Interval, r::_IA_AiorOi,     X::Integer) = 
	Iterators.flatten([enumAccReprAggr(f, a, w, r, X) for r in IA72IARelations(IA_AiorOi)])
enumAccReprAggr(f::FeatureTypeFun, a::Function, w::Interval, r::_IA_DorBorE,    X::Integer) = 
	Iterators.flatten([enumAccReprAggr(f, a, w, r, X) for r in IA72IARelations(IA_DorBorE)])
enumAccReprAggr(f::FeatureTypeFun, a::Function, w::Interval, r::_IA_DiorBiorEi, X::Integer) = 
	Iterators.flatten([enumAccReprAggr(f, a, w, r, X) for r in IA72IARelations(IA_DiorBiorEi)])
enumAccReprAggr(f::FeatureTypeFun, a::Function, w::Interval, r::_IA_I,          X::Integer) = 
	Iterators.flatten([enumAccReprAggr(f, a, w, r, X) for r in IA72IARelations(IA_I)])







 
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
