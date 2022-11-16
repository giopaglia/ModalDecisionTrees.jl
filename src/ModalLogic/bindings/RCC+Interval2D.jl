goes_with(::Type{Interval2D}, ::RCCRelation) = true

############################################################################################
# Methods for RCC8 relations and Interval2D's can be obtained by combining their 1D versions.
# Consider the following table:
#
#                      .-------------------------------------------------------.
#                      |         DC   EC   PO   TPP   T̅P̅P̅   NTPP   N̅T̅P̅P̅    Id  |
#                      |-------------------------------------------------------|
#                      | DC   |  DC | DC | DC | DC  | DC  |  DC  |  DC  |  DC  |
#                      | EC   |  DC | EC | EC | EC  | EC  |  EC  |  EC  |  EC  |
#                      | PO   |  DC | EC | PO | PO  | PO  |  PO  |  PO  |  PO  |
#                      | TPP  |  DC | EC | PO | TPP | PO  |  TPP |  PO  |  TPP |
#                      | T̅P̅P̅  |  DC | EC | PO | PO  | T̅P̅P̅ |  PO  |  T̅P̅P̅ |  T̅P̅P̅ |
#                      | NTPP |  DC | EC | PO | TPP | PO  | NTPP |  PO  |  TPP |
#                      | N̅T̅P̅P̅ |  DC | EC | PO | PO  | T̅P̅P̅ |  PO  | N̅T̅P̅P̅ |  T̅P̅P̅ |
#                      |  Id  |  DC | EC | PO | TPP | T̅P̅P̅ |  TPP |  T̅P̅P̅ |  Id  |
#                      '-------------------------------------------------------'
#
############################################################################################

_accessibles(w::Interval2D, ::_Topo_DC,    X::Integer, Y::Integer) =
	IterTools.distinct(
		Iterators.flatten((
			Iterators.product(_accessibles(w.x, Topo_DC,    X), _accessibles__(w.y, RelationGlob,Y)),
			Iterators.product(_accessibles__(w.x, RelationGlob,X), _accessibles(w.y, Topo_DC,    Y)),
			# TODO try avoiding the distinct, replacing the second line (RelationGlob,_accessibles) with 7 combinations of RelationGlob with Topo_EC, Topo_PO, Topo_TPP, Topo_TPPi, Topo_NTPP, Topo_NTPPi
		))
	)
_accessibles(w::Interval2D, ::_Topo_EC,    X::Integer, Y::Integer) =
	Iterators.flatten((
		Iterators.product(_accessibles(w.x, Topo_EC,    X), _accessibles(w.y, Topo_EC,    Y)),
		#
		Iterators.product(_accessibles(w.x, Topo_EC,    X), _accessibles(w.y, Topo_PO,    Y)),
		Iterators.product(_accessibles(w.x, Topo_EC,    X), _accessibles(w.y, Topo_TPP,   Y)),
		Iterators.product(_accessibles(w.x, Topo_EC,    X), _accessibles(w.y, Topo_TPPi,  Y)),
		Iterators.product(_accessibles(w.x, Topo_EC,    X), _accessibles(w.y, Topo_NTPP,  Y)),
		Iterators.product(_accessibles(w.x, Topo_EC,    X), _accessibles(w.y, Topo_NTPPi, Y)),
		Iterators.product(_accessibles(w.x, Topo_EC,    X), _accessibles(w.y, RelationId, Y)),
		#
		Iterators.product(_accessibles(w.x, Topo_PO,    X), _accessibles(w.y, Topo_EC,    Y)),
		Iterators.product(_accessibles(w.x, Topo_TPP,   X), _accessibles(w.y, Topo_EC,    Y)),
		Iterators.product(_accessibles(w.x, Topo_TPPi,  X), _accessibles(w.y, Topo_EC,    Y)),
		Iterators.product(_accessibles(w.x, Topo_NTPP,  X), _accessibles(w.y, Topo_EC,    Y)),
		Iterators.product(_accessibles(w.x, Topo_NTPPi, X), _accessibles(w.y, Topo_EC,    Y)),
		Iterators.product(_accessibles(w.x, RelationId, X), _accessibles(w.y, Topo_EC,    Y)),
	))
_accessibles(w::Interval2D, ::_Topo_PO,    X::Integer, Y::Integer) =
	Iterators.flatten((
		Iterators.product(_accessibles(w.x, Topo_PO,    X), _accessibles(w.y, Topo_PO,    Y)),
		#
		Iterators.product(_accessibles(w.x, Topo_TPP,   X), _accessibles(w.y, Topo_PO,    Y)),
		Iterators.product(_accessibles(w.x, Topo_TPPi,  X), _accessibles(w.y, Topo_PO,    Y)),
		Iterators.product(_accessibles(w.x, Topo_NTPP,  X), _accessibles(w.y, Topo_PO,    Y)),
		Iterators.product(_accessibles(w.x, Topo_NTPPi, X), _accessibles(w.y, Topo_PO,    Y)),
		Iterators.product(_accessibles(w.x, RelationId, X), _accessibles(w.y, Topo_PO,    Y)),
		#
		Iterators.product(_accessibles(w.x, Topo_PO,    X), _accessibles(w.y, Topo_TPP,   Y)),
		Iterators.product(_accessibles(w.x, Topo_PO,    X), _accessibles(w.y, Topo_TPPi,  Y)),
		Iterators.product(_accessibles(w.x, Topo_PO,    X), _accessibles(w.y, Topo_NTPP,  Y)),
		Iterators.product(_accessibles(w.x, Topo_PO,    X), _accessibles(w.y, Topo_NTPPi, Y)),
		Iterators.product(_accessibles(w.x, Topo_PO,    X), _accessibles(w.y, RelationId, Y)),
		#
		Iterators.product(_accessibles(w.x, Topo_TPP,   X), _accessibles(w.y, Topo_TPPi,  Y)),
		Iterators.product(_accessibles(w.x, Topo_TPPi,  X), _accessibles(w.y, Topo_TPP,   Y)),
		#
		Iterators.product(_accessibles(w.x, Topo_TPPi,  X), _accessibles(w.y, Topo_NTPP,  Y)),
		Iterators.product(_accessibles(w.x, Topo_NTPP,  X), _accessibles(w.y, Topo_TPPi,  Y)),
		#
		Iterators.product(_accessibles(w.x, Topo_TPP,   X), _accessibles(w.y, Topo_NTPPi, Y)),
		Iterators.product(_accessibles(w.x, Topo_NTPP,  X), _accessibles(w.y, Topo_NTPPi, Y)),
		Iterators.product(_accessibles(w.x, Topo_NTPPi, X), _accessibles(w.y, Topo_TPP,   Y)),
		Iterators.product(_accessibles(w.x, Topo_NTPPi, X), _accessibles(w.y, Topo_NTPP,  Y)),
	))
_accessibles(w::Interval2D, ::_Topo_TPP,   X::Integer, Y::Integer) =
	Iterators.flatten((
		Iterators.product(_accessibles(w.x, Topo_TPP,   X), _accessibles(w.y, Topo_TPP,   Y)),
		#
		Iterators.product(_accessibles(w.x, Topo_TPP,   X), _accessibles(w.y, Topo_NTPP,  Y)),
		Iterators.product(_accessibles(w.x, Topo_NTPP,  X), _accessibles(w.y, Topo_TPP,   Y)),
		#
		Iterators.product(_accessibles(w.x, Topo_TPP,   X), _accessibles(w.y, RelationId, Y)),
		Iterators.product(_accessibles(w.x, RelationId, X), _accessibles(w.y, Topo_TPP,   Y)),
		#
		Iterators.product(_accessibles(w.x, Topo_NTPP,  X), _accessibles(w.y, RelationId, Y)),
		Iterators.product(_accessibles(w.x, RelationId, X), _accessibles(w.y, Topo_NTPP,  Y)),
	))

_accessibles(w::Interval2D, ::_Topo_TPPi,  X::Integer, Y::Integer) =
	Iterators.flatten((
		Iterators.product(_accessibles(w.x, Topo_TPPi,  X), _accessibles(w.y, Topo_TPPi,  Y)),
		#
		Iterators.product(_accessibles(w.x, Topo_TPPi,  X), _accessibles(w.y, Topo_NTPPi, Y)),
		Iterators.product(_accessibles(w.x, Topo_NTPPi, X), _accessibles(w.y, Topo_TPPi,  Y)),
		#
		Iterators.product(_accessibles(w.x, Topo_TPPi,  X), _accessibles(w.y, RelationId, Y)),
		Iterators.product(_accessibles(w.x, RelationId, X), _accessibles(w.y, Topo_TPPi,  Y)),
		#
		Iterators.product(_accessibles(w.x, Topo_NTPPi, X), _accessibles(w.y, RelationId, Y)),
		Iterators.product(_accessibles(w.x, RelationId, X), _accessibles(w.y, Topo_NTPPi, Y)),
	))

_accessibles(w::Interval2D, ::_Topo_NTPP,  X::Integer, Y::Integer) =
	# Iterators.flatten((
		Iterators.product(_accessibles(w.x, Topo_NTPP,  X), _accessibles(w.y, Topo_NTPP,  Y))
		# , ))
_accessibles(w::Interval2D, ::_Topo_NTPPi, X::Integer, Y::Integer) =
	# Iterators.flatten((
		Iterators.product(_accessibles(w.x, Topo_NTPPi, X), _accessibles(w.y, Topo_NTPPi, Y))
	# , ))

############################################################################################

_accessibles(w::Interval2D, r::_TopoRelRCC5,  XYZ::Vararg{Integer,2}) =
    Iterators.flatten((_accessibles(w, IA_r,  XYZ...) for IA_r in RCC52IARelations(r)))
    # Iterators.flatten((_accessibles(w, RCC8_r,  XYZ...) for RCC8_r in RCC52RCC8Relations(r)))
    # Iterators.flatten((_accessibles(w, IA_r,  XYZ...) for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)))

############################################################################################
