############################################################################################
# RCC topological relations + definitions for Interval
############################################################################################

abstract type RCCRelation <: TopologicalRelation end

goes_with(::Type{Interval}, ::RCCRelation) = true

############################################################################################
# RCC8 topological relations (plus equality, i.e. RelationId):
# - Externally connected
# - Partially overlapping
# - Tangential proper part
# - Tangential proper part inverse
# - Non-tangential proper part
# - Non-tangential proper part inverse
############################################################################################
# Graphical representation of R((x,y),(z,t)) for R ∈ RCC8
# 
#                                                  x                   y                    
#                                                  |-------------------|                    
#                                                  .                   .                    
#                                                  .                   .  z        t        
# Disconnected                         (DC)        .                   . |--------|         
#                                                  .                   .                    
#                                                  .                   z         t          
# Externally connected                 (EC)        .                   |---------|          
#                                                  .                   .                    
#                                                  .                z     t                 
# Partially overlapping                (PO)        .                |-----|                 
#                                                  .                   .                    
#                                                  .             z     t                    
# Tangential proper part               (TPP)       .             |-----|                    
#                                                  .                   .                    
#                                                  z                   .     t              
# Tangential proper part inverse       (T̅P̅P̅)       |-------------------------|              
#                                                  .                   .                    
#                                                  .           z       .                    
# Non-tangential proper part           (NTPP)      .           |-----| .                    
#                                                  .                   .                    
#                                                z .                   . t                  
# Non-tangential proper part inverse   (N̅T̅P̅P̅)    |-----------------------|                  
# 
############################################################################################
#
# Relations for RCC8
abstract type _TopoRelRCC8 <: RCCRelation end
struct _Topo_DC     <: _TopoRelRCC8 end; const Topo_DC     = _Topo_DC();     # Disconnected
struct _Topo_EC     <: _TopoRelRCC8 end; const Topo_EC     = _Topo_EC();     # Externally connected
struct _Topo_PO     <: _TopoRelRCC8 end; const Topo_PO     = _Topo_PO();     # Partially overlapping
struct _Topo_TPP    <: _TopoRelRCC8 end; const Topo_TPP    = _Topo_TPP();    # Tangential proper part
struct _Topo_TPPi   <: _TopoRelRCC8 end; const Topo_TPPi   = _Topo_TPPi();   # Tangential proper part inverse
struct _Topo_NTPP   <: _TopoRelRCC8 end; const Topo_NTPP   = _Topo_NTPP();   # Non-tangential proper part
struct _Topo_NTPPi  <: _TopoRelRCC8 end; const Topo_NTPPi  = _Topo_NTPPi();  # Non-tangential proper part inverse

Base.show(io::IO, ::_Topo_DC)    = print(io, "DC")
Base.show(io::IO, ::_Topo_EC)    = print(io, "EC")
Base.show(io::IO, ::_Topo_PO)    = print(io, "PO")
Base.show(io::IO, ::_Topo_TPP)   = print(io, "TPP")
Base.show(io::IO, ::_Topo_TPPi)  = print(io, "T̅P̅P̅")
Base.show(io::IO, ::_Topo_NTPP)  = print(io, "NTPP")
Base.show(io::IO, ::_Topo_NTPPi) = print(io, "N̅T̅P̅P̅")

############################################################################################

# Coarser relations for RCC5
abstract type _TopoRelRCC5 <: RCCRelation end
struct _Topo_DR     <: _TopoRelRCC5 end; const Topo_DR     = _Topo_DR();     # Disjointed
struct _Topo_PP     <: _TopoRelRCC5 end; const Topo_PP     = _Topo_PP();     # Proper part
struct _Topo_PPi    <: _TopoRelRCC5 end; const Topo_PPi    = _Topo_PPi();    # Proper part inverse

Base.show(io::IO, ::_Topo_DR)    = print(io, "DR")
Base.show(io::IO, ::_Topo_PP)    = print(io, "PP")
Base.show(io::IO, ::_Topo_PPi)   = print(io, "P̅P̅")

############################################################################################

# 7 RCC8 Relations
const RCC8Relations = [Topo_DC, Topo_EC, Topo_PO, Topo_TPP, Topo_TPPi, Topo_NTPP, Topo_NTPPi]

# 4 RCC5 Relations
const RCC5Relations = [Topo_DR, Topo_PO, Topo_PP, Topo_PPi]

############################################################################################

# It is conveniente to define RCC relations as unions of IA relations
const _TopoRelRCC8FromIA = Union{_Topo_DC,_Topo_EC,_Topo_PO,_Topo_TPP,_Topo_TPPi}

topo2IARelations(::_Topo_DC)     = [IA_L,  IA_Li]
topo2IARelations(::_Topo_EC)     = [IA_A,  IA_Ai]
topo2IARelations(::_Topo_PO)     = [IA_O,  IA_Oi]
topo2IARelations(::_Topo_TPP)    = [IA_B,  IA_E]
topo2IARelations(::_Topo_TPPi)   = [IA_Bi, IA_Ei]
topo2IARelations(::_Topo_NTPP)   = [IA_D]
topo2IARelations(::_Topo_NTPPi)  = [IA_Di]

# TODO RCC5 can be better written as a combination of IA7 relations!
RCC52RCC8Relations(::_Topo_DR)   = [Topo_DC,    Topo_EC]
RCC52RCC8Relations(::_Topo_PP)   = [Topo_TPP,   Topo_NTPP]
RCC52RCC8Relations(::_Topo_PPi)  = [Topo_TPPi,  Topo_NTPPi]

RCC52IARelations(::_Topo_DR)   = [IA_L,  IA_Li,  IA_A,  IA_Ai]
RCC52IARelations(::_Topo_PP)   = [IA_B,  IA_E,   IA_D]
RCC52IARelations(::_Topo_PPi)  = [IA_Bi, IA_Ei,  IA_Di]

# Enumerate accessible worlds from a single world
_accessibles(w::Interval, r::_TopoRelRCC8FromIA,    X::Integer) = Iterators.flatten((_accessibles(w, IA_r,  X) for IA_r in topo2IARelations(r)))
# _accessibles(w::Interval, ::_Topo_DC,    X::Integer) = Iterators.flatten((_accessibles(w, IA_L,  X), _accessibles(w, IA_Li, X)))
# _accessibles(w::Interval, ::_Topo_EC,    X::Integer) = Iterators.flatten((_accessibles(w, IA_A,  X), _accessibles(w, IA_Ai, X)))
# _accessibles(w::Interval, ::_Topo_PO,    X::Integer) = Iterators.flatten((_accessibles(w, IA_O,  X), _accessibles(w, IA_Oi, X)))
# _accessibles(w::Interval, ::_Topo_TPP,   X::Integer) = Iterators.flatten((_accessibles(w, IA_B,  X), _accessibles(w, IA_E,  X)))
# _accessibles(w::Interval, ::_Topo_TPPi,  X::Integer) = Iterators.flatten((_accessibles(w, IA_Bi, X), _accessibles(w, IA_Ei, X)))
_accessibles(w::Interval, ::_Topo_NTPP,  X::Integer) = _accessibles(w, IA_D, X)
_accessibles(w::Interval, ::_Topo_NTPPi, X::Integer) = _accessibles(w, IA_Di, X)

# RCC5 computed as a combination
_accessibles(w::Interval, r::_TopoRelRCC5,  XYZ::Vararg{Integer,1}) =
	# Iterators.flatten((_accessibles(w, IA_r,  XYZ...) for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)))
    Iterators.flatten((_accessibles(w, IA_r,  XYZ...) for IA_r in RCC52IARelations(r)))

#=

computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, r::_TopoRelRCC8FromIA, channel::DimensionalChannel{T,1}) where {T} = begin
	maxExtrema(
		map((IA_r)->(yieldReprs(test_operator, enumAccRepr(test_operator, w, IA_r, length(channel)), channel)), topo2IARelations(r))
	)
end
compute_modal_gamma(test_operator::_TestOpGeq, w::Interval, r::_TopoRelRCC8FromIA, channel::DimensionalChannel{T,1}) where {T} = begin
	maximum(
		map((IA_r)->(yieldRepr(test_operator, enumAccRepr(test_operator, w, IA_r, length(channel)), channel)), topo2IARelations(r))
	)
end
compute_modal_gamma(test_operator::_TestOpLeq, w::Interval, r::_TopoRelRCC8FromIA, channel::DimensionalChannel{T,1}) where {T} = begin
	mininimum(
		map((IA_r)->(yieldRepr(test_operator, enumAccRepr(test_operator, w, IA_r, length(channel)), channel)), topo2IARelations(r))
	)
end

enumAccRepr(test_operator::TestOperator, w::Interval, ::_Topo_NTPP,  X::Integer) = enumAccRepr(test_operator, w, IA_D, X)
enumAccRepr(test_operator::TestOperator, w::Interval, ::_Topo_NTPPi, X::Integer) = enumAccRepr(test_operator, w, IA_Di, X)

computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, r::_TopoRelRCC5, channel::DimensionalChannel{T,1}) where {T} = begin
	maxExtrema(
		map((IA_r)->(yieldReprs(test_operator, enumAccRepr(test_operator, w, IA_r, size(channel)...), channel)), [IA_r for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)])
	)
end
compute_modal_gamma(test_operator::_TestOpGeq, w::Interval, r::_TopoRelRCC5, channel::DimensionalChannel{T,1}) where {T} = begin
	maximum(
		map((IA_r)->(yieldRepr(test_operator, enumAccRepr(test_operator, w, IA_r, size(channel)...), channel)), [IA_r for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)])
	)
end
compute_modal_gamma(test_operator::_TestOpLeq, w::Interval, r::_TopoRelRCC5, channel::DimensionalChannel{T,1}) where {T} = begin
	mininimum(
		map((IA_r)->(yieldRepr(test_operator, enumAccRepr(test_operator, w, IA_r, size(channel)...), channel)), [IA_r for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)])
	)
end

=#
	
