using SoleLogics.Relations

############################################################################################
# RCC topological relations + definitions for Interval
############################################################################################

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

computeModalThresholdDual(test_operator::CanonicalFeatureGeq, w::Interval, r::_TopoRelRCC8FromIA, channel::DimensionalChannel{T,1}) where {T} = begin
    maxExtrema(
        map((IA_r)->(yieldReprs(test_operator, enum_acc_repr(test_operator, w, IA_r, length(channel)), channel)), topo2IARelations(r))
    )
end
compute_modal_gamma(test_operator::CanonicalFeatureGeq, w::Interval, r::_TopoRelRCC8FromIA, channel::DimensionalChannel{T,1}) where {T} = begin
    maximum(
        map((IA_r)->(yieldRepr(test_operator, enum_acc_repr(test_operator, w, IA_r, length(channel)), channel)), topo2IARelations(r))
    )
end
compute_modal_gamma(test_operator::CanonicalFeatureLeq, w::Interval, r::_TopoRelRCC8FromIA, channel::DimensionalChannel{T,1}) where {T} = begin
    mininimum(
        map((IA_r)->(yieldRepr(test_operator, enum_acc_repr(test_operator, w, IA_r, length(channel)), channel)), topo2IARelations(r))
    )
end

enum_acc_repr(test_operator::TestOperator, w::Interval, ::_Topo_NTPP,  X::Integer) = enum_acc_repr(test_operator, w, IA_D, X)
enum_acc_repr(test_operator::TestOperator, w::Interval, ::_Topo_NTPPi, X::Integer) = enum_acc_repr(test_operator, w, IA_Di, X)

computeModalThresholdDual(test_operator::CanonicalFeatureGeq, w::Interval, r::_TopoRelRCC5, channel::DimensionalChannel{T,1}) where {T} = begin
    maxExtrema(
        map((IA_r)->(yieldReprs(test_operator, enum_acc_repr(test_operator, w, IA_r, size(channel)...), channel)), [IA_r for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)])
    )
end
compute_modal_gamma(test_operator::CanonicalFeatureGeq, w::Interval, r::_TopoRelRCC5, channel::DimensionalChannel{T,1}) where {T} = begin
    maximum(
        map((IA_r)->(yieldRepr(test_operator, enum_acc_repr(test_operator, w, IA_r, size(channel)...), channel)), [IA_r for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)])
    )
end
compute_modal_gamma(test_operator::CanonicalFeatureLeq, w::Interval, r::_TopoRelRCC5, channel::DimensionalChannel{T,1}) where {T} = begin
    mininimum(
        map((IA_r)->(yieldRepr(test_operator, enum_acc_repr(test_operator, w, IA_r, size(channel)...), channel)), [IA_r for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)])
    )
end

=#
