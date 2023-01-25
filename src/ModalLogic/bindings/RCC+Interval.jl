goeswith(::Type{Interval}, ::RCCRelation) = true

# Enumerate accessible worlds from a single world
_accessibles(fr::Full1DFrame, w::Interval, r::RCC8RelationFromIA) = Iterators.flatten((_accessibles(fr, w, IA_r) for IA_r in topo2IARelations(r)))
# _accessibles(fr::Full1DFrame, w::Interval, ::_Topo_DC) = Iterators.flatten((_accessibles(fr, w, IA_L), _accessibles(fr, w, IA_Li)))
# _accessibles(fr::Full1DFrame, w::Interval, ::_Topo_EC) = Iterators.flatten((_accessibles(fr, w, IA_A), _accessibles(fr, w, IA_Ai)))
# _accessibles(fr::Full1DFrame, w::Interval, ::_Topo_PO) = Iterators.flatten((_accessibles(fr, w, IA_O), _accessibles(fr, w, IA_Oi)))
# _accessibles(fr::Full1DFrame, w::Interval, ::_Topo_TPP) = Iterators.flatten((_accessibles(fr, w, IA_B), _accessibles(fr, w, IA_E)))
# _accessibles(fr::Full1DFrame, w::Interval, ::_Topo_TPPi) = Iterators.flatten((_accessibles(fr, w, IA_Bi), _accessibles(fr, w, IA_Ei)))
_accessibles(fr::Full1DFrame, w::Interval, ::_Topo_NTPP) = _accessibles(fr, w, IA_D)
_accessibles(fr::Full1DFrame, w::Interval, ::_Topo_NTPPi) = _accessibles(fr, w, IA_Di)

# RCC5 computed as a combination
_accessibles(fr::Full1DFrame, w::Interval, r::RCC5Relation) =
    # Iterators.flatten((_accessibles(Full1DFrame(fr), w, IA_r, ) for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)))
    Iterators.flatten((_accessibles(Full1DFrame(fr), w, IA_r, ) for IA_r in RCC52IARelations(r)))

#=

computeModalThresholdDual(test_operator::CanonicalFeatureGeq, w::Interval, r::RCC8RelationFromIA, channel::DimensionalChannel{T,1}) where {T} = begin
    maxExtrema(
        map((IA_r)->(yieldReprs(test_operator, enum_acc_repr(test_operator, w, IA_r, length(channel)), channel)), topo2IARelations(r))
    )
end
compute_modal_gamma(test_operator::CanonicalFeatureGeq, w::Interval, r::RCC8RelationFromIA, channel::DimensionalChannel{T,1}) where {T} = begin
    maximum(
        map((IA_r)->(yieldRepr(test_operator, enum_acc_repr(test_operator, w, IA_r, length(channel)), channel)), topo2IARelations(r))
    )
end
compute_modal_gamma(test_operator::CanonicalFeatureLeq, w::Interval, r::RCC8RelationFromIA, channel::DimensionalChannel{T,1}) where {T} = begin
    mininimum(
        map((IA_r)->(yieldRepr(test_operator, enum_acc_repr(test_operator, w, IA_r, length(channel)), channel)), topo2IARelations(r))
    )
end

enum_acc_repr(fr::Full1DFrame, test_operator::TestOperator, w::Interval, ::_Topo_NTPP,) = enum_acc_repr(test_operator, fr, w, IA_D)
enum_acc_repr(fr::Full1DFrame, test_operator::TestOperator, w::Interval, ::_Topo_NTPPi,) = enum_acc_repr(test_operator, fr, w, IA_Di)

computeModalThresholdDual(test_operator::CanonicalFeatureGeq, w::Interval, r::RCC5Relation, channel::DimensionalChannel{T,1}) where {T} = begin
    maxExtrema(
        map((IA_r)->(yieldReprs(test_operator, enum_acc_repr(test_operator, w, IA_r, size(channel)...), channel)), [IA_r for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)])
    )
end
compute_modal_gamma(test_operator::CanonicalFeatureGeq, w::Interval, r::RCC5Relation, channel::DimensionalChannel{T,1}) where {T} = begin
    maximum(
        map((IA_r)->(yieldRepr(test_operator, enum_acc_repr(test_operator, w, IA_r, size(channel)...), channel)), [IA_r for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)])
    )
end
compute_modal_gamma(test_operator::CanonicalFeatureLeq, w::Interval, r::RCC5Relation, channel::DimensionalChannel{T,1}) where {T} = begin
    mininimum(
        map((IA_r)->(yieldRepr(test_operator, enum_acc_repr(test_operator, w, IA_r, size(channel)...), channel)), [IA_r for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)])
    )
end

=#
    
