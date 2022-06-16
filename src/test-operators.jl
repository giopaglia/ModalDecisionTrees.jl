export EQ, GT, LT, GEQ, LEQ,
        existential_aggregator, aggregator_bottom,
        TestOperatorFun, Aggregator

const Aggregator = Function

const TestOperatorFun = Function
################################################################################
################################################################################

# (Rational(60,100))

# # TODO improved version for Rational numbers
# # TODO check
# @inline test_op_partialsort!(test_op::_CanonicalFeatureGeqSoft, vals::Vector{T}) where {T} = 
#   partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)); rev=true)
# @inline test_op_partialsort!(test_op::_CanonicalFeatureLeqSoft, vals::Vector{T}) where {T} = 
#   partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)))

# @inline computePropositionalThreshold(test_op::Union{_CanonicalFeatureGeqSoft,_CanonicalFeatureLeqSoft}, w::AbstractWorld, channel::DimensionalChannel{T,N}) where {T,N} = begin
#   vals = vec(ch_readWorld(w,channel))
#   test_op_partialsort!(test_op,vals)
# end
# @inline computePropositionalThresholdMany(test_ops::Vector{<:TestOperator}, w::AbstractWorld, channel::DimensionalChannel{T,N}) where {T,N} = begin
#   vals = vec(ch_readWorld(w,channel))
#   (test_op_partialsort!(test_op,vals) for test_op in test_ops)
# end

# @inline test_decision(test_operator::_CanonicalFeatureGeqSoft, w::AbstractWorld, channel::DimensionalChannel{T,N}, threshold::Real) where {T,N} = begin 
#   ys = 0
#   # TODO write with reduce, and optimize it (e.g. by stopping early if the decision is reached already)
#   vals = ch_readWorld(w,channel)
#   for x in vals
#     if x >= threshold
#       ys+=1
#     end
#   end
#   (ys/length(vals)) >= test_operator.alpha
# end

# @inline test_decision(test_operator::_CanonicalFeatureLeqSoft, w::AbstractWorld, channel::DimensionalChannel{T,N}, threshold::Real) where {T,N} = begin 
#   ys = 0
#   # TODO write with reduce, and optimize it (e.g. by stopping early if the decision is reached already)
#   vals = ch_readWorld(w,channel)
#   for x in vals
#     if x <= threshold
#       ys+=1
#     end
#   end
#   (ys/length(vals)) >= test_operator.alpha
# end

# const all_lowlevel_test_operators = [
#     CanonicalFeatureGeq, CanonicalFeatureLeq,
#     SoftenedOperators...
#   ]

# const all_ordered_test_operators = [
#     CanonicalFeatureGeq, CanonicalFeatureLeq,
#     SoftenedOperators...
#   ]
# const all_test_operators_order = [
#     CanonicalFeatureGeq, CanonicalFeatureLeq,
#     SoftenedOperators...
#   ]
# sort_test_operators!(x::Vector{TO}) where {TO<:TestOperator} = begin
#   intersect(all_test_operators_order, x)
# end

# crisp operators


equality_operator(x::S,  y::S)       where {S} = ==(x,y)  # =
greater_than_operator(x::S,  y::S)   where {S} =  >(x,y)  # >
lesser_than_operator(x::S,  y::S)    where {S} =  <(x,y)  # <
greater_eq_than_operator(x::S, y::S) where {S} =  ≥(x,y)  # ≥
lesser_eq_than_operator(x::S, y::S)  where {S} =  ≤(x,y)  # ≤


test_operator_inverse(::typeof(≥))  = <
test_operator_inverse(::typeof(≤))  = >
test_operator_inverse(::typeof(<))  = ≥
test_operator_inverse(::typeof(>))  = ≤
test_operator_inverse(::typeof(==)) = !=
test_operator_inverse(::typeof(!=)) = ==


existential_aggregator(::typeof(equality_operator))        = ∪
existential_aggregator(::typeof(greater_than_operator))    = maximum
existential_aggregator(::typeof(lesser_than_operator))     = minimum
existential_aggregator(::typeof(greater_eq_than_operator)) = maximum
existential_aggregator(::typeof(lesser_eq_than_operator))  = minimum

# bottom_aggregator(::typeof(∪))          = TODO
aggregator_bottom(::typeof(maximum), T::Type) = typemin(T)
aggregator_bottom(::typeof(minimum), T::Type) = typemax(T)

evaluate_thresh_decision(operator::TestOperatorFun, gamma::T, a::T) where {T} = operator(gamma, a)

aggregator_to_binary(::typeof(maximum)) = max
aggregator_to_binary(::typeof(minimum)) = min

dual_test_operator(::typeof(≥)) = ≤
dual_test_operator(::typeof(≤)) = ≥

const EQ  = equality_operator
const GT  = greater_than_operator
const LT  = lesser_than_operator
const GEQ = greater_eq_than_operator
const LEQ = lesser_eq_than_operator


OrderingTestOperator = Union{
    typeof(GT),
    typeof(LT),
    typeof(GEQ),
    typeof(LEQ),
}

# TODO findu out whether these are needed or preferred
existential_aggregator(::typeof(==)) = ∪
existential_aggregator(::typeof(>))  = maximum
existential_aggregator(::typeof(<))  = minimum
existential_aggregator(::typeof(≥))  = maximum
existential_aggregator(::typeof(≤))  = minimum

# fuzzy operators

# =ₕ
function get_fuzzy_linear_eq(h::T, fuzzy_type::Type{<:Real} = Float64) where {T}
  fun = function (x::S, y::S) where {S}
    Δ = y-x
    if abs(Δ) ≥ h
      zero(fuzzy_type)
    else
      fuzzy_type(1-(abs(Δ)/h))
    end
  end
  @eval global existential_aggregator(::typeof($fun)) = ∪
  fun
end


# >ₕ
function get_fuzzy_linear_gt(h::T, fuzzy_type::Type{<:Real} = Float64) where {T}
  fun = function (x::S, y::S) where {S}
    Δ = y-x
    if Δ ≥ 0
      zero(fuzzy_type)
    elseif Δ ≤ -h
      one(fuzzy_type)
    else
      fuzzy_type(Δ/h)
    end
  end
  @eval global existential_aggregator(::typeof($fun)) = maximum
  fun
end

# <ₕ
function get_fuzzy_linear_lt(h::T, fuzzy_type::Type{<:Real} = Float64) where {T}
  fun = function (x::S, y::S) where {S}
    Δ = y-x
    if Δ ≥ h
      one(fuzzy_type)
    elseif Δ ≤ 0
      zero(fuzzy_type)
    else
      fuzzy_type(Δ/h)
    end
  end
  @eval global existential_aggregator(::typeof($fun)) = minimum
  fun
end


# ≧ₕ
function get_fuzzy_linear_geq(h::T, fuzzy_type::Type{<:Real} = Float64) where {T}
  fun = function (x::S, y::S) where {S}
    Δ = y-x
    if Δ ≤ 0
      one(fuzzy_type)
    elseif Δ ≥ h
      zero(fuzzy_type)
    else
      fuzzy_type(1-Δ/h)
    end
  end
  @eval global existential_aggregator(::typeof($fun)) = maximum
  fun
end


# ≦ₕ
function get_fuzzy_linear_leq(h::T, fuzzy_type::Type{<:Real} = Float64) where {T}
  fun = function (x::S, y::S) where {S}
    Δ = x-y
    if Δ ≤ 0
      one(fuzzy_type)
    elseif Δ ≥ h
      zero(fuzzy_type)
    else
      fuzzy_type(1-Δ/h)
    end
  end
  @eval global existential_aggregator(::typeof($fun)) = minimum
  fun
end

# ≥ₕ
function get_fuzzy_linear_geqt(h::T, fuzzy_type::Type{<:Real} = Float64) where {T}
  h_2 = h/2
  fun = function (x::S, y::S) where {S}
    Δ = y-x
    if Δ ≥ h_2
      zero(fuzzy_type)
    elseif Δ ≤ -h_2
      one(fuzzy_type)
    else
      fuzzy_type((h_2-Δ)/h)
    end
  end
  @eval global existential_aggregator(::typeof($fun)) = maximum
  fun
end

# ≤ₕ
function get_fuzzy_linear_leqt(h::T, fuzzy_type::Type{<:Real} = Float64) where {T}
  h_2 = h/2
  fun = function (x::S, y::S) where {S}
    Δ = y-x
    if Δ ≥ h_2
      one(fuzzy_type)
    elseif Δ ≤ -h_2
      zero(fuzzy_type)
    else
      fuzzy_type((Δ+h_2)/h)
    end
  end
  @eval global existential_aggregator(::typeof($fun)) = minimum
  fun
end

# h = 4
# v1 = 0
# v2 = -4:4

# op_fuzzy_eq = get_fuzzy_linear_eq(h)
# op_fuzzy_gt = get_fuzzy_linear_gt(h)
# op_fuzzy_lt = get_fuzzy_linear_lt(h)
# op_fuzzy_geqt = get_fuzzy_linear_geqt(h)
# op_fuzzy_leqt = get_fuzzy_linear_leqt(h)
# op_fuzzy_geq = get_fuzzy_linear_geq(h)
# op_fuzzy_leq = get_fuzzy_linear_leq(h)

# zip(v2, eq.(v1, v2)) |> collect
# zip(v2, gt.(v1, v2)) |> collect
# zip(v2, lt.(v1, v2)) |> collect
# zip(v2, geq.(v1, v2)) |> collect
# zip(v2, leq.(v1, v2)) |> collect
# zip(v2, op_fuzzy_eq.(v1, v2)) |> collect
# zip(v2, op_fuzzy_gt.(v1, v2)) |> collect
# zip(v2, op_fuzzy_lt.(v1, v2)) |> collect
# zip(v2, op_fuzzy_geqt.(v1, v2)) |> collect
# zip(v2, op_fuzzy_leqt.(v1, v2)) |> collect
# zip(v2, op_fuzzy_geq.(v1, v2)) |> collect
# zip(v2, op_fuzzy_leq.(v1, v2)) |> collect

################################################################################
################################################################################

export MixedFeature, CanonicalFeature, CanonicalFeatureGeq, CanonicalFeatureLeq

abstract type CanonicalFeature end

# ⪴ and ⪳, that is, "*all* of the values on this world are at least, or at most ..."
struct _CanonicalFeatureGeq <: CanonicalFeature end; const CanonicalFeatureGeq  = _CanonicalFeatureGeq();
struct _CanonicalFeatureLeq <: CanonicalFeature end; const CanonicalFeatureLeq  = _CanonicalFeatureLeq();

export CanonicalFeatureGeq_95, CanonicalFeatureGeq_90, CanonicalFeatureGeq_85, CanonicalFeatureGeq_80, CanonicalFeatureGeq_75, CanonicalFeatureGeq_70, CanonicalFeatureGeq_60,
       CanonicalFeatureLeq_95, CanonicalFeatureLeq_90, CanonicalFeatureLeq_85, CanonicalFeatureLeq_80, CanonicalFeatureLeq_75, CanonicalFeatureLeq_70, CanonicalFeatureLeq_60

# ⪴_α and ⪳_α, that is, "*at least α⋅100 percent* of the values on this world are at least, or at most ..."

struct _CanonicalFeatureGeqSoft  <: CanonicalFeature
  alpha :: AbstractFloat
  _CanonicalFeatureGeqSoft(a::T) where {T<:Real} = (a > 0 && a < 1) ? new(a) : throw_n_log("Invalid instantiation for test operator: _CanonicalFeatureGeqSoft($(a))")
end;
struct _CanonicalFeatureLeqSoft  <: CanonicalFeature
  alpha :: AbstractFloat
  _CanonicalFeatureLeqSoft(a::T) where {T<:Real} = (a > 0 && a < 1) ? new(a) : throw_n_log("Invalid instantiation for test operator: _CanonicalFeatureLeqSoft($(a))")
end;

const CanonicalFeatureGeq_95  = _CanonicalFeatureGeqSoft((Rational(95,100)));
const CanonicalFeatureGeq_90  = _CanonicalFeatureGeqSoft((Rational(90,100)));
const CanonicalFeatureGeq_85  = _CanonicalFeatureGeqSoft((Rational(85,100)));
const CanonicalFeatureGeq_80  = _CanonicalFeatureGeqSoft((Rational(80,100)));
const CanonicalFeatureGeq_75  = _CanonicalFeatureGeqSoft((Rational(75,100)));
const CanonicalFeatureGeq_70  = _CanonicalFeatureGeqSoft((Rational(70,100)));
const CanonicalFeatureGeq_60  = _CanonicalFeatureGeqSoft((Rational(60,100)));

const CanonicalFeatureLeq_95  = _CanonicalFeatureLeqSoft((Rational(95,100)));
const CanonicalFeatureLeq_90  = _CanonicalFeatureLeqSoft((Rational(90,100)));
const CanonicalFeatureLeq_85  = _CanonicalFeatureLeqSoft((Rational(85,100)));
const CanonicalFeatureLeq_80  = _CanonicalFeatureLeqSoft((Rational(80,100)));
const CanonicalFeatureLeq_75  = _CanonicalFeatureLeqSoft((Rational(75,100)));
const CanonicalFeatureLeq_70  = _CanonicalFeatureLeqSoft((Rational(70,100)));
const CanonicalFeatureLeq_60  = _CanonicalFeatureLeqSoft((Rational(60,100)));

MixedFeature = Union{ModalFeature,CanonicalFeature,Function,Tuple{TestOperatorFun,Function},Tuple{TestOperatorFun,ModalFeature}}

############################################################################################


display_feature_test_operator_pair(feature::ModalFeature,     test_operator::TestOperatorFun)        = "$(feature) $(test_operator)"

display_feature_test_operator_pair(feature::SingleAttributeMin,     test_operator::typeof(≥))        = "A$(feature.i_attribute) ⪴"
display_feature_test_operator_pair(feature::SingleAttributeMax,     test_operator::typeof(≤))        = "A$(feature.i_attribute) ⪳"
display_feature_test_operator_pair(feature::SingleAttributeSoftMin, test_operator::typeof(≥))        = "A$(feature.i_attribute) $("⪴" * util.subscriptnumber(rstrip(rstrip(string(alpha(feature)*100), '0'), '.')))"
display_feature_test_operator_pair(feature::SingleAttributeSoftMax, test_operator::typeof(≤))        = "A$(feature.i_attribute) $("⪳" * util.subscriptnumber(rstrip(rstrip(string(alpha(feature)*100), '0'), '.')))"

display_feature_test_operator_pair(feature::SingleAttributeMin,     test_operator::typeof(<))        = "A$(feature.i_attribute) ⪶"
display_feature_test_operator_pair(feature::SingleAttributeMax,     test_operator::typeof(>))        = "A$(feature.i_attribute) ⪵"
display_feature_test_operator_pair(feature::SingleAttributeSoftMin, test_operator::typeof(<))        = "A$(feature.i_attribute) $("⪶" * util.subscriptnumber(rstrip(rstrip(string(alpha(feature)*100), '0'), '.')))"
display_feature_test_operator_pair(feature::SingleAttributeSoftMax, test_operator::typeof(>))        = "A$(feature.i_attribute) $("⪵" * util.subscriptnumber(rstrip(rstrip(string(alpha(feature)*100), '0'), '.')))"

display_feature_test_operator_pair(feature::SingleAttributeMin,     test_operator::typeof(≤))        = "A$(feature.i_attribute) ↘"
display_feature_test_operator_pair(feature::SingleAttributeMax,     test_operator::typeof(≥))        = "A$(feature.i_attribute) ↗"
display_feature_test_operator_pair(feature::SingleAttributeSoftMin, test_operator::typeof(≤))        = "A$(feature.i_attribute) $("↘" * util.subscriptnumber(rstrip(rstrip(string(alpha(feature)*100), '0'), '.')))"
display_feature_test_operator_pair(feature::SingleAttributeSoftMax, test_operator::typeof(≥))        = "A$(feature.i_attribute) $("↗" * util.subscriptnumber(rstrip(rstrip(string(alpha(feature)*100), '0'), '.')))"
