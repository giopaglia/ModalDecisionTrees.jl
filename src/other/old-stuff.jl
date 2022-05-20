export TestOperator,
        TestOpNone,
        TestOpGeq, TestOpLeq,
        _TestOpGeqSoft, _TestOpLeqSoft

abstract type TestOperator end

################################################################################
################################################################################

struct _TestOpNone  <: TestOperator end; const TestOpNone  = _TestOpNone();

################################################################################
################################################################################

abstract type TestOperatorPositive <: TestOperator end
abstract type TestOperatorNegative <: TestOperator end

polarity(::TestOperatorPositive) = true
polarity(::TestOperatorNegative) = false

@inline bottom(::TestOperatorPositive, T::Type) = typemin(T)
@inline bottom(::TestOperatorNegative, T::Type) = typemax(T)

@inline opt(::TestOperatorPositive) = max
@inline opt(::TestOperatorNegative) = min

# Warning: I'm assuming all operators are "closed" (= not strict, like >= and <=)
@inline evaluate_thresh_decision(::TestOperatorPositive, t::T, gamma::T) where {T} = (t <= gamma)
@inline evaluate_thresh_decision(::TestOperatorNegative, t::T, gamma::T) where {T} = (t >= gamma)

compute_modal_gamma(test_operator::Union{TestOperatorPositive,TestOperatorNegative}, w::WorldType, relation::AbstractRelation, channel::DimensionalChannel{T,N}) where {WorldType<:AbstractWorld,T,N} = begin
    worlds = accessibles([w], relation, channel)
    # TODO rewrite as reduce(opt(test_operator), (computePropositionalThreshold(test_operator, w, channel) for w in worlds); init=bottom(test_operator, T))
    v = bottom(test_operator, T)
    for w in worlds
        e = computePropositionalThreshold(test_operator, w, channel)
        v = opt(test_operator)(v,e)
    end
    v
end
computeModalThresholdDual(test_operator::TestOperatorPositive, w::WorldType, relation::AbstractRelation, channel::DimensionalChannel{T,N}) where {WorldType<:AbstractWorld,T,N} = begin
    worlds = accessibles([w], relation, channel)
    extr = (typemin(T),typemax(T))
    for w in worlds
        e = computePropositionalThresholdDual(test_operator, w, channel)
        extr = (min(extr[1],e[1]), max(extr[2],e[2]))
    end
    extr
end
computeModalThresholdMany(test_ops::Vector{<:TestOperator}, w::WorldType, relation::AbstractRelation, channel::DimensionalChannel{T,N}) where {WorldType<:AbstractWorld,T,N} = begin
    [compute_modal_gamma(test_op, w, relation, channel) for test_op in test_ops]
end

################################################################################
################################################################################

# ⪴ and ⪳, that is, "*all* of the values on this world are at least, or at most ..."
struct _TestOpGeq  <: TestOperatorPositive end; const TestOpGeq  = _TestOpGeq();
struct _TestOpLeq  <: TestOperatorNegative end; const TestOpLeq  = _TestOpLeq();

dual_test_operator(::_TestOpGeq) = TestOpLeq
dual_test_operator(::_TestOpLeq) = TestOpGeq

# TODO introduce singleton design pattern for these constants
primary_test_operator(x::_TestOpGeq) = TestOpGeq # x
primary_test_operator(x::_TestOpLeq) = TestOpGeq # dual_test_operator(x)

siblings(::_TestOpGeq) = []
siblings(::_TestOpLeq) = []

Base.show(io::IO, test_operator::_TestOpGeq) = print(io, "⪴")
Base.show(io::IO, test_operator::_TestOpLeq) = print(io, "⪳")

@inline computePropositionalThreshold(::_TestOpGeq, w::AbstractWorld, channel::DimensionalChannel{T,N}) where {T,N} = begin
    # println(_TestOpGeq)
    # println(w)
    # println(channel)
    # println(maximum(ch_readWorld(w,channel)))
    # readline()
    minimum(ch_readWorld(w,channel))
end
@inline computePropositionalThreshold(::_TestOpLeq, w::AbstractWorld, channel::DimensionalChannel{T,N}) where {T,N} = begin
    # println(_TestOpLeq)
    # println(w)
    # println(channel)
    # readline()
    maximum(ch_readWorld(w,channel))
end
@inline computePropositionalThresholdDual(::_TestOpGeq, w::AbstractWorld, channel::DimensionalChannel{T,N}) where {T,N} = extrema(ch_readWorld(w,channel))

@inline test_decision(test_operator::_TestOpGeq, w::AbstractWorld, channel::DimensionalChannel{T,N}, threshold::Number) where {T,N} = begin # TODO maybe this becomes SIMD, or sum/all(ch_readWorld(w,channel)  .<= threshold)
    # Source: https://stackoverflow.com/questions/47564825/check-if-all-the-elements-of-a-julia-array-are-equal
    # @inbounds
    # TODO try:
    # all(ch_readWorld(w,channel) .>= threshold)
    for x in ch_readWorld(w,channel)
        x >= threshold || return false
    end
    return true
end
@inline test_decision(test_operator::_TestOpLeq, w::AbstractWorld, channel::DimensionalChannel{T,N}, threshold::Number) where {T,N} = begin # TODO maybe this becomes SIMD, or sum/all(ch_readWorld(w,channel)  .<= threshold)
    # Source: https://stackoverflow.com/questions/47564825/check-if-all-the-elements-of-a-julia-array-are-equal
    # @info "WLes" w threshold #n ch_readWorld(w,channel)
    # @inbounds
    # TODO try:
    # all(ch_readWorld(w,channel) .<= threshold)
    for x in ch_readWorld(w,channel)
        x <= threshold || return false
    end
    return true
end

################################################################################
################################################################################

export TestOpGeq_95, TestOpGeq_90, TestOpGeq_85, TestOpGeq_80, TestOpGeq_75, TestOpGeq_70, TestOpGeq_60,
                TestOpLeq_95, TestOpLeq_90, TestOpLeq_85, TestOpLeq_80, TestOpLeq_75, TestOpLeq_70, TestOpLeq_60

# ⪴_α and ⪳_α, that is, "*at least α⋅100 percent* of the values on this world are at least, or at most ..."

struct _TestOpGeqSoft  <: TestOperatorPositive
  alpha :: AbstractFloat
  _TestOpGeqSoft(a::T) where {T<:Real} = (a > 0 && a < 1) ? new(a) : throw_n_log("Invalid instantiation for test operator: _TestOpGeqSoft($(a))")
end;
struct _TestOpLeqSoft  <: TestOperatorNegative
  alpha :: AbstractFloat
  _TestOpLeqSoft(a::T) where {T<:Real} = (a > 0 && a < 1) ? new(a) : throw_n_log("Invalid instantiation for test operator: _TestOpLeqSoft($(a))")
end;

const TestOpGeq_95  = _TestOpGeqSoft((Rational(95,100)));
const TestOpGeq_90  = _TestOpGeqSoft((Rational(90,100)));
const TestOpGeq_85  = _TestOpGeqSoft((Rational(85,100)));
const TestOpGeq_80  = _TestOpGeqSoft((Rational(80,100)));
const TestOpGeq_75  = _TestOpGeqSoft((Rational(75,100)));
const TestOpGeq_70  = _TestOpGeqSoft((Rational(70,100)));
const TestOpGeq_60  = _TestOpGeqSoft((Rational(60,100)));

const TestOpLeq_95  = _TestOpLeqSoft((Rational(95,100)));
const TestOpLeq_90  = _TestOpLeqSoft((Rational(90,100)));
const TestOpLeq_85  = _TestOpLeqSoft((Rational(85,100)));
const TestOpLeq_80  = _TestOpLeqSoft((Rational(80,100)));
const TestOpLeq_75  = _TestOpLeqSoft((Rational(75,100)));
const TestOpLeq_70  = _TestOpLeqSoft((Rational(70,100)));
const TestOpLeq_60  = _TestOpLeqSoft((Rational(60,100)));

alpha(x::_TestOpGeqSoft) = x.alpha
alpha(x::_TestOpLeqSoft) = x.alpha

dual_test_operator(x::_TestOpGeqSoft) = TestOpNone
dual_test_operator(x::_TestOpLeqSoft) = TestOpNone
# TODO The dual_test_operators for TestOpGeqSoft(alpha) is TestOpLeSoft(1-alpha), which is not defined yet.
# Define it, together with their dual_test_operator and computePropositionalThresholdDual
# dual_test_operator(x::_TestOpGeqSoft) = throw_n_log("If you use $(x), need to write computeModalThresholdDual for the primal test operator.")
# dual_test_operator(x::_TestOpLeqSoft) = throw_n_log("If you use $(x), need to write computeModalThresholdDual for the primal test operator.")

primary_test_operator(x::_TestOpGeqSoft) = x
primary_test_operator(x::_TestOpLeqSoft) = dual_test_operator(x)

const SoftenedOperators = [
                                            TestOpGeq_95, TestOpLeq_95,
                                            TestOpGeq_90, TestOpLeq_90,
                                            TestOpGeq_80, TestOpLeq_80,
                                            TestOpGeq_85, TestOpLeq_85,
                                            TestOpGeq_75, TestOpLeq_75,
                                            TestOpGeq_70, TestOpLeq_70,
                                            TestOpGeq_60, TestOpLeq_60,
                                        ]

siblings(x::Union{_TestOpGeqSoft,_TestOpLeqSoft}) = SoftenedOperators

Base.show(io::IO, test_operator::_TestOpGeqSoft) = print(io, "⪴" * subscriptnumber(rstrip(rstrip(string(alpha(test_operator)*100), '0'), '.')))
Base.show(io::IO, test_operator::_TestOpLeqSoft) = print(io, "⪳" * subscriptnumber(rstrip(rstrip(string(alpha(test_operator)*100), '0'), '.')))

# TODO improved version for Rational numbers
# TODO check
@inline test_op_partialsort!(test_op::_TestOpGeqSoft, vals::Vector{T}) where {T} = 
    partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)); rev=true)
@inline test_op_partialsort!(test_op::_TestOpLeqSoft, vals::Vector{T}) where {T} = 
    partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)))

@inline computePropositionalThreshold(test_op::Union{_TestOpGeqSoft,_TestOpLeqSoft}, w::AbstractWorld, channel::DimensionalChannel{T,N}) where {T,N} = begin
    vals = vec(ch_readWorld(w,channel))
    test_op_partialsort!(test_op,vals)
end
# @inline computePropositionalThresholdDual(test_op::_TestOpGeqSoft, w::AbstractWorld, channel::DimensionalChannel{T,N}) where {T,N} = begin
#   vals = vec(ch_readWorld(w,channel))
#   xmin = test_op_partialsort!(test_op,vec(ch_readWorld(w,channel)))
#   xmin = partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)); rev=true)
#   xmax = partialsort!(vals,ceil(Int, (alpha(test_op))*length(vals)))
#   xmin,xmax
# end
@inline computePropositionalThresholdMany(test_ops::Vector{<:TestOperator}, w::AbstractWorld, channel::DimensionalChannel{T,N}) where {T,N} = begin
    vals = vec(ch_readWorld(w,channel))
    (test_op_partialsort!(test_op,vals) for test_op in test_ops)
end

@inline test_decision(test_operator::_TestOpGeqSoft, w::AbstractWorld, channel::DimensionalChannel{T,N}, threshold::Number) where {T,N} = begin 
    ys = 0
    # TODO write with reduce, and optimize it (e.g. by stopping early if the decision is reached already)
    vals = ch_readWorld(w,channel)
    for x in vals
        if x >= threshold
            ys+=1
        end
    end
    (ys/length(vals)) >= test_operator.alpha
end

@inline test_decision(test_operator::_TestOpLeqSoft, w::AbstractWorld, channel::DimensionalChannel{T,N}, threshold::Number) where {T,N} = begin 
    ys = 0
    # TODO write with reduce, and optimize it (e.g. by stopping early if the decision is reached already)
    vals = ch_readWorld(w,channel)
    for x in vals
        if x <= threshold
            ys+=1
        end
    end
    (ys/length(vals)) >= test_operator.alpha
end

################################################################################
################################################################################


const all_lowlevel_test_operators = [
        TestOpGeq, TestOpLeq,
        SoftenedOperators...
    ]

const all_ordered_test_operators = [
        TestOpGeq, TestOpLeq,
        SoftenedOperators...
    ]
const all_test_operators_order = [
        TestOpGeq, TestOpLeq,
        SoftenedOperators...
    ]
sort_test_operators!(x::Vector{TO}) where {TO<:TestOperator} = begin
    intersect(all_test_operators_order, x)
end

################################################################################
################################################################################



function test_decision(
        X::DimensionalDataset{T},
        i_sample::Integer,
        w::AbstractWorld,
        feature::ModalFeature,
        test_operator::OrderingTestOperator,
        threshold::T) where {T}
    test_decision(X, i_sample, w, feature, existential_aggregator(test_operator), threshold)
end

function test_decision(
        X::DimensionalDataset{T},
        i_sample::Integer,
        w::AbstractWorld,
        feature::ModalFeature... ,
        aggregator::typeof(maximum),
        threshold::T) where {T}
    values = get_values ... (X, i_sample, w, feature.i_attribute...) ch_readWorld(w,channel)
    all_broadcast_sc(values, test_operator, threshold)
end

function test_decision(
        X::InterpretedModalDataset{T},
        i_sample::Integer,
        w::AbstractWorld,
        feature::ModalFeature,
        test_operator::TestOperatorFun,
        threshold::T) where {T}
    test_decision(X.domain, i_sample, w, feature, test_operator, threshold)
end


############################################################################################
############################################################################################


function computePropositionalThreshold(feature::ModalFeature, w::AbstractWorld, instance::DimensionalInstance{T,N}) where {T,N}
    interpret_feature(feature, inst_readWorld(w, instance)::DimensionalChannel{T,N-1})::T
end

computeModalThresholdDual(test_operator::TestOperatorFun, w::WorldType, relation::R where R<:_UnionOfRelations{relsTuple}, channel::DimensionalChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
  computePropositionalThresholdDual(test_operator, w, channel)
  fieldtypes(relsTuple)
compute_modal_gamma(test_operator::TestOperatorFun, w::WorldType, relation::R where R<:_UnionOfRelations{relsTuple}, channel::DimensionalChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
  computePropositionalThreshold(test_operator, w, channel)
  fieldtypes(relsTuple)

#=

# needed for GAMMAS

yieldReprs(test_operator::_TestOpGeq, repr::_ReprMax{Interval},  channel::DimensionalChannel{T,1}) where {T} =
    reverse(extrema(ch_readWorld(repr.w, channel)))::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprMin{Interval},  channel::DimensionalChannel{T,1}) where {T} =
    extrema(ch_readWorld(repr.w, channel))::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprVal{Interval},  channel::DimensionalChannel{T,1}) where {T} =
    (channel[repr.w.x],channel[repr.w.x])::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprNone{Interval}, channel::DimensionalChannel{T,1}) where {T} =
    (typemin(T),typemax(T))::NTuple{2,T}

yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, repr::_ReprMax{Interval},  channel::DimensionalChannel{T,1}) where {T} =
    maximum(ch_readWorld(repr.w, channel))::T
yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, repr::_ReprMin{Interval},  channel::DimensionalChannel{T,1}) where {T} =
    minimum(ch_readWorld(repr.w, channel))::T
yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, repr::_ReprVal{Interval},  channel::DimensionalChannel{T,1}) where {T} =
    channel[repr.w.x]::T
yieldRepr(test_operator::_TestOpGeq, repr::_ReprNone{Interval}, channel::DimensionalChannel{T,1}) where {T} =
    typemin(T)::T
yieldRepr(test_operator::_TestOpLeq, repr::_ReprNone{Interval}, channel::DimensionalChannel{T,1}) where {T} =
    typemax(T)::T

enum_acc_repr(test_operator::_TestOpGeq, w::Interval, ::_RelationGlob, X::Integer) = _ReprMax(Interval(1,X+1))
enum_acc_repr(test_operator::_TestOpLeq, w::Interval, ::_RelationGlob, X::Integer) = _ReprMin(Interval(1,X+1))

# TODO optimize relationGlob
computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, r::R where R<:AbstractRelation, channel::DimensionalChannel{T,1}) where {T} =
    yieldReprs(test_operator, enum_acc_repr(test_operator, w, r, size(channel)...), channel)
computeModalThreshold(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval, r::R where R<:AbstractRelation, channel::DimensionalChannel{T,1}) where {T} =
    yieldRepr(test_operator, enum_acc_repr(test_operator, w, r, size(channel)...), channel)

# TODO optimize relationGlob?
# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_RelationGlob, channel::DimensionalChannel{T,1}) where {T} = begin
#   # X = length(channel)
#   # println("Check!")
#   # println(test_operator)
#   # println(w)
#   # println(relation)
#   # println(channel)
#   # println(computePropositionalThresholdDual(test_operator, Interval(1,X+1), channel))
#   # readline()
#   # computePropositionalThresholdDual(test_operator, Interval(1,X+1), channel)
#   reverse(extrema(channel))
# end
# computeModalThreshold(test_operator::_TestOpGeq, w::Interval, ::_RelationGlob, channel::DimensionalChannel{T,1}) where {T} = begin
#   # TODO optimize this by replacing readworld with channel[1:X]...
#   # X = length(channel)
#   # maximum(ch_readWorld(Interval(1,X+1),channel))
#   maximum(channel)
# end
# computeModalThreshold(test_operator::_TestOpLeq, w::Interval, ::_RelationGlob, channel::DimensionalChannel{T,1}) where {T} = begin
#   # TODO optimize this by replacing readworld with channel[1:X]...
#   # X = length(channel)
#   # minimum(ch_readWorld(Interval(1,X+1),channel))
#   minimum(channel)
# end

    
ch_readWorld(w::Interval, channel::DimensionalChannel{T,1}) where {T} = channel[w.x:w.y-1]

=#



#=
# needed for GAMMAS

yieldReprs(test_operator::_TestOpGeq, repr::_ReprMax{Interval2D},  channel::DimensionalChannel{T,2}) where {T} =
  reverse(extrema(ch_readWorld(repr.w, channel)))::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprMin{Interval2D},  channel::DimensionalChannel{T,2}) where {T} =
  extrema(ch_readWorld(repr.w, channel))::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprVal{Interval2D},  channel::DimensionalChannel{T,2}) where {T} =
  (channel[repr.w.x.x, repr.w.y.x],channel[repr.w.x.x, repr.w.y.x])::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprNone{Interval2D}, channel::DimensionalChannel{T,2}) where {T} =
  (typemin(T),typemax(T))::NTuple{2,T}

yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, repr::_ReprMax{Interval2D},  channel::DimensionalChannel{T,2}) where {T} =
  maximum(ch_readWorld(repr.w, channel))::T
yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, repr::_ReprMin{Interval2D},  channel::DimensionalChannel{T,2}) where {T} =
  minimum(ch_readWorld(repr.w, channel))::T
yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, repr::_ReprVal{Interval2D},  channel::DimensionalChannel{T,2}) where {T} =
  channel[repr.w.x.x, repr.w.y.x]::T
yieldRepr(test_operator::_TestOpGeq, repr::_ReprNone{Interval2D}, channel::DimensionalChannel{T,2}) where {T} =
  typemin(T)::T
yieldRepr(test_operator::_TestOpLeq, repr::_ReprNone{Interval2D}, channel::DimensionalChannel{T,2}) where {T} =
  typemax(T)::T

enum_acc_repr(test_operator::_TestOpGeq, w::Interval2D, ::_RelationGlob, X::Integer, Y::Integer) = _ReprMax(Interval2D(Interval(1,X+1), Interval(1,Y+1)))
enum_acc_repr(test_operator::_TestOpLeq, w::Interval2D, ::_RelationGlob, X::Integer, Y::Integer) = _ReprMin(Interval2D(Interval(1,X+1), Interval(1,Y+1)))

# TODO write only one ExtremeModal/ExtremaModal
# TODO optimize relationGlob
computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval2D, r::R where R<:AbstractRelation, channel::DimensionalChannel{T,2}) where {T} = begin
  # if (channel == [412 489 559 619 784; 795 771 1317 854 1256; 971 874 878 1278 560] && w.x.x==1 && w.x.y==3 && w.y.x==3 && w.y.y==4)
  #   println(enum_acc_repr(test_operator, w, r, size(channel)...))
  #   readline()
  # end
  yieldReprs(test_operator, enum_acc_repr(test_operator, w, r, size(channel)...), channel)
end
compute_modal_gamma(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval2D, r::R where R<:AbstractRelation, channel::DimensionalChannel{T,2}) where {T} =
  yieldRepr(test_operator, enum_acc_repr(test_operator, w, r, size(channel)...), channel)
# channel = [1,2,3,2,8,349,0,830,7290,298,20,29,2790,27,90279,270,2722,79072,0]
# w = ModalLogic.Interval(3,9)
# # w = ModalLogic.Interval(3,4)
# for relation in ModalLogic.IARelations
#   ModalLogic.computeModalThresholdDual(TestOpGeq, w, relation, channel)
# end

# channel2 = randn(3,4)
# channel2[1:3,1]
# channel2[1:3,2]
# channel2[1:3,3]
# channel2[1:3,4]
# vals=channel2
# mapslices(maximum, vals, dims=1)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval2D, ::_RelationGlob, channel::DimensionalChannel{T,2}) where {T} = begin
#   # X = size(channel, 1)
#   # Y = size(channel, 2)
#   # println("Check!")
#   # println(test_operator)
#   # println(w)
#   # println(relation)
#   # println(channel)
#   # println(computePropositionalThresholdDual(test_operator, Interval2D(Interval(1,X+1), Interval(1, Y+1)), channel))
#   # readline()
#   # computePropositionalThresholdDual(test_operator, Interval2D(Interval(1,X+1), Interval(1, Y+1)), channel)
#   reverse(extrema(channel))
# end
# compute_modal_gamma(test_operator::_TestOpGeq, w::Interval2D, ::_RelationGlob, channel::DimensionalChannel{T,2}) where {T} = begin
#   # TODO optimize this by replacing readworld with channel[1:X]...
#   # X = size(channel, 1)
#   # Y = size(channel, 2)
#   # maximum(channel[1:X,1:Y])
#   maximum(channel)
# end
# compute_modal_gamma(test_operator::_TestOpLeq, w::Interval2D, ::_RelationGlob, channel::DimensionalChannel{T,2}) where {T} = begin
#   # TODO optimize this by replacing readworld with channel[1:X]...
#   # X = size(channel, 1)
#   # Y = size(channel, 2)
#   # println(channel)
#   # println(w)
#   # println(minimum(channel[1:X,1:Y]))
#   # readline()
#   # minimum(channel[1:X,1:Y])
#   minimum(channel)
# end


@inline ch_readWorld(w::Interval2D, channel::DimensionalChannel{T,2}) where {T} = channel[w.x.x:w.x.y-1,w.y.x:w.y.y-1]

=#


# Other options:
# accessibles2_1_2(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) =
#   IterTools.imap(Interval, _accessibles(Base.argmin((w.y for w in S)), IA_L, X))
# accessibles2_1_2(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) =
#   IterTools.imap(Interval, _accessibles(Base.argmax((w.x for w in S)), IA_Li, X))
# accessibles2_2(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) = begin
#   m = argmin(map((w)->w.y, S))
#   IterTools.imap(Interval, _accessibles([w for (i,w) in enumerate(S) if i == m][1], IA_L, X))
# end
# accessibles2_2(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) = begin
#   m = argmax(map((w)->w.x, S))
#   IterTools.imap(Interval, _accessibles([w for (i,w) in enumerate(S) if i == m][1], IA_Li, X))
# end
# # This makes sense if we have 2-Tuples instead of intervals
# function snd((a,b)::Tuple) b end
# function fst((a,b)::Tuple) a end
# accessibles2_1(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) = 
#   IterTools.imap(Interval,
#       _accessibles(S[argmin(map(snd, S))], IA_L, X)
#   )
# accessibles2_1(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) = 
#   IterTools.imap(Interval,
#       _accessibles(S[argmax(map(fst, S))], IA_Li, X)
#   )


#=
# TODO parametrize on the test_operator. These are wrong anyway...
# Note: these conditions are the ones that make a modal_step inexistent
enum_acc_repr(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)                 ? _ReprVal(Interval(w.y, w.y+1)   ) : _ReprNone{Interval}() # [Interval(w.y, X+1)]     : Interval[]
enum_acc_repr(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)                   ? _ReprVal(Interval(w.x-1, w.x)   ) : _ReprNone{Interval}() # [Interval(1, w.x)]       : Interval[]
enum_acc_repr(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)               ? _ReprVal(Interval(w.x, w.x+1)   ) : _ReprNone{Interval}() # [Interval(w.x, w.y-1)]   : Interval[]
enum_acc_repr(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)               ? _ReprVal(Interval(w.y-1, w.y)   ) : _ReprNone{Interval}() # [Interval(w.x+1, w.y)]   : Interval[]

enum_acc_repr(test_operator::_TestOpGeq, w::Interval, ::_IA_L,  X::Integer) = (w.y+1 < X+1)               ? _ReprMax(Interval(w.y+1, X+1)   ) : _ReprNone{Interval}() # [Interval(w.y+1, X+1)]   : Interval[]
enum_acc_repr(test_operator::_TestOpGeq, w::Interval, ::_IA_Li, X::Integer) = (1 < w.x-1)                 ? _ReprMax(Interval(1, w.x-1)     ) : _ReprNone{Interval}() # [Interval(1, w.x-1)]     : Interval[]
enum_acc_repr(test_operator::_TestOpGeq, w::Interval, ::_IA_D,  X::Integer) = (w.x+1 < w.y-1)             ? _ReprMax(Interval(w.x+1, w.y-1) ) : _ReprNone{Interval}() # [Interval(w.x+1, w.y-1)] : Interval[]
enum_acc_repr(test_operator::_TestOpLeq, w::Interval, ::_IA_L,  X::Integer) = (w.y+1 < X+1)               ? _ReprMin(Interval(w.y+1, X+1)   ) : _ReprNone{Interval}() # [Interval(w.y+1, X+1)]   : Interval[]
enum_acc_repr(test_operator::_TestOpLeq, w::Interval, ::_IA_Li, X::Integer) = (1 < w.x-1)                 ? _ReprMin(Interval(1, w.x-1)     ) : _ReprNone{Interval}() # [Interval(1, w.x-1)]     : Interval[]
enum_acc_repr(test_operator::_TestOpLeq, w::Interval, ::_IA_D,  X::Integer) = (w.x+1 < w.y-1)             ? _ReprMin(Interval(w.x+1, w.y-1) ) : _ReprNone{Interval}() # [Interval(w.x+1, w.y-1)] : Interval[]

enum_acc_repr(test_operator::_TestOpGeq, w::Interval, ::_IA_Bi, X::Integer) = (w.y < X+1)                 ? _ReprMin(Interval(w.x, w.y+1)   ) : _ReprNone{Interval}() # [Interval(w.x, X+1)]     : Interval[]
enum_acc_repr(test_operator::_TestOpGeq, w::Interval, ::_IA_Ei, X::Integer) = (1 < w.x)                   ? _ReprMin(Interval(w.x-1, w.y)   ) : _ReprNone{Interval}() # [Interval(1, w.y)]       : Interval[]
enum_acc_repr(test_operator::_TestOpGeq, w::Interval, ::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ? _ReprMin(Interval(w.x-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(1, X+1)]       : Interval[]
enum_acc_repr(test_operator::_TestOpGeq, w::Interval, ::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ? _ReprMin(Interval(w.y-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(w.x+1, X+1)]   : Interval[]
enum_acc_repr(test_operator::_TestOpGeq, w::Interval, ::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ? _ReprMin(Interval(w.x-1, w.x+1) ) : _ReprNone{Interval}() # [Interval(1, w.y-1)]     : Interval[]
enum_acc_repr(test_operator::_TestOpLeq, w::Interval, ::_IA_Bi, X::Integer) = (w.y < X+1)                 ? _ReprMax(Interval(w.x, w.y+1)   ) : _ReprNone{Interval}() # [Interval(w.x, X+1)]     : Interval[]
enum_acc_repr(test_operator::_TestOpLeq, w::Interval, ::_IA_Ei, X::Integer) = (1 < w.x)                   ? _ReprMax(Interval(w.x-1, w.y)   ) : _ReprNone{Interval}() # [Interval(1, w.y)]       : Interval[]
enum_acc_repr(test_operator::_TestOpLeq, w::Interval, ::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ? _ReprMax(Interval(w.x-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(1, X+1)]       : Interval[]
enum_acc_repr(test_operator::_TestOpLeq, w::Interval, ::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ? _ReprMax(Interval(w.y-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(w.x+1, X+1)]   : Interval[]
enum_acc_repr(test_operator::_TestOpLeq, w::Interval, ::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ? _ReprMax(Interval(w.x-1, w.x+1) ) : _ReprNone{Interval}() # [Interval(1, w.y-1)]     : Interval[]
=#


# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_A, channel::DimensionalChannel{T,1}) where {T} =
#   (w.y < length(channel)+1) ? (channel[w.y],channel[w.y]) : (typemax(T),typemin(T))
# compute_modal_gamma(test_operator::_TestOpGeq, w::Interval, ::_IA_A, channel::DimensionalChannel{T,1}) where {T} =
#   (w.y < length(channel)+1) ? channel[w.y] : typemax(T)
# compute_modal_gamma(test_operator::_TestOpLeq, w::Interval, ::_IA_A, channel::DimensionalChannel{T,1}) where {T} =
#   (w.y < length(channel)+1) ? channel[w.y] : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_Ai, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x) ? (channel[w.x-1],channel[w.x-1]) : (typemax(T),typemin(T))
# compute_modal_gamma(test_operator::_TestOpGeq, w::Interval, ::_IA_Ai, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x) ? channel[w.x-1] : typemax(T)
# compute_modal_gamma(test_operator::_TestOpLeq, w::Interval, ::_IA_Ai, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x) ? channel[w.x-1] : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_L, channel::DimensionalChannel{T,1}) where {T} =
#   (w.y+1 < length(channel)+1) ? reverse(extrema(channel[w.y+1:length(channel)])) : (typemax(T),typemin(T))
# compute_modal_gamma(test_operator::_TestOpGeq, w::Interval, ::_IA_L, channel::DimensionalChannel{T,1}) where {T} =
#   (w.y+1 < length(channel)+1) ? maximum(channel[w.y+1:length(channel)]) : typemax(T)
# compute_modal_gamma(test_operator::_TestOpLeq, w::Interval, ::_IA_L, channel::DimensionalChannel{T,1}) where {T} =
#   (w.y+1 < length(channel)+1) ? minumum(channel[w.y+1:length(channel)]) : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_Li, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x-1) ? reverse(extrema(channel[1:w.x-2])) : (typemax(T),typemin(T))
# compute_modal_gamma(test_operator::_TestOpGeq, w::Interval, ::_IA_Li, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x-1) ? maximum(channel[1:w.x-2]) : typemax(T)
# compute_modal_gamma(test_operator::_TestOpLeq, w::Interval, ::_IA_Li, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x-1) ? minumum(channel[1:w.x-2]) : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_B, channel::DimensionalChannel{T,1}) where {T} =
#   (w.x < w.y-1) ? (channel[w.x],channel[w.x]) : (typemax(T),typemin(T))
# compute_modal_gamma(test_operator::_TestOpGeq, w::Interval, ::_IA_B, channel::DimensionalChannel{T,1}) where {T} =
#   (w.x < w.y-1) ? channel[w.x] : typemax(T)
# compute_modal_gamma(test_operator::_TestOpLeq, w::Interval, ::_IA_B, channel::DimensionalChannel{T,1}) where {T} =
#   (w.x < w.y-1) ? channel[w.x] : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_Bi, channel::DimensionalChannel{T,1}) where {T} =
#   (w.y < length(channel)+1) ? (minimum(channel[w.x:w.y-1+1]),maximum(channel[w.x:w.y-1+1])) : (typemax(T),typemin(T))
# compute_modal_gamma(test_operator::_TestOpGeq, w::Interval, ::_IA_Bi, channel::DimensionalChannel{T,1}) where {T} =
#   (w.y < length(channel)+1) ? minimum(channel[w.x:w.y-1+1]) : typemax(T)
# compute_modal_gamma(test_operator::_TestOpLeq, w::Interval, ::_IA_Bi, channel::DimensionalChannel{T,1}) where {T} =
#   (w.y < length(channel)+1) ? maximum(channel[w.x:w.y-1+1]) : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_E, channel::DimensionalChannel{T,1}) where {T} =
#   (w.x+1 < w.y) ? (channel[w.y-1],channel[w.y-1]) : (typemax(T),typemin(T))
# compute_modal_gamma(test_operator::_TestOpGeq, w::Interval, ::_IA_E, channel::DimensionalChannel{T,1}) where {T} =
#   (w.x+1 < w.y) ? channel[w.y-1] : typemax(T)
# compute_modal_gamma(test_operator::_TestOpLeq, w::Interval, ::_IA_E, channel::DimensionalChannel{T,1}) where {T} =
#   (w.x+1 < w.y) ? channel[w.y-1] : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_Ei, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x) ? (minimum(channel[w.x-1:w.y-1]),maximum(channel[w.x-1:w.y-1])) : (typemax(T),typemin(T))
# compute_modal_gamma(test_operator::_TestOpGeq, w::Interval, ::_IA_Ei, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x) ? minimum(channel[w.x-1:w.y-1]) : typemax(T)
# compute_modal_gamma(test_operator::_TestOpLeq, w::Interval, ::_IA_Ei, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x) ? maximum(channel[w.x-1:w.y-1]) : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_D, channel::DimensionalChannel{T,1}) where {T} =
#   (w.x+1 < w.y-1) ? reverse(extrema(channel[w.x+1:w.y-1-1])) : (typemax(T),typemin(T))
# compute_modal_gamma(test_operator::_TestOpGeq, w::Interval, ::_IA_D, channel::DimensionalChannel{T,1}) where {T} =
#   (w.x+1 < w.y-1) ? maximum(channel[w.x+1:w.y-1-1]) : typemax(T)
# compute_modal_gamma(test_operator::_TestOpLeq, w::Interval, ::_IA_D, channel::DimensionalChannel{T,1}) where {T} =
#   (w.x+1 < w.y-1) ? minumum(channel[w.x+1:w.y-1-1]) : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_Di, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x && w.y < length(channel)+1) ? (minimum(channel[w.x-1:w.y-1+1]),maximum(channel[w.x-1:w.y-1+1])) : (typemax(T),typemin(T))
# compute_modal_gamma(test_operator::_TestOpGeq, w::Interval, ::_IA_Di, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x && w.y < length(channel)+1) ? minimum(channel[w.x-1:w.y-1+1]) : typemax(T)
# compute_modal_gamma(test_operator::_TestOpLeq, w::Interval, ::_IA_Di, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x && w.y < length(channel)+1) ? maximum(channel[w.x-1:w.y-1+1]) : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_O, channel::DimensionalChannel{T,1}) where {T} =
#   (w.x+1 < w.y && w.y < length(channel)+1) ? (minimum(channel[w.y-1:w.y-1+1]),maximum(channel[w.y-1:w.y-1+1])) : (typemax(T),typemin(T))
# compute_modal_gamma(test_operator::_TestOpGeq, w::Interval, ::_IA_O, channel::DimensionalChannel{T,1}) where {T} =
#   (w.x+1 < w.y && w.y < length(channel)+1) ? minimum(channel[w.y-1:w.y-1+1]) : typemax(T)
# compute_modal_gamma(test_operator::_TestOpLeq, w::Interval, ::_IA_O, channel::DimensionalChannel{T,1}) where {T} =
#   (w.x+1 < w.y && w.y < length(channel)+1) ? maximum(channel[w.y-1:w.y-1+1]) : typemin(T)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval, ::_IA_Oi, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x && w.x+1 < w.y) ? (minimum(channel[w.x-1:w.x]),maximum(channel[w.x-1:w.x])) : (typemax(T),typemin(T))
# compute_modal_gamma(test_operator::_TestOpGeq, w::Interval, ::_IA_Oi, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x && w.x+1 < w.y) ? minimum(channel[w.x-1:w.x]) : typemax(T)
# compute_modal_gamma(test_operator::_TestOpLeq, w::Interval, ::_IA_Oi, channel::DimensionalChannel{T,1}) where {T} =
#   (1 < w.x && w.x+1 < w.y) ? maximum(channel[w.x-1:w.x]) : typemin(T)


# enum_acc_repr for _IA2D_URelations
# 3 operator categories for the 13+1 relations
const _IA2DRelMaximizer = Union{_RelationGlob,_IA_L,_IA_Li,_IA_D}
const _IA2DRelMinimizer = Union{_RelationId,_IA_O,_IA_Oi,_IA_Bi,_IA_Ei,_IA_Di}
const _IA2DRelSingleVal = Union{_IA_A,_IA_Ai,_IA_B,_IA_E}

#=

################################################################################
################################################################################
# TODO remove (needed for GAMMAS)
# Utility type for enhanced computation of thresholds
abstract type _ReprTreatment end
struct _ReprFake{WorldType<:AbstractWorld} <: _ReprTreatment w :: WorldType end
struct _ReprMax{WorldType<:AbstractWorld}  <: _ReprTreatment w :: WorldType end
struct _ReprMin{WorldType<:AbstractWorld}  <: _ReprTreatment w :: WorldType end
struct _ReprVal{WorldType<:AbstractWorld}  <: _ReprTreatment w :: WorldType end
struct _ReprNone{WorldType<:AbstractWorld} <: _ReprTreatment end
# enum_acc_repr(::_TestOpGeq, w::WorldType, ::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = _ReprMin(w)
# enum_acc_repr(::_TestOpLeq, w::WorldType, ::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = _ReprMax(w)

@inline enum_acc_repr2D(test_operator::TestOperator, w::Interval2D, rx::R1 where R1<:AbstractRelation, ry::R2 where R2<:AbstractRelation, X::Integer, Y::Integer, _ReprConstructor::Type{rT}) where {rT<:_ReprTreatment} = begin
    x = enum_acc_repr(test_operator, w.x, rx, X)
    # println(x)
    if x == _ReprNone{Interval}()
        return _ReprNone{Interval2D}()
    end
    y = enum_acc_repr(test_operator, w.y, ry, Y)
    # println(y)
    if y == _ReprNone{Interval}()
        return _ReprNone{Interval2D}()
    end
    return _ReprConstructor(Interval2D(x.w, y.w))
end

# 3*3 = 9 cases ((13+1)^2 = 196 relations)
# Maximizer operators
enum_acc_repr(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelMaximizer}, X::Integer, Y::Integer) =
    enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprMax)
enum_acc_repr(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelMinimizer}, X::Integer, Y::Integer) = begin
    # println(enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin))
    enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)
end
enum_acc_repr(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
    enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprVal)

enum_acc_repr(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
    enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprMax)
enum_acc_repr(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
    enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)
enum_acc_repr(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelMaximizer}, X::Integer, Y::Integer) =
    enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprMax)
enum_acc_repr(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelMinimizer}, X::Integer, Y::Integer) =
    enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)

enum_acc_repr(test_operator::_TestOpLeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelMaximizer}, X::Integer, Y::Integer) =
    enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)
enum_acc_repr(test_operator::_TestOpLeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelMinimizer}, X::Integer, Y::Integer) =
    enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprMax)
enum_acc_repr(test_operator::_TestOpLeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
    enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprVal)

enum_acc_repr(test_operator::_TestOpLeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
    enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)
enum_acc_repr(test_operator::_TestOpLeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
    enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprMax)
enum_acc_repr(test_operator::_TestOpLeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelMaximizer}, X::Integer, Y::Integer) =
    enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)
enum_acc_repr(test_operator::_TestOpLeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelMinimizer}, X::Integer, Y::Integer) =
    enum_acc_repr2D(test_operator, w, r.x, r.y, X, Y, _ReprMax)

# The last two cases are difficult to express with enum_acc_repr, better do it at computeModalThresholdDual instead

# TODO create a dedicated min/max combination representation?
yieldMinMaxCombinations(test_operator::_TestOpGeq, productRepr::_ReprTreatment, channel::DimensionalChannel{T,2}, dims::Integer) where {T} = begin
    if productRepr == _ReprNone{Interval2D}()
        return typemin(T),typemax(T)
    end
    vals = ch_readWorld(productRepr.w, channel)
    # TODO try: maximum(mapslices(minimum, vals, dims=1)),minimum(mapslices(maximum, vals, dims=1))
    extr = vec(mapslices(extrema, vals, dims=dims))
    # println(extr)
    maxExtrema(extr)
end

yieldMinMaxCombination(test_operator::_TestOpGeq, productRepr::_ReprTreatment, channel::DimensionalChannel{T,2}, dims::Integer) where {T} = begin
    if productRepr == _ReprNone{Interval2D}()
        return typemin(T)
    end
    vals = ch_readWorld(productRepr.w, channel)
    maximum(mapslices(minimum, vals, dims=dims))
end

yieldMinMaxCombination(test_operator::_TestOpLeq, productRepr::_ReprTreatment, channel::DimensionalChannel{T,2}, dims::Integer) where {T} = begin
    if productRepr == _ReprNone{Interval2D}()
        return typemax(T)
    end
    vals = ch_readWorld(productRepr.w, channel)
    minimum(mapslices(maximum, vals, dims=dims))
end

computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelMaximizer}, channel::DimensionalChannel{T,2}) where {T} = begin
    yieldMinMaxCombinations(test_operator, enum_acc_repr2D(test_operator, w, r.x, r.y, size(channel)..., _ReprFake), channel, 1)
end
compute_modal_gamma(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelMaximizer}, channel::DimensionalChannel{T,2}) where {T} = begin
    yieldMinMaxCombination(test_operator, enum_acc_repr2D(test_operator, w, r.x, r.y, size(channel)..., _ReprFake), channel, 1)
end
computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelMinimizer}, channel::DimensionalChannel{T,2}) where {T} = begin
    yieldMinMaxCombinations(test_operator, enum_acc_repr2D(test_operator, w, r.x, r.y, size(channel)..., _ReprFake), channel, 2)
end
compute_modal_gamma(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelMinimizer}, channel::DimensionalChannel{T,2}) where {T} = begin
    yieldMinMaxCombination(test_operator, enum_acc_repr2D(test_operator, w, r.x, r.y, size(channel)..., _ReprFake), channel, 2)
end

=#

# TODO: per _TestOpLeq gli operatori si invertono

const _IA2DRelMax = Union{_RelationGlob,_IA_L,_IA_Li,_IA_D}
const _IA2DRelMin = Union{_RelationId,_IA_O,_IA_Oi,_IA_Bi,_IA_Ei,_IA_Di}
const _IA2DRelVal = Union{_IA_A,_IA_Ai,_IA_B,_IA_E}

# accessibles_aggr(f::Union{SingleAttributeMin,SingleAttributeMax}, a::Union{typeof(minimum),typeof(maximum)}, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMax,R2<:_IA2DRelMax},  X::Integer) = IterTools.imap(Interval2D, Iterators.product(accessibles_aggr(f, a, w.x, rx, X), accessibles_aggr(f, a, w.y, ry, Y)))

#=

computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval2D, r::_TopoRelRCC5, channel::DimensionalChannel{T,2}) where {T} = begin
  maxExtrema(
    map((RCC8_r)->(computeModalThresholdDual(test_operator, w, RCC8_r, channel)), RCC52RCC8Relations(r))
  )
end
compute_modal_gamma(test_operator::_TestOpGeq, w::Interval2D, r::_TopoRelRCC5, channel::DimensionalChannel{T,2}) where {T} = begin
  maximum(
    map((RCC8_r)->(compute_modal_gamma(test_operator, w, RCC8_r, channel)), RCC52RCC8Relations(r))
  )
end
compute_modal_gamma(test_operator::_TestOpLeq, w::Interval2D, r::_TopoRelRCC5, channel::DimensionalChannel{T,2}) where {T} = begin
  mininimum(
    map((RCC8_r)->(compute_modal_gamma(test_operator, w, RCC8_r, channel)), RCC52RCC8Relations(r))
  )
end

=#

# More efficient implementations for edge cases
# ?


#=
# TODO optimize RCC5

# Virtual relation used for computing Topo_DC on Interval2D
struct _Virtual_Enlarge <: AbstractRelation end; const Virtual_Enlarge = _Virtual_Enlarge();     # Virtual_Enlarge
enlargeInterval(w::Interval, X::Integer) = Interval(max(1,w.x-1),min(w.y+1,X+1))

enum_acc_repr(test_operator::_TestOpGeq, w::Interval, ::_Virtual_Enlarge,  X::Integer) = _ReprMin(enlargeInterval(w,X))
enum_acc_repr(test_operator::_TestOpLeq, w::Interval, ::_Virtual_Enlarge,  X::Integer) = _ReprMax(enlargeInterval(w,X))


# Topo2D2Topo1D(::_Topo_DC) = [
#                               (RelationGlob , Topo_DC),
#                               # TODO many many others but for now let's just say...
#                               (Topo_DC     , Virtual_Enlarge),
# ]
Topo2D2Topo1D(::_Topo_EC) = [
                              (Topo_EC     , Topo_EC),
                              #
                              (Topo_PO     , Topo_EC),
                              (Topo_TPP    , Topo_EC),
                              (Topo_TPPi   , Topo_EC),
                              (Topo_NTPP   , Topo_EC),
                              (Topo_NTPPi  , Topo_EC),
                              (RelationId  , Topo_EC),
                              #
                              (Topo_EC     , Topo_PO),
                              (Topo_EC     , Topo_TPP),
                              (Topo_EC     , Topo_TPPi),
                              (Topo_EC     , Topo_NTPP),
                              (Topo_EC     , Topo_NTPPi),
                              (Topo_EC     , RelationId),
]
Topo2D2Topo1D(::_Topo_PO) = [
                              (Topo_PO     , Topo_PO),
                              #
                              (Topo_PO     , Topo_TPP),
                              (Topo_PO     , Topo_TPPi),
                              (Topo_PO     , Topo_NTPP),
                              (Topo_PO     , Topo_NTPPi),
                              (Topo_PO     , RelationId),
                              #
                              (Topo_TPP    , Topo_PO),
                              (Topo_TPPi   , Topo_PO),
                              (Topo_NTPP   , Topo_PO),
                              (Topo_NTPPi  , Topo_PO),
                              (RelationId  , Topo_PO),
                              #
                              (Topo_TPPi   , Topo_TPP),
                              (Topo_TPP    , Topo_TPPi),
                              #
                              (Topo_NTPP   , Topo_TPPi),
                              (Topo_TPPi   , Topo_NTPP),
                              #
                              (Topo_NTPPi  , Topo_TPP),
                              (Topo_NTPPi  , Topo_NTPP),
                              (Topo_TPP    , Topo_NTPPi),
                              (Topo_NTPP   , Topo_NTPPi),
]
Topo2D2Topo1D(::_Topo_TPP) = [
                              (Topo_TPP    , Topo_TPP),   
                              #
                              (Topo_NTPP   , Topo_TPP),   
                              (Topo_TPP    , Topo_NTPP),  
                              #
                              (RelationId  , Topo_TPP),   
                              (Topo_TPP    , RelationId), 
                              #
                              (RelationId  , Topo_NTPP),  
                              (Topo_NTPP   , RelationId), 
]
Topo2D2Topo1D(::_Topo_TPPi) = [
                              (Topo_TPPi   , Topo_TPPi),
                              #
                              (Topo_NTPPi  , Topo_TPPi),
                              (Topo_TPPi   , Topo_NTPPi),
                              #
                              (RelationId  , Topo_TPPi),
                              (Topo_TPPi   , RelationId),
                              #
                              (RelationId  , Topo_NTPPi),
                              (Topo_NTPPi  , RelationId),
]
Topo2D2Topo1D(::_Topo_NTPP) = [
                              (Topo_NTPP   , Topo_NTPP),
]
Topo2D2Topo1D(::_Topo_NTPPi) = [
                              (Topo_NTPPi  , Topo_NTPPi),
]


computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_DC, channel::DimensionalChannel{T,2}) where {T} = begin
  reprx1 = enum_acc_repr2D(test_operator, w, RelationGlob, IA_L,         size(channel)..., _ReprMax)
  reprx2 = enum_acc_repr2D(test_operator, w, RelationGlob, IA_Li,        size(channel)..., _ReprMax)
  repry1 = enum_acc_repr2D(test_operator, w, IA_L,     Virtual_Enlarge, size(channel)..., _ReprMax)
  repry2 = enum_acc_repr2D(test_operator, w, IA_Li,    Virtual_Enlarge, size(channel)..., _ReprMax)
  extr = yieldReprs(test_operator, reprx1, channel),
         yieldReprs(test_operator, reprx2, channel),
         yieldReprs(test_operator, repry1, channel),
         yieldReprs(test_operator, repry2, channel)
  maxExtrema(extr)
end
compute_modal_gamma(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_DC, channel::DimensionalChannel{T,2}) where {T} = begin
  # reprx1 = enum_acc_repr2D(test_operator, w, IA_L,         RelationGlob, size(channel)..., _ReprMax)
  # reprx2 = enum_acc_repr2D(test_operator, w, IA_Li,        RelationGlob, size(channel)..., _ReprMax)
  # repry1 = enum_acc_repr2D(test_operator, w, RelationGlob,  IA_L,        size(channel)..., _ReprMax)
  # repry2 = enum_acc_repr2D(test_operator, w, RelationGlob,  IA_Li,       size(channel)..., _ReprMax)
  reprx1 = enum_acc_repr2D(test_operator, w, RelationGlob, IA_L,         size(channel)..., _ReprMax)
  reprx2 = enum_acc_repr2D(test_operator, w, RelationGlob, IA_Li,        size(channel)..., _ReprMax)
  repry1 = enum_acc_repr2D(test_operator, w, IA_L,     Virtual_Enlarge, size(channel)..., _ReprMax)
  repry2 = enum_acc_repr2D(test_operator, w, IA_Li,    Virtual_Enlarge, size(channel)..., _ReprMax)
  # if channel == [819 958 594; 749 665 383; 991 493 572] && w.x.x==1 && w.x.y==2 && w.y.x==1 && w.y.y==3
  #   println(max(yieldRepr(test_operator, reprx1, channel),
  #        yieldRepr(test_operator, reprx2, channel),
  #        yieldRepr(test_operator, repry1, channel),
  #        yieldRepr(test_operator, repry2, channel)))
  #   readline()
  # end
  max(yieldRepr(test_operator, reprx1, channel),
       yieldRepr(test_operator, reprx2, channel),
       yieldRepr(test_operator, repry1, channel),
       yieldRepr(test_operator, repry2, channel))
end
compute_modal_gamma(test_operator::_TestOpLeq, w::Interval2D, r::_Topo_DC, channel::DimensionalChannel{T,2}) where {T} = begin
  reprx1 = enum_acc_repr2D(test_operator, w, RelationGlob, IA_L,         size(channel)..., _ReprMin)
  reprx2 = enum_acc_repr2D(test_operator, w, RelationGlob, IA_Li,        size(channel)..., _ReprMin)
  repry1 = enum_acc_repr2D(test_operator, w, IA_L,     Virtual_Enlarge, size(channel)..., _ReprMin)
  repry2 = enum_acc_repr2D(test_operator, w, IA_Li,    Virtual_Enlarge, size(channel)..., _ReprMin)
  min(yieldRepr(test_operator, reprx1, channel),
       yieldRepr(test_operator, reprx2, channel),
       yieldRepr(test_operator, repry1, channel),
       yieldRepr(test_operator, repry2, channel))
end

# EC: Just optimize the values on the outer boundary
computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_EC, channel::DimensionalChannel{T,2}) where {T} = begin
  X,Y = size(channel)
  reprs = [
    ((w.x.x-1 >= 1)   ? [Interval2D(Interval(w.x.x-1,w.x.x),enlargeInterval(w.y,Y))] : Interval2D[])...,
    ((w.x.y+1 <= X+1) ? [Interval2D(Interval(w.x.y,w.x.y+1),enlargeInterval(w.y,Y))] : Interval2D[])...,
    ((w.y.x-1 >= 1)   ? [Interval2D(enlargeInterval(w.x,X),Interval(w.y.x-1,w.y.x))] : Interval2D[])...,
    ((w.y.y+1 <= Y+1) ? [Interval2D(enlargeInterval(w.x,X),Interval(w.y.y,w.y.y+1))] : Interval2D[])...,
  ]
  extr = map(w->yieldReprs(test_operator, _ReprMax(w), channel), reprs)
  maxExtrema(extr)
end
compute_modal_gamma(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_EC, channel::DimensionalChannel{T,2}) where {T} = begin
  X,Y = size(channel)
  reprs = [
    ((w.x.x-1 >= 1)   ? [Interval2D(Interval(w.x.x-1,w.x.x),enlargeInterval(w.y,Y))] : Interval2D[])...,
    ((w.x.y+1 <= X+1) ? [Interval2D(Interval(w.x.y,w.x.y+1),enlargeInterval(w.y,Y))] : Interval2D[])...,
    ((w.y.x-1 >= 1)   ? [Interval2D(enlargeInterval(w.x,X),Interval(w.y.x-1,w.y.x))] : Interval2D[])...,
    ((w.y.y+1 <= Y+1) ? [Interval2D(enlargeInterval(w.x,X),Interval(w.y.y,w.y.y+1))] : Interval2D[])...,
  ]
  extr = map(w->yieldRepr(test_operator, _ReprMax(w), channel), reprs)
  maximum([extr..., typemin(T)])
end
compute_modal_gamma(test_operator::_TestOpLeq, w::Interval2D, r::_Topo_EC, channel::DimensionalChannel{T,2}) where {T} = begin
  X,Y = size(channel)
  reprs = [
    ((w.x.x-1 >= 1)   ? [Interval2D(Interval(w.x.x-1,w.x.x),enlargeInterval(w.y,Y))] : Interval2D[])...,
    ((w.x.y+1 <= X+1) ? [Interval2D(Interval(w.x.y,w.x.y+1),enlargeInterval(w.y,Y))] : Interval2D[])...,
    ((w.y.x-1 >= 1)   ? [Interval2D(enlargeInterval(w.x,X),Interval(w.y.x-1,w.y.x))] : Interval2D[])...,
    ((w.y.y+1 <= Y+1) ? [Interval2D(enlargeInterval(w.x,X),Interval(w.y.y,w.y.y+1))] : Interval2D[])...,
  ]
  extr = map(w->yieldRepr(test_operator, _ReprMin(w), channel), reprs)
  minimum([extr..., typemax(T)])
end

# PO: For each pair crossing the border, perform a minimization step and then a maximization step
computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_PO, channel::DimensionalChannel{T,2}) where {T} = begin
  # if true &&
  #   # (channel == [1620 1408 1343; 1724 1398 1252; 1177 1703 1367] && w.x.x==1 && w.x.y==3 && w.y.x==3 && w.y.y==4) ||
  #   # (channel == [412 489 559 619 784; 795 771 1317 854 1256; 971 874 878 1278 560] && w.x.x==1 && w.x.y==3 && w.y.x==3 && w.y.y==4)
  #   (channel == [2405 2205 1898 1620 1383; 1922 1555 1383 1393 1492; 1382 1340 1434 1640 1704] && w.x.x==1 && w.x.y==3 && w.y.x==3 && w.y.y==4)
    
  #   x_singleton = ! (w.x.x < w.x.y-1)
  #   y_singleton = ! (w.y.x < w.y.y-1)
  #   if x_singleton && y_singleton
  #     println(typemin(T),typemax(T))
  #   else
  #     rx1,rx2 = x_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)
  #     ry1,ry2 = y_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)
  #     println(rx1)
  #     println(rx2)
  #     println(ry1)
  #     println(ry2)
  #     # reprx1 = enum_acc_repr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprMin)
  #     # reprx2 = enum_acc_repr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprMin)
  #     # repry1 = enum_acc_repr2D(test_operator, w, RelationId, ry1,        size(channel)..., _ReprMin)
  #     # repry2 = enum_acc_repr2D(test_operator, w, RelationId, ry2,        size(channel)..., _ReprMin)
  #     # println(reprx1)
  #     # println(reprx2)
  #     # println(repry1)
  #     # println(repry2)

  #     println(
  #     yieldMinMaxCombinations(test_operator, enum_acc_repr2D(test_operator, w, RelationId, ry1,    size(channel)..., _ReprFake), channel, 2)
  #     )
  #     println(
  #     yieldMinMaxCombinations(test_operator, enum_acc_repr2D(test_operator, w, RelationId, ry2,    size(channel)..., _ReprFake), channel, 2)
  #     )
  #     println(
  #     yieldMinMaxCombinations(test_operator, enum_acc_repr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake), channel, 1)
  #     )
  #     println(
  #     yieldMinMaxCombinations(test_operator, enum_acc_repr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake), channel, 1)
  #     )
  #     println(
  #     maxExtrema((
  #       yieldMinMaxCombinations(test_operator, enum_acc_repr2D(test_operator, w, RelationId, ry1,    size(channel)..., _ReprFake), channel, 2),
  #       yieldMinMaxCombinations(test_operator, enum_acc_repr2D(test_operator, w, RelationId, ry2,    size(channel)..., _ReprFake), channel, 2),
  #       yieldMinMaxCombinations(test_operator, enum_acc_repr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake), channel, 1),
  #       yieldMinMaxCombinations(test_operator, enum_acc_repr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake), channel, 1),
  #     ))
  #     )

  #     # println(computeModalThresholdDual(test_operator, w, RectangleRelation(rx1        , RelationId), channel))
  #     # println(computeModalThresholdDual(test_operator, w, RectangleRelation(rx2        , RelationId), channel))
  #     # println(computeModalThresholdDual(test_operator, w, RectangleRelation(RelationId , ry1),        channel))
  #     # println(computeModalThresholdDual(test_operator, w, RectangleRelation(RelationId , ry2),        channel))
  #     # println(maxExtrema((
  #     #   computeModalThresholdDual(test_operator, w, RectangleRelation(rx1        , RelationId), channel),
  #     #   computeModalThresholdDual(test_operator, w, RectangleRelation(rx2        , RelationId), channel),
  #     #   computeModalThresholdDual(test_operator, w, RectangleRelation(RelationId , ry1),        channel),
  #     #   computeModalThresholdDual(test_operator, w, RectangleRelation(RelationId , ry2),        channel),
  #     #   ))
  #     # )
  #   end

  #   readline()
  # end
  x_singleton = ! (w.x.x < w.x.y-1)
  y_singleton = ! (w.y.x < w.y.y-1)
  if x_singleton && y_singleton
    return typemin(T),typemax(T)
  end

  rx1,rx2 = x_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)
  ry1,ry2 = y_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)

  # reprx1 = enum_acc_repr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake)
  # reprx2 = enum_acc_repr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake)
  # repry1 = enum_acc_repr2D(test_operator, w, RelationId, ry1,        size(channel)..., _ReprFake)
  # repry2 = enum_acc_repr2D(test_operator, w, RelationId, ry2,        size(channel)..., _ReprFake)

  maxExtrema(
    yieldMinMaxCombinations(test_operator, enum_acc_repr2D(test_operator, w, RelationId, ry1,    size(channel)..., _ReprFake), channel, 2),
    yieldMinMaxCombinations(test_operator, enum_acc_repr2D(test_operator, w, RelationId, ry2,    size(channel)..., _ReprFake), channel, 2),
    yieldMinMaxCombinations(test_operator, enum_acc_repr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake), channel, 1),
    yieldMinMaxCombinations(test_operator, enum_acc_repr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake), channel, 1),
  )
end
compute_modal_gamma(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_PO, channel::DimensionalChannel{T,2}) where {T} = begin
  # if channel == [1620 1408 1343; 1724 1398 1252; 1177 1703 1367] && w.x.x==1 && w.x.y==3 && w.y.x==3 && w.y.y==4
  #   println(! (w.x.x < w.x.y-1) && ! (w.y.x < w.y.y-1))
  #   println(max(
  #     computeModalThresholdDual(test_operator, w, RectangleRelation(RelationId , IA_O),       channel),
  #     computeModalThresholdDual(test_operator, w, RectangleRelation(RelationId , IA_Oi),      channel),
  #     computeModalThresholdDual(test_operator, w, RectangleRelation(IA_Oi      , RelationId), channel),
  #     computeModalThresholdDual(test_operator, w, RectangleRelation(IA_O       , RelationId), channel),
  #   ))
  #   readline()
  # end
  x_singleton = ! (w.x.x < w.x.y-1)
  y_singleton = ! (w.y.x < w.y.y-1)
  if x_singleton && y_singleton
    return typemin(T)
  end

  rx1,rx2 = x_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)
  ry1,ry2 = y_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)

  max(
    yieldMinMaxCombination(test_operator, enum_acc_repr2D(test_operator, w, RelationId, ry1,    size(channel)..., _ReprFake), channel, 2),
    yieldMinMaxCombination(test_operator, enum_acc_repr2D(test_operator, w, RelationId, ry2,    size(channel)..., _ReprFake), channel, 2),
    yieldMinMaxCombination(test_operator, enum_acc_repr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake), channel, 1),
    yieldMinMaxCombination(test_operator, enum_acc_repr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake), channel, 1),
  )
end
compute_modal_gamma(test_operator::_TestOpLeq, w::Interval2D, r::_Topo_PO, channel::DimensionalChannel{T,2}) where {T} = begin
  x_singleton = ! (w.x.x < w.x.y-1)
  y_singleton = ! (w.y.x < w.y.y-1)
  if x_singleton && y_singleton
    return typemax(T)
  end
  
  rx1,rx2 = x_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)
  ry1,ry2 = y_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)

  min(
    yieldMinMaxCombination(test_operator, enum_acc_repr2D(test_operator, w, RelationId, ry1,    size(channel)..., _ReprFake), channel, 2),
    yieldMinMaxCombination(test_operator, enum_acc_repr2D(test_operator, w, RelationId, ry2,    size(channel)..., _ReprFake), channel, 2),
    yieldMinMaxCombination(test_operator, enum_acc_repr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake), channel, 1),
    yieldMinMaxCombination(test_operator, enum_acc_repr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake), channel, 1),
  )
end

# TPP: Just optimize the values on the inner boundary
computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_TPP, channel::DimensionalChannel{T,2}) where {T} = begin
  reprs = if (w.x.x < w.x.y-1) && (w.y.x < w.y.y-1)
      [Interval2D(Interval(w.x.x,w.x.x+1),w.y), Interval2D(Interval(w.x.y-1,w.x.y),w.y), Interval2D(w.x,Interval(w.y.x,w.y.x+1)), Interval2D(w.x,Interval(w.y.y-1,w.y.y))]
    elseif (w.x.x < w.x.y-1) || (w.y.x < w.y.y-1)
      [w]
    else Interval2D[]
  end
  extr = map(w->yieldReprs(test_operator, _ReprMax(w), channel), reprs)
  maxExtrema(extr)
end
compute_modal_gamma(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_TPP, channel::DimensionalChannel{T,2}) where {T} = begin
  reprs = if (w.x.x < w.x.y-1) && (w.y.x < w.y.y-1)
      [Interval2D(Interval(w.x.x,w.x.x+1),w.y), Interval2D(Interval(w.x.y-1,w.x.y),w.y), Interval2D(w.x,Interval(w.y.x,w.y.x+1)), Interval2D(w.x,Interval(w.y.y-1,w.y.y))]
    elseif (w.x.x < w.x.y-1) || (w.y.x < w.y.y-1)
      [w]
    else Interval2D[]
  end
  extr = map(w->yieldRepr(test_operator, _ReprMax(w), channel), reprs)
  maximum([extr..., typemin(T)])
end
compute_modal_gamma(test_operator::_TestOpLeq, w::Interval2D, r::_Topo_TPP, channel::DimensionalChannel{T,2}) where {T} = begin
  reprs = if (w.x.x < w.x.y-1) && (w.y.x < w.y.y-1)
      [Interval2D(Interval(w.x.x,w.x.x+1),w.y), Interval2D(Interval(w.x.y-1,w.x.y),w.y), Interval2D(w.x,Interval(w.y.x,w.y.x+1)), Interval2D(w.x,Interval(w.y.y-1,w.y.y))]
    elseif (w.x.x < w.x.y-1) || (w.y.x < w.y.y-1)
      [w]
    else Interval2D[]
  end
  extr = map(w->yieldRepr(test_operator, _ReprMin(w), channel), reprs)
  minimum([extr..., typemax(T)])
end

# TPPi: check 4 possible extensions of the box and perform a minimize+maximize step
computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_TPPi, channel::DimensionalChannel{T,2}) where {T} = begin
  X,Y = size(channel)
  reprs = [
    ((w.x.x-1 >= 1)   ? [Interval2D(Interval(w.x.x-1,w.x.y),w.y)] : Interval2D[])...,
    ((w.x.y+1 <= X+1) ? [Interval2D(Interval(w.x.x,w.x.y+1),w.y)] : Interval2D[])...,
    ((w.y.x-1 >= 1)   ? [Interval2D(w.x,Interval(w.y.x-1,w.y.y))] : Interval2D[])...,
    ((w.y.y+1 <= Y+1) ? [Interval2D(w.x,Interval(w.y.x,w.y.y+1))] : Interval2D[])...,
  ]
  extr = map(w->yieldReprs(test_operator, _ReprMin(w), channel), reprs)
  maxExtrema(extr)
end
compute_modal_gamma(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_TPPi, channel::DimensionalChannel{T,2}) where {T} = begin
  X,Y = size(channel)
  reprs = [
    ((w.x.x-1 >= 1)   ? [Interval2D(Interval(w.x.x-1,w.x.y),w.y)] : Interval2D[])...,
    ((w.x.y+1 <= X+1) ? [Interval2D(Interval(w.x.x,w.x.y+1),w.y)] : Interval2D[])...,
    ((w.y.x-1 >= 1)   ? [Interval2D(w.x,Interval(w.y.x-1,w.y.y))] : Interval2D[])...,
    ((w.y.y+1 <= Y+1) ? [Interval2D(w.x,Interval(w.y.x,w.y.y+1))] : Interval2D[])...,
  ]
  extr = map(w->yieldRepr(test_operator, _ReprMin(w), channel), reprs)
  maximum([extr..., typemin(T)])
end
compute_modal_gamma(test_operator::_TestOpLeq, w::Interval2D, r::_Topo_TPPi, channel::DimensionalChannel{T,2}) where {T} = begin
  X,Y = size(channel)
  reprs = [
    ((w.x.x-1 >= 1)   ? [Interval2D(Interval(w.x.x-1,w.x.y),w.y)] : Interval2D[])...,
    ((w.x.y+1 <= X+1) ? [Interval2D(Interval(w.x.x,w.x.y+1),w.y)] : Interval2D[])...,
    ((w.y.x-1 >= 1)   ? [Interval2D(w.x,Interval(w.y.x-1,w.y.y))] : Interval2D[])...,
    ((w.y.y+1 <= Y+1) ? [Interval2D(w.x,Interval(w.y.x,w.y.y+1))] : Interval2D[])...,
  ]
  extr = map(w->yieldRepr(test_operator, _ReprMax(w), channel), reprs)
  minimum([extr..., typemax(T)])
end

enum_acc_repr(test_operator::TestOperator, w::Interval2D, ::_Topo_NTPP,  X::Integer, Y::Integer) = enum_acc_repr(test_operator, w, RectangleRelation(IA_D,IA_D), X, Y)
enum_acc_repr(test_operator::TestOperator, w::Interval2D, ::_Topo_NTPPi, X::Integer, Y::Integer) = enum_acc_repr(test_operator, w, RectangleRelation(IA_Di,IA_Di), X, Y)

=#
#=


# To test optimizations
fn1 = ModalLogic.enum_acc_repr
fn2 = ModalLogic.enum_acc_repr2
rel = ModalLogic.Topo_EC
X = 4
Y = 3
while(true)
  a = randn(4,4);
  wextr = (x)->ModalLogic.computePropositionalThresholdDual([TestOpGeq, TestOpLeq], x,a);
  # TODO try all rectangles, avoid randominzing like this... Also try all channel sizes
  x1 = rand(1:X);
  x2 = x1+rand(1:(X+1-x1));
  x3 = rand(1:Y);
  x4 = x3+rand(1:(Y+1-x3));
  for i in 1:X
    println(a[i,:]);
  end
  println(x1,",",x2);
  println(x3,",",x4);
  println(a[x1:x2-1,x3:x4-1]);
  print("[")
  print(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> (y)->map((x)->ModalLogic.print_world(x),y));
  println("]")
  print("[")
  print(fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> (y)->map((x)->ModalLogic.print_world(x),y));
  println("]")
  println(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr);
  println(fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr);
  (fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr) == (fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr) || break;
end

fn1 = ModalLogic.enum_acc_repr
fn2 = ModalLogic.enum_acc_repr2
rel = ModalLogic.Topo_EC
a = [253 670 577; 569 730 931; 633 850 679];
X,Y = size(a)
while(true)
  wextr = (x)->ModalLogic.computePropositionalThresholdDual([TestOpGeq, TestOpLeq], x,a);
  # TODO try all rectangles, avoid randominzing like this... Also try all channel sizes
  x1 = rand(1:X);
  x2 = x1+rand(1:(X+1-x1));
  x3 = rand(1:Y);
  x4 = x3+rand(1:(Y+1-x3));
  for i in 1:X
    println(a[i,:]);
  end
  println(x1,",",x2);
  println(x3,",",x4);
  println(a[x1:x2-1,x3:x4-1]);
  print("[")
  print(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> (y)->map((x)->ModalLogic.print_world(x),y));
  println("]")
  print("[")
  print(fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> (y)->map((x)->ModalLogic.print_world(x),y));
  println("]")
  println(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr);
  println(fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr);
  (fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr) == (fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr) || break;
end

fn1 = ModalLogic.enum_acc_repr
fn2 = ModalLogic.enum_acc_repr2
rel = ModalLogic.Topo_EC
a = [253 670 577; 569 730 931; 633 850 679];
X,Y = size(a)
while(true)
wextr = (x)->ModalLogic.computePropositionalThresholdDual([TestOpGeq, TestOpLeq], x,a);
# TODO try all rectangles, avoid randominzing like this... Also try all channel sizes
x1 = 2
x2 = 3
x3 = 2
x4 = 3
for i in 1:X
  println(a[i,:]);
end
println(x1,",",x2);
println(x3,",",x4);
println(a[x1:x2-1,x3:x4-1]);
print("[")
print(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> (y)->map((x)->ModalLogic.print_world(x),y));
println("]")
print("[")
print(fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> (y)->map((x)->ModalLogic.print_world(x),y));
println("]")
println(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr);
println(fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr);
(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr) == (fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr) || break;
end

=#

################################################################################
# END 2D Topological relations
################################################################################



# const DimensionalUniDataset{T<:Number,UD} = AbstractArray{T,UD}# getUniChannel(ud::DimensionalUniDataset{T,1},  idx::Integer) where T = @views ud[idx]           # N=0
# getUniChannel(ud::DimensionalUniDataset{T,2},  idx::Integer) where T = @views ud[:, idx]        # N=1
# getUniChannel(ud::DimensionalUniDataset{T,3},  idx::Integer) where T = @views ud[:, :, idx]     # N=2
# Initialize DimensionalUniDataset by slicing across the attribute dimension
# DimensionalUniDataset(::UndefInitializer, d::DimensionalDataset{T,2}) where T = Array{T, 1}(undef, n_samples(d))::DimensionalUniDataset{T, 1}
# DimensionalUniDataset(::UndefInitializer, d::DimensionalDataset{T,3}) where T = Array{T, 2}(undef, size(d)[1:end-1])::DimensionalUniDataset{T, 2}
# DimensionalUniDataset(::UndefInitializer, d::DimensionalDataset{T,4}) where T = Array{T, 3}(undef, size(d)[1:end-1])::DimensionalUniDataset{T, 3}


# get_channel(d::DimensionalDataset{T,2},      idx_i::Integer, idx_a::Integer) where T = @views d[      idx_a, idx_i]::T                       # N=0
# get_channel(d::DimensionalDataset{T,3},      idx_i::Integer, idx_a::Integer) where T = @views d[:,    idx_a, idx_i]::DimensionalChannel{T,1} # N=1
# get_channel(d::DimensionalDataset{T,4},      idx_i::Integer, idx_a::Integer) where T = @views d[:, :, idx_a, idx_i]::DimensionalChannel{T,2} # N=2

# channel_size(d::DimensionalDataset{T,2}, idx_i::Integer) where T = size(d[      1, idx_i])
# channel_size(d::DimensionalDataset{T,3}, idx_i::Integer) where T = size(d[:,    1, idx_i])
# channel_size(d::DimensionalDataset{T,4}, idx_i::Integer) where T = size(d[:, :, 1, idx_i])
# channel_size(d::DimensionalDataset{T,D}, idx_i::Integer) where {T,D} = size(d[idx_i])[1:end-2]

# @computed get_channel(X::InterpretedModalDataset{T,N}, idxs::AbstractVector{Integer}, attribute::Integer) where T = X[idxs, attribute, fill(:, N)...]::AbstractArray{T,N-1}
# # get_channel(X::InterpretedModalDataset,   args...)    = get_channel(X.domain, args...)
# # get_channel(X::MultiFrameModalDataset,   i_frame::Integer, idx_i::Integer, idx_f::Integer, args...)  = get_channel(X.frames[i_frame], idx_i, idx_f, args...)
# 
# 
# TODO maybe using views can improve performances
# attributeview(X::DimensionalDataset{T,2}, idxs::AbstractVector{Integer}, attribute::Integer) = d[idxs, attribute]
# attributeview(X::DimensionalDataset{T,3}, idxs::AbstractVector{Integer}, attribute::Integer) = view(d, idxs, attribute, :)
# attributeview(X::DimensionalDataset{T,4}, idxs::AbstractVector{Integer}, attribute::Integer) = view(d, idxs, attribute, :, :)


# strip_domain(d::DimensionalDataset{T,2}) where T = d  # N=0
# strip_domain(d::DimensionalDataset{T,3}) where T = dropdims(d; dims=1)      # N=1
# strip_domain(d::DimensionalDataset{T,4}) where T = dropdims(d; dims=(1,2))  # N=2

# function prepare_featsnaggrs(grouped_featsnops::AbstractVector{<:AbstractVector{<:TestOperatorFun}})
    
#   # Pairs of feature ids + set of aggregators
#   grouped_featsnaggrs = Vector{<:Aggregator}[
#       ModalLogic.existential_aggregator.(test_operators) for (i_feature, test_operators) in enumerate(grouped_featsnops)
#   ]

#   # grouped_featsnaggrs = [grouped_featsnaggrs[i_feature] for i_feature in 1:length(features)]

#   # # Flatten dictionary, and enhance aggregators in dictionary with their relative indices
#   # flattened_featsnaggrs = Tuple{<:ModalFeature,<:Aggregator}[]
#   # i_featsnaggr = 1
#   # for (i_feature, aggregators) in enumerate(grouped_featsnaggrs)
#   #   for aggregator in aggregators
#   #       push!(flattened_featsnaggrs, (features[i_feature],aggregator))
#   #       i_featsnaggr+=1
#   #   end
#   # end

#   grouped_featsnaggrs
# end
