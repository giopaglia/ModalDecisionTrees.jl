export FeatureTypeFun,
        FeatureTypeNone,
        AttributeMinimumFeatureType, AttributeMaximumFeatureType,
        AttributeSoftMinimumFeatureType, AttributeSoftMaximumFeatureType,
        AttributeFunctionFeatureType, ChannelFunctionFeatureType,
        ExternalFWDFeatureType

import Base.vec

Base.vec(x::Number) = [x]

################################################################################
################################################################################

abstract type FeatureTypeFun end

struct _FeatureTypeNone  <: FeatureTypeFun end; const FeatureTypeNone  = _FeatureTypeNone();
yieldFunction(f::_FeatureTypeNone) = @error " Can't intepret FeatureTypeNone in any possible form"
# Base.show(io::IO, f::_FeatureTypeNone) = Base.print(io, "<Empty FeatureType>")
################################################################################
################################################################################
################################################################################

struct AttributeMinimumFeatureType <: FeatureTypeFun
    i_attribute::Integer
end
struct AttributeMaximumFeatureType <: FeatureTypeFun
    i_attribute::Integer
end

yieldFunction(f::AttributeMinimumFeatureType) = minimum ∘ (x)->ModalLogic.getInstanceAttribute(x,f.i_attribute)
yieldFunction(f::AttributeMaximumFeatureType) = maximum ∘ (x)->ModalLogic.getInstanceAttribute(x,f.i_attribute)

Base.show(io::IO, f::AttributeMinimumFeatureType) = Base.print(io, "min(A$(f.i_attribute))")
Base.show(io::IO, f::AttributeMaximumFeatureType) = Base.print(io, "max(A$(f.i_attribute))")

################################################################################
################################################################################
################################################################################

struct AttributeSoftMinimumFeatureType{T<:AbstractFloat} <: FeatureTypeFun
    i_attribute::Integer
    alpha::T
    function AttributeSoftMinimumFeatureType(
        i_attribute::Integer,
        alpha::T,
    ) where {T}
        @assert !(alpha > 1.0 || alpha < 0.0) "Can't instantiate AttributeSoftMinimumFeatureType with alpha = $(alpha)"
        @assert !isone(alpha) "Can't instantiate AttributeSoftMinimumFeatureType with alpha = $(alpha). Use AttributeMinimumFeatureType instead!"
        new{T}(i_attribute, alpha)
    end
end

struct AttributeSoftMaximumFeatureType{T<:AbstractFloat} <: FeatureTypeFun
    i_attribute::Integer
    alpha::T
    function AttributeSoftMaximumFeatureType(
        i_attribute::Integer,
        alpha::T,
    ) where {T}
        @assert !(alpha > 1.0 || alpha < 0.0) "Can't instantiate AttributeSoftMaximumFeatureType with alpha = $(alpha)"
        @assert !isone(alpha) "Can't instantiate AttributeSoftMaximumFeatureType with alpha = $(alpha). Use AttributeMaximumFeatureType instead!"
        new{T}(i_attribute, alpha)
    end
end


yieldFunction(f::AttributeSoftMinimumFeatureType) =
    (x)->(vals = vec(ModalLogic.getInstanceAttribute(x,f.i_attribute)); partialsort!(vals,ceil(Int, f.alpha*length(vals)); rev=true))
yieldFunction(f::AttributeSoftMaximumFeatureType) =
    (x)->(vals = vec(ModalLogic.getInstanceAttribute(x,f.i_attribute)); partialsort!(vals,ceil(Int, f.alpha*length(vals))))

alpha(f::AttributeSoftMinimumFeatureType) = f.alpha
alpha(f::AttributeSoftMaximumFeatureType) = f.alpha

# TODO simplify OneWorld case!! Maybe features must dispatch on WorldType as well or on the type of underlying data!
# For now, OneWorld falls into the generic case through this definition of vec()
# yieldFunction(f::AttributeSoftMinimumFeatureType) = ModalLogic.getInstanceAttribute(x,f.i_attribute)
# yieldFunction(f::AttributeSoftMaximumFeatureType) = ModalLogic.getInstanceAttribute(x,f.i_attribute)

Base.show(io::IO, f::AttributeSoftMinimumFeatureType) = Base.print(io, "min" * subscriptnumber(rstrip(rstrip(string(f.alpha*100), '0'), '.')) * "(A$(f.i_attribute))")
Base.show(io::IO, f::AttributeSoftMaximumFeatureType) = Base.print(io, "max" * subscriptnumber(rstrip(rstrip(string(f.alpha*100), '0'), '.')) * "(A$(f.i_attribute))")

################################################################################
################################################################################
################################################################################

struct AttributeFunctionFeatureType <: FeatureTypeFun
    i_attribute::Integer
    f::Function
end

yieldFunction(f::AttributeFunctionFeatureType) =
    f.f ∘ (x)->(vals = vec(ModalLogic.getInstanceAttribute(x,f.i_attribute));)

Base.show(io::IO, f::AttributeFunctionFeatureType) = Base.print(io, "$(f.f)(A$(f.i_attribute))")

################################################################################
################################################################################
################################################################################

struct ChannelFunctionFeatureType <: FeatureTypeFun
    f::Function
end
yieldFunction(f::ChannelFunctionFeatureType) = f.f
Base.show(io::IO, f::ChannelFunctionFeatureType) = Base.print(io, "$(f.f)")

################################################################################
################################################################################
################################################################################

abstract type FWDFeatureTypeFun<:FeatureTypeFun end


struct FWDFunctionFeatureType <: FWDFeatureTypeFun # TODO test
    fwd_f::Function
end
yieldFunction(f::FWDFunctionFeatureType) = error("yieldFunction(::FWDFunctionFeatureType) should never be called. Check code") # TODO breaks typechecker?
Base.show(io::IO, f::FWDFunctionFeatureType) = Base.print(io, "FWDFunctionFeatureType($(fwd_f))")

struct ExternalFWDFeatureType <: FWDFeatureTypeFun
    name::String
    fwd::Any
end
yieldFunction(f::ExternalFWDFeatureType) = error("yieldFunction(::ExternalFWDFeatureType) should never be called. Check code") # TODO breaks typechecker?
# Base.show(io::IO, f::ExternalFWDFeatureType) = Base.print(io, "ExternalFWDFeatureType(fwd of type $(typeof(f.fwd)))")
Base.show(io::IO, f::ExternalFWDFeatureType) = Base.print(io, "$(f.name)")

################################################################################
################################################################################
################################################################################

# yieldFunction(AttributeSoftMaximumFeatureType(1,0.8))

# aggr_union = ∪
# aggr_min = minimum
# aggr_max = maximum

# aggr_soft_min
# get_aggr_softmin_f(alpha::AbstractFloat) = begin
#   @inline f(vals::AbstractVector{T}) where {T} = begin
#       partialsort!(vals,ceil(Int, alpha*length(vals)); rev=true)
#   end
# end

# aggr_soft_max
# get_aggr_softmax_f(alpha::AbstractFloat) = begin
#   @inline f(vals::AbstractVector{T}) where {T} = begin
#       partialsort!(vals,ceil(Int, alpha*length(vals)))
#   end
# end


# @inline computePropositionalThreshold(::_TestOpGeq, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
#   minimum(ch_readWorld(w,channel))
# end
# @inline computePropositionalThreshold(::_TestOpLeq, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
#   maximum(ch_readWorld(w,channel))
# end

# @inline test_op_partialsort!(test_op::_TestOpGeqSoft, vals::Vector{T}) where {T} = 
#   partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)); rev=true)
    
# @inline test_op_partialsort!(test_op::_TestOpLeqSoft, vals::Vector{T}) where {T} = 
#   partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)))


# @inline computePropositionalThreshold(test_op::Union{_TestOpGeqSoft,_TestOpLeqSoft}, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
#   vals = vec(ch_readWorld(w,channel))
#   test_op_partialsort!(test_op,vals)
# end


# @inline computePropositionalThresholdMany(test_ops::Vector{<:TestOperator}, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
#   vals = vec(ch_readWorld(w,channel))
#   (test_op_partialsort!(test_op,vals) for test_op in test_ops)
# end




# AggregateFeatureType{Interval}(minimum)
# AggregateFeatureType{Interval}(maximum)
# AggregateFeatureType{Interval}(soft_minimum_f(80))
# AggregateFeatureType{Interval}(soft_maximum_f(80))


# f(x) = getindex(x,1,:) |> maximum
