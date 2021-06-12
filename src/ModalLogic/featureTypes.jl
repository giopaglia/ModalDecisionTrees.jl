export FeatureType, FeatureTypeFun,
				AttributeMinimumFeatureType, AttributeMaximumFeatureType,
				AttributeSoftMinimumFeatureType, AttributeSoftMaximumFeatureType

const FeatureType = Integer

SimpleFeatureType(a, feature) = feature

display_feature(feature) = "V$(feature)"

################################################################################
################################################################################

abstract type FeatureTypeFun end

struct _FeatureTypeNone  <: FeatureTypeFun end; const FeatureTypeNone  = _FeatureTypeNone();

# struct AggregateFeatureType{worldType<:AbstractWorld} <: FeatureTypeFun
# struct SingleAttributeAggregateFeatureType <: FeatureTypeFun
# 	i_attribute::Integer
# 	aggregator::Function
# end

# yieldFunction(f::SingleAttributeAggregateFeatureType) = f.aggregator ∘ (x)->ModalLogic.getInstanceAttribute(x,f.i_attr)



# TODO
# AttributeSoftMinimumFeatureType
# AttributeSoftMaximumFeatureType

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

struct AttributeSoftMinimumFeatureType{T<:AbstractFloat} <: FeatureTypeFun
	i_attribute::Integer
	alpha::T
	function AttributeSoftMinimumFeatureType(
		i_attribute::Integer,
		alpha::T,
	) where {T}
		@assert !iszero(alpha) "Can't instantiate AttributeSoftMinimumFeatureType with alpha = $(alpha)"
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
		@assert !iszero(alpha) "Can't instantiate AttributeSoftMaximumFeatureType with alpha = $(alpha)"
		@assert !isone(alpha) "Can't instantiate AttributeSoftMaximumFeatureType with alpha = $(alpha). Use AttributeMaximumFeatureType instead!"
		new{T}(i_attribute, alpha)
	end
end

yieldFunction(f::AttributeSoftMinimumFeatureType{T}) where T =
	(x)->(vals = vec(ModalLogic.getInstanceAttribute(x,f.i_attribute)); partialsort!(vals,ceil(Int, f.alpha*length(vals)); rev=true))
yieldFunction(f::AttributeSoftMaximumFeatureType{T}) where T =
	(x)->(vals = vec(ModalLogic.getInstanceAttribute(x,f.i_attribute)); partialsort!(vals,ceil(Int, f.alpha*length(vals))))

Base.show(io::IO, f::AttributeSoftMinimumFeatureType) = Base.print(io, "min" * subscriptnumber(rstrip(rstrip(string(f.alpha*100), '0'), '.')) * "(A$(f.i_attribute))")
Base.show(io::IO, f::AttributeSoftMaximumFeatureType) = Base.print(io, "max" * subscriptnumber(rstrip(rstrip(string(f.alpha*100), '0'), '.')) * "(A$(f.i_attribute))")

# yieldFunction(AttributeSoftMaximumFeatureType(1,0.8))

# aggr_union = ∪
# aggr_min = minimum
# aggr_max = maximum

# aggr_soft_min
# get_aggr_softmin_f(alpha::AbstractFloat) = begin
# 	@inline f(vals::AbstractVector{T}) where {T} = begin
# 		partialsort!(vals,ceil(Int, alpha*length(vals)); rev=true)
# 	end
# end

# aggr_soft_max
# get_aggr_softmax_f(alpha::AbstractFloat) = begin
# 	@inline f(vals::AbstractVector{T}) where {T} = begin
# 		partialsort!(vals,ceil(Int, alpha*length(vals)))
# 	end
# end


# @inline computePropositionalThreshold(::_TestOpGeq, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
# 	minimum(ch_readWorld(w,channel))
# end
# @inline computePropositionalThreshold(::_TestOpLeq, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
# 	maximum(ch_readWorld(w,channel))
# end

# @inline test_op_partialsort!(test_op::_TestOpGeqSoft, vals::Vector{T}) where {T} = 
# 	partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)); rev=true)
	
# @inline test_op_partialsort!(test_op::_TestOpLeqSoft, vals::Vector{T}) where {T} = 
# 	partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)))


# @inline computePropositionalThreshold(test_op::Union{_TestOpGeqSoft,_TestOpLeqSoft}, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
# 	vals = vec(ch_readWorld(w,channel))
# 	test_op_partialsort!(test_op,vals)
# end


# @inline computePropositionalThresholdMany(test_ops::Vector{<:TestOperator}, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
# 	vals = vec(ch_readWorld(w,channel))
# 	(test_op_partialsort!(test_op,vals) for test_op in test_ops)
# end




# AggregateFeatureType{Interval}(minimum)
# AggregateFeatureType{Interval}(maximum)
# AggregateFeatureType{Interval}(soft_minimum_f(80))
# AggregateFeatureType{Interval}(soft_maximum_f(80))


# f(x) = getindex(x,1,:) |> maximum
