export FeatureType, FeatureTypeFun

const FeatureType = Integer

SimpleFeatureType(a, feature) = feature

display_feature(feature) = "V$(feature)"

################################################################################
################################################################################

abstract type FeatureTypeFun end

# struct _FeatureTypeNone  <: FeatureTypeFun end; const FeatureTypeNone  = _FeatureTypeNone();

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


aggr_union = ∪
aggr_min = minimum
aggr_max = maximum

# aggr_min
get_aggr_softmin_f(alpha::AbstractFloat) = begin
	@inline f(vals::AbstractVector{T}) where {T} = begin
		partialsort!(vals,ceil(Int, alpha*length(vals)); rev=true)
	end
end

get_aggr_softmax_f(alpha::AbstractFloat) = begin
	@inline f(vals::AbstractVector{T}) where {T} = begin
		partialsort!(vals,ceil(Int, alpha*length(vals)))
	end
end


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
