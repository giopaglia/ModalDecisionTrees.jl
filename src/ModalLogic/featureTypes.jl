export FeatureType

const FeatureType = Integer
# abstract type FeatureType end

SimpleFeatureType(a, feature) = feature

display_feature(feature) = "V$(feature)"

################################################################################
################################################################################

const FeatureFun = Function

# struct _FeatureTypeNone  <: FeatureType end; const FeatureTypeNone  = _FeatureTypeNone();

# struct AggregateFeatureType{worldType<:AbstractWorld} <: FeatureType
# struct AggregateFeatureType <: FeatureType
# 	aggregate_function::Function
# end

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
# 	minimum(readWorld(w,channel))
# end
# @inline computePropositionalThreshold(::_TestOpLeq, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
# 	maximum(readWorld(w,channel))
# end

# @inline test_op_partialsort!(test_op::_TestOpGeqSoft, vals::Vector{T}) where {T} = 
# 	partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)); rev=true)
	
# @inline test_op_partialsort!(test_op::_TestOpLeqSoft, vals::Vector{T}) where {T} = 
# 	partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)))


# @inline computePropositionalThreshold(test_op::Union{_TestOpGeqSoft,_TestOpLeqSoft}, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
# 	vals = vec(readWorld(w,channel))
# 	test_op_partialsort!(test_op,vals)
# end


# @inline computePropositionalThresholdMany(test_ops::Vector{<:TestOperator}, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
# 	vals = vec(readWorld(w,channel))
# 	(test_op_partialsort!(test_op,vals) for test_op in test_ops)
# end




# AggregateFeatureType{Interval}(minimum)
# AggregateFeatureType{Interval}(maximum)
# AggregateFeatureType{Interval}(soft_minimum_f(80))
# AggregateFeatureType{Interval}(soft_maximum_f(80))


# f(x) = getindex(x,1,:) |> maximum
