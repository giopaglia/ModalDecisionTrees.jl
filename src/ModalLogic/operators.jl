export EQ, GT, LT, GEQ, LEQ,
				existential_aggregator, aggregator_bottom,
				TestOperatorFun

const TestOperatorFun = Function
################################################################################
################################################################################

# abstract type AbstractTestOperator end

################################################################################
################################################################################

# @inline evaluateThreshCondition(::TestOperatorPositive, t::T, gamma::T) where {T} = (t <= gamma)
# @inline evaluateThreshCondition(::TestOperatorNegative, t::T, gamma::T) where {T} = (t >= gamma)

# @inline bottom(::TestOperatorPositive, T::Type) = typemin(T)

# @inline opt(::TestOperatorPositive) = max


# @inline testCondition(test_operator::_TestOpGeq, w::AbstractWorld, channel::MatricialChannel{T,N}, threshold::Number) where {T,N} = begin # TODO maybe this becomes SIMD, or sum/all(ch_readWorld(w,channel)  .<= threshold)
# 	# Source: https://stackoverflow.com/questions/47564825/check-if-all-the-elements-of-a-julia-array-are-equal
# 	# @inbounds
# 	# TODO try:
# 	# all(ch_readWorld(w,channel) .>= threshold)
# 	for x in ch_readWorld(w,channel)
# 		x >= threshold || return false
# 	end
# 	return true
# end
# @inline testCondition(test_operator::_TestOpLeq, w::AbstractWorld, channel::MatricialChannel{T,N}, threshold::Number) where {T,N} = begin # TODO maybe this becomes SIMD, or sum/all(ch_readWorld(w,channel)  .<= threshold)
# 	# Source: https://stackoverflow.com/questions/47564825/check-if-all-the-elements-of-a-julia-array-are-equal
# 	# @info "WLes" w threshold #n ch_readWorld(w,channel)
# 	# @inbounds
# 	# TODO try:
# 	# all(ch_readWorld(w,channel) .<= threshold)
# 	for x in ch_readWorld(w,channel)
# 		x <= threshold || return false
# 	end
# 	return true
# end

# Base.show(io::IO, test_operator::_TestOpGeq) = print(io, "⫺")
# Base.show(io::IO, test_operator::_TestOpLeq) = print(io, "⫹")
 

 
# struct _TestOpGeqSoft  <: TestOperatorPositive
#   alpha :: AbstractFloat
#   _TestOpGeqSoft(a::T) where {T<:Real} = (a > 0 && a < 1) ? new(a) : error("Invalid instantiation for test operator: _TestOpGeqSoft($(a))")
# end;
# struct _TestOpLeqSoft  <: TestOperatorNegative
#   alpha :: AbstractFloat
#   _TestOpLeqSoft(a::T) where {T<:Real} = (a > 0 && a < 1) ? new(a) : error("Invalid instantiation for test operator: _TestOpLeqSoft($(a))")
# end;

# Base.show(io::IO, test_operator::_TestOpGeqSoft) = print(io, "⫺" * subscriptnumber(rstrip(rstrip(string(alpha(test_operator)*100), '0'), '.')))
# Base.show(io::IO, test_operator::_TestOpLeqSoft) = print(io, "⫹" * subscriptnumber(rstrip(rstrip(string(alpha(test_operator)*100), '0'), '.')))


# @inline testCondition(test_operator::_TestOpGeqSoft, w::AbstractWorld, channel::MatricialChannel{T,N}, threshold::Number) where {T,N} = begin 
# 	ys = 0
# 	# TODO write with reduce, and optimize it (e.g. by stopping early if the condition is reached already)
# 	vals = ch_readWorld(w,channel)
# 	for x in vals
# 		if x >= threshold
# 			ys+=1
# 		end
# 	end
# 	(ys/length(vals)) >= test_operator.alpha
# end

# @inline testCondition(test_operator::_TestOpLeqSoft, w::AbstractWorld, channel::MatricialChannel{T,N}, threshold::Number) where {T,N} = begin 
# 	ys = 0
# 	# TODO write with reduce, and optimize it (e.g. by stopping early if the condition is reached already)
# 	vals = ch_readWorld(w,channel)
# 	for x in vals
# 		if x <= threshold
# 			ys+=1
# 		end
# 	end
# 	(ys/length(vals)) >= test_operator.alpha
# end


# struct _NO_OP  <: AbstractTestOperator end; const NO_OP = _NO_OP();

# TODO add NEQ


# crisp operators

equality_operator(x::S,  y::S)       where {S} = ==(x,y)  # =
greater_than_operator(x::S,  y::S)   where {S} =  >(x,y)  # >
lesser_than_operator(x::S,  y::S)    where {S} =  <(x,y)  # <
greater_eq_than_operator(x::S, y::S) where {S} =  ≥(x,y)  # ≥
lesser_eq_than_operator(x::S, y::S)  where {S} =  ≤(x,y)  # ≤

existential_aggregator(::typeof(equality_operator))        = ∪
existential_aggregator(::typeof(greater_than_operator))    = maximum
existential_aggregator(::typeof(lesser_than_operator))     = minimum
existential_aggregator(::typeof(greater_eq_than_operator)) = maximum
existential_aggregator(::typeof(lesser_eq_than_operator))  = minimum

# bottom_aggregator(::typeof(∪))          = TODO
aggregator_bottom(::typeof(maximum), T::Type) = typemin(T)
aggregator_bottom(::typeof(minimum), T::Type) = typemax(T)

aggregator_to_binary(::typeof(maximum)) = max
aggregator_to_binary(::typeof(minimum)) = min


const EQ  = equality_operator
const GT  = greater_than_operator
const LT  = lesser_than_operator
const GEQ = greater_eq_than_operator
const LEQ = lesser_eq_than_operator

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
