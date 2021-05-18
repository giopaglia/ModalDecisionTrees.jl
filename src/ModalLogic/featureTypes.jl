export FeatureType

const FeatureType = Integer
# abstract type FeatureType end

SimpleFeatureType(a, feature) = feature

display_feature(feature) = "V$(feature)"
################################################################################
################################################################################

# struct _FeatureTypeNone  <: FeatureType end; const FeatureTypeNone  = _FeatureTypeNone();

# crisp operators

eq(x::S,  y::S) where {S} = ==(x,y)  # =
gt(x::S,  y::S) where {S} =  >(x,y)  # >
lt(x::S,  y::S) where {S} =  <(x,y)  # <
geq(x::S, y::S) where {S} =  ≥(x,y)  # ≥
leq(x::S, y::S) where {S} =  ≤(x,y)  # ≤

# fuzzy operators

# =ₕ
function get_fuzzy_linear_eq(h::T, fuzzy_type::Type{<:Real} = Float64) where {T}
	function f(x::S, y::S) where {S}
		Δ = y-x
		if abs(Δ) ≥ h
			zero(fuzzy_type)
		else
			fuzzy_type(1-(abs(Δ)/h))
		end
	end
end


# >ₕ
function get_fuzzy_linear_gt(h::T, fuzzy_type::Type{<:Real} = Float64) where {T}
	function f(x::S, y::S) where {S}
		Δ = y-x
		if Δ ≥ 0
			zero(fuzzy_type)
		elseif Δ ≤ -h
			one(fuzzy_type)
		else
			fuzzy_type(Δ/h)
		end
	end
end

# <ₕ
function get_fuzzy_linear_lt(h::T, fuzzy_type::Type{<:Real} = Float64) where {T}
	function f(x::S, y::S) where {S}
		Δ = y-x
		if Δ ≥ h
			one(fuzzy_type)
		elseif Δ ≤ 0
			zero(fuzzy_type)
		else
			fuzzy_type(Δ/h)
		end
	end
end


# ≧ₕ
function get_fuzzy_linear_geq(h::T, fuzzy_type::Type{<:Real} = Float64) where {T}
	function f(x::S, y::S) where {S}
		Δ = y-x
		if Δ ≤ 0
			one(fuzzy_type)
		elseif Δ ≥ h
			zero(fuzzy_type)
		else
			fuzzy_type(1-Δ/h)
		end
	end
end


# ≦ₕ
function get_fuzzy_linear_leq(h::T, fuzzy_type::Type{<:Real} = Float64) where {T}
	function f(x::S, y::S) where {S}
		Δ = x-y
		if Δ ≤ 0
			one(fuzzy_type)
		elseif Δ ≥ h
			zero(fuzzy_type)
		else
			fuzzy_type(1-Δ/h)
		end
	end
end

# ≥ₕ
function get_fuzzy_linear_geqt(h::T, fuzzy_type::Type{<:Real} = Float64) where {T}
	h_2 = h/2
	function f(x::S, y::S) where {S}
		Δ = y-x
		if Δ ≥ h_2
			zero(fuzzy_type)
		elseif Δ ≤ -h_2
			one(fuzzy_type)
		else
			fuzzy_type((h_2-Δ)/h)
		end
	end
end

# ≤ₕ
function get_fuzzy_linear_leqt(h::T, fuzzy_type::Type{<:Real} = Float64) where {T}
	h_2 = h/2
	function f(x::S, y::S) where {S}
		Δ = y-x
		if Δ ≥ h_2
			one(fuzzy_type)
		elseif Δ ≤ -h_2
			zero(fuzzy_type)
		else
			fuzzy_type((Δ+h_2)/h)
		end
	end
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

# struct AggregateFeatureType{worldType<:AbstractWorld} <: FeatureType
# 	aggregator_function::Function
# end

# AggregateFeatureType{Interval}(minimum)
# AggregateFeatureType{Interval}(maximum)
# AggregateFeatureType{Interval}(soft_minimum_f(80))
# AggregateFeatureType{Interval}(soft_maximum_f(80))


# f(x) = getindex(x,1,:) |> maximum
