
# partially written by Poom Chiarawongse <eight1911@gmail.com> 

module util

using LinearAlgebra
using Random
using StatsBase

export entropy, subscriptnumber, spawn_rng


############################################################################################
############################################################################################
############################################################################################

# Generate a new rng by peeling the seed from another rng
# (if used correctly, this ensures reproducibility)
spawn_rng(rng) = Random.MersenneTwister(abs(rand(rng, Int)))

################################################################################
# Loss functions for regression and classification
# These functions return the additive inverse of entropy measures
# 
# For each measure, three versions are defined:
# - A single version, computing the loss for a single dataset
# - A combined version, computing the loss for a dataset split, equivalent to (ws_l*entropy_l + ws_r*entropy_r)
# - A final version, which corrects the loss and is only computed after the optimization step.
# 
# Note: regression losses are defined in the weighted & unweigthed versions
# TODO: write a loss based on gini index
################################################################################

################################################################################
# Classification: Shannon entropy
# (ps = normalize(ws, 1); return -sum(ps.*log.(ps)))
# Source: _shannon_entropy from https://github.com/bensadeghi/DecisionTree.jl/blob/master/src/util.jl, with inverted sign

# Single
Base.@propagate_inbounds @inline function _shannon_entropy_mod(ws :: AbstractVector{U}, t :: U) where {U <: Real}
    s = 0.0
    @simd for k in ws
        if k > 0
            s += k * log(k)
        end
    end
    return -(log(t) - s / t)
end

# Double
Base.@propagate_inbounds @inline function _shannon_entropy_mod(
    ws_l :: AbstractVector{U}, tl :: U,
    ws_r :: AbstractVector{U}, tr :: U,
) where {U <: Real}
    (tl * _shannon_entropy_mod(ws_l, tl) +
     tr * _shannon_entropy_mod(ws_r, tr))
end

# Correction
Base.@propagate_inbounds @inline function _shannon_entropy_mod(e :: AbstractFloat)
    e
end

# ShannonEntropy() = _shannon_entropy
ShannonEntropy() = _shannon_entropy_mod

################################################################################
# Classification: Shannon (second untested version)

# # Single
# Base.@propagate_inbounds @inline function _shannon_entropy(ws :: AbstractVector{U}, t :: U) where {U <: Real}
#     log(t) + _shannon_entropy(ws) / t
# end

# Base.@propagate_inbounds @inline function _shannon_entropy(ws :: AbstractVector{U}) where {U <: Real}
#     s = 0.0
#     for k in filter((k)->k > 0, ws)
#         s += k * log(k)
#     end
#     s
# end

# # Double
# Base.@propagate_inbounds @inline function _shannon_entropy(
#     ws_l :: AbstractVector{U}, tl :: U,
#     ws_r :: AbstractVector{U}, tr :: U,
# ) where {U <: Real}
#     (tl * log(tl) + _shannon_entropy(ws_l) +
#      tr * log(tr) + _shannon_entropy(ws_r))
# end

# # Correction
# Base.@propagate_inbounds @inline function _shannon_entropy(e :: AbstractFloat)
#     e*log2(ℯ)
# end

################################################################################
# Classification: Tsallis entropy
# (ps = normalize(ws, 1); return -log(sum(ps.^alpha))/(1.0-alpha)) with (alpha > 1.0)

# Single
Base.@propagate_inbounds @inline function _tsallis_entropy(alpha :: AbstractFloat, ws :: AbstractVector{U}, t :: U) where {U <: Real}
    log(sum(ps = normalize(ws, 1).^alpha))
end

# Double
Base.@propagate_inbounds @inline function _tsallis_entropy(
    alpha :: AbstractFloat,
    ws_l :: AbstractVector{U}, tl :: U,
    ws_r :: AbstractVector{U}, tr :: U,
) where {U <: Real}
    (tl * _tsallis_entropy(alpha, ws_l, tl) +
     tr * _tsallis_entropy(alpha, ws_r, tr))
end

# Correction
Base.@propagate_inbounds @inline function _tsallis_entropy(alpha :: AbstractFloat, e :: AbstractFloat)
    e*(1/(alpha-1.0))
end

TsallisEntropy(alpha::AbstractFloat) = (args...)->_tsallis_entropy(alpha, args...)

################################################################################
# Classification: Renyi entropy
# (ps = normalize(ws, 1); -(1.0-sum(ps.^alpha))/(alpha-1.0)) with (alpha > 1.0)

# Single
Base.@propagate_inbounds @inline function _renyi_entropy(alpha :: AbstractFloat, ws :: AbstractVector{U}, t :: U) where {U <: Real}
    (sum(normalize(ws, 1).^alpha)-1.0)
end

# Double
Base.@propagate_inbounds @inline function _renyi_entropy(
    alpha :: AbstractFloat,
    ws_l :: AbstractVector{U}, tl :: U,
    ws_r :: AbstractVector{U}, tr :: U,
) where {U <: Real}
    (tl * _renyi_entropy(alpha, ws_l, tl) +
     tr * _renyi_entropy(alpha, ws_r, tr))
end

# Correction
Base.@propagate_inbounds @inline function _renyi_entropy(alpha :: AbstractFloat, e :: AbstractFloat)
    e*(1/(alpha-1.0))
end

RenyiEntropy(alpha::AbstractFloat) = (args...)->_renyi_entropy(alpha, args...)

################################################################################
# Regression: Variance (weighted & unweigthed, see https://en.m.wikipedia.org/wiki/Weighted_arithmetic_mean)

# Single
# sum(ws .* ((ns .- (sum(ws .* ns)/t)).^2)) / (t)
Base.@propagate_inbounds @inline function _variance(ns :: AbstractVector{L}, s :: L, t :: Integer) where {L, U <: Real}
    # @btime sum((ns .- mean(ns)).^2) / (1 - t)
    # @btime (sum(ns.^2)-s^2/t) / (1 - t)
    (sum(ns.^2)-s^2/t) / (1 - t)
    # TODO remove / (1 - t) from here, and move it to the correction-version of _variance, but it must be for single-version only!
end

# Single weighted (non-frequency weigths interpretation)
# sum(ws .* ((ns .- (sum(ws .* ns)/t)).^2)) / (t)
Base.@propagate_inbounds @inline function _variance(ns :: AbstractVector{L}, ws :: AbstractVector{U}, wt :: U) where {L, U <: Real}
    # @btime (sum(ws .* ns)/wt)^2 - sum(ws .* (ns.^2))/wt
    # @btime (wns = ws .* ns; (sum(wns)/wt)^2 - sum(wns .* ns)/wt)
    # @btime (wns = ws .* ns; sum(wns)^2/wt^2 - sum(wns .* ns)/wt)
    # @btime (wns = ws .* ns; (sum(wns)^2/wt - sum(wns .* ns))/wt)
    (wns = ws .* ns; (sum(wns .* ns) - sum(wns)^2/wt)/wt)
end

# Double
Base.@propagate_inbounds @inline function _variance(
    ns_l :: AbstractVector{U}, sl :: L, tl :: U,
    ns_r :: AbstractVector{U}, sr :: L, tr :: U,
) where {L, U <: Real}
    ((tl*sum(ns_l.^2)-sl^2) / (1 - tl)) +
    ((tr*sum(ns_l.^2)-sr^2) / (1 - tr))
end

# Correction
Base.@propagate_inbounds @inline function _variance(e :: AbstractFloat)
    e
end

# TODO write double non weigthed

############################################################################################

# The default classification loss is Shannon's entropy
entropy = ShannonEntropy()
# The default regression loss is variance
variance = _variance


############################################################################################
############################################################################################
############################################################################################

# Translate a list of labels into categorical form
Base.@propagate_inbounds @inline function get_categorical_form(Y :: AbstractVector{T}) where {T}
    class_names = unique(Y)

    dict = Dict{T, Int64}()
    @simd for i in 1:length(class_names)
        @inbounds dict[class_names[i]] = i
    end

    _Y = Array{Int64}(undef, length(Y))
    @simd for i in 1:length(Y)
        @inbounds _Y[i] = dict[Y[i]]
    end

    return class_names, _Y
end

################################################################################
# Sort utils
################################################################################

# adapted from the Julia Base.Sort Library
Base.@propagate_inbounds @inline function partition!(v::AbstractVector, w::AbstractVector{T}, pivot::T, region::UnitRange{<:Integer}) where T
    i, j = 1, length(region)
    r_start = region.start - 1
    @inbounds while true
        while i <= length(region) && w[i] <= pivot; i += 1; end;
        while j >= 1              && w[j]  > pivot; j -= 1; end;
        i >= j && break
        ri = r_start + i
        rj = r_start + j
        v[ri], v[rj] = v[rj], v[ri]
        w[i], w[j] = w[j], w[i]
        i += 1; j -= 1
    end
    return j
end

function nat_sort(x, y)
    # https://titanwolf.org/Network/Articles/Article?AID=969b78b2-141a-43ef-9391-7c55b3c513c7
    splitbynum(x) = split(x, r"(?<=\D)(?=\d)|(?<=\d)(?=\D)")
    numstringtonum(arr) = [(n = tryparse(Float32, e)) != nothing ? n : e for e in arr]
    
    xarr = numstringtonum(splitbynum(string(x)))
    yarr = numstringtonum(splitbynum(string(y)))
    for i in 1:min(length(xarr), length(yarr))
        if typeof(xarr[i]) != typeof(yarr[i])
            a = string(xarr[i]); b = string(yarr[i])
        else
             a = xarr[i]; b = yarr[i]
        end
        if a == b
            continue
        else
            return a < b
        end
    end
    return length(xarr) < length(yarr)
end

################################################################################
# I/O utils
################################################################################

# Source: https://stackoverflow.com/questions/46671965/printing-variable-subscripts-in-julia/46674866
# '₀'
subscriptnumber(i::Int) = begin
    join([
        (if i < 0
            [Char(0x208B)]
        else [] end)...,
        [Char(0x2080+d) for d in reverse(digits(abs(i)))]...
    ])
end
# https://www.w3.org/TR/xml-entity-names/020.html
# '․', 'ₑ', '₋'
subscriptnumber(s::AbstractString) = begin
    char_to_subscript(ch) = begin
        if ch == 'e'
            'ₑ'
        elseif ch == '.'
            '․'
        elseif ch == '.'
            '․'
        elseif ch == '-'
            '₋'
        else
            subscriptnumber(parse(Int, ch))
        end
    end

    try
        join(map(char_to_subscript, [string(ch) for ch in s]))
    catch
        s
    end
end

subscriptnumber(i::AbstractFloat) = subscriptnumber(string(i))

################################################################################
# Others
################################################################################


function minify(d::AbstractVector{T}) where {T<:Union{Number,Missing,Nothing}}
    vals = unique(d)
    n_unique_vals = length(vals)
    new_T = UInt8
    for (nbits, _T) in [(8 => UInt8), (16 => UInt16), (32 => UInt32), (64 => UInt64), (128 => UInt128)]
        if n_unique_vals <= 2^nbits
            new_T = _T
            break
        end
    end
    if new_T == T
        d, identity
    else
        sort!(vals)
        new_d = map((x)->findfirst((v)->(v==x), vals), d)
        # backmap = Dict{T,new_T}([i => v for (i,v) in enumerate(vals)])
        backmap = (x)->vals[x]
        # backmap = vals
        new_d, backmap
    end
end


function minify(d::AbstractVector{<:MID}) where {MID<:Array}
    @assert all((x)->(eltype(x) <: Union{Number,Missing,Nothing}), d)
    vals = unique(Iterators.flatten(d))
    n_unique_vals = length(vals)
    new_T = UInt8
    for (nbits, _T) in [(8 => UInt8), (16 => UInt16), (32 => UInt32), (64 => UInt64), (128 => UInt128)]
        if n_unique_vals <= 2^nbits
            new_T = _T
            break
        end
    end
    sort!(vals)
    new_d = map((x1)->begin
        # new_dict_type = typeintersect(AbstractDict{Int64,Float64},MID{ID,new_T})
        Array{new_T}([findfirst((v)->(v==x2), vals) for x2 in x1])
    end, d)
    # backmap = Dict{T,new_T}([i => v for (i,v) in enumerate(vals)])
    backmap = (x)->vals[x]
    # backmap = vals
    new_d, backmap
end


function minify(d::AbstractVector{<:MID}) where {ID,MID<:Dict{<:ID,T where T<:Union{Number,Missing,Nothing}}}
    vals = unique(Iterators.flatten([values(x) for x in d]))
    n_unique_vals = length(vals)
    new_T = UInt8
    for (nbits, _T) in [(8 => UInt8), (16 => UInt16), (32 => UInt32), (64 => UInt64), (128 => UInt128)]
        if n_unique_vals <= 2^nbits
            new_T = _T
            break
        end
    end
    if new_T == T
        d, identity
    else
        sort!(vals)
        new_d = map((x1)->begin
            # new_dict_type = typeintersect(AbstractDict{Int64,Float64},MID{ID,new_T})
            Dict{ID,new_T}([id => findfirst((v)->(v==x2), vals) for (id, x2) in x1])
        end, d)
        # backmap = Dict{T,new_T}([i => v for (i,v) in enumerate(vals)])
        backmap = (x)->vals[x]
        # backmap = vals
        new_d, backmap
    end
end

# const Minifiable = Union{
#     AbstractArray{T} where {T<:Union{Number,Missing,Nothing}},
#     AbstractArray{<:AbstractArray{T}} where {T<:Union{Number,Missing,Nothing}},
#     AbstractArray{<:MID} where {T<:Union{Number,Missing,Nothing},ID,MID<:Dict{ID,T}},
# }
# const Backmap = Union{
#     Vector{<:Integer}
# }


vectorize(x::Real) = [x]
vectorize(x::AbstractVector) = x

@inline function softminimum(vals, alpha)
    _vals = util.vectorize(vals);
    partialsort!(_vals,ceil(Int, alpha*length(_vals)); rev=true)
end

@inline function softmaximum(vals, alpha)
    _vals = util.vectorize(vals);
    partialsort!(_vals,ceil(Int, alpha*length(_vals)))
end

function all_broadcast_sc(test_operator, values, threshold)
    # Note: this is faster than all(broadcast(test_operator, values, threshold))
    for x in values
        test_operator(x,threshold) || return false
    end
    return true
end


# minExtrema(extr::Union{NTuple{N,NTuple{2,T}},AbstractVector{NTuple{2,T}}}) where {T<:Real,N} = reduce(((fst,snd),(f,s))->(min(fst,f),max(snd,s)), extr; init=(typemax(T),typemin(T)))
# maxExtrema(extr::Union{NTuple{N,NTuple{2,T}},AbstractVector{NTuple{2,T}}}) where {T<:Real,N} = reduce(((fst,snd),(f,s))->(max(fst,f),min(snd,s)), extr; init=(typemin(T),typemax(T)))
# minExtrema(extr::Vararg{NTuple{2,T}}) where {T<:Real} = minExtrema(extr)
# maxExtrema(extr::Vararg{NTuple{2,T}}) where {T<:Real} = maxExtrema(extr)

end
