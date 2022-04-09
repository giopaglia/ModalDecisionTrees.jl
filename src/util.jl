
# written by Poom Chiarawongse <eight1911@gmail.com>

module util

export gini, entropy, zero_one, q_bi_sort!, subscriptnumber, spawn_rng
export R2, majority_vote

using Random
using LinearAlgebra
using StatsBase

################################################################################
################################################################################
################################################################################

# Generate a new rng from a random pick from a given one.
spawn_rng(rng) = Random.MersenneTwister(abs(rand(rng, Int)))

# This function translates a list of labels into categorical form
Base.@propagate_inbounds @inline function assign(Y :: AbstractVector{T}) where T
    function assign(Y :: AbstractVector{T}, list :: AbstractVector{T}) where T
        dict = Dict{T, Int64}()
        @simd for i in 1:length(list)
            @inbounds dict[list[i]] = i
        end

        _Y = Array{Int64}(undef, length(Y))
        @simd for i in 1:length(Y)
            @inbounds _Y[i] = dict[Y[i]]
        end

        return list, _Y
    end

    set = Set{T}()
    for y in Y
        push!(set, y)
    end
    list = collect(set)
    return assign(Y, list)
end

Base.@propagate_inbounds @inline function zero_one(ns :: AbstractVector{T}, n :: T) where {T <: Real}
    return 1.0 - maximum(ns) / n
end

Base.@propagate_inbounds @inline function gini(ns :: AbstractVector{T}, n :: T) where {T <: Real}
    s = 0.0
    @simd for k in ns
        s += k * (n - k)
    end
    return s / (n * n)
end

# a = [5,3,9,2]
# a = [0,0,9,0]
# entropy(a, sum(a))
# maximum(a)/sum(a)

# returns the entropy of ns/n, ns is an array of integers
# and entropy_terms are precomputed entropy terms

################################################################################
# Entropy measures for regression and classification
# TODO explain: single and double version
# TODO explain: regression has separated weighted & unweigthed versions
# Note: code is equivalent to: (ws_l*entropy_l + ws_r*entropy_r)
# Note: these functions return the additive inverse of entropy measures
################################################################################
# Shannon entropy: (ps = normalize(ws, 1); return -sum(ps.*log.(ps)))
# Single
Base.@propagate_inbounds @inline function _shannon_entropy(ws :: AbstractVector{U}, t :: U) where {U <: Real}
    log(t) + _shannon_entropy(ws) / t
end

Base.@propagate_inbounds @inline function _shannon_entropy(ws :: AbstractVector{U}) where {U <: Real}
    s = 0.0
    for k in filter((k)->k > 0, ws)
        s += k * log(k)
    end
    s
end

# Double
Base.@propagate_inbounds @inline function _shannon_entropy(
    ws_l :: AbstractVector{U}, tl :: U,
    ws_r :: AbstractVector{U}, tr :: U,
) where {U <: Real}
    (tl * log(tl) + _shannon_entropy(ws_l) +
     tr * log(tr) + _shannon_entropy(ws_r))
end

# Correction
Base.@propagate_inbounds @inline function _shannon_entropy(e :: AbstractFloat)
    e*log2(ℯ)
end

# TODO fix: this is _shannon_entropy from https://github.com/bensadeghi/DecisionTree.jl/blob/master/src/util.jl, with inverted sign

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

# Tsallis entropy (alpha > 1.0) (ps = normalize(ws, 1); return -log(sum(ps.^alpha))/(1.0-alpha))
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

# Renyi entropy (alpha > 1.0) (ps = normalize(ws, 1); -(1.0-sum(ps.^alpha))/(alpha-1.0))
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

entropy = ShannonEntropy()

################################################################################
# Variance (weighted & unweigthed, see https://en.m.wikipedia.org/wiki/Weighted_arithmetic_mean)

# Single
# sum(ws .* ((ns .- (sum(ws .* ns)/t)).^2)) / (t)
Base.@propagate_inbounds @inline function _variance(ns :: AbstractVector{L}, s :: L, t :: Integer) where {L, U <: Real}
    # @btime sum((ns .- StatsBase.mean(ns)).^2) / (1 - t)
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

variance = _variance

################################################################################
# adapted from the Julia Base.Sort Library
Base.@propagate_inbounds @inline function partition!(v::AbstractVector, w::AbstractVector{T}, pivot::T, region::Union{AbstractVector{<:Integer},UnitRange{<:Integer}}) where T
    i, j = 1, length(region)
    r_start = region.start - 1
    @inbounds while true
        while i <= length(region) && w[i] <= pivot; i += 1; end; # TODO check this i <= ... sign
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

# adapted from the Julia Base.Sort Library
Base.@propagate_inbounds @inline function insert_sort!(v::AbstractVector, w::AbstractVector, lo::Integer, hi::Integer, offset::Integer)
    @inbounds for i = lo+1:hi
        j = i
        x = v[i]
        y = w[offset+i]
        while j > lo
            if x < v[j-1]
                v[j] = v[j-1]
                w[offset+j] = w[offset+j-1]
                j -= 1
                continue
            end
            break
        end
        v[j] = x
        w[offset+j] = y
    end
    return v
end

Base.@propagate_inbounds @inline function _selectpivot!(v::AbstractVector, w::AbstractVector, lo::Integer, hi::Integer, offset::Integer)
    @inbounds begin
        mi = (lo+hi)>>>1

        # sort the values in v[lo], v[mi], v[hi]

        if v[mi] < v[lo]
            v[mi], v[lo] = v[lo], v[mi]
            w[offset+mi], w[offset+lo] = w[offset+lo], w[offset+mi]
        end
        if v[hi] < v[mi]
            if v[hi] < v[lo]
                v[lo], v[mi], v[hi] = v[hi], v[lo], v[mi]
                w[offset+lo], w[offset+mi], w[offset+hi] = w[offset+hi], w[offset+lo], w[offset+mi]
            else
                v[hi], v[mi] = v[mi], v[hi]
                w[offset+hi], w[offset+mi] = w[offset+mi], w[offset+hi]
            end
        end

        # move v[mi] to v[lo] and use it as the pivot
        v[lo], v[mi] = v[mi], v[lo]
        w[offset+lo], w[offset+mi] = w[offset+mi], w[offset+lo]
        v_piv = v[lo]
        w_piv = w[offset+lo]
    end

    # return the pivot
    return v_piv, w_piv
end

################################################################################
################################################################################
################################################################################

# Coefficient of determination
function R2(actual::AbstractVector, predicted::AbstractVector)
  @assert length(actual) == length(predicted)
  ss_residual = sum((actual - predicted).^2)
  ss_total = sum((actual .- mean(actual)).^2)
  return 1.0 - ss_residual/ss_total
end


################################################################################
################################################################################
################################################################################

# adapted from the Julia Base.Sort Library
Base.@propagate_inbounds @inline function _bi_partition!(v::AbstractVector, w::AbstractVector, lo::Integer, hi::Integer, offset::Integer)
    pivot, w_piv = _selectpivot!(v, w, lo, hi, offset)
    # pivot == v[lo], v[hi] > pivot
    i, j = lo, hi
    @inbounds while true
        i += 1; j -= 1
        while v[i] < pivot; i += 1; end;
        while pivot < v[j]; j -= 1; end;
        i >= j && break
        v[i], v[j] = v[j], v[i]
        w[offset+i], w[offset+j] = w[offset+j], w[offset+i]
    end
    v[j], v[lo] = pivot, v[j]
    w[offset+j], w[offset+lo] = w_piv, w[offset+j]

    # v[j] == pivot
    # v[k] >= pivot for k > j
    # v[i] <= pivot for i < j
    return j
end


# adapted from the Julia Base.Sort Library
# adapted from the Julia Base.Sort Library
# this sorts v[lo:hi] and w[offset+lo, offset+hi]
# simultaneously by the values in v[lo:hi]
const SMALL_THRESHOLD  = 20
Base.@propagate_inbounds @inline function q_bi_sort!(v::AbstractVector, w::AbstractVector, lo::Integer, hi::Integer, offset::Integer)
    @inbounds while lo < hi
        hi-lo <= SMALL_THRESHOLD && return insert_sort!(v, w, lo, hi, offset)
        j = _bi_partition!(v, w, lo, hi, offset)
        if j-lo < hi-j
            # recurse on the smaller chunk
            # this is necessary to preserve O(log(n))
            # stack space in the worst case (rather than O(n))
            lo < (j-1) && q_bi_sort!(v, w, lo, j-1, offset)
            lo = j+1
        else
            j+1 < hi && q_bi_sort!(v, w, j+1, hi, offset)
            hi = j-1
        end
    end
    return v
end

################################################################################
################################################################################
################################################################################

# https://titanwolf.org/Network/Articles/Article?AID=969b78b2-141a-43ef-9391-7c55b3c513c7
splitbynum(x) = split(x, r"(?<=\D)(?=\d)|(?<=\d)(?=\D)")
numstringtonum(arr) = [(n = tryparse(Float32, e)) != nothing ? n : e for e in arr]
function nat_sort(x, y)
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
################################################################################
################################################################################

# https://stackoverflow.com/questions/46671965/printing-variable-subscripts-in-julia/46674866
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

end

