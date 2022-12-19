export entropy

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
Base.@propagate_inbounds @inline function _variance(ns :: AbstractVector{L}, s :: L, t :: Integer) where {L}
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
