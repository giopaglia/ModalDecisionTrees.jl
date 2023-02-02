
# partially written by Poom Chiarawongse <eight1911@gmail.com> 

module utils

using LinearAlgebra
using Random
using StatsBase

############################################################################################
# Sort utils
############################################################################################

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

# TODO figure out what to do with this.
function all_broadcast_sc(_test_operator, values, threshold)
    # Note: this is faster than all(broadcast(_test_operator, values, threshold))
    for x in values
        (_test_operator)(x,threshold) || return false
    end
    return true
end


# minExtrema(extr::Union{NTuple{N,NTuple{2,T}},AbstractVector{NTuple{2,T}}}) where {T<:Real,N} = reduce(((fst,snd),(f,s))->(min(fst,f),max(snd,s)), extr; init=(typemax(T),typemin(T)))
# maxExtrema(extr::Union{NTuple{N,NTuple{2,T}},AbstractVector{NTuple{2,T}}}) where {T<:Real,N} = reduce(((fst,snd),(f,s))->(max(fst,f),min(snd,s)), extr; init=(typemin(T),typemax(T)))
# minExtrema(extr::Vararg{NTuple{2,T}}) where {T<:Real} = minExtrema(extr)
# maxExtrema(extr::Vararg{NTuple{2,T}}) where {T<:Real} = maxExtrema(extr)

end
