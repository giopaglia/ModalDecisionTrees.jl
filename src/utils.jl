# partially written by Poom Chiarawongse <eight1911@gmail.com>
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
