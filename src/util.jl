
# partially written by Poom Chiarawongse <eight1911@gmail.com> 

module util

using LinearAlgebra
using Random
using StatsBase

export subscriptnumber

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
subscriptnumber(i::Any) = i

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
