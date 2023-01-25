
goeswith(::Type{Interval}, ::IntervalRelation) = true

_accessibles(fr::Full1DFrame, w::Interval, ::_IA_A) = zip(Iterators.repeated(w.y), w.y+1:X(fr)+1)
_accessibles(fr::Full1DFrame, w::Interval, ::_IA_Ai) = zip(1:w.x-1, Iterators.repeated(w.x))
_accessibles(fr::Full1DFrame, w::Interval, ::_IA_L) = _intervals_in(w.y+1, X(fr)+1)
_accessibles(fr::Full1DFrame, w::Interval, ::_IA_Li) = _intervals_in(1, w.x-1)
_accessibles(fr::Full1DFrame, w::Interval, ::_IA_B) = zip(Iterators.repeated(w.x), w.x+1:w.y-1)
_accessibles(fr::Full1DFrame, w::Interval, ::_IA_Bi) = zip(Iterators.repeated(w.x), w.y+1:X(fr)+1)
_accessibles(fr::Full1DFrame, w::Interval, ::_IA_E) = zip(w.x+1:w.y-1, Iterators.repeated(w.y))
_accessibles(fr::Full1DFrame, w::Interval, ::_IA_Ei) = zip(1:w.x-1, Iterators.repeated(w.y))
_accessibles(fr::Full1DFrame, w::Interval, ::_IA_D) = _intervals_in(w.x+1, w.y-1)
_accessibles(fr::Full1DFrame, w::Interval, ::_IA_Di) = Iterators.product(1:w.x-1, w.y+1:X(fr)+1)
_accessibles(fr::Full1DFrame, w::Interval, ::_IA_O) = Iterators.product(w.x+1:w.y-1, w.y+1:X(fr)+1)
_accessibles(fr::Full1DFrame, w::Interval, ::_IA_Oi) = Iterators.product(1:w.x-1, w.x+1:w.y-1)

_accessibles(fr::Full1DFrame, w::Interval, ::_IA_AorO) = Iterators.product(w.x+1:w.y, w.y+1:X(fr)+1)
_accessibles(fr::Full1DFrame, w::Interval, ::_IA_AiorOi) = Iterators.product(1:w.x-1,   w.x:w.y-1)

_accessibles(fr::Full1DFrame, w::Interval, ::_IA_DorBorE) = Iterators.flatten((_accessibles(fr, w, IA_B), _accessibles(fr, w, IA_D), _accessibles(fr, w, IA_E)))
_accessibles(fr::Full1DFrame, w::Interval, ::_IA_DiorBiorEi) = Iterators.flatten((_accessibles(fr, w, IA_Bi), _accessibles(fr, w, IA_Di), _accessibles(fr, w, IA_Ei)))

_accessibles(fr::Full1DFrame, w::Interval, ::_IA_I) = Iterators.flatten((
    # Iterators.product(1:w.x-1, w.y+1:X(fr)+1),   # Di
    # Iterators.product(w.x:w.y, w.y+1:X(fr)+1),   # A+O+Bi
    Iterators.product(1:w.y, w.y+1:X(fr)+1),       # Di+A+O+Bi
    Iterators.product(1:w.x-1, w.x:w.y),       # Ai+Oi+Ei
    zip(Iterators.repeated(w.x), w.x+1:w.y-1), # B
    zip(w.x+1:w.y-1, Iterators.repeated(w.y)), # E
    _intervals_in(w.x+1, w.y-1),               # D
))


# More efficient implementations for edge cases (Later)
accessibles(fr::Full1DFrame, S::AbstractWorldSet{Interval}, ::_IA_L) =
    accessibles(fr, nth(S, argmin(map((w)->w.y, S))), IA_L)
accessibles(fr::Full1DFrame, S::AbstractWorldSet{Interval}, ::_IA_Li) =
    accessibles(fr, nth(S, argmax(map((w)->w.x, S))), IA_Li)

# More efficient implementations for edge cases (After)
accessibles(fr::Full1DFrame, S::AbstractWorldSet{Interval}, ::_IA_A) =
    IterTools.imap(Interval,
        Iterators.flatten(
            IterTools.imap((y)->zip(Iterators.repeated(y), y+1:X(fr)+1),
                IterTools.distinct(map((w)->w.y, S))
            )
        )
    )
accessibles(fr::Full1DFrame, S::AbstractWorldSet{Interval}, ::_IA_Ai) =
    IterTools.imap(Interval,
        Iterators.flatten(
            IterTools.imap((x)->zip(1:x-1, Iterators.repeated(x)),
                IterTools.distinct(map((w)->w.x, S))
            )
        )
    )

############################################################################################
# When defining `accessibles_aggr` for minimum & maximum features, we find that we can
#  categorized  interval relations according to their behavior.
# Consider the decision ⟨R⟩ (minimum(A1) ≥ 10) evaluated on a world w = (x,y):
#  - With R = RelationId, it requires computing minimum(A1) on w;
#  - With R = RelationGlob, it requires computing maximum(A1) on 1:(X(fr)+1) (the largest world);
#  - With R = Begins inverse, it requires computing minimum(A1) on (x,y+1), if such interval exists;
#  - With R = During, it requires computing maximum(A1) on (x+1,y-1), if such interval exists;
#  - With R = After, it requires reading the single value in (y,y+1) (or, alternatively, computing minimum(A1) on it), if such interval exists;
#
# Here is the categorization assuming feature = minimum and test_operator = ≥:
#
#                                    .----------------------.
#                                    |(  Id  minimum)       |
#                                    |IA_Bi  minimum        |
#                                    |IA_Ei  minimum        |
#                                    |IA_Di  minimum        |
#                                    |IA_O   minimum        |
#                                    |IA_Oi  minimum        |
#                                    |----------------------|
#                                    |(Glob  maximum)       |
#                                    |IA_L   maximum        |
#                                    |IA_Li  maximum        |
#                                    |IA_D   maximum        |
#                                    |----------------------|
#                                    |IA_A   single-value   |
#                                    |IA_Ai  single-value   |
#                                    |IA_B   single-value   |
#                                    |IA_E   single-value   |
#                                    '----------------------'
#
# When feature = maximum, the two categories minimum and maximum swap roles.
# Furthermore, if test_operator = ≤, or, more generally, existential_aggregator(test_operator)
#  is minimum instead of maximum, again, the two categories minimum and maximum swap roles.
############################################################################################

# e.g., minimum + ≥
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Bi) = (w.y < X(fr)+1)                 ?  Interval[Interval(w.x,   w.y+1)] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Ei) = (1 < w.x)                   ?  Interval[Interval(w.x-1, w.y  )] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Di) = (1 < w.x && w.y < X(fr)+1)      ?  Interval[Interval(w.x-1, w.y+1)] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_O) = (w.x+1 < w.y && w.y < X(fr)+1)  ?  Interval[Interval(w.y-1, w.y+1)] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Oi) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(w.x-1, w.x+1)] : Interval[]

# e.g., maximum + ≤
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Bi) = (w.y < X(fr)+1)                 ?  Interval[Interval(w.x,   w.y+1)] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Ei) = (1 < w.x)                   ?  Interval[Interval(w.x-1, w.y  )] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Di) = (1 < w.x && w.y < X(fr)+1)      ?  Interval[Interval(w.x-1, w.y+1)] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_O) = (w.x+1 < w.y && w.y < X(fr)+1)  ?  Interval[Interval(w.y-1, w.y+1)] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Oi) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(w.x-1, w.x+1)] : Interval[]

# e.g., minimum + ≥
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Bi) = (w.y < X(fr)+1)                 ?  Interval[Interval(w.x,   X(fr)+1)]   : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Ei) = (1 < w.x)                   ?  Interval[Interval(1,     w.y  )] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Di) = (1 < w.x && w.y < X(fr)+1)      ?  Interval[Interval(1,     X(fr)+1  )] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_O) = (w.x+1 < w.y && w.y < X(fr)+1)  ?  Interval[Interval(w.x+1, X(fr)+1  )] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Oi) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(1,     w.y-1)] : Interval[]

# e.g., maximum + ≤
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Bi) = (w.y < X(fr)+1)                 ?  Interval[Interval(w.x,   X(fr)+1)]   : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Ei) = (1 < w.x)                   ?  Interval[Interval(1,     w.y  )] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Di) = (1 < w.x && w.y < X(fr)+1)      ?  Interval[Interval(1,     X(fr)+1  )] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_O) = (w.x+1 < w.y && w.y < X(fr)+1)  ?  Interval[Interval(w.x+1, X(fr)+1  )] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Oi) = (1 < w.x && w.x+1 < w.y)    ?  Interval[Interval(1,     w.y-1)] : Interval[]

############################################################################################

# e.g., minimum + ≥
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_L) = (w.y+1 < X(fr)+1)   ? short_intervals_in(w.y+1, X(fr)+1)   : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_Li) = (1 < w.x-1)     ? short_intervals_in(1, w.x-1)     : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(maximum), w::Interval, r::_IA_D) = (w.x+1 < w.y-1) ? short_intervals_in(w.x+1, w.y-1) : Interval[]

# e.g., maximum + ≤
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_L) = (w.y+1 < X(fr)+1)   ? short_intervals_in(w.y+1, X(fr)+1)   : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_Li) = (1 < w.x-1)     ? short_intervals_in(1, w.x-1)     : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(minimum), w::Interval, r::_IA_D) = (w.x+1 < w.y-1) ? short_intervals_in(w.x+1, w.y-1) : Interval[]

# e.g., minimum + ≥
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_L) = (w.y+1 < X(fr)+1)   ? Interval[Interval(w.y+1, X(fr)+1)  ] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_Li) = (1 < w.x-1)     ? Interval[Interval(1, w.x-1)    ] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(minimum), w::Interval, r::_IA_D) = (w.x+1 < w.y-1) ? Interval[Interval(w.x+1, w.y-1)] : Interval[]

# e.g., maximum + ≤
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_L) = (w.y+1 < X(fr)+1)   ? Interval[Interval(w.y+1, X(fr)+1)  ] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_Li) = (1 < w.x-1)     ? Interval[Interval(1, w.x-1)    ] : Interval[]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(maximum), w::Interval, r::_IA_D) = (w.x+1 < w.y-1) ? Interval[Interval(w.x+1, w.y-1)] : Interval[]

############################################################################################

# e.g., minimum + ≥
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(maximum), w::Interval, ::_IA_A) = (w.y < X(fr)+1)     ?   Interval[Interval(w.y,   w.y+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.y, X(fr)+1)]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(maximum), w::Interval, ::_IA_Ai) = (1 < w.x)       ?   Interval[Interval(w.x-1, w.x  )] : Interval[] #  _ReprVal(Interval   )# [Interval(1, w.x)]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(maximum), w::Interval, ::_IA_B) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.x+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x, w.y-1)]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(maximum), w::Interval, ::_IA_E) = (w.x+1 < w.y)   ?   Interval[Interval(w.y-1, w.y  )] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x+1, w.y)]

# e.g., maximum + ≤
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(minimum), w::Interval, ::_IA_A) = (w.y < X(fr)+1)     ?   Interval[Interval(w.y,   w.y+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.y, X(fr)+1)]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(minimum), w::Interval, ::_IA_Ai) = (1 < w.x)       ?   Interval[Interval(w.x-1, w.x  )] : Interval[] #  _ReprVal(Interval   )# [Interval(1, w.x)]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(minimum), w::Interval, ::_IA_B) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.x+1)] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x, w.y-1)]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(minimum), w::Interval, ::_IA_E) = (w.x+1 < w.y)   ?   Interval[Interval(w.y-1, w.y  )] : Interval[] #  _ReprVal(Interval   )# [Interval(w.x+1, w.y)]

# e.g., minimum + ≥
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(minimum), w::Interval, ::_IA_A) = (w.y < X(fr)+1)     ?   Interval[Interval(w.y,   X(fr)+1  )] : Interval[] #  _ReprVal(Interval(w.y, w.y+1)   )# [Interval(w.y, X(fr)+1)]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(minimum), w::Interval, ::_IA_Ai) = (1 < w.x)       ?   Interval[Interval(1,     w.x  )] : Interval[] #  _ReprVal(Interval(w.x-1, w.x)   )# [Interval(1, w.x)]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(minimum), w::Interval, ::_IA_B) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.y-1)] : Interval[] #  _ReprVal(Interval(w.x, w.x+1)   )# [Interval(w.x, w.y-1)]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMin, a::typeof(minimum), w::Interval, ::_IA_E) = (w.x+1 < w.y)   ?   Interval[Interval(w.x+1, w.y  )] : Interval[] #  _ReprVal(Interval(w.y-1, w.y)   )# [Interval(w.x+1, w.y)]

# e.g., maximum + ≤
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(maximum), w::Interval, ::_IA_A) = (w.y < X(fr)+1)     ?   Interval[Interval(w.y,   X(fr)+1  )] : Interval[] #  _ReprVal(Interval(w.y, w.y+1)   )# [Interval(w.y, X(fr)+1)]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(maximum), w::Interval, ::_IA_Ai) = (1 < w.x)       ?   Interval[Interval(1,     w.x  )] : Interval[] #  _ReprVal(Interval(w.x-1, w.x)   )# [Interval(1, w.x)]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(maximum), w::Interval, ::_IA_B) = (w.x < w.y-1)   ?   Interval[Interval(w.x,   w.y-1)] : Interval[] #  _ReprVal(Interval(w.x, w.x+1)   )# [Interval(w.x, w.y-1)]
accessibles_aggr(fr::Full1DFrame, f::SingleAttributeMax, a::typeof(maximum), w::Interval, ::_IA_E) = (w.x+1 < w.y)   ?   Interval[Interval(w.x+1, w.y  )] : Interval[] #  _ReprVal(Interval(w.y-1, w.y)   )# [Interval(w.x+1, w.y)]

############################################################################################
# Similarly, here is the categorization for IA7 & IA3 assuming feature = minimum and test_operator = ≥:
#
#                               .-----------------------------.
#                               |(  Id         minimum)       |
#                               |IA_AorO       minimum        |
#                               |IA_AiorOi     minimum        |
#                               |IA_DiorBiorEi minimum        |
#                               |-----------------------------|
#                               |IA_DorBorE    maximum        |
#                               |-----------------------------|
#                               |IA_I          ?              |
#                               '-----------------------------'
# TODO write the correct `accessibles_aggr` methods, instead of these fallbacks:
accessibles_aggr(fr::Full1DFrame, f::AbstractFeature, a::Aggregator, w::Interval, r::_IA_AorO) =
    Iterators.flatten([accessibles_aggr(fr, f, a, w, r) for r in IA72IARelations(IA_AorO)])
accessibles_aggr(fr::Full1DFrame, f::AbstractFeature, a::Aggregator, w::Interval, r::_IA_AiorOi) =
    Iterators.flatten([accessibles_aggr(fr, f, a, w, r) for r in IA72IARelations(IA_AiorOi)])
accessibles_aggr(fr::Full1DFrame, f::AbstractFeature, a::Aggregator, w::Interval, r::_IA_DorBorE) =
    Iterators.flatten([accessibles_aggr(fr, f, a, w, r) for r in IA72IARelations(IA_DorBorE)])
accessibles_aggr(fr::Full1DFrame, f::AbstractFeature, a::Aggregator, w::Interval, r::_IA_DiorBiorEi) =
    Iterators.flatten([accessibles_aggr(fr, f, a, w, r) for r in IA72IARelations(IA_DiorBiorEi)])
accessibles_aggr(fr::Full1DFrame, f::AbstractFeature, a::Aggregator, w::Interval, r::_IA_I) =
    Iterators.flatten([accessibles_aggr(fr, f, a, w, r) for r in IA72IARelations(IA_I)])
