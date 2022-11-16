############################################################################################
# Allen's Interval Algebra relations
############################################################################################

# Interval directional relations
abstract type IntervalRelation <: GeometricalRelation end

############################################################################################
# Interval algebra comprehends 12 relations (plus equality, i.e. RelationId):
#  - the 6 relations After, Later, Begins, Ends, During, Overlaps
#  - their inverses
############################################################################################
# Graphical representation of R((x,y),(z,t)) for R ∈ {After, Later, Begins, Ends, During, Overlaps}:
# 
#                       x                   y                                               
#                       |-------------------|                                               
#                       .                   .                                               
#                       .                   z        t            y = z                     
# After       (A)       .                   |--------|                                      
#                       .                   .                                               
#                       .                   .   z         t       y < z                     
# Later       (L)       .                   .   |---------|                                 
#                       .                   .                                               
#                       z     t             .                     x = z, t < y              
# Begins      (B)       |-----|             .                                               
#                       .                   .                                               
#                       .             z     t                     y = t, x < z              
# Ends        (E)       .             |-----|                                               
#                       .                   .                                               
#                       .   z        t      .                     x < z, t < y              
# During      (D)       .   |--------|      .                                               
#                       .                   .                                               
#                       .           z       .    t                x < z < y < t             
# Overlaps    (O)       .           |------------|                                          
# 
############################################################################################


struct _IA_A  <: IntervalRelation end; const IA_A  = _IA_A();  # After
struct _IA_L  <: IntervalRelation end; const IA_L  = _IA_L();  # Later
struct _IA_B  <: IntervalRelation end; const IA_B  = _IA_B();  # Begins
struct _IA_E  <: IntervalRelation end; const IA_E  = _IA_E();  # Ends
struct _IA_D  <: IntervalRelation end; const IA_D  = _IA_D();  # During
struct _IA_O  <: IntervalRelation end; const IA_O  = _IA_O();  # Overlaps

struct _IA_Ai <: IntervalRelation end; const IA_Ai = _IA_Ai(); # After inverse
struct _IA_Li <: IntervalRelation end; const IA_Li = _IA_Li(); # Later inverse
struct _IA_Bi <: IntervalRelation end; const IA_Bi = _IA_Bi(); # Begins inverse
struct _IA_Ei <: IntervalRelation end; const IA_Ei = _IA_Ei(); # Ends inverse
struct _IA_Di <: IntervalRelation end; const IA_Di = _IA_Di(); # During inverse
struct _IA_Oi <: IntervalRelation end; const IA_Oi = _IA_Oi(); # Overlaps inverse

Base.show(io::IO, ::_IA_A)  = print(io, "A")
Base.show(io::IO, ::_IA_L)  = print(io, "L")
Base.show(io::IO, ::_IA_B)  = print(io, "B")
Base.show(io::IO, ::_IA_E)  = print(io, "E")
Base.show(io::IO, ::_IA_D)  = print(io, "D")
Base.show(io::IO, ::_IA_O)  = print(io, "O")
Base.show(io::IO, ::_IA_Ai) = print(io, "A̅")
Base.show(io::IO, ::_IA_Li) = print(io, "L̅")
Base.show(io::IO, ::_IA_Bi) = print(io, "B̅")
Base.show(io::IO, ::_IA_Ei) = print(io, "E̅")
Base.show(io::IO, ::_IA_Di) = print(io, "D̅")
Base.show(io::IO, ::_IA_Oi) = print(io, "O̅")

# Properties
is_transitive(r::_IA_L) = true
is_transitive(r::_IA_Li) = true
is_transitive(r::_IA_D) = true
is_transitive(r::_IA_Di) = true
is_transitive(r::_IA_B) = true
is_transitive(r::_IA_Bi) = true
is_transitive(r::_IA_E) = true
is_transitive(r::_IA_Ei) = true
is_topological(r::_IA_D) = true
is_topological(r::_IA_Di) = true

############################################################################################

# Coarser relations: IA7
abstract type _IA7Rel <: IntervalRelation  end
struct _IA_AorO       <: _IA7Rel end; const IA_AorO       = _IA_AorO();       # After ∪ Overlaps
struct _IA_DorBorE    <: _IA7Rel end; const IA_DorBorE    = _IA_DorBorE();    # During ∪ Begins ∪ Ends
struct _IA_AiorOi     <: _IA7Rel end; const IA_AiorOi     = _IA_AiorOi();     # (After ∪ Overlaps) inverse
struct _IA_DiorBiorEi <: _IA7Rel end; const IA_DiorBiorEi = _IA_DiorBiorEi(); # (During ∪ Begins ∪ Ends) inverse

# Even coarser relations: IA3
abstract type _IA3Rel <: IntervalRelation  end
struct _IA_I          <: _IA3Rel end; const IA_I          = _IA_I();   # Intersecting (ABEDO ∪ ABEDO inverse)

# Properties
is_transitive(r::_IA_DorBorE) = true
is_transitive(r::_IA_DiorBiorEi) = true
is_topological(r::_IA_I) = true

IA72IARelations(::_IA_AorO)       = [IA_A,  IA_O]
IA72IARelations(::_IA_AiorOi)     = [IA_Ai, IA_Oi]
IA72IARelations(::_IA_DorBorE)    = [IA_D,  IA_B,  IA_E]
IA72IARelations(::_IA_DiorBiorEi) = [IA_Di, IA_Bi, IA_Ei]
IA72IARelations(::_IA_I)          = [
    IA_A,  IA_O,  IA_D,  IA_B,  IA_E,
    IA_Ai, IA_Oi, IA_Di, IA_Bi, IA_Ei
]

Base.show(io::IO, r::_IA7Rel)       = print(io, join(IA72IARelations(r), "∨"))
Base.show(io::IO, ::_IA_I)          = print(io, "I")

############################################################################################

# 12 IA relations
const IARelations = [IA_A,  IA_L,  IA_B,  IA_E,  IA_D,  IA_O,
                     IA_Ai, IA_Li, IA_Bi, IA_Ei, IA_Di, IA_Oi]

# 7 IA7 relations
const IA7Relations = [IA_AorO,   IA_L,  IA_DorBorE,
                      IA_AiorOi, IA_Li, IA_DiorBiorEi]

# 3 IA3 relations
const IA3Relations = [IA_I, IA_L, IA_Li]

# 13 Interval Algebra extended with universal
const IARelations_extended = [RelationGlob, IARelations...]

