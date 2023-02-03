using SoleLogics: _RelationId, _RelationGlob

using SoleModels: AbstractRelation, AbstractFeature, TestOperatorFun, FeatCondition
using SoleModels: alpha, display_feature, display_feature_test_operator_pair

export ExistentialDimensionalDecision,
       #
       relation, feature, test_operator, threshold,
       is_propositional_decision,
       is_global_decision,
       #
       display_decision, display_decision_inverse

############################################################################################
# Decision
############################################################################################

# A decision inducing a branching/split (e.g., ⟨L⟩ (minimum(A2) ≥ 10) )
abstract type AbstractDecision end

function Base.show(io::IO, decision::AbstractDecision)
    println(io, display_decision(decision))
end
function display_decision(
    i_frame::Integer,
    decision::AbstractDecision;
    attribute_names_map::Union{Nothing,AbstractVector{<:AbstractVector},AbstractVector{<:AbstractDict}} = nothing,
    kwargs...,
)
    _attribute_names_map = isnothing(attribute_names_map) ? nothing : attribute_names_map[i_frame]
    "{$i_frame} $(display_decision(decision; attribute_names_map = _attribute_names_map, kwargs...))"
end

############################################################################################

abstract type SimpleDecision <: AbstractDecision end

function display_decision_inverse(decision::SimpleDecision, kwargs...; args...)
    display_decision(inverse(decision), kwargs...; args...)
end

function display_decision_inverse(i_frame::Integer, decision::SimpleDecision, kwargs...; args...)
    display_decision(i_frame, inverse(decision), kwargs...; args...)
end

display_existential(rel::AbstractRelation) = SoleLogics.display_operator(DiamondRelationalOperator{typeof(rel)}())
display_universal(rel::AbstractRelation)   = SoleLogics.display_operator(BoxRelationalOperator{typeof(rel)}())

############################################################################################

# ⊤
struct TopDecision <: SimpleDecision end
display_decision(::TopDecision) = "⊤"
inverse(::TopDecision) = BotDecision()

# ⊥
struct BotDecision <: SimpleDecision end
display_decision(::BotDecision) = "⊥"
inverse(::BotDecision) = TopDecision()

# ⟨R⟩⊤
struct ExistentialTopDecision{R<:AbstractRelation} <: SimpleDecision end
display_decision(::ExistentialTopDecision{R}) where {R<:AbstractRelation} = "$(display_existential(R))⊤"
inverse(::ExistentialTopDecision{R}) where {R<:AbstractRelation} = UniversalBotDecision{R}()

# [R]⊥
struct UniversalBotDecision{R<:AbstractRelation} <: SimpleDecision end
display_decision(::UniversalBotDecision{R}) where {R<:AbstractRelation} = "$(display_universal(R))⊥"
inverse(::UniversalBotDecision{R}) where {R<:AbstractRelation} = ExistentialTopDecision{R}()

############################################################################################
############################################################################################
############################################################################################

# Decisions based on dimensional conditions
abstract type DimensionalDecision{T} <: SimpleDecision end

# p
struct PropositionalDimensionalDecision{T} <: DimensionalDecision{T}
    p :: FeatCondition{T}
end

proposition(d::PropositionalDimensionalDecision) = d.p
feature(d::PropositionalDimensionalDecision) = feature(proposition(d))
test_operator(d::PropositionalDimensionalDecision) = test_operator(proposition(d))
threshold(d::PropositionalDimensionalDecision) = threshold(proposition(d))

inverse(p::PropositionalDimensionalDecision{T}) where {T} = 
    PropositionalDimensionalDecision{T}(inverse(p))

############################################################################################

abstract type ModalDimensionalDecision{T} <: DimensionalDecision{T} end

relation(d::ModalDimensionalDecision) = d.relation
proposition(d::ModalDimensionalDecision) = d.p
feature(d::ModalDimensionalDecision) = feature(proposition(d))
test_operator(d::ModalDimensionalDecision) = test_operator(proposition(d))
threshold(d::ModalDimensionalDecision) = threshold(proposition(d))

is_propositional_decision(d::ModalDimensionalDecision) = (relation(d) isa _RelationId)
is_global_decision(d::ModalDimensionalDecision) = (relation(d) isa _RelationGlob)

# ⟨R⟩p
struct ExistentialDimensionalDecision{T} <: ModalDimensionalDecision{T}

    # Relation, interpreted as an existential modal operator
    relation  :: AbstractRelation
    
    p         :: FeatCondition{T}

    function ExistentialDimensionalDecision{T}() where {T}
        new{T}()
    end

    function ExistentialDimensionalDecision{T}(
        relation      :: AbstractRelation,
        p             :: FeatCondition{T}
    ) where {T}
        new{T}(relation, p)
    end

    function ExistentialDimensionalDecision(
        relation      :: AbstractRelation,
        p             :: FeatCondition{T}
    ) where {T}
        ExistentialDimensionalDecision{T}(relation, p)
    end

    function ExistentialDimensionalDecision{T}(
        relation      :: AbstractRelation,
        feature       :: AbstractFeature,
        test_operator :: TestOperatorFun,
        threshold     :: T
    ) where {T}
        p = FeatCondition(feature, test_operator, threshold)
        ExistentialDimensionalDecision{T}(relation, p)
    end

    function ExistentialDimensionalDecision(
        relation      :: AbstractRelation,
        feature       :: AbstractFeature,
        test_operator :: TestOperatorFun,
        threshold     :: T
    ) where {T}
        ExistentialDimensionalDecision{T}(relation, feature, test_operator, threshold)
    end

    function ExistentialDimensionalDecision(
        decision      :: ExistentialDimensionalDecision{T},
        threshold_f   :: Function
    ) where {T}
        q = FeatCondition(decision.p, threshold_f(threshold(decision.p)))
        ExistentialDimensionalDecision{T}(relation(decision), q)
    end
end

# [R]p
struct UniversalDimensionalDecision{T} <: ModalDimensionalDecision{T}
    relation  :: AbstractRelation
    p         :: FeatCondition{T}
end

function inverse(decision::ExistentialDimensionalDecision{T}) where {T}
    UniversalDimensionalDecision{T}(
        relation(decision),
        inverse(proposition(decision))
    )
end
function inverse(decision::UniversalDimensionalDecision{T}) where {T}
    ExistentialDimensionalDecision{T}(
        relation(decision),
        inverse(proposition(decision))
    )
end

function display_decision(
        decision::Union{ExistentialDimensionalDecision,UniversalDimensionalDecision};
        threshold_display_method::Function = x -> x,
        attribute_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing,
        use_feature_abbreviations::Bool = false,
    )
    prop_decision_str = "$(
        display_feature_test_operator_pair(
            feature(decision),
            test_operator(decision);
            attribute_names_map = attribute_names_map,
            use_feature_abbreviations = use_feature_abbreviations,
        )
    ) $(threshold_display_method(threshold(decision)))"
    if !is_propositional_decision(decision)
        rel_display_fun = (decision isa ExistentialDimensionalDecision ? display_existential : display_universal)
        "$(rel_display_fun(relation(decision))) ($prop_decision_str)"
    else
        "$prop_decision_str"
    end
end


############################################################################################
