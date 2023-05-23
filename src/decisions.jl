using SoleLogics: identityrel, globalrel

using SoleLogics: AbstractRelation
using SoleModels: AbstractFeature, TestOperator, FeatCondition
using SoleModels: syntaxstring
using SoleModels.DimensionalDatasets: alpha

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

import SoleModels: negation

function Base.show(io::IO, decision::AbstractDecision)
    println(io, display_decision(decision))
end
function display_decision(
    frameid::FrameId,
    decision::AbstractDecision;
    attribute_names_map::Union{Nothing,AbstractVector{<:AbstractVector},AbstractVector{<:AbstractDict}} = nothing,
    kwargs...,
)
    _attribute_names_map = isnothing(attribute_names_map) ? nothing : attribute_names_map[frameid]
    "{$frameid} $(display_decision(decision; attribute_names_map = _attribute_names_map, kwargs...))"
end

############################################################################################

abstract type SimpleDecision <: AbstractDecision end

function display_decision_inverse(decision::SimpleDecision, kwargs...; args...)
    display_decision(negation(decision), kwargs...; args...)
end

function display_decision_inverse(frameid::FrameId, decision::SimpleDecision, kwargs...; args...)
    display_decision(frameid, negation(decision), kwargs...; args...)
end

display_existential(rel::AbstractRelation; kwargs...) = SoleLogics.syntaxstring(DiamondRelationalOperator{typeof(rel)}(); kwargs...)
display_universal(rel::AbstractRelation; kwargs...)   = SoleLogics.syntaxstring(BoxRelationalOperator{typeof(rel)}(); kwargs...)

############################################################################################

# ⊤
struct TopDecision <: SimpleDecision end
display_decision(::TopDecision) = "⊤"
negation(::TopDecision) = BotDecision()

# ⊥
struct BotDecision <: SimpleDecision end
display_decision(::BotDecision) = "⊥"
negation(::BotDecision) = TopDecision()

# ⟨R⟩⊤
struct ExistentialTopDecision{R<:AbstractRelation} <: SimpleDecision end
display_decision(::ExistentialTopDecision{R}) where {R<:AbstractRelation} = "$(display_existential(R))⊤"
negation(::ExistentialTopDecision{R}) where {R<:AbstractRelation} = UniversalBotDecision{R}()

# [R]⊥
struct UniversalBotDecision{R<:AbstractRelation} <: SimpleDecision end
display_decision(::UniversalBotDecision{R}) where {R<:AbstractRelation} = "$(display_universal(R))⊥"
negation(::UniversalBotDecision{R}) where {R<:AbstractRelation} = ExistentialTopDecision{R}()

############################################################################################
############################################################################################
############################################################################################

# Decisions based on dimensional conditions
abstract type DimensionalDecision{U} <: SimpleDecision end

# p
struct PropositionalDimensionalDecision{U} <: DimensionalDecision{U}
    p :: FeatCondition{U}
end

proposition(d::PropositionalDimensionalDecision) = d.p
feature(d::PropositionalDimensionalDecision) = feature(proposition(d))
test_operator(d::PropositionalDimensionalDecision) = test_operator(proposition(d))
threshold(d::PropositionalDimensionalDecision) = threshold(proposition(d))

negation(p::PropositionalDimensionalDecision{U}) where {U} =
    PropositionalDimensionalDecision{U}(negation(p))

############################################################################################

abstract type ModalDimensionalDecision{U} <: DimensionalDecision{U} end

relation(d::ModalDimensionalDecision) = d.relation
proposition(d::ModalDimensionalDecision) = d.p
feature(d::ModalDimensionalDecision) = feature(proposition(d))
test_operator(d::ModalDimensionalDecision) = test_operator(proposition(d))
threshold(d::ModalDimensionalDecision) = threshold(proposition(d))

is_propositional_decision(d::ModalDimensionalDecision) = (relation(d) == identityrel)
is_global_decision(d::ModalDimensionalDecision) = (relation(d) == globalrel)

# ⟨R⟩p
struct ExistentialDimensionalDecision{U} <: ModalDimensionalDecision{U}

    # Relation, interpreted as an existential modal operator
    relation  :: AbstractRelation

    p         :: FeatCondition{U}

    function ExistentialDimensionalDecision{U}() where {U}
        new{U}()
    end

    function ExistentialDimensionalDecision{U}(
        relation      :: AbstractRelation,
        p             :: FeatCondition{U}
    ) where {U}
        new{U}(relation, p)
    end

    function ExistentialDimensionalDecision(
        relation      :: AbstractRelation,
        p             :: FeatCondition{U}
    ) where {U}
        ExistentialDimensionalDecision{U}(relation, p)
    end

    function ExistentialDimensionalDecision{U}(
        relation      :: AbstractRelation,
        feature       :: AbstractFeature,
        test_operator :: TestOperator,
        threshold     :: U
    ) where {U}
        p = FeatCondition(feature, test_operator, threshold)
        ExistentialDimensionalDecision{U}(relation, p)
    end

    function ExistentialDimensionalDecision(
        relation      :: AbstractRelation,
        feature       :: AbstractFeature,
        test_operator :: TestOperator,
        threshold     :: U
    ) where {U}
        ExistentialDimensionalDecision{U}(relation, feature, test_operator, threshold)
    end

    function ExistentialDimensionalDecision(
        decision      :: ExistentialDimensionalDecision{U},
        threshold_f   :: Function
    ) where {U}
        q = FeatCondition(decision.p, threshold_f(threshold(decision.p)))
        ExistentialDimensionalDecision{U}(relation(decision), q)
    end
end

# [R]p
struct UniversalDimensionalDecision{U} <: ModalDimensionalDecision{U}
    relation  :: AbstractRelation
    p         :: FeatCondition{U}
end

function negation(decision::ExistentialDimensionalDecision{U}) where {U}
    UniversalDimensionalDecision{U}(
        relation(decision),
        negation(proposition(decision))
    )
end
function negation(decision::UniversalDimensionalDecision{U}) where {U}
    ExistentialDimensionalDecision{U}(
        relation(decision),
        negation(proposition(decision))
    )
end

function display_decision(
    decision::Union{ExistentialDimensionalDecision,UniversalDimensionalDecision};
    threshold_display_method::Function = x -> x,
    attribute_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing,
    use_feature_abbreviations::Bool = false,
)
    prop_decision_str = syntaxstring(
        decision.p;
        threshold_display_method = threshold_display_method,
        attribute_names_map = attribute_names_map,
        use_feature_abbreviations = use_feature_abbreviations,
    )
    if !is_propositional_decision(decision)
        rel_display_fun = (decision isa ExistentialDimensionalDecision ? display_existential : display_universal)
        "$(rel_display_fun(relation(decision))) ($prop_decision_str)"
    else
        "$prop_decision_str"
    end
end


############################################################################################
