using SoleModels: AbstractRelation, AbstractFeature, TestOperatorFun
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

display_existential(rel::AbstractRelation) = "⟨$(rel)⟩"
display_universal(rel::AbstractRelation)   = "[$(rel)]"

############################################################################################

# ⊤
struct TopDecision <: AbstractDecision end
display_decision(::TopDecision) = "⊤"

# ⊥
struct BotDecision <: AbstractDecision end
display_decision(::BotDecision) = "⊥"

# ⟨R⟩⊤
struct ExistentialTopDecision{R<:AbstractRelation} <: AbstractDecision end
display_decision(::ExistentialTopDecision{R}) where {R<:AbstractRelation} = "$(display_existential(R))⊤"

# [R]⊥
struct UniversalBotDecision{R<:AbstractRelation} <: AbstractDecision end
display_decision(::UniversalBotDecision{R}) where {R<:AbstractRelation} = "$(display_universal(R))⊥"

############################################################################################

# Decisions based on dimensional conditions
abstract type DimensionalDecision{T} <: AbstractDecision end

# p
struct DimensionalDecision{T} <: DimensionalDecision{T}
    p :: Condition{<:AbstractMetaCondition,T}
end

feature(d::DimensionalDecision) = feature(d)
test_operator(d::DimensionalDecision) = test_operator(d)
threshold(d::DimensionalDecision) = threshold(d)

# ⟨R⟩p
struct ExistentialDimensionalDecision{T} <: DimensionalDecision{T}

    # Relation, interpreted as an existential modal operator
    relation  :: AbstractRelation
    
    p         :: Condition{M,T} where {M<:AbstractMetaCondition}

    function ExistentialDimensionalDecision{T}() where {T}
        new{T}()
    end

    function ExistentialDimensionalDecision{T}(
        relation      :: AbstractRelation,
        p             :: Condition{M,T}
    ) where {M<:AbstractMetaCondition,T}
        new{T}(relation, p)
    end

    function ExistentialDimensionalDecision(
        relation      :: AbstractRelation,
        p             :: Condition{M,T}
    ) where {M<:AbstractMetaCondition,T}
        ExistentialDimensionalDecision{T}(relation, p)
    end

    function ExistentialDimensionalDecision{T}(
        relation      :: AbstractRelation,
        feature       :: AbstractFeature,
        test_operator :: TestOperatorFun,
        threshold     :: T
    ) where {T}
        p = Condition(feature, test_operator, threshold)
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
        q = Condition(decision.p, threshold_f(threshold(decision.p)))
        ExistentialDimensionalDecision{T}(relation(decision), q)
    end
end

relation(d::ExistentialDimensionalDecision) = d.relation
feature(d::ExistentialDimensionalDecision) = feature(d.p)
test_operator(d::ExistentialDimensionalDecision) = test_operator(d.p)
threshold(d::ExistentialDimensionalDecision) = threshold(d.p)

is_propositional_decision(d::ExistentialDimensionalDecision) = (relation(d) isa ModalLogic._RelationId)
is_global_decision(d::ExistentialDimensionalDecision) = (relation(d) isa ModalLogic._RelationGlob)

function display_decision(
        decision::ExistentialDimensionalDecision;
        threshold_display_method::Function = x -> x,
        universal = false,
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
        "$((universal ? display_universal : display_existential)(relation(decision))) ($prop_decision_str)"
    else
        "$prop_decision_str"
    end
end


function display_decision(
        i_frame::Integer,
        decision::ExistentialDimensionalDecision;
        attribute_names_map::Union{Nothing,AbstractVector{<:AbstractVector},AbstractVector{<:AbstractDict}} = nothing,
        kwargs...)
    _attribute_names_map = isnothing(attribute_names_map) ? nothing : attribute_names_map[i_frame]
    "{$i_frame} $(display_decision(decision; attribute_names_map = _attribute_names_map, kwargs...))"
end

function display_decision_inverse(i_frame::Integer, decision::ExistentialDimensionalDecision{T}; args...) where {T}
    inv_decision = ExistentialDimensionalDecision{T}(relation(decision), feature(decision), test_operator_inverse(test_operator(decision)), threshold(decision))
    display_decision(i_frame, inv_decision; universal = true, args...)
end

############################################################################################
