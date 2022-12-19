
############################################################################################
# Decision
############################################################################################

abstract type AbstractDecision{T} end

export Decision,
       #
       relation, feature, test_operator, threshold,
       is_propositional_decision,
       is_global_decision,
       #
       display_decision, display_decision_inverse

# A decision inducing a branching/split (e.g., ⟨L⟩ (minimum(A2) ≥ 10) )
struct Decision{T} <: AbstractDecision{T}

    # Relation, interpreted as an existential modal operator
    #  Note: RelationId for propositional decisions
    relation      :: AbstractRelation

    # Modal feature (a scalar function that can be computed on a world)
    feature       :: ModalFeature

    # Test operator (e.g. ≥)
    test_operator :: TestOperatorFun

    # Threshold value
    threshold     :: T

    function Decision{T}() where {T}
        new{T}()
    end

    function Decision{T}(
        relation      :: AbstractRelation,
        feature       :: ModalFeature,
        test_operator :: TestOperatorFun,
        threshold     :: T
    ) where {T}
        new{T}(relation, feature, test_operator, threshold)
    end

    function Decision(
        relation      :: AbstractRelation,
        feature       :: ModalFeature,
        test_operator :: TestOperatorFun,
        threshold     :: T
    ) where {T}
        Decision{T}(relation, feature, test_operator, threshold)
    end

    function Decision(
        decision      :: Decision{T},
        threshold_f   :: Function
    ) where {T}
        Decision{T}(decision.relation, decision.feature, decision.test_operator, threshold_f(decision.threshold))
    end
end

is_propositional_decision(d::Decision) = (d.relation isa ModalLogic._RelationId)
is_global_decision(d::Decision) = (d.relation isa ModalLogic._RelationGlob)

function Base.show(io::IO, decision::Decision)
    println(io, display_decision(decision))
end

function display_decision(
        decision::Decision;
        threshold_display_method::Function = x -> x,
        universal = false,
        attribute_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing,
        use_feature_abbreviations::Bool = false,
    )
    prop_decision_str = "$(
        display_feature_test_operator_pair(
            decision.feature,
            decision.test_operator;
            attribute_names_map = attribute_names_map,
            use_feature_abbreviations = use_feature_abbreviations,
        )
    ) $(threshold_display_method(decision.threshold))"
    if !is_propositional_decision(decision)
        "$((universal ? display_universal : display_existential)(decision.relation)) ($prop_decision_str)"
    else
        "$prop_decision_str"
    end
end

display_existential(rel::AbstractRelation) = "⟨$(rel)⟩"
display_universal(rel::AbstractRelation)   = "[$(rel)]"

############################################################################################

function display_decision(
        i_frame::Integer,
        decision::Decision;
        attribute_names_map::Union{Nothing,AbstractVector{<:AbstractVector},AbstractVector{<:AbstractDict}} = nothing,
        kwargs...)
    _attribute_names_map = isnothing(attribute_names_map) ? nothing : attribute_names_map[i_frame]
    "{$i_frame} $(display_decision(decision; attribute_names_map = _attribute_names_map, kwargs...))"
end

function display_decision_inverse(i_frame::Integer, decision::Decision{T}; args...) where {T}
    inv_decision = Decision{T}(decision.relation, decision.feature, test_operator_inverse(decision.test_operator), decision.threshold)
    display_decision(i_frame, inv_decision; universal = true, args...)
end
