############################################################################################
using SoleLogics

const Consequent = Any

struct Rule{L<:Logic,C<:Consequent}
    antecedent :: Formula{L}
    consequent :: C
end

antecedent(rule::Rule) = rule.antecedent
consequent(rule::Rule) = rule.consequent

const ClassificationRule = Rule{L,CLabel} where {L}
const RegressionRule     = Rule{L,RLabel} where {L}

struct RuleBasedModel{L<:Logic,C<:Consequent}
    rules :: Vector{<:Rule{L,C}}
end

const RuleBasedClassifier = RuleBasedModel{L,CLabel} where {L}
const RuleBasedRegressor  = RuleBasedModel{L,RLabel} where {L}

"""
TODO document
"""
function list_rules(tree::DTree)
    list_rules(tree.root) # TODO what about world_types and init_conditions?
end

function list_rules(node::DTInternal)
    pos_lambda = get_lambda(node.decision)
    neg_lambda = get_inverse_lambda(node.decision)
    return [
        [advance_rule(rule, pos_lambda) for rule in list_rules(node.left)]...,
        [advance_rule(rule, neg_lambda) for rule in list_rules(node.right)]...,
    ]
end

function list_rules(leaf::DTLeaf{L})
    Rule{TODO, L}(TODO: SoleLogics.True, prediction(leaf))
end

function list_rules(leaf::NSDTLeaf{L})
    Rule{TODO, PredictingFunction{L}}(TODO: SoleLogics.True, predicting_function(leaf))
end


function get_lambda(decision::Decision)
    Formula( # TODO formula of any logic, or formula of a specific logic?
        if is_propositional_decision(decision)
            return PropositionalDimensionalLetter( # TODO
                decision.feature,
                decision.test_operator,
                decision.threshold
            )
        else
            ModalExistentialOperator{decision.relation}(
                PropositionalDimensionalLetter( # TODO
                    decision.feature,
                    decision.test_operator,
                    decision.threshold
                )
            )
        end
    )
end
function get_inverse_lambda(decision::Decision)
    Formula( # TODO formula of any logic, or formula of a specific logic?
        if is_propositional_decision(decision)
            return Operator{negation...}(
                PropositionalDimensionalLetter( # TODO
                    decision.feature,
                    decision.test_operator,
                    decision.threshold
                )
            )
        else
            ModalUniversalOperator{decision.relation}(
                Operator{negation...}(
                    PropositionalDimensionalLetter( # TODO
                        decision.feature,
                        decision.test_operator,
                        decision.threshold
                    )
                )
            )
        end
    )
end

function advance_rule(rule::Rule{L,C}, lambda::Formula{L}) where {L,C}
    Rule{L,C}(advance_rule(antecedent(rule), lambda), consequent(rule))
end

function advance_rule(rule_antecedent::Formula{L}, lambda::Formula{L}) where {L}
    conjuncts = begin
        Formula{L}[TODO derive conjuncts from rule_antecedent...]
        una sorta di `conjuncts = split(formula, :(\wedge))`
    end
    Formula{L}(tipo join(advance_conjunct.(conjuncts, lambda), :(\wedge)))
end

function advance_conjunct(conjunct::Formula{L}, lambda::Formula{L}) where {L}
    # TODO
    is_positive(conjunct)
    is_left(lambda)
end

function is_positive(conjunct::Formula{L}) where {L}
    # TODO
end

function is_left(lambda::Formula{L}) where {L}
    # TODO
end

############################################################################################

# frequency_rule(  dataset     starting dataset
#                  rule        rule on which to determine the frequency
#               ) =
#     number of instances satisfying the rule condition divided by total number of instances

# error_rule(    CLabel   classification problem
#                dataset  starting dataset
#                rule     rule on which to determine the error
#           ) =
#     number of incorrectly classified instances (y' obtained from the evaluation of the
#     rule antecedent is different from y we had in the dataset) divided by number of
#     instances satisfying the rule condition

# error_rule(    RLabel   regression problem
#                dataset  starting dataset
#                rule     rule on which to determine the error
#           ) =
#     Mean Squared Error (MSE)
#     MSE = (1/k) * sum(t_i - t')^2   # k = number of instances satisfying rule condition
#                                     # t_i (i = 1,...,k) = target value of the i-th
#                                     #                   instance satisfying the antecedent
#                                     # t' = (1/k) * sum(t_i (i = 1,...,k))

# length_rule(    rule    rule on which to determine the complexity of the rule
#            ) =
#     number of variable-value pairs in the condition

############################################################################################

############################################################################################
# prune_ruleset(
#         Label        CLabel or RLabel
#         dataset      starting dataset
#         ruleset      forest rules
#         s = nothing->1.0e-6  parameter limiting the value of decay_i when x is 0
#                                      or very small  (s in the paper)
#         decay_threshold = nothing->0.05   threshold below which a variable-value pair is
#                                           dropped from the rule condition
#     )
#     for every rule in ruleset
#         E_zero = error_rule(Label,dataset,rule)   #error of original rule
#         for every variable-value pair in rule in inverse order
#             E_minus_i = error_rule(Label,dataset,rule\{i-th pair})   #rule's error without
#                                                                      # i-th pair
#             decay_i = (E_minus_i - E_zero) / max(E_zero,s)
#             if decay_i < decay_threshold
#                 i-th pair of the rule is eliminated from the antecedent
#                 recalculation of E_zero
#             end
#         end
#     end
# end
############################################################################################

############################################################################################
#
# # STEL - Simplified Tree Ensemble Learner
# simplified_tree_ensemble_learner(   CLabel       classification problem
#                                     best_rules   better rules from feature selection
#                                     min_support = 0.01  threshold below which a is
#                                                       dropped from S to avoid overfitting
#                                 ) =
#     R = {}  #ordered rule list
#     t* = average_label(Y)     #default rule
#     S = {t*,best_rules}
#     delete rules from S with frequency < min_frequency
#     D = copy(dataSet)    #insieme delle istanze di allenamento
#     while true
#         si calcola le metriche error, frequency e length per ogni regola sulla base delle istanze (rimaste) in D
#         per andare a selezionare la regola con l'errore minimo o, se ci sono legami, quella con la
#         frequenza più alta e la condizione più corta, per aggiungerla poi alla fine di R
#         Le istanze che soddisfano la miglior regola sono rimosse da D
#         t* = average_label(Y[ids delle istanze rimaste in D])
#         if good_rule == t*
#             return R
#         else if length(best_rules) == 0
#                 t* = classe più frequente su dataSet
#                 R = {R,t*}
#                 return R
#         end
#     end
# end
############################################################################################
using StatsBase
using Statistics
using SoleLogics

evaluate_rule(rule::Rule,X:MultiFrameModalDataset) = nothing #TODO

function length_rule(node::Node,operators_set::Operators)
    left = leftchild(node)
    right = rightchild(node)
    left_size = 0
    right_size = 0

    if isnothing(left) && isnothing(right)
        #leaf
        if token(node) ∉ operators_set
            return 1
        else
            return 0
        end
    end

    !isnothing(left) && (left_size = length_rule(left,operators_set))
    !isnothing(right) && (right_size = length_rule(right,operators_set))

    if token(node) ∉ operators_set
        return 1 + left_size + right_size
    else
        return left_size + right_size
    end

end

#metrics_rule -> confidence, error for classification problem and regression problem
#TODO: support
function metrics_rule(rule::Rule{L,C},X::MultiFrameModalDataset,Y::AbstractVector) where {L,C}
    metrics = Float64[]

    predictions = evaluate_rule(rule,X)
    n_instances = size(X,1)

    #confidence
    confidence = sum(predictions .== Y) / n_instances
    append!(metrics,confidence)

    #error
    if C <: CLabel
        error = sum(abs.(predictions .- Y)) / n_instances
    elseif C <: RLabel
        error = msd(predictions,Y)
    end
    append!(metrics,error)

    #length of the rule
    length = length_rule(rule.tree,operators(L))  #to check
    append!(metrics,length)

    return metrics
end

#prune_ruleset -> prune of rule in ruleset
function prune_ruleset(
        ruleset::RuleBasedModel{L,C},
        X::MultiFrameModalDataset,
        Y::AbstractVector;
        s = 1.0e-6,
        decay_threshold = 0.05
    )

    isnothing(s) && (s = 1.0e-6)
    isnothing(decay_threshold) && (decay_threshold = 0.05)

    for rule in ruleset
        E_zero::Float64 = metrics_rule(rule,X,Y)[2]  #error in second position
        for every variable-value pair in reverse(antecedent(rule))
            E_minus_i = metrics_rule(rule,X,Y)[2]
            decay_i = (E_minus_i-E_zero)/max(E_zero,s)
            if decay_i < decay_threshold
                #TODO: delete i-th pair in rule
                E_zero = metrics_rule(rule,X,Y)[2]
            end
        end
    end
end

default(C <: CLabel, Y::AbstractVector) = mean(Y) #TODO: Rounding

#TODO: default per regression problem

#TODO: return nothing
delete_rules_by_frequency(S::RuleBasedModel) = nothing

#stel -> learner to get a rule list for future predictions
function simplified_tree_ensemble_learner(
        best_rules::RuleBasedModel{L,C},
        X::MultiFrameModalDataset,
        Y::AbstractVector;
        min_support=0.01
    ) where {L,C}

    R = RuleBasedModel()  #vector of ordered list
    rule_default = default(C,Y)
    S = RuleBasedModel()  #vector of rules left
    append!(S.rules,best_rules)
    append!(S.rules,rule_default)

    idx_delete_rule = delete_rules_by_frequency(S)
    D = copy(X)

    #to finish

end

# Patch single-frame _-> multi-frame
extract_rules(model::Any, X::ModalDataset, args...; kwargs...) =
    extract_rules(model, MultiFrameModalDataset(X), args...; kwargs...)

# Extract rules from a forest, with respect to a dataset
function extract_rules(
        forest::DForest,
        X::MultiFrameModalDataset,
        Y::AbstractVector;
        prune_rules = false,
        s = nothing,
        decay_threshold = nothing,
        #
        method = :TODO_give_a_proper_name_like_CBC_or_something_like_that,
        min_support = 0.01,

    )
    # Update supporting labels
    _, forest = apply(forest, X, Y; kwargs...)

    ########################################################################################
    # Extract rules from each tree
    ########################################################################################
    # Obtain full ruleset
    ruleset = begin
        ruleset = []
        for every tree in the forest
            tree_rules = list_rules(tree) # TODO implement
            append!(ruleset, tree_rules)
        end
        unique(ruleset) # TODO maybe also sort (which requires a definition of isless(formula1, formula2))
    end
    ########################################################################################

    ########################################################################################
    # Prune rules according to the confidence metric (with respect to a dataset)
    #  (and similar metrics: support, confidence, and length)
    if prune_rules
        ruleset = prune_ruleset(ruleset, X, Y; s = s, decay_threshold = decay_threshold)
    end
    ########################################################################################

    ########################################################################################
    # Obtain the best rules
    best_rules = begin
        if method == :TODO_give_a_proper_name_like_CBC_or_something_like_that
            # Extract antecedents
            # TODO: implement antecedent(rule)
            antset = antecedent.(ruleset)
            # Build the binary satisfuction matrix (m × j+1, with m instances and j antecedents
            M = begin
                TODO use (antset, X, Y) accordingly and compute M
            end
            # TODO implement CBC
            best_rules_idxs = CBC(M)
            M = M[:, best_rules_idxs] (or M[best_rules_idxs, :])
            ruleset[best_rules_idxs]
        else
            error("Unexpected method specified: $(method)")
        end
    end
    ########################################################################################

    ########################################################################################
    # Construct a rule-based model from the set of best rules
    simplified_tree_ensemble_learner(best_rules, X, Y, min_support)
    ########################################################################################
end
