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
#                                     min_frequency = 0.01  threshold below which a is
#                                                       dropped from S to avoid overfitting
#                                 ) =
#     R = {}  #ordered rule list
#     t* = average_label(Y)     #default rule
#     S = {t*,best_rules}
#     delete rules from S with frequency < min_frequency
#     D = copy(dataSet)    #copy of original dataset
#     while true
#         use error, frequency and length to select the rule with the least error based on
#         the instances in D or, if multiple instances have the same value, we select the
#         rule with the highest frequency and the shortest condition; this rule will then
#         be added to the end of R
#         Instances that satisfy the best rule are removed from D
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
using Metrics
using Statistics
using SoleLogics
using SoleFeatures

evaluate_rule(rule::Formula{L},X:MultiFrameModalDataset) = nothing #TODO

#length_rule -> number of pairs in a rule
function length_rule(node::Node,operators_set::Operators)
    left_size = 0
    right_size = 0

    if !isdefined(node,:leftchild) && !isdefined(node,:rightchild)
        #leaf
        if token(node) ∉ operators_set
            return 1
        else
            return 0
        end
    end

    isdefined(node,:leftchild) && (left_size = length_rule(leftchild(node),operators_set))
    isdefined(node,:rightchild) && (right_size = length_rule(rightchild(node),operators_set))

    if token(node) ∉ operators_set
        return 1 + left_size + right_size
    else
        return left_size + right_size
    end
end

#metrics_rule -> return frequency, error and length of the rule
#TODO: fix evaluate_rule
function metrics_rule(
        rule::Rule{L,C},
        X::MultiFrameModalDataset,
        Y::AbstractVector
    ) where {L,C}
    metrics = Float64[]

    predictions = evaluate_rule(rule,X)
    n_instances = size(X,1)
    n_instances_satisfy = sum(predictions)

    #frequency of the rule
    frequency_rule =  n_instances_satisfy / n_instances
    append!(metrics,frequency_rule)

    #error of the rule
    if C <: CLabel
        #number of incorrectly classified instances divided by number of instances
        #satisfying the rule condition
        error_rule = sum(predictions .!= Y) / n_instances_satisfy
    elseif C <: RLabel
        #Mean Squared Error (mse)
        error_rule = mse(predictions,Y)
    end
    append!(metrics,error_rule)

    #length of the rule
    n_pairs = length_rule(rule.tree,SoleLogics.operators(L))
    append!(metrics,n_pairs)

    return metrics
end

#extract decisions from rule
function extract_decisions(node::Node,operators_set::Operators,decs::Vector)

    if !isdefined(node,:leftchild) && !isdefined(node,:rightchild)
        #leaf
        if token(node) ∉ operators_set
            return append!(decs,token(node))
        else
            return nothing
        end
    end

    isdefined(node,:leftchild) && extract_decisions(leftchild(node),operators_set,decs)
    isdefined(node,:rightchild) && extract_decisions(rightchild(node),operators_set,decs)

    if token(node) ∉ operators_set
        return append!(decs,token(node))
    else
        return nothing
    end
end

#prune_ruleset -> prune of rule in ruleset
function prune_ruleset(
        ruleset::RuleBasedModel{L,C},
        X::MultiFrameModalDataset,
        Y::AbstractVector;
        s = nothing,
        decay_threshold = nothing
    )

    isnothing(s) && (s = 1.0e-6)
    isnothing(decay_threshold) && (decay_threshold = 0.05)

    for rule in ruleset.rules
        E_zero::Float64 = metrics_rule(rule,X,Y)[2]  #error in second position
        #extract decisions from rule
        decs = []
        extract_decisions(antecedent(rule).tree,operations(L),decs)
        for conds in reverse(decs)
            #temp_formula = SoleModelChecking.gen_formula(ceil(length(decs)/2),) TODO
            E_minus_i = metrics_rule(rule,X,Y)[2]
            decay_i = (E_minus_i-E_zero)/max(E_zero,s)
            if decay_i < decay_threshold
                #TODO: delete i-th pair in rule    #TODO
                E_zero = metrics_rule(rule,X,Y)[2]
            end
        end
    end
end

#StatsBase.mode(x::Vector) -> return the most frequent number in a vector
#default rule for classification problem
default(C <: CLabel, Y::AbstractVector) = mode(Y)
#default rule for regression problem
default(C <: RLabel, Y::AbstractVector) = mean(Y)

#stel -> learner to get a rule list for future predictions
function simplified_tree_ensemble_learner(
        best_rules::RuleBasedModel{L,C},
        X::MultiFrameModalDataset,
        Y::AbstractVector;
        min_frequency = nothing
    ) where {L,C}

    isnothing(min_frequency) && (min_frequency = 0.01)

    R = RuleBasedModel()  #vector of ordered list
    rule_default = default(C,Y)
    S = RuleBasedModel()  #vector of rules left
    append!(S.rules,best_rules.rules)
    append!(S.rules,rule_default)

    #delete rules that have a frequency less than 0.01 (min_frequency)
    S = begin
        #reduce(hcat,metrics_rule.(S,X,Y))' -> transpose of the matrix of rules metrics
        freq_rules = reduce(hcat,metrics_rule.(S,X,Y))'[:,1]
        idx_not_delete_rules = findall(freq_rules .>= min_frequency)
        S[idx_not_delete_rules]
    end

    #D -> copy of the original dataset
    D = copy(X)

    while true
        metrics = reduce(hcat,metrics_rule.(S,D,Y))'

        idx_best_rule = begin
            #first: find the rule with minimum error
            idx = findall(metrics[:,2] .== min(metrics[:,2]...))
            (length(idx) == 0) && (return idx)

            #if not one, find the rule with maximum frequency
            idx = findall(metrics[idx,1] .== max(metrics[idx,1]...))
            (length(idx) == 0) && (return idx)

            #if not one, find the rule with minimum length
            idx = findall(metrics[idx,3] .== min(metrics[idx,3]...))
            (length(idx) == 0) && (return idx)

            #TODO: final case, more than one rule with minimum length
        end

        #add at the end the best rule
        append!(R,S[idx_best_rule])

        #delete the instances satisfying the best rule
        idx_remain_rule = begin
            #there remain instances that do not meet the best rule's condition
            #(S[idx_best_rule])
            predictions = evaluate_rule(S[idx_best_rule],D)
            #remain in D the rule that not satisfying the best rule's condition
            findall(predictions .== 0)
        end

        D = D[idx_remain_rule,:]
        rule_default = default(C,Y[idx_remain_rule])

        if S[idx_best_rule,:] == rule_default
            return R
        end

        if size(D,1) == 0  #there are no instances in D
            append!(R,default(C,Y))
            return R
        end

    end

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
        min_frequency = 0.01,
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
            # Build the binary satisfuction matrix (m × j+1, with m instances and j antecedents)
            M = begin
                #TODO use (antset, X, Y) accordingly and compute M
                M = Matrix{Bool}(undef,size(X,1),length(antset)+1)
                for rule in antset
                    hcat(M,evaluate_rule(rule,X))
                end
                hcat(M,Y)
            end
            #correlation() -> function in SoleFeatures
            best_rules_idxs = correlation(M,cor)
            M = M[:, best_rules_idxs] #(or M[best_rules_idxs, :])
            ruleset[best_rules_idxs]
        else
            error("Unexpected method specified: $(method)")
        end
    end
    ########################################################################################

    ########################################################################################
    # Construct a rule-based model from the set of best rules
    simplified_tree_ensemble_learner(best_rules, X, Y; min_frequency = min_frequency)
    ########################################################################################
end
