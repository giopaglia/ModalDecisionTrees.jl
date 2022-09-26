############################################################################################
using Metrics: mse
using SoleFeatures: correlation
using SoleLogics: operators
using Statistics: mean
using StatsBase: mode

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

rules(model::RuleBasedModel) = model.rules

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

evaluate_rule(rule::Formula{L},X:MultiFrameModalDataset) = nothing #TODO

# Extract decisions from rule
function extract_decisions(node::Node,operators_set::Operators,decs::AbstractVector)
    if !isdefined(node, :leftchild) && !isdefined(node, :rightchild)
        # Leaf
        if token(node) in operators_set
            return nothing
        else
            return append!(decs, token(node))
        end
    end

    isdefined(node, :leftchild) && extract_decisions(leftchild(node), operators_set, decs)
    isdefined(node, :rightchild) && extract_decisions(rightchild(node), operators_set, decs)

    if token(node) in operators_set
        return nothing
    else
        return append!(decs,token(node))
    end
end

"""
    default(::CLabel, Y::AbstractVector) -> Any

    Compute the most frequent class in the vector Y, default rule for classification
    problem

# Arguments
- `::CLabel`: classification problem
- `Y::AbstractVector`: target values of starting dataset

# Returns
- `Any`: most frequent class that can be a number or string
"""
default(::CLabel, Y::AbstractVector) = mode(Y)

"""
    default(::RLabel, Y::AbstractVector) -> AbstractFloat

    Compute the mean of the vector Y, default rule for regression problem

# Arguments
- `::RLabel`: regression problem
- `Y::AbstractVector`: target values of starting dataset

# Returns
- `AbstractFloat`: mean
"""
default(::RLabel, Y::AbstractVector) = mean(Y)

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
        method = :CBC,
        min_frequency = nothing,
)
    """
        length_rule(node::Node, operators::Operators) -> Int

        Computer the number of pairs in a rule (length of the rule)

    # Arguments
    - `node::Node`: node on which you refer
    - `operators::Operators`: set of operators of the considered logic

    # Returns
    - `Int`: number of pairs
    """
    function length_rule(node::Node, operators::Operators)
        left_size = 0
        right_size = 0

        if !isdefined(node, :leftchild) && !isdefined(node, :rightchild)
            # Leaf
            if token(node) in operators
                return 0
            else
                return 1
            end
        end

        isdefined(node, :leftchild) && (left_size = length_rule(leftchild(node), operators))
        isdefined(node, :rightchild) && (right_size = length_rule(rightchild(node), operators))

        if token(node) in operators
            return left_size + right_size
        else
            return 1 + left_size + right_size
        end
    end

    """
        metrics_rule(args...) -> AbstractVector

        Compute frequency, error and length of the rule

    # Arguments
    - `rule::Rule{L,C}`: rule
    - `X::MultiFrameModalDataset`: dataset
    - `Y::AbstractVector`: target values of X

    # Returns
    - `AbstractVector`: metrics values vector of the rule
    """
    function metrics_rule(
        rule::Rule{L,C},
        X::MultiFrameModalDataset,
        Y::AbstractVector
    ) where {L,C}
        metrics = (;)
        predictions = evaluate_rule(rule, X) # Fix evaluate_rule
        n_instances = size(X, 1)
        n_instances_satisfy =

        # Frequency of the rule
        frequency_rule =  n_instances_satisfy / n_instances
        metrics = merge(metrics, (frequency_rule = frequency_rule))

        # Error of the rule
        error_rule = begin
            if C <: CLabel
                # Number of incorrectly classified instances divided by number of instances
                # satisfying the rule condition.
                sum(predictions .!= Y) / n_instances_satisfy
            elseif C <: RLabel
                # Mean Squared Error (mse)
                mse(predictions, Y)
            end
        end
        metrics = merge(metrics, (error_rule = error_rule))

        # Length of the rule
        n_pairs = length_rule(rule.tree, operators(L))
        metrics = merge(metrics, (n_pairs = n_pairs))

        return metrics
    end

    """
        prune_ruleset(ruleset::RuleBasedModel{L,C}) -> RuleBasedModel

        Prune the rules in ruleset with error metric

    If `s` and `decay_threshold` is unspecified, their values are set to nothing and the
    first two rows of the function set s and decay_threshold with their default values

    # Arguments
    - `ruleset::RuleBasedModel{L,C}`: rules to prune

    # Returns
    - `RuleBasedModel`: rules after the prune
    """
    function prune_ruleset(
        ruleset::RuleBasedModel{L,C}
    ) where {L,C}
        isnothing(s) && (s = 1.0e-6)
        isnothing(decay_threshold) && (decay_threshold = 0.05)

        for rule in ruleset.rules
            E_zero::AbstractFloat = metrics_rule(rule, X, Y)[2]  #error in second position
            # Extract decisions from rule
            decs = []
            extract_decisions(antecedent(rule).tree, operations(L), decs)
            for conds in reverse(decs)
                # temp_formula = SoleModelChecking.gen_formula(ceil(length(decs)/2),) TODO
                E_minus_i = metrics_rule(rule, X, Y)[2]
                decay_i = (E_minus_i - E_zero) / max(E_zero, s)
                if decay_i < decay_threshold
                    # TODO: delete i-th pair in rule    #TODO
                    E_zero = metrics_rule(rule, X, Y)[2]
                end
            end
        end
    end

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
        ruleset = prune_ruleset(ruleset)
    end
    ########################################################################################

    ########################################################################################
    # Obtain the best rules
    best_rules = begin
        if method == :CBC
            # Extract antecedents
            # TODO: implement antecedent(rule)
            antset = antecedent.(ruleset)
            # Build the binary satisfuction matrix (m × j, with m instances and j antecedents)
            M = hcat([evaluate_antecedent(antecent(rule), X) for rule in antset]...)
            # correlation() -> function in SoleFeatures
            best_idxs = correlation(M,cor)
            M = M[:, best_idxs] # (or M[best_rules_idxs, :])
            ruleset[best_idxs]
        else
            error("Unexpected method specified: $(method)")
        end
    end
    ########################################################################################

    ########################################################################################
    # Construct a rule-based model from the set of best rules
    isnothing(min_frequency) && (min_frequency = 0.01)

    R = RuleBasedModel()  # Vector of ordered list
    S = RuleBasedModel()  # Vector of rules left
    append!(S.rules, best_rules)
    append!(S.rules, majority_vote(Y))

    # Delete rules that have a frequency less than min_frequency
    S = begin
        # reduce(hcat,metrics_rule.(S,X,Y))' -> transpose of the matrix of rules metrics
        freq_rules = reduce(hcat, metrics_rule.(S, X, Y))'[:,1]
        idx_undeleted = findall(freq_rules .>= min_frequency) # Undeleted rule indexes
        S[idx_undeleted]
    end

    # D -> copy of the original dataset
    D = copy(X)

    while true
        metrics = reduce(hcat, metrics_rule.(S, D, Y))'

        # Best rule index
        idx_best = begin
            # First: find the rule with minimum error
            idx = findall(metrics[:,2] .== min(metrics[:,2]...))
            (length(idx) == 0) && (return idx)

            # If not one, find the rule with maximum frequency
            idx = findall(metrics[idx,1] .== max(metrics[idx,1]...))
            (length(idx) == 0) && (return idx)

            # If not one, find the rule with minimum length
            idx = findall(metrics[idx,3] .== min(metrics[idx,3]...))
            (length(idx) == 0) && (return idx)

            # TODO: final case, more than one rule with minimum length
        end

        # Add at the end the best rule
        append!(R,S[idx_best])

        # Delete the instances satisfying the best rule
        idx_remaining = begin
            # There remain instances that do not meet the best rule's condition
            # (S[idx_best]).
            predictions = evaluate_rule(S[idx_best], D)
            #remain in D the rule that not satisfying the best rule's condition
            findall(predictions .== 0)
        end

        D = D[idx_remaining,:]
        rule_default = majority_vote(Y[idx_remaining])

        if S[idx_best,:] == rule_default
            return R
        end

        if size(D, 1) == 0  #there are no instances in D
            append!(R, majority_vote(Y))
            return R
        end
    end
    ########################################################################################
end
