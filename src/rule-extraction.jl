############################################################################################
using Metrics: mse
using SoleFeatures: findcorrelation
using SoleLogics: operators

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

# Evaluation for single decision
# TODO
function evaluate_decision(dec::Decision, X::MultiFrameModalDataset) end

# TODO remove
logic = extract_logic

############################################################################################
############################################################################################
############################################################################################

# Extract decisions from rule
function extract_decisions(formula::Formula{L}) where {L<:Logic}
    # TODO remove in favor of operators_set = operators(L)
    operators_set = operators(logic(antecedent(rule)))
    function _extract_decisions(node::FNode, decs::AbstractVector{<:Decision})
        # Leaf or internal node
        if !isdefined(node, :leftchild) && !isdefined(node, :rightchild)
            if token(node) in operators_set
                return decs
            else
                return push!(decs, token(node))
            end
        else
            isdefined(node, :leftchild)  && _extract_decisions(leftchild(node),  decs)
            isdefined(node, :rightchild) && _extract_decisions(rightchild(node), decs)

            if !(token(node) in operators_set)
                return push!(decs, token(node))
            end
            decs
        end
    end
    _extract_decisions(formula.tree, [])
end


############################################################################################
# Rule evaluation
############################################################################################

# Evaluation for an antecedent

evaluate_antecedent(antecedent::Formula{L}, X::MultiFrameModalDataset) where {L<:Logic} =
    evaluate_antecedent(extract_decisions(antecedent), X)

function evaluate_antecedent(decs::AbstractVector{<:Decision}, X::MultiFrameModalDataset)
    D = hcat([evaluate_decision(d, X) for d in decs]...)
    # If all values in a row is true, then true (and logical)
    return map(all, eachrow(D))
end

# Evaluation for a rule

# From rule to antecedent and consequent
evaluate_rule(rule::Rule, X::MultiFrameModalDataset, Y::AbstractVector{<:Consequent}) =
    evaluate_rule(antecedent(rule), consequent(rule), X, Y)

# From antecedent to decision
evaluate_rule(
    ant::Formula{L},
    cons::Consequent,
    X::MultiFrameModalDataset,
    Y::AbstractVector{<:Consequent}
) where {L<:Logic} = evaluate_rule(extract_decisions(ant),cons,X,Y)

# Use decision and consequent
function evaluate_rule(
    decs::AbstractVector{<:Decision},
    cons::Consequent,
    X::MultiFrameModalDataset,
    Y::AbstractVector{<:Consequent}
)
    # Antecedent satisfaction. For each instances in X:
    #  - `false` when not satisfiable,
    #  - `true` when satisfiable.
    ant_sat = evaluate_antecedent(decs,X)

    # Indices of satisfiable instances
    idxs_sat = findall(ant_sat .== true)

    # Consequent satisfaction. For each instances in X:
    #  - `false` when not satisfiable,
    #  - `true` when satisfiable,
    #  - `nothing` when antecedent does not hold.
    cons_sat = begin
        cons_sat = Vector{Union{Bool, Nothing}}(fill(nothing, length(Y)))
        idxs_true = begin
            idx_cons = findall(cons .== Y)
            intersect(idxs_sat,idx_cons)
        end
        idxs_false = begin
            idx_cons = findall(cons .!= Y)
            intersect(idxs_sat,idx_cons)
        end
        cons_sat[idxs_true]  .= true
        cons_sat[idxs_false] .= false
    end

    y_pred = begin
        y_pred = Vector{Union{Consequent, Nothing}}(fill(nothing, length(Y)))
        y_pred[idxs_sat] .= C
        y_pred
    end

    return (;
        ant_sat   = ant_sat,
        idxs_sat  = idxs_sat,
        cons_sat  = cons_sat,
        y_pred    = y_pred,
    )
end

############################################################################################
# Rule extraction from random forest
############################################################################################

# Patch single-frame _-> multi-frame
extract_rules(model::Any, X::ModalDataset, args...; kwargs...) =
    extract_rules(model, MultiFrameModalDataset(X), args...; kwargs...)

# Extract rules from a forest, with respect to a dataset
function extract_rules(
        forest::DForest,
        X::MultiFrameModalDataset,
        Y::AbstractVector{<:Consequent};
        prune_rules = false,
        s = nothing,
        decay_threshold = nothing,
        #
        method = :CBC,
        min_frequency = nothing,
)
    # """
    #     length_rule(node::FNode, operators::Operators) -> Int

    #     Computer the number of pairs in a rule (length of the rule)

    # # Arguments
    # - `node::FNode`: node on which you refer
    # - `operators::Operators`: set of operators of the considered logic

    # # Returns
    # - `Int`: number of pairs
    # """
    # function length_rule(node::FNode, operators::Operators)
    #     left_size = 0
    #     right_size = 0

    #     if !isdefined(node, :leftchild) && !isdefined(node, :rightchild)
    #         # Leaf
    #         if token(node) in operators
    #             return 0
    #         else
    #             return 1
    #         end
    #     end

    #     isdefined(node, :leftchild) && (left_size = length_rule(leftchild(node), operators))
    #     isdefined(node, :rightchild) && (right_size = length_rule(rightchild(node), operators))

    #     if token(node) in operators
    #         return left_size + right_size
    #     else
    #         return 1 + left_size + right_size
    #     end
    # end

    rule_metrics(rule::Rule{L,C}, X::MultiFrameModalDataset, Y::AbstractVector{<:Consequent}) =
        rule_metrics(extract_decisions(antecedent(rule)),cons,X,Y)

    """
        rule_metrics(args...) -> AbstractVector

        Compute frequency, error and length of the rule

    # Arguments
    - `decs::AbstractVector{<:Decision}`: vector of decisions
    - `cons::Consequent`: rule's consequent
    - `X::MultiFrameModalDataset`: dataset
    - `Y::AbstractVector{<:Consequent}`: target values of X

    # Returns
    - `AbstractVector`: metrics values vector of the rule
    """
    function rule_metrics(
        decs::AbstractVector{<:Decision},
        cons::Consequent,
        X::MultiFrameModalDataset,
        Y::AbstractVector{<:Consequent}
    ) where {L,C}
        metrics = (;) #empty named tuple
        eval_result = evaluate_rule(decs, cons, X, Y)
        n_instances = size(X, 1)
        n_satisfy = sum(eval_result[:ant_sat])

        # Support of the rule
        rule_support =  n_satisfy / n_instances

        # Error of the rule
        rule_error = begin
            if typeof(cons) <: CLabel
                # Number of incorrectly classified instances divided by number of instances
                # satisfying the rule condition.
                misclassified_instances = length(findall(eval_result[:y_pred] .== Y))
                misclassified_instances / n_satisfy
            elseif typeof(cons) <: RLabel
                # Mean Squared Error (mse)
                idxs_sat = eval_result[:idxs_sat]
                mse(eval_result[:y_pred][idxs_sat], Y[idxs_sat])
            end
        end

        # Length of the rule
        rule_length = length(decs)

        return (;
            support   = rule_support,
            error     = rule_error,
            length    = rule_length,
        )
    end

    """
        prune_ruleset(ruleset::AbstractVector{<:Rule}) -> RuleBasedModel

        Prune the rules in ruleset with error metric

    If `s` and `decay_threshold` is unspecified, their values are set to nothing and the
    first two rows of the function set s and decay_threshold with their default values

    # Arguments
    - `ruleset::AbstractVector{<:Rule}`: rules to prune

    # Returns
    - `RuleBasedModel`: rules after the prune
    """
    function prune_ruleset(
        ruleset::AbstractVector{<:Rule}
    )
        isnothing(s) && (s = 1.0e-6)
        isnothing(decay_threshold) && (decay_threshold = 0.05)

        [begin
            # Extract decisions from rule
            decs = extract_decisions(antecedent(rule))
            cons = consequent(rule)

            E_zero = rule_metrics(decs, cons, X, Y)[:error]

            for idx in length(decs):1
                # Indices to be considered to evaluate the rule
                other_idxs = vcat(1:(idx-1), (idx+1):length(decs))
                # Return error of the rule without idx-th pair
                E_minus_i = rule_metrics(decs[other_idxs], cons, X, Y)[:error]
                decay_i = (E_minus_i - E_zero) / max(E_zero, s)
                if decay_i < decay_threshold
                    # Remove the idx-th pair in the vector of decisions
                    deleteat!(decs, idx)
                    E_zero = rule_metrics(decs, cons, X, Y)[:error]
                end
            end
            # Assemble formula from vector of decisions (decs)
            antecedent = TODO
            #TODO check if this works:
            Rule(antecedent, cons)
            # Rule{typeof(antecedent),typeof(cons)}(antecedent, cons)
        end for rule in ruleset]
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
            antset = antecedent.(ruleset)
            # Build the binary satisfuction matrix (m × j, with m instances and j antecedents)
            M = hcat([evaluate_antecedent(antecedent(rule), X) for rule in antset]...)
            # correlation() -> function in SoleFeatures
            best_idxs = findcorrelation(M)
            M = M[:, best_idxs]
            ruleset[best_idxs]
        else
            error("Unexpected method specified: $(method)")
        end
    end
    ########################################################################################

    ########################################################################################
    # Construct a rule-based model from the set of best rules
    isnothing(min_frequency) && (min_frequency = 0.01)

    # D -> copy of the original dataset
    D = copy(X)
    R = RuleBasedModel()  # Vector of ordered list
    S = RuleBasedModel()  # Vector of rules left
    append!(S.rules, best_rules)
    push!(S.rules, Rule{L}(Formula{L}(), majority_vote(Y)))

    # Delete rules that have a frequency less than min_frequency
    S = begin
        # "metrics" is a vector of dictionaries that contain the metrics of a specific rule
        metrics = rule_metrics.(S, X, Y)
        freq_rules = [metrics[i][:support] for i in collect(1:length(metrics))]
        idx_undeleted = findall(freq_rules .>= min_frequency) # Undeleted rule indexes
        S[idx_undeleted]
    end

    while true
        # Metrics update based on remaining instances
        metrics = rule_metrics.(S, D, Y)
        rules_support = [metrics[i][:support] for i in eachindex(metrics)]
        rules_error = [metrics[i][:error] for i in eachindex(metrics)]
        rules_length = [metrics[i][:length] for i in eachindex(metrics)]

        # Best rule index
        idx_best = begin
            # First: find the rule with minimum error
            idx = findall(rules_error .== min(rules_error...))
            (length(idx) == 1) && (return idx)

            # If not one, find the rule with maximum frequency
            idx_support = findall(rules_support .== max(rules_support[idx]...))
            (length(intersect!(idx, idx_support)) == 1) && (return idx)

            # If not one, find the rule with minimum length
            idx_length = findall(rules_length .== min(rules_length[idx]...))
            (length(intersect!(idx, idx_length)) == 1) && (return idx)

            # Final case: more than one rule with minimum length
            # Randomly choose a rule
            rand(idx)
        end

        # Add at the end the best rule
        push!(R.rules, S[idx_best])
        # Delete the best rule from S
        deleteat!(S,idx_best)

        # Delete the instances satisfying the best rule
        idx_remaining = begin
            # There remain instances that do not meet the best rule's condition
            # (S[idx_best]).
            eval_result = evaluate_rule(S[idx_best], D, Y)
            sat_unsat = eval_result[:ant_sat]
            #remain in D the rule that not satisfying the best rule's condition
            findall(sat_unsat .== false)
        end

        D = D[idx_remaining,:]
        # Update of the default rule
        S.rules[length(S.rules)] = Rule{L}(Formula{L}(), majority_vote(Y[idx_remaining]))

        if idx_best == length(S.rules)
            return R
        end

        if size(D, 1) == 0  #there are no instances in D
            push!(R, Rule{L}(Formula{L}(),majority_vote(Y)))
            return R
        end
    end
    ########################################################################################
end
