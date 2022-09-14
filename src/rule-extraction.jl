############################################################################################
# rules(
#     ruleset  it will contain all the rules of the tree
#     node     current node, start with root of the tree
#     C       conjunction of the variable-value pairs from the root node to the current node
# )
############################################################################################
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
            tree_rules = rules(tree) # TODO implement
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
