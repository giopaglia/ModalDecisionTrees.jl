using Revise

using SoleLogics
using SoleModels
using SoleModels: info, ScalarCondition, ScalarMetaCondition

using SoleModels: AbstractFeature
using SoleModels: relation, feature, test_operator, threshold, inverse_test_operator

using ModalDecisionTrees: DTInternal, DTNode, DTLeaf, NSDTLeaf
using ModalDecisionTrees: isleftchild, isrightchild, isinleftsubtree, isinrightsubtree

using ModalDecisionTrees: left, right

using FunctionWrappers: FunctionWrapper

using Memoization

############################################################################################
# MDTv1 translation
############################################################################################

function build_antecedent(a::MultiFormula{F}, initconditions) where {F<:AbstractSyntaxStructure}
    MultiFormula(Dict([i_mod => anchor(f, initconditions[i_mod]) for (i_mod, f) in modforms(a)]))
end

function translate(
    forest::DForest,
    info = (;),
)
    pure_trees = [translate(tree) for tree in trees(forest)]

    info = merge(info, (;
        metrics = metrics(forest),
    ))

    return DecisionForest(pure_trees, info)
end

function translate(
    tree::DTree,
    info = (;),
)
    pure_root = translate(ModalDecisionTrees.root(tree), ModalDecisionTrees.initconditions(tree))

    info = merge(info, SoleModels.info(pure_root))
    info = merge(info, (;))

    return DecisionTree(pure_root, info)
end

function translate(
    node::DTInternal,
    initconditions,
    ancestors::Vector{<:DTInternal} = DTInternal[],
    info = (;),
    shortform::Union{Nothing,MultiFormula} = nothing,
)
    new_ancestors = [ancestors..., node]
    φl = pathformula(new_ancestors, left(node), false)
    φr = pathformula(new_ancestors, right(node), false)
    pos_shortform, neg_shortform = begin
        if isnothing(shortform)
            φl, φr
        else
            dl, dr = deepcopy(modforms(shortform)), deepcopy(modforms(shortform))
            dl[i_modality(node)] = modforms(φl)[i_modality(node)]
            dr[i_modality(node)] = modforms(φr)[i_modality(node)]
            MultiFormula(dl), MultiFormula(dr)
        end
    end
    info = merge(info, (;
        this = translate(ModalDecisionTrees.this(node), initconditions, new_ancestors, (;), shortform),
        supporting_labels = ModalDecisionTrees.supp_labels(node),
        # pos_shortform = build_antecedent(pos_shortform, initconditions),
        # neg_shortform = build_antecedent(neg_shortform, initconditions),
    ))
    if !isnothing(shortform)
        info = merge(info, (;
            shortform = build_antecedent(shortform, initconditions),
        ))
    end

    SoleModels.Branch(
        build_antecedent(φl, initconditions),
        translate(left(node), initconditions, new_ancestors, (;), pos_shortform),
        translate(right(node), initconditions, new_ancestors, (;), neg_shortform),
        info
    )
end

function translate(
    tree::DTLeaf,
    initconditions,
    ancestors::Vector{<:DTInternal} = DTInternal[],
    info = (;),
    shortform = nothing
)
    info = merge(info, (;
        supporting_labels      = ModalDecisionTrees.supp_labels(tree),
        supporting_predictions = ModalDecisionTrees.predictions(tree),
    ))
    if !isnothing(shortform)
        info = merge(info, (;
            shortform = build_antecedent(shortform, initconditions),
        ))
    end
    return SoleModels.ConstantModel(ModalDecisionTrees.prediction(tree), info)
end

function translate(
    tree::NSDTLeaf,
    initconditions,
    ancestors::Vector{<:DTInternal} = DTInternal[],
    info = (;),
    shortform = nothing
)
    info = merge(info, (;
        supporting_labels      = ModalDecisionTrees.supp_labels(tree),
        supporting_predictions = ModalDecisionTrees.predictions(tree),
    ))
    if !isnothing(shortform)
        info = merge(info, (;
            shortform = build_antecedent(shortform, initconditions),
        ))
    end
    return SoleModels.FunctionModel(ModalDecisionTrees.predicting_function(tree), info)
end

############################################################################################
############################################################################################
############################################################################################

function _condition(feature::AbstractFeature, test_op, threshold::T) where {T}
    # t = FunctionWrapper{Bool,Tuple{U,T}}(test_op)
    metacond = ScalarMetaCondition(feature, test_op)
    cond = ScalarCondition(metacond, threshold)
    return cond
end

function get_proposition(φ::ScalarExistentialFormula)
    test_op = test_operator(φ)
    return Proposition(_condition(feature(φ), test_op, threshold(φ)))
end

function get_proposition_inv(φ::ScalarExistentialFormula)
    test_op = inverse_test_operator(test_operator(φ))
    return Proposition(_condition(feature(φ), test_op, threshold(φ)))
end

function get_diamond_op(φ::ScalarExistentialFormula)
    return DiamondRelationalOperator{typeof(relation(φ))}()
end

function get_box_op(φ::ScalarExistentialFormula)
    return BoxRelationalOperator{typeof(relation(φ))}()
end


function get_lambda(parent::DTNode, child::DTNode)
    f = formula(ModalDecisionTrees.decision(parent))
    isprop = (relation(f) == identityrel)
    if isinleftsubtree(child, parent)
        p = get_proposition(f)
        diamond_op = get_diamond_op(f)
        return isprop ? SyntaxTree(p) : diamond_op(p)
    elseif isinrightsubtree(child, parent)
        p_inv = get_proposition_inv(f)
        box_op = get_box_op(f)
        return isprop ? SyntaxTree(p_inv) : box_op(p_inv)
    else
        error("Cannot compute pathformula on malformed path: $((child, parent)).")
    end
end

############################################################################################
############################################################################################
############################################################################################

# Compute path formula using semantics from TODO cite
@memoize function pathformula(
    ancestors::Vector{<:DTInternal{L,<:SimpleDecision{<:ScalarExistentialFormula}}},
    node::DTNode{LL},
    multi::Bool
) where {L,LL}

    if length(ancestors) == 0
        return error("pathformula cannot accept 0 ancestors. node = $(node).")
    else
        # Compute single-modality formula to check.
        if !multi
            ancestors = filter(a->i_modality(a) == i_modality(last(ancestors)), ancestors)
            # @assert length(ancestors) > 0
        end

        if length(ancestors) == 1
            anc = first(ancestors)
            return MultiFormula(i_modality(anc), get_lambda(anc, node))
        else
            φ = pathformula(Vector{DTInternal{Union{L,LL},<:SimpleDecision{<:ScalarExistentialFormula}}}(ancestors[2:end]), node, multi)

            nodes = [ancestors..., node]

            # @assert length(unique(anc_mods)) == 1 "At the moment, translate does not work " *
            #     "for MultiFormula formulas $(unique(anc_mods))."

            if isinleftsubtree(nodes[2], nodes[1])
                f = formula(ModalDecisionTrees.decision(nodes[1]))
                p = MultiFormula(i_modality(nodes[1]), SyntaxTree(get_proposition(f)))
                isprop = (relation(f) == identityrel)

                if isinleftsubtree(nodes[3], nodes[2])
                    if isprop
                        return p ∧ φ
                    else
                        ◊ = get_diamond_op(f)
                        return ◊(p ∧ φ)
                    end
                elseif isinrightsubtree(nodes[3], nodes[2])
                    if isprop
                        return p ∧ (p → φ)
                    else
                        ◊ = get_diamond_op(f)
                        □ = get_box_op(f)
                        return ◊(p) ∧ □(p → φ)
                    end
                else
                    error("Cannot compute pathformula on malformed path: $((nodes[3], nodes[2])).")
                end
            elseif isinrightsubtree(nodes[2], nodes[1])
                λ = MultiFormula(i_modality(nodes[1]), get_lambda(nodes[1], nodes[2]))
                return λ ∧ φ
            else
                error("Cannot compute pathformula on malformed path: $((nodes[2], nodes[1])).")
            end
        end
    end
end

# lambda(node::DTInternal) = decision2formula(decision(node))
# lambda_inv(node::DTInternal) = ¬decision2formula(decision(node))

# isback(backnode::DTInternal, back::DTInternal) = (backnode == back(node))
# isleft(leftnode::DTInternal, node::DTInternal) = (leftnode == left(node))
# isright(rightnode::DTInternal, node::DTInternal) = (rightnode == right(node))

# function lambda(node::DTInternal, parent::DTInternal)
#     if isleft(node, parent)
#         lambda(parent)
#     elseif isright(node, parent)
#         lambda_inv(parent)
#     else
#         error("Cannot compute lambda of two nodes that are not parent-child: $(node) and $(parent).")
#     end
# end


# function isimplicative(f::AbstractFormula)
#     t = tree(f)
#     return token(t) == → ||
#         (any(isa.(token(t), [BoxRelationalOperator, □])) && first(children(t)) == →)
# end

# function pathformula(ancestors::Vector{<:DTInternal{L,<:DoubleEdgedDecision}}, leaf::DTNode{L}) where {L}
#     nodes = [ancestors..., leaf]
#     depth = length(ancestors)

#     if depth == 0
#         SoleLogics.⊤
#     elseif depth == 1
#         lambda(node, first(ancestor))
#     else
#         _lambda = lambda(first(ancestors), second(ancestors))
#         pi1, pi2, ctr, ctr_child = begin
#         TODO
#         for a in ancestors...
#         isback
#         ctr
#         end
#         agreement = !xor(isleft(second(ancestors), first(ancestors)), isleft(ctr_child, ctr))

#         f1 = pureformula(pi1)
#         f2 = pureformula(pi2)

#         if !(_lambda isa... ExistsTrueDecision)
#             if !xor(agreement, !isimplicative(f2))
#                 _lambda ∧ (f1 ∧ f2)
#             else
#                 _lambda → (f1 → f2)
#             end
#         else
#             relation = relation(_lambda)
#             if !xor(agreement, !isimplicative(f2))
#                 DiamondRelationalOperator(relation)()(f1 ∧ f2)
#             else
#                 BoxRelationalOperator(relation)()(f1 → f2)
#             end
#         end
#     end
# end

############################################################################################
############################################################################################
############################################################################################
