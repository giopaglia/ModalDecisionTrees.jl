using Revise

using SoleLogics
using SoleModels
using SoleModels: info, ScalarCondition, ScalarMetaCondition

using ModalDecisionTrees: left, right

using ModalDecisionTrees: relation, feature, test_operator, threshold
using ModalDecisionTrees: inverse_test_operator
using ModalDecisionTrees: AbstractFeature
using ModalDecisionTrees: DTInternal, DTNode, DTLeaf, NSDTLeaf
using ModalDecisionTrees: isleftchild, isrightchild

using FunctionWrappers: FunctionWrapper

############################################################################################
# MDTv1 translation
############################################################################################

########################### NOTE: functions to test ########################################

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
    pure_root = translate(ModalDecisionTrees.root(tree))

    info = merge(info, SoleModels.info(pure_root))
    info = merge(info, (;))

    return DecisionTree(pure_root, info)
end

function translate(
    node::DTInternal,
    ancestors::Vector{<:DTInternal} = DTInternal[],
    info = (;),
)
    new_ancestors = [ancestors..., node]
    formula = pathformula(new_ancestors, left(node))
    info = merge(info, (;
        this = translate(ModalDecisionTrees.this(node), new_ancestors),
        supp_labels = ModalDecisionTrees.supp_labels(node),
    ))
    SoleModels.Branch(
        formula,
        translate(left(node), new_ancestors),
        translate(right(node), new_ancestors),
        info
    )
end

function translate(
    tree::DTLeaf,
    ancestors::Vector{<:DTInternal} = DTInternal[],
    info = (;),
)
    info = merge(info, (;
        supp_labels = ModalDecisionTrees.supp_labels(tree)
    ))
    return SoleModels.ConstantModel(ModalDecisionTrees.prediction(tree), info)
end

function translate(
    tree::NSDTLeaf,
    ancestors::Vector{<:DTInternal} = DTInternal[],
    info = (;),
)
    info = merge(info, (;
        supp_labels = ModalDecisionTrees.supp_labels(tree)
    ))
    return SoleModels.FunctionModel(ModalDecisionTrees.predicting_function(tree), info)
end

############################################################################################
############################################################################################
############################################################################################

function _condition(feature::AbstractFeature{U}, test_op, threshold::T) where {U,T}
    # t = FunctionWrapper{Bool,Tuple{U,T}}(test_op)
    metacond = ScalarMetaCondition(feature, test_op)
    cond = ScalarCondition(metacond, threshold)
    return cond
end

function get_proposition(dec::ExistentialDimensionalDecision)
    test_op = test_operator(dec)
    return Proposition(_condition(feature(dec), test_op, threshold(dec)))
end

function get_proposition_inv(dec::ExistentialDimensionalDecision)
    test_op = inverse_test_operator(test_operator(dec))
    return Proposition(_condition(feature(dec), test_op, threshold(dec)))
end

function get_diamond_op(dec::ExistentialDimensionalDecision)
    return DiamondRelationalOperator{typeof(relation(dec))}()
end

function get_box_op(dec::ExistentialDimensionalDecision)
    return BoxRelationalOperator{typeof(relation(dec))}()
end


function get_lambda(parent::DTNode, child::DTNode)
    dec = ModalDecisionTrees.decision(parent)
    if isleftchild(child, parent)
        p = get_proposition(dec)
        diamond_op = get_diamond_op(dec)
        return diamond_op(p)
    elseif isrightchild(child, parent)
        p_inv = get_proposition_inv(dec)
        box_op = get_box_op(dec)
        return box_op(p_inv)
    else
        error("Cannot compute pathformula on malformed path: $(nodes).")
    end
end

############################################################################################
############################################################################################
############################################################################################

# Compute path formula using semantics from TODO cite
function pathformula(ancestors::Vector{<:DTInternal{L,<:ExistentialDimensionalDecision}}, leaf::DTNode{LL}) where {L,LL}
    nodes = [ancestors..., leaf]
    # dispatch a seconda del numero di nodi degli ancestors
    if length(nodes) == 0
        error("Cannot compute pathformula on empty path.")
    elseif length(nodes) == 1
        return SyntaxTree(TOP)
    elseif length(nodes) == 2
        return get_lambda(nodes[1], nodes[2])
    else
        φ = pathformula(Vector{DTInternal{Union{L,LL},<:ExistentialDimensionalDecision}}(nodes[2:end-1]), nodes[end])

        if isleftchild(nodes[2], nodes[1])
            dec = ModalDecisionTrees.decision(nodes[1])
            p = get_proposition(dec)
            isprop = (relation(dec) == identityrel)

            if isleftchild(nodes[3], nodes[2])
                if isprop
                    return p ∧ φ
                else
                    ◊ = get_diamond_op(dec)
                    return ◊(p ∧ φ)
                end
            elseif isrightchild(nodes[3], nodes[2])
                if isprop
                    return p ∧ (p → φ)
                else
                    ◊ = get_diamond_op(dec)
                    □ = get_box_op(dec)
                    return ◊(p) ∧ □(p → φ)
                end
            else
                error("Cannot compute pathformula on malformed path: $(nodes).")
            end
        else
            λ = get_lambda(nodes[1], nodes[2])
            return λ ∧ φ
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
