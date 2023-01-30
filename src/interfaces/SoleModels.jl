using Revise

using SoleLogics
using SoleModels
using SoleModels: info, FeatCondition, FeatMetaCondition

using ModalDecisionTrees: left, right

using ModalDecisionTrees: relation, feature, test_operator, threshold
using ModalDecisionTrees: test_operator_inverse
using ModalDecisionTrees: AbstractFeature
using ModalDecisionTrees: DTInternal, DTNode, DTLeaf, NSDTLeaf

using FunctionWrappers: FunctionWrapper


############################################################################################
# MDTv1 translation
############################################################################################

########################### NOTE: functions to test ########################################

function translate_mdtv1(
    forest::DForest,
    info = (;),
)
    pure_trees = [translate_mdtv1(tree) for tree in trees(forest)]

    info = merge(info, (;
        metrics = metrics(forest),
    ))

    return DecisionForest(pure_trees, info)
end

function translate_mdtv1(
    tree::DTree,
    info = (;),
)
    pure_root = translate_mdtv1(ModalDecisionTrees.root(tree))

    info = merge(info, SoleModels.info(pure_root))
    info = merge(info, (;))
    
    return DecisionTree(pure_root, info)
end

function translate_mdtv1(
    node::DTInternal, 
    ancestors::Vector{<:DTInternal} = DTInternal[],
    info = (;),
)
    new_ancestors = [ancestors..., node]
    formula = pathformula(DTNode[new_ancestors..., left(node)])
    info = merge(info, (;
        this = translate_mdtv1(ModalDecisionTrees.this(node), new_ancestors),
        supp_labels = ModalDecisionTrees.supp_labels(node),
    ))
    SoleModels.Branch(
        formula,
        translate_mdtv1(left(node), new_ancestors),
        translate_mdtv1(right(node), new_ancestors),
        info
    )
end

function translate_mdtv1(
    tree::DTLeaf,
    ancestors::Vector{<:DTInternal} = DTInternal[],
    info = (;),
)
    info = merge(info, (;
        supp_labels = ModalDecisionTrees.supp_labels(tree)
    ))
    return SoleModels.ConstantModel(ModalDecisionTrees.prediction(tree), info)
end

function translate_mdtv1(
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
    metacond = FeatMetaCondition(feature, test_op)
    cond = FeatCondition(metacond, threshold)
    return cond
end

function get_proposition(dec::ExistentialDimensionalDecision)
    test_op = test_operator(dec)
    return Proposition(_condition(feature(dec), test_op, threshold(dec)))
end

function get_proposition_inv(dec::ExistentialDimensionalDecision)
    test_op = test_operator_inverse(test_operator(dec))
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

isleftchild(node::DTNode, parent::DTNode) = (left(parent) == node)
isrightchild(node::DTNode, parent::DTNode) = (right(parent) == node)

function pathformula(nodes::Vector{<:DTNode})
    # dispatch a seconda del numero di nodi degli ancestors
    if length(nodes) == 0
        error("Cannot compute pathformula on empty path.")
    elseif length(nodes) == 1
        return SyntaxTree(TOP)
    elseif length(nodes) == 2
        return get_lambda(nodes[1], nodes[2])
    else
        φ = pathformula(nodes[2:end])

        if isleftchild(nodes[2], nodes[1])
            dec = ModalDecisionTrees.decision(nodes[1])
            p = get_proposition(dec)
            isprop = (relation(dec) == RelationId)

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
