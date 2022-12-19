using SoleLogics
using SoleModels

############################################################################################
# MDTv1 translation
############################################################################################
function translate_mdtv1(tree:DTree)
    condition = LogicalTruthCondition(SyntaxTree(SoleLogics.TOP))
    root_tree = root(tree)
    pure_root =
        is_leaf_node(root_tree) ?
            # TODO: find supp_labels for leaf node
            ConstantModel(prediction(root_tree), (; supp_labels = supp_labels(root_tree))) :
            Branch(condition,
                translate_mdtv1(condition,lefchild(root_tree)),
                translate_mdtv1(condition,rightchild(root_tree)),
                # TODO: in NamedTuple also `prediction`?
                (; supp_labels = supp_labels(root_tree))
            )

    return DecisionTree(pure_root)
end

function translate_mdtv1(prev_condition::AbstractBooleanCondition, node::DTInternal)
    # `node` comes from a non-pure DTree, while `prev_condition` is part of a pure DecisionTree

    # condition in `node` is modified, depending on its parent formula
    # accordingly to ModalDecisionTree paper (pg. 17)
    # TODO: modify compose_purecondition
    condition  = compose_purecondition(node,prev_condition)

    #define a branch with new condition
    return Branch(condition,
            translate_mdtv1(condition,lefchild(node)),
            translate_mdtv1(condition,rightchild(node)),
            # TODO: in NamedTuple also `prediction`?
            (; supp_labels = supp_labels(node))
        )
end

# TODO: models
function translate_mdtv1(
    prev_condition::AbstractBooleanCondition,
    node::Union{DTLeaf, NSFTLeaf}
)
    condition  = compose_purecondition(node,prev_condition)

    # if `node` is a DTLeaf or NSFTLeaf, stop the recursion
    return Branch(condition,
            translate_mdtv1(condition,lefchild(node)),
            translate_mdtv1(condition,rightchild(node)),
            # TODO: in NamedTuple also `prediction`?
            (; supp_labels = supp_labels(node))
        )

    # Where is the position of model?
    #ConstantModel(prediction(node), (; supp_labels = supp_labels(node)))
end

agreement(node::DTInternal,parent::DTInternal) =
    is_lchild(parent) && is_lchild(node) ? true : false

# From here to fix
function compose_purecondition(node::DTInternal, λ::AbstractBooleanCondition)
    # ModalDecisionTree paper, pg.17
    # base case (h=1) and "λ ∈ S⁻" case ("p ∧ φπ" is returned)
    if height(node) == 1 || is_rchild(node)
        p = formula(λ) #extract formula from logical condition
        φ = SyntaxTree(Proposition(is_lchild(node) ?
                                    parent(node).decision :            #TODO: fix
                                    NEGATION(parent(node).decision)
            ))

        return SyntaxTree(CONJUNCTION,(λ,φ))
    else
        # other cases dispatch
        return compose_purecondition(node,λ,Val(agreement(node, parent(node))))
    end
end

# first and second case
function compose_purecondition(node::DTInternal, λ::DTInternal, ::Val{true})
    p_decision = parent(node).decision

    p = formula(λ) #extract formula from logical condition
    φ = SyntaxTree(Proposition(is_lchild(node) ?
                                p_decision :            #TODO: fix
                                NEGATION(p_decision)
        ))

    conj = SyntaxTree(CONJUNCTION,(p,φ))

    if is_propositional_decision(p_decision)
        # first case (p ∧ φπ)
        return conj
    else
        # second case (⟨X⟩(p ∧ φπ))
        return SyntaxTree(EXMODOP(relation(p_decision)),(conj))
    end
end

# third and fourth case
function compose_purecondition(node::DTInternal, lambda::DTInternal, ::Val{false})
    p_decision = parent(node).decision

    p = formula(lambda) #extract formula from logical condition
    φ = SyntaxTree(Proposition(is_lchild(node) ?
                                p_decision :            #TODO: fix
                                NEGATION(p_decision)
        ))

    # NOTE: p2 is identical to p, but we need double the memory
    # to correctly represent every link in the tree.
    # See if-else statement to better understand p2 role.
    p2 = formula(lambda)

    # In both cases the fragment p => φπ is common
    impl = SyntaxTree(IMPLICATION,(p,φ))

    if is_propositional_decision(p_decision)
        # third case p2 ∧ (p => φπ)
        return SyntaxTree(CONJUNCTION,(p2,impl))
    else
        # fourth case ⟨X⟩p2 ∧ [X](p => φπ)
        diamond = SyntaxTree(EXMODOP(relation(p_decision)),(p2))
        box = SyntaxTree(UNIVMODOP(relation(p_decision)),(impl))

        return SyntaxTree(CONJUNCTION,(diamond,box))
    end
end

#=
# utility to link one parent to exactly one children
function link_nodes(p::FNode{L}, r::FNode{L}) where {L <: AbstractLogic}
    rightchild!(p, r)
    parent!(r, p)
end

# utility to link one parent to its left and right childrens
function link_nodes(p::FNode{L}, l::FNode{L}, r::FNode{L}) where {L <: AbstractLogic}
    leftchild!(p, l)
    rightchild!(p, r)
    parent!(l, p)
    parent!(r, p)
end
=#
