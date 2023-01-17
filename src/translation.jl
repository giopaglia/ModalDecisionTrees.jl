using Revise

using SoleLogics
using SoleModels
using SoleModels: info, LogicalTruthCondition, Condition

############################################################################################
# MDTv1 translation
############################################################################################

############################ NOTE: function to try #########################################

function translate_mdtv1(forest::DForest)
    pure_trees = [translate_mdtv1(tree) for tree in trees(forest)]

    info = (;
        metrics = metrics(forest),
    )

    return DecisionForest(pure_trees,info)
end

function translate_mdtv1(tree::DTree)
    pure_root = translate_mdtv1(ModalDecisionTrees.root(tree))
    info = SoleModels.info(pure_root)
    # TODO
    # info = merge(info(pure_root),
    #     tree.metrics
    # )
    return DecisionTree(pure_root, info)
end


function translate_mdtv1(tree::DTLeaf)
    # TODO: info(tree)
    return SoleModels.ConstantModel(
        ModalDecisionTrees.prediction(tree),
        (; supp_labels = ModalDecisionTrees.supp_labels(tree))
    )
end

function translate_mdtv1(tree::NSDTLeaf)
    # TODO: info(tree)
    return SoleModels.FunctionModel(
        ModalDecisionTrees.predicting_function(tree),
        (; supp_labels = ModalDecisionTrees.supp_labels(tree))
    )
end

function translate_mdtv1(node::DTInternal, ancestors::Vector{DTInternal} = [])
    formula = compose_pureformula(node, ancestors)
    SoleModels.Branch(LogicalTruthCondition{SyntaxTree}(formula),
        typeof(ModalDecisionTrees.left(node)) <: Union{DTLeaf,NSDTLeaf} ?
            translate_mdtv1(ModalDecisionTrees.left(node)) :
            translate_mdtv1(ModalDecisionTrees.left(node),[ancestors...,node]),
        typeof(ModalDecisionTrees.right(node)) <: Union{DTLeaf,NSDTLeaf} ?
            translate_mdtv1(ModalDecisionTrees.right(node)) :
            translate_mdtv1(ModalDecisionTrees.right(node),[ancestors...,node]),
        (;
            this = translate_mdtv1(ModalDecisionTrees.this(node)),
            supp_labels = ModalDecisionTrees.supp_labels(node)
        )
    )
end

function diamond_proposition(node::DTInternal)
    dec = ModalDecisionTrees.decision(node)
    metacond = FeaturedMetaCondition(feature(dec),FunctionWrapper(test_operator(dec)))
    cond = Condition(metacond,threshold(dec))
    diamond = DiamondRelationalOperator{typeof(ModalDecisionTrees.relation(dec))}

    return SyntaxTree(diamond,(cond))
end

function box_proposition(node::DTInternal)
    dec = ModalDecisionTrees.decision(node)
    metacond = FeaturedMetaCondition(feature(dec),FunctionWrapper(test_operator(dec)))
    cond = Condition(metacond,threshold(dec))
    box = BoxRelationalOperator{typeof(ModalDecisionTrees.relation(dec))}

    return SyntaxTree(box,(cond))
end

function compose_pureformula(node::DTInternal,ancestors::Vector{DTInternal})
    # dispatch a seconda del numero di nodi degli ancestors
    if length == 0
        dec = ModalDecisionTrees.decision(node)
        metacond = FeaturedMetaCondition(feature(dec),FunctionWrapper(test_operator(dec)))
        cond = Condition(metacond,threshold(dec))
        diamond = DiamondRelationalOperator{typeof(ModalDecisionTrees.relation(dec))}

        return SyntaxTree(diamond,(cond))
    else if length == 1
        if is_lchild(node,ancestors[1])
            φ = begin
                dec = ModalDecisionTrees.decision(node)
                metacond = FeaturedMetaCondition(feature(dec),FunctionWrapper(test_operator(dec)))
                cond = Condition(metacond,threshold(dec))
                diamond = DiamondRelationalOperator{typeof(ModalDecisionTrees.relation(dec))}

                SyntaxTree(diamond,(cond))
            end

            λ = begin
                dec = ModalDecisionTrees.decision(ancestors[1])
                metacond = FeaturedMetaCondition(feature(dec),FunctionWrapper(test_operator(dec)))
                cond = Condition(metacond,threshold(dec))

                SyntaxTree(cond)
            end

            diamond = DiamondRelationalOperator{typeof(ModalDecisionTrees.relation(dec))}

            return SyntaxTree(diamond,(∧(λ,φ)))
        else
            λ = begin
                dec = ModalDecisionTrees.decision(node)
                metacond = FeaturedMetaCondition(feature(dec),FunctionWrapper(test_operator(dec)))
                cond = Condition(metacond,threshold(dec))
                diamond = DiamondRelationalOperator{typeof(ModalDecisionTrees.relation(dec))}

                SyntaxTree(diamond,(cond))
            end

            φ = begin
                dec = ModalDecisionTrees.decision(ancestors[1])
                metacond = FeaturedMetaCondition(feature(dec),FunctionWrapper(test_operator(dec)))
                cond = Condition(metacond,threshold(dec))
                diamond = DiamondRelationalOperator{typeof(ModalDecisionTrees.relation(dec))}

                SyntaxTree(diamond,(cond))
            end

            return ∧(λ,φ)
        end
    else
        return compose_pureformula([ancestors...,node])
    end
end

function compose_pureformula(nodes::Vector{DTInternal})

end

#=
# Other cases
function compose_pureformula(nodes::Vector{DTInternal})
    #Funzione ricorsiva che considera i primi due ancestors e poi lascia aperto così si possa calcolare la sottoformula

    # Attraversato ramo destro (ultimo caso)
    if is_rchild(nodes[2],nodes[1])
        neg = begin
            dec = decision(nodes[1])
            box = BoxRelationalOperator{typeof(relation(dec))}
            metacond = FeaturedMetaCondition(feature(dec),FunctionWrapper(test_operator(dec)))
            cond = Condition(metacond,threshold(dec))

            SyntaxTree(box,(cond))
        end

        pos = begin
            if length(nodes) > 2
                compose_formula(nodes[2:end])
            else
                dec = decision(nodes[2])
                diamond = DiamondRelationalOperator{typeof(relation(dec))}
                metacond = FeaturedMetaCondition(
                    feature(dec),FunctionWrapper(test_operator(dec))
                )
                cond = Condition(metacond,threshold(dec))

                SyntaxTree(diamond,(cond))
            end
        end

        return ∧(neg,pos)
    else
        #Attraversato Ramo Sinistro

    end

    λ = lambda(node,parent)

    if is_lchild(node,parent)
        # TODO: define relation
        φ = Proposition(ModalDecisionTrees.decision(node))
        return ∧(λ,φ)
    else
        # Caso λ ∈ S⁻
        φ = Proposition(ModalDecisionTrees.decision(node))
        return ∧(λ,φ)
    end
end
=#

function lambda(node::DTInternal,parent::DTInternal)
    prop = Proposition(ModalDecisionTrees.decision(parent))

    return is_lchild(node,parent) ? SyntaxTree(prop) : ¬(prop)
end

is_lchild(node::DTInternal,parent::DTInternal) =
    ModalDecisionTrees.left(parent) == node ? true : false
is_rchild(node::DTInternal,parent::DTInternal) =
    ModalDecisionTrees.right(parent) == node ? true : false

#=
function translate_mdtv1(prev_condition::AbstractBooleanCondition, node::DTInternal)
    # `node` comes from a non-pure DTree, while `prev_condition` is part of a pure DecisionTree

    # condition in `node` is modified, depending on its parent formula
    # accordingly to ModalDecisionTree paper (pg. 17)
    # TODO: modify compose_purecondition
    condition  = compose_purecondition(node,prev_condition)

    #define a branch with new condition
    return Branch(condition,
            translate_mdtv1(condition,left(node)),
            translate_mdtv1(condition,right(node)),
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
            translate_mdtv1(condition,left(node)),
            translate_mdtv1(condition,right(node)),
            # TODO: in NamedTuple also `prediction`?
            (; supp_labels = supp_labels(node))
        )

    # Where is the position of model?
    #ConstantModel(prediction(node), (; supp_labels = supp_labels(node)))
end
=#

#=
# From here to fix
function compose_purecondition(node::DTInternal, λ::AbstractBooleanCondition)
    # ModalDecisionTree paper, pg.17
    # base case (h=1) and "λ ∈ S⁻" case ("p ∧ φπ" is returned)
    if height(node) == 1 || is_rchild(node)
        p = formula(λ) #extract formula from logical condition
        φ = SyntaxTree(Proposition(is_lchild(node) ?
                                    decision(parent(node)) :            #TODO: fix
                                    NEGATION(decision(parent(node)))
            ))

        return SyntaxTree(CONJUNCTION,(λ,φ))
    else
        # other cases dispatch
        return compose_purecondition(node,λ,Val(agreement(node, parent(node))))
    end
end

# first and second case
function compose_purecondition(node::DTInternal, λ::DTInternal, ::Val{true})
    p_decision = decision(parent(node))

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
        return SyntaxTree(DiamondRelationalOperator{typeof(relation(p_decision))}(),(conj))
    end
end

# third and fourth case
function compose_purecondition(node::DTInternal, lambda::DTInternal, ::Val{false})
    p_decision = decision(parent(node))

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
=#

#=
# utility to link one parent to exactly one children
function link_nodes(p::FNode{L}, r::FNode{L}) where {L <: AbstractLogic}
    right!(p, r)
    parent!(r, p)
end

# utility to link one parent to its left and right childrens
function link_nodes(p::FNode{L}, l::FNode{L}, r::FNode{L}) where {L <: AbstractLogic}
    left!(p, l)
    right!(p, r)
    parent!(l, p)
    parent!(r, p)
end
=#
