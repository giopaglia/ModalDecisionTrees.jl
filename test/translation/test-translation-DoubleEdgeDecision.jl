using Revise

############################################################################################
############################################################################################
############################################################################################

using SoleLogics
using SoleModels
using ModalDecisionTrees
using ModalDecisionTrees: DTLeaf, DTInternal

reg_leaf, cls_leaf = DTLeaf([1.0,2.0]), DTLeaf([1,2])

decision1 = ExistentialDimensionalDecision(global_rel, SingleAttributeMin(1), >=, 10)
decision2 = ExistentialDimensionalDecision(IA_A, SingleAttributeMin(2), <, 0)
decision3 = ExistentialDimensionalDecision(IA_L, SingleAttributeMin(3), <=, 0)

branch = DTInternal(2, decision1, cls_leaf, cls_leaf)
branch = DTInternal(2, decision2, cls_leaf, branch)
branch = DTInternal(2, decision3, branch, cls_leaf)

############################################################################################
############################################################################################
############################################################################################

using ModalDecisionTrees: DoubleEdgedDecision, _back!, _forth!

ded1 = DoubleEdgedDecision(decision1)
ded2 = DoubleEdgedDecision(decision2)
ded3 = DoubleEdgedDecision(decision3)

branch = DTInternal(2, ded1, cls_leaf, cls_leaf)
_back!(ded1, Ref(branch))
_forth!(ded1, Ref(branch))
branch = DTInternal(2, ded2, cls_leaf, branch)
_back!(ded2, Ref(branch))
_forth!(ded2, Ref(branch))
branch = DTInternal(2, ded3, branch, cls_leaf)
_back!(ded3, Ref(branch))
_forth!(ded3, Ref(branch))

tree = branch


pure_tree = ModalDecisionTrees.translate(tree)
