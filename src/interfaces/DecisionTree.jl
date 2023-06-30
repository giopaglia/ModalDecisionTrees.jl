# DecisionTree.jl (https://github.com/JuliaAI/DecisionTree.jl) is the main package
#  for decision tree learning in Julia. These definitions allow for ModalDecisionTrees.jl
#  to act as a drop-in replacement for DecisionTree.jl. Well, more or less.

depth(t::ModalDecisionTrees.ModalDecisionTree) = depth(t)
depth(t::ModalDecisionTrees.DTree) = height(t)
