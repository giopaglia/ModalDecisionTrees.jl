using Revise
using SoleModels
using SoleLogics

using ModalDecisionTrees
using ModalDecisionTrees.experimentals: parse_tree

include("../../src/translation.jl")

tree_str1 = """
{1} ⟨G⟩ (min(A4) >= 0.04200671690893693)                        NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 37/74 (conf = 0.5000)
✔ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 32/37 (conf = 0.8649)
✘ {1} ⟨G⟩ (min(A22) >= 470729.9023515756)                       YES_WITH_COUGH : 32/37 (conf = 0.8649)
 ✔ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 5/6 (conf = 0.8333)
 ✘ YES_WITH_COUGH : 31/31 (conf = 1.0000)
"""

tree1 = parse_tree(tree_str1)

pure_tree1 = translate_mdtv1(tree1)
