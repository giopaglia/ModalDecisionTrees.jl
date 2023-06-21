@testset "parse-and-translate" begin

using SoleModels
using SoleModels: printmodel
using SoleLogics

using ModalDecisionTrees
using ModalDecisionTrees: translate
using ModalDecisionTrees.experimentals: parse_tree

tree_str1 = """
{1} ⟨G⟩ (min[V4] >= 0.04200671690893693)                        NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 37/74 (conf = 0.5000)
✔ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 32/37 (conf = 0.8649)
✘ {1} ⟨G⟩ (min[V22] >= 470729.9023515756)                       YES_WITH_COUGH : 32/37 (conf = 0.8649)
 ✔ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 5/6 (conf = 0.8333)
 ✘ YES_WITH_COUGH : 31/31 (conf = 1.0000)
"""

tree_str2 = """
{1} ⟨G⟩ (max[V28] <= 7.245112655929639)                 YES : 78/141 (conf = 0.5532)
✔ {1} ⟨L̅⟩ (min[V20] >= 4222.6591159789605)                      NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 55/88 (conf = 0.6250)
│✔ {1} ⟨L̅⟩ (max[V11] <= 0.0038141608453366675)                  NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 34/66 (conf = 0.5152)
││✔ {1} ⟨A̅⟩ (max[V29] <= 178.31522392540964)                    NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 19/22 (conf = 0.8636)
│││✔ YES : 3/3 (conf = 1.0000)
│││✘ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 19/19 (conf = 1.0000)
││✘ {1} ⟨B⟩ (min[V26] >= 217902.31767535824)                    YES : 29/44 (conf = 0.6591)
││ ✔ {1} max[V6] <= 0.011319891844101688                        NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 12/14 (conf = 0.8571)
││ │✔ YES : 2/2 (conf = 1.0000)
││ │✘ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 12/12 (conf = 1.0000)
││ ✘ {1} ⟨L⟩ (min[V6] >= 1.2154505217391558)                    YES : 27/30 (conf = 0.9000)
││  ✔ YES : 24/24 (conf = 1.0000)
││  ✘ {1} ⟨A̅⟩ (max[V16] <= 81.4665167044706)                    YES : 3/6 (conf = 0.5000)
││   ✔ YES : 3/3 (conf = 1.0000)
││   ✘ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 3/3 (conf = 1.0000)
│✘ {1} ⟨A̅⟩ (min[V24] >= 10.975911723366615)                     NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 21/22 (conf = 0.9545)
│ ✔ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 21/21 (conf = 1.0000)
│ ✘ YES : 1/1 (conf = 1.0000)
✘ {1} ⟨G⟩ (min[V8] >= 494.33421895459713)                       YES : 45/53 (conf = 0.8491)
 ✔ {1} ⟨L̅⟩ (min[V27] >= 87446.39318797569)                      NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 7/13 (conf = 0.5385)
 │✔ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 6/6 (conf = 1.0000)
 │✘ {1} max[V2] <= 42.36525041432014                    YES : 6/7 (conf = 0.8571)
 │ ✔ YES : 6/6 (conf = 1.0000)
 │ ✘ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 1/1 (conf = 1.0000)
 ✘ {1} ⟨G⟩ (min[V13] >= 31.231588457748384)                     YES : 39/40 (conf = 0.9750)
  ✔ YES : 39/39 (conf = 1.0000)
  ✘ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 1/1 (conf = 1.0000)
"""

tree1 = parse_tree(tree_str1)

tree2 = parse_tree(tree_str2)

pure_tree1 = translate(tree1)

pure_tree2 = translate(tree2)


end
