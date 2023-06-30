using Test
using SoleModels
using ModalDecisionTrees
using ModalDecisionTrees: DTLeaf, prediction
using ModalDecisionTrees: DTInternal, decision

# Creation of decision leaves, nodes, decision trees, forests

# Construct a leaf from a label
# @test DTLeaf(1)        == DTLeaf{Int64}(1, Int64[])
# @test DTLeaf{Int64}(1) == DTLeaf{Int64}(1, Int64[])

# @test DTLeaf("Class_1")           == DTLeaf{String}("Class_1", String[])
# @test DTLeaf{String}("Class_1")   == DTLeaf{String}("Class_1", String[])

# Construct a leaf from a label & supporting labels
# @test DTLeaf(1, [])               == DTLeaf{Int64}(1, Int64[])
# @test DTLeaf{Int64}(1, [1.0])     == DTLeaf{Int64}(1, Int64[1])

@test repr( DTLeaf(1.0, [1.0]))   == repr(DTLeaf{Float64}(1.0, [1.0]))
@test_nowarn DTLeaf{Float32}(1, [1])
@test_nowarn DTLeaf{Float32}(1.0, [1.5])

@test_throws MethodError DTLeaf(1, ["Class1"])
@test_throws InexactError DTLeaf(1, [1.5])

@test_nowarn DTLeaf{String}("1.0", ["0.5", "1.5"])

# Inferring the label from supporting labels
@test prediction(DTLeaf{String}(["Class_1", "Class_1", "Class_2"])) == "Class_1"

@test_nowarn DTLeaf(["1.5"])
@test_throws MethodError DTLeaf([1.0,"Class_1"])

# Check robustness
@test_nowarn DTLeaf{Int64}(1, 1:10)
@test_nowarn DTLeaf{Int64}(1, 1.0:10.0)
@test_nowarn DTLeaf{Float32}(1, 1:10)

# @test prediction(DTLeaf(1:10)) == 5
@test prediction(DTLeaf{Float64}(1:10)) == 5.5
@test prediction(DTLeaf{Float32}(1:10)) == 5.5f0
@test prediction(DTLeaf{Float64}(1:11)) == 6

# Check edge parity case (aggregation biased towards the first class)
@test prediction(DTLeaf{String}(["Class_1", "Class_2"])) == "Class_1"
@test prediction(DTLeaf(["Class_1", "Class_2"])) == "Class_1"

# TODO test NSDT Leaves

# Decision internal node (DTInternal) + Decision Tree & Forest (DTree & DForest)

formula = SoleModels.ScalarExistentialFormula(SoleModels.globalrel, UnivariateMin(1), >=, 10)

_decision = SimpleDecision(formula)

reg_leaf, cls_leaf = DTLeaf([1.0,2.0]), DTLeaf([1,2])

# create node
cls_node = @test_nowarn DTInternal(2, _decision, cls_leaf, cls_leaf, cls_leaf)

# composite node
cls_node = @test_nowarn DTInternal(2, _decision, cls_leaf, cls_leaf, cls_leaf)
cls_node = DTInternal(2, _decision, cls_leaf, cls_leaf, cls_leaf)

# Note: modality is required
@test_throws MethodError DTInternal(_decision, cls_leaf, cls_leaf, cls_leaf)
@test_throws MethodError DTInternal(_decision, reg_leaf, reg_leaf, reg_leaf)
@test_throws MethodError DTInternal(_decision, cls_node, cls_leaf)

# create node without local _decision
# cls_node = @test_nowarn DTInternal(2, _decision, cls_leaf, cls_leaf)
cls_node = @test_logs (:warn,) DTInternal(2, _decision, cls_leaf, cls_leaf)

# Mixed tree
@test_throws AssertionError DTInternal(2, _decision, reg_leaf, cls_leaf)

cls_tree = @test_nowarn DTree(cls_node, [Interval], [ModalDecisionTrees.start_without_world])
cls_forest = @test_nowarn DForest([cls_tree, cls_tree, cls_tree])

