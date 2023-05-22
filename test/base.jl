using SoleModels
using ModalDecisionTrees: DTLeaf, prediction

@testset "Creation of decision leaves, nodes, decision trees, forests" begin

    # @testset "Decision leaves (DTLeaf)" begin

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

    # end

    # TODO test NSDT Leaves

    # @testset "Decision internal node (DTInternal) + Decision Tree & Forest (DTree & DForest)" begin

        decision = ExistentialDimensionalDecision(SoleModels.globalrel, UnivariateMin(1), >=, 10)

        reg_leaf, cls_leaf = DTLeaf([1.0,2.0]), DTLeaf([1,2])

        # # create node
        # # cls_node = @test_nowarn DTInternal(decision, cls_leaf, cls_leaf, cls_leaf)
        # # cls_node = @test_nowarn DTInternal(2, decision, cls_leaf, cls_leaf, cls_leaf)
        # # create node without local decision
        # cls_node = @test_nowarn DTInternal(2, decision, cls_leaf, cls_leaf)
        # @test_throws MethodError DTInternal(2, decision, reg_leaf, cls_leaf)
        # # create node without frame
        # # @test_nowarn DTInternal(decision, reg_leaf, reg_leaf, reg_leaf)
        # # create node without frame nor local decision
        # cls_node = @test_nowarn DTInternal(decision, cls_node, cls_leaf)

        # cls_tree = @test_nowarn DTree(cls_node, [ModalLogic.Interval], [start_without_world])
        # cls_forest = @test_nowarn DForest([cls_tree, cls_tree, cls_tree])
    # end

end
