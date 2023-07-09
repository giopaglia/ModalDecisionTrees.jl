using Test

using ModalDecisionTrees
using MLJ
using MLJBase
using SoleModels
using SoleModels.DimensionalDatasets
using DataFrames

using Random
using CategoricalArrays
using StatsBase

using ModalDecisionTrees: build_stump, build_tree, build_forest

println("Julia version: ", VERSION)

function run_tests(list)
    println("\n" * ("#"^50))
    for test in list
        println("TEST: $test")
        include(test)
        println("=" ^ 50)
    end
end

test_suites = [
    ("Base", ["base.jl"]),
    ("Classification", [
        "classification/iris.jl",
        "classification/iris-params.jl",
        "classification/japanesevowels.jl",
        "classification/digits.jl",
        "classification/mnist.jl",
        # "classification/demo-juliacon2022.jl",
        # 
        # "classification/random.jl",
        # "classification/low_precision.jl",
        # "classification/heterogeneous.jl",
        # "classification/scikitlearn.jl"
    ]),
    ("Regression", [
        # "regression/simple.jl",
        "regression/ames.jl",
        "regression/digits-regression.jl",
        # "regression/random.jl",
        # "regression/low_precision.jl",
    ]),
    ("Miscellaneous", [
        "multimodal-datasets.jl"
        "multiformulas-construction.jl"
        # "miscellaneous/convert.jl"
        "translation/parse-and-translate-existentialdecisions.jl"
    ]),
]

@testset "ModalDecisionTrees.jl" begin
    for ts in 1:length(test_suites)
        name = test_suites[ts][1]
        list = test_suites[ts][2]
        let
            @testset "$name" begin
                run_tests(list)
            end
        end
    end
end
