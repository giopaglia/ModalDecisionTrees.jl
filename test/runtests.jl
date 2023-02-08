using Test

using ModalDecisionTrees
using MLJ
using SoleModels
using DataFrames

using Random
using CategoricalArrays
using StatsBase


include("load_data.jl")

println("Julia version: ", VERSION)

function run_tests(list)
    for test in list
        println("TEST: $test \n")
        include(test)
        println("=" ^ 50)
    end
end

test_suites = [
    ("Classification", [
        "classification/japanesevowels.jl",
        "classification/iris.jl",
        # "classification/demo-juliacon2022.jl",
        # 
        # "classification/random.jl",
        # "classification/low_precision.jl",
        # "classification/heterogeneous.jl",
        # "classification/digits.jl",
        # "classification/adult.jl",
        # "classification/scikitlearn.jl"
    ]),
    ("Regression", [
        "regression/ames.jl",
        "regression/digits-regression.jl",
        # "regression/random.jl",
        # "regression/low_precision.jl",
    ]),
    ("Miscellaneous", [
        # "miscellaneous/convert.jl"
        "translation/parse-and-translate-existentialdecisions.jl"
    ]),
]

@testset "Test Suites" begin
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
