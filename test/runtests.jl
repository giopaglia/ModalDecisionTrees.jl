using ModalDecisionTrees
using DelimitedFiles
using Random
using ScikitLearnBase
using StatsBase
using Test

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
        # "classification/random.jl",
        # "classification/low_precision.jl",
        # "classification/heterogeneous.jl",
        # "classification/digits.jl",
        # "classification/iris.jl",
        # "classification/adult.jl",
        # "classification/scikitlearn.jl"
    ]),
    ("Regression", [
        # "regression/random.jl",
        # "regression/low_precision.jl",
        # "regression/digits.jl",
        # "regression/scikitlearn.jl"
    ]),
    ("Miscellaneous", [
        # "miscellaneous/convert.jl"
        #    "miscellaneous/parallel.jl"
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
