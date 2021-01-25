julia

using Pkg
Pkg.activate("DecisionTree.jl")
using Revise

using DecisionTree
using DecisionTree.ModalLogic

import Random
my_rng() = Random.MersenneTwister(1) # Random.GLOBAL_RNG

using Logging

using BenchmarkTools
using ScikitLearnBase
using Statistics
using Test
using Profile
using PProf

include("example-datasets.jl")

function testDataset((name,dataset), timeit::Bool = true; post_pruning_purity_thresholds = [], args = (), kwargs = ())
	println("Benchmarking dataset '$name'...")
	global_logger(ConsoleLogger(stderr, Logging.Warn));
	length(dataset) == 4 || error(length(dataset))
	X_train, Y_train, X_test, Y_test = dataset
	
	println("args = ", args)
	println("kwargs = ", kwargs)

	println(" n_samples = $(size(X_train)[end-1])")
	if timeit
		@btime build_tree($Y_train, $X_train, args...; kwargs..., rng = my_rng());
	end

	# global_logger(ConsoleLogger(stderr, Logging.Info))
	T = build_tree(Y_train, X_train, args...; kwargs..., rng = my_rng());

	if !timeit
		print(T)
	end

	for pruning_purity_threshold in sort(unique([(Float64.(post_pruning_purity_thresholds))...,1.0]))
		println(" Purity threshold $pruning_purity_threshold")
		
		global_logger(ConsoleLogger(stderr, Logging.Warn));

		T_pruned = prune_tree(T, pruning_purity_threshold)
		preds = apply_tree(T_pruned, X_test);
		cm = confusion_matrix(Y_test, preds)
		# @test cm.accuracy > 0.99

		print("  acc: ", round(cm.accuracy*100, digits=2), "% kappa: ", round(cm.kappa*100, digits=2), "% ")
		display(cm.matrix)

		global_logger(ConsoleLogger(stderr, Logging.Info));

		if timeit
			println("nodes: ($(num_nodes(T_pruned)), height: $(height(T_pruned)))")
		end
	end
	T
end

# testDatasets(d, timeit::Bool = true) = map((x)->testDataset(x, timeit), d);

datasets = Tuple{String,Tuple{Array,Array,Array,Array}}[
	# ("simpleDataset",traintestsplit(simpleDataset(200,n_variables = 50,rng = my_rng())...,0.8)),
	# ("simpleDataset2",traintestsplit(simpleDataset2(200,n_variables = 5,rng = my_rng())...,0.8)),
	# ("Eduard-5",EduardDataset(5)),
	# ("Eduard-10",EduardDataset(10)),
	("PaviaDataset, 1x1",traintestsplit(SampleLandCoverDataset(9*30, 1, "Pavia", rng = my_rng())...,0.8)),
	("PaviaDataset, 3x3 flattened",traintestsplit(SampleLandCoverDataset(9*30, 3, "Pavia", flattened = true, rng = my_rng())...,0.8)),
	("PaviaDataset, 3x3",traintestsplit(SampleLandCoverDataset(9*30, 3, "Pavia", rng = my_rng())...,0.8)),
	# ("PaviaDataset",traintestsplit(SampleLandCoverDataset(9*30, 5, "Pavia", rng = my_rng())...,0.8)),
	("IndianPinesCorrectedDataset, 1x1",traintestsplit(SampleLandCoverDataset(16*30, 1, "IndianPinesCorrected", rng = my_rng())...,0.8)),
	("IndianPinesCorrectedDataset, 3x3 flattened",traintestsplit(SampleLandCoverDataset(16*30, 3, flattened = true, "IndianPinesCorrected", rng = my_rng())...,0.8)),
	("IndianPinesCorrectedDataset, 3x3",traintestsplit(SampleLandCoverDataset(16*30, 3, "IndianPinesCorrected", rng = my_rng())...,0.8)),
	# ("PaviaDataset",traintestsplit(SampleLandCoverDataset(9*30, 7, "Pavia", n_variables = 5, rng = my_rng())...,0.8)),
];

args = (
	max_depth=-1,
	min_samples_leaf=4,
	min_samples_split=8,
	# min_purity_increase=0.02,
	# max_purity_split=1.0, # TODO there's something wrong here, I think this sets min_purity_increase.
)
kwargs = (
	initCondition=DecisionTree.startAtCenter,
)
T = testDataset(datasets[1], false, args=args, kwargs=kwargs)
T = testDataset(datasets[2], false, args=args, kwargs=kwargs)
T = testDataset(datasets[3], false, args=args, kwargs=kwargs)

# @profview T = testDataset(datasets[2], false)
# T = testDataset(datasets[1], false)
# @profile T = testDataset(datasets[1], false)
# pprof()
# Profile.print()
# testDatasets(datasets);

# X_train, Y_train, X_test, Y_test = traintestsplit(simpleDataset(200,n_variables = 50,rng = my_rng())...,0.8)
# model = fit!(DecisionTreeClassifier(pruning_purity_threshold=pruning_purity_threshold), X_train, Y_train)
# cm = confusion_matrix(Y_test, predict(model, X_test))
# @test cm.accuracy > 0.99
