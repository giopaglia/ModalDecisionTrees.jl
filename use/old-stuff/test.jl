# julia-1.5.4
# julia

include("runner.jl")

# exec_runs(d, timing_mode::Bool = true) = map((x)->exec_run(x, timing_mode), d);

rng = my_rng()

# run(`say 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'`)

# @profview T = exec_run(datasets[2], false)
# T = exec_run(datasets[1], false)
# @profile T = exec_run(datasets[1], false)
# pprof()
# Profile.print()
# exec_runs(datasets);

# X_train, Y_train, X_test, Y_test = traintestsplit(simpleDataset(200,n_attributes = 50,rng = my_rng),0.8)
# model = fit!(DecisionTreeClassifier(pruning_purity_threshold=pruning_purity_threshold), X_train, Y_train)
# cm = confusion_matrix(Y_test, predict(model, X_test))
# @test overall_accuracy(cm) > 0.99

# for relations in [ModalLogic.RCC8Relations, ModalLogic.IA2DRelations]
# 	for (X,Y) in Iterators.product(4:6,4:9)
# 		sum = 0
# 		for rel in relations
# 			sum += (ModalLogic.enumAccessibles(S, rel, X,Y) |> collect |> length)
# 			end
# 		# println(X, " ", Y, " ", (X*(X+1))/2 * (Y*(Y+1))/2 - 1, " ", sum)
# 		@assert sum == ((X*(X+1))/2 * (Y*(Y+1))/2 - 1)
# 	end
# 	for (X,Y) in Iterators.product(4:6,4:9)
# 		sum = 0
# 		for rel in relations
# 			sum += (ModalLogic.enumAccessibles(S, rel, X,Y) |> distinct |> collect |> length)
# 			end
# 		# println(X, " ", Y, " ", (X*(X+1))/2 * (Y*(Y+1))/2 - 1, " ", sum)
# 		@assert sum == ((X*(X+1))/2 * (Y*(Y+1))/2 - 1)
# 	end
# end

# S = [ModalLogic.Interval2D((2,3),(3,4))]
# S = [ModalLogic.Interval2D((2,4),(2,4))]
S = [ModalLogic.Interval2D((2,3),(2,3))]
relations = ModalLogic.RCC8Relations
(X,Y) = (3,3)
SUM = 0
for rel in relations
	println(rel)
	map(ModalLogic.print_world, ModalLogic.enumAccessibles(S, rel, X,Y) |> collect)
	global SUM
	SUM += (ModalLogic.enumAccessibles(S, rel, X,Y) |> collect |> length)
end
# println(X, " ", Y, " ", (X*(X+1))/2 * (Y*(Y+1))/2 - 1, " ", sum)
@assert SUM == ((X*(X+1))/2 * (Y*(Y+1))/2 - 1)

# Test that T = exec_run(datasets[1], timing_mode, args=args, kwargs=kwargs); with test_operators=[TestOpLeq] and without is equivalent
