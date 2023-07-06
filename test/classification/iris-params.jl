
using ModalDecisionTrees
using MLJ
using Random

X, y = @load_iris

model = ModalDecisionTree(; max_depth = 0)
mach = machine(model, X, y) |> fit!
@test height(fitted_params(mach).model) == 0
@test depth(fitted_params(mach).model) == 0

model = ModalDecisionTree(; max_depth = 2, )
mach = machine(model, X, y) |> fit!
@test depth(fitted_params(mach).model) == 2

model = ModalDecisionTree(;
	min_samples_leaf = 2,
	min_samples_split = 4,
	min_purity_increase = 0.1,
	max_purity_at_leaf = 1.0,
	print_progress = true,
	rng = 2
)
mach = machine(model, X, y) |> fit!
@test depth(fitted_params(mach).tree) == 4

################################################################################

using ModalDecisionTrees
using MLJ
using Random

X, y = @load_iris

model = ModalDecisionTree(;
	max_purity_at_leaf = 1.0,
	print_progress = true,
	display_depth = 1,
	rng = Random.MersenneTwister(2)
)
mach = machine(model, X, y) |> fit!
report(mach).printmodel()
################################################################################

using ModalDecisionTrees
using MLJ
using Random

X, y = @load_iris

model = ModalDecisionTree(;
	max_purity_at_leaf = 1.0,
	print_progress = true,
	max_modal_depth = 2,
	# display_depth = nothing,
	display_depth = 2,
	rng = Random.MersenneTwister(2)
)
mach = machine(model, X, y) |> fit!
@test depth(fitted_params(mach).tree) == 4
@test_nowarn report(mach).printmodel()
@test_nowarn report(mach).printmodel(false, 0)
@test_nowarn report(mach).printmodel(true, 0)
@test_nowarn report(mach).printmodel(false, 2)
@test_nowarn report(mach).printmodel(true, 2)
@test_nowarn report(mach).printmodel(0)
@test_nowarn report(mach).printmodel(1)
@test_nowarn report(mach).printmodel(4)
@test_nowarn report(mach).printmodel(10)

report(mach).printmodel(; hidemodality=false)
report(mach).printmodel(hidemodality=false)


@test_nowarn report(mach).printmodel(show_metrics = true)
@test_nowarn report(mach).printmodel(show_intermediate_finals = true)
@test_nowarn report(mach).printmodel(show_metrics = true, show_intermediate_finals = true)
@test_nowarn report(mach).printmodel(show_metrics = true, show_intermediate_finals = true, max_depth=nothing)
@test_nowarn report(mach).printmodel(show_metrics = (;), show_intermediate_finals = 200, max_depth=nothing)
printmodel.(listrules(report(mach).solemodel); show_metrics=true);

out1 = (io = IOBuffer(); report(mach).printmodel(io, true); String(take!(io)))
out2 = (io = IOBuffer(); report(mach).printmodel(io, false); String(take!(io)))

@test occursin("petal", out1)
@test occursin("petal", out2)
# @test occursin("petal", displaymodel(report(mach).solemodel))

@test_nowarn listrules(report(mach).solemodel)
@test_nowarn listrules(report(mach).solemodel; use_shortforms=true)
@test_nowarn listrules(report(mach).solemodel; use_shortforms=false)
@test_nowarn listrules(report(mach).solemodel; use_shortforms=true, use_leftmostlinearform = true)
@test_nowarn listrules(report(mach).solemodel; use_shortforms=false, use_leftmostlinearform = true)
@test_throws ErrorException listrules(report(mach).solemodel; use_shortforms=false, use_leftmostlinearform = true, force_syntaxtree = true)


@test_nowarn report(mach).printmodel(true, 3; syntaxstring_kwargs = (;hidemodality = true))

model = ModalRandomForest()

mach = machine(model, X, y, w) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
		sepal_width = [2.8, 3.0, 2.8],
		petal_length = [5.6, 5.8, 6.1],
		petal_width = [2.1, 1.6, 1.9],)
yhat = MLJ.predict(mach, Xnew)

yhat = MLJ.predict(mach, X)

@test MLJBase.accuracy(y, yhat) > 0.8

