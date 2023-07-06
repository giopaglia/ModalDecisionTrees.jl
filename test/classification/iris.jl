using ModalDecisionTrees
using MLJ

################################################################################

X, y = @load_iris

w = abs.(randn(length(y)))
# w = fill(1, length(y))
# w = rand([1,2], length(y))
model = ModalDecisionTree()

mach = machine(model, X, y, w) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)

yhat = MLJ.predict(mach, Xnew)
yhat = MLJ.predict_mode(mach, Xnew)

yhat = MLJ.predict_mode(mach, X)

@test MLJBase.accuracy(y, yhat) > 0.8

@test_nowarn fitted_params(mach).model
@test_nowarn report(mach).solemodel

@test_nowarn printmodel(prune(fitted_params(mach).model, simplify=true, min_samples_leaf = 20), max_depth = 3)
@test_nowarn printmodel(prune(fitted_params(mach).model, simplify=true, min_samples_leaf = 20))

@test_nowarn printmodel(report(mach).solemodel, header = false)
@test_nowarn printmodel(report(mach).solemodel, header = :brief)
@test_nowarn printmodel(report(mach).solemodel, header = true)
@test_nowarn printmodel(report(mach).solemodel, show_subtree_info = true)

@test_nowarn printmodel.((SoleModels.listrules(report(mach).solemodel,)));

################################################################################
