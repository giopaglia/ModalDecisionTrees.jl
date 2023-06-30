@testset "iris.jl" begin

# Import ModalDecisionTrees.jl & MLJ
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

yhat = MLJ.predict(mach, X)

@test MLJBase.accuracy(y, yhat) > 0.8

################################################################################

# Import ModalDecisionTrees.jl & MLJ
using ModalDecisionTrees
using MLJ

X, y = @load_iris

model = ModalDecisionTree(; max_depth = 0)
mach = machine(model, X, y) |> fit!
@test height(fitted_params(mach).model) == 0
@test depth(fitted_params(mach).model) == 0

model = ModalDecisionTree(; max_depth = 2)
mach = machine(model, X, y) |> fit!
@test depth(fitted_params(mach).model) == 2

# fitted_params(mach).model
# report(mach).solemodel

model = ModalDecisionTree()
mach = machine(model, X, y) |> fit!
@test depth(fitted_params(mach).model) == 2

model = ModalRandomForest()

mach = machine(model, X, y, w) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)
yhat = MLJ.predict(mach, Xnew)

yhat = MLJ.predict(mach, X)

@test MLJBase.accuracy(y, yhat) > 0.8


end
