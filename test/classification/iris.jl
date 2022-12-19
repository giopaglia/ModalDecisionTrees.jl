@testset "iris.jl" begin

# Import ModalDecisionTrees.jl & MLJ
using ModalDecisionTrees
using SoleModels
using SoleModels: ConfusionMatrix
using MLJ

################################################################################

X, y = @load_iris

w = abs.(randn(length(y)))
# w = fill(1, length(y))
# w = rand([1,2], length(y))
model = ModalDecisionTree()
# model = ModalRandomForest()

mach = machine(model, X, y, w) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)
yhat = MLJ.predict(mach, Xnew) # probabilistic predictions

yhat = MLJ.predict(mach, X)

cm = ConfusionMatrix(Vector{String}(y), yhat);
@test overall_accuracy(cm) > 0.8

end
