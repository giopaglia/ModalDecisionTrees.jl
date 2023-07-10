using MLJBase
using StatsBase
rng = Random.MersenneTwister(1)
X, y = make_regression(100, 2; rng = rng)

# model = ModalDecisionTree(min_purity_increase = 0.001)
model = ModalDecisionTree()

mach = machine(model, X, y)

train_idxs = 1:div(length(y), 2)
test_idxs = div(length(y), 2)+1:100

MLJBase.fit!(mach, rows=train_idxs)
ypreds = MLJBase.predict_mean(mach, rows=test_idxs)

println(StatsBase.cor(ypreds, y[test_idxs]))
@test StatsBase.cor(ypreds, y[test_idxs]) > 0.45
