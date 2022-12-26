@testset "ames.jl" begin

X, y = MLJ.@load_ames

# Only consider non-categorical variables
mask = BitVector((!).((<:).(eltype.(values(X)), CategoricalValue)))

# X = filter(((i,c),)->(i in mask), collect(enumerate(X))[mask])
X = NamedTuple(zip(keys(X)[mask], values(X)[mask]))
X = DataFrame(X)

X = Float64.(X)

p = randperm(Random.MersenneTwister(1), 100)
X, y = X[p, :], y[p]

n_instances = size(X, 1)
n_train = Int(floor(n_instances*.8))
p = randperm(Random.MersenneTwister(1), n_instances)
train_idxs = p[1:n_train]
test_idxs = p[n_train+1:end]

X_train, y_train = X[train_idxs,:], y[train_idxs]
X_test, y_test = X[test_idxs,:], y[test_idxs]

model = ModalDecisionTree(min_purity_increase = 0.001)

mach = machine(model, X_train, y_train) |> fit!

yhat = MLJ.predict(mach, X_test)

m = SoleModels.compute_metrics(yhat, y_test)

@test m.cor > 0.6

model = ModalRandomForest(n_trees = 15)

mach = machine(model, X_train, y_train) |> fit!

yhat = MLJ.predict_mean(mach, X_test)

m = SoleModels.compute_metrics(yhat, y_test)

@test m.cor > 0.7

# using Plots
# p = sortperm(y_test)
# scatter(y_test[p], label = "y")
# scatter!(yhat[p], label = "ŷ")
# k = 20
# plot!([mean(yhat[p][i:i+k]) for i in 1:length(yhat[p])-k], label = "ŷ, moving average")

end
