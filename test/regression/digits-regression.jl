@testset "digits-regression.jl" begin

X, y = load_data("digits")

y = float.(y)

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

@test StatsBase.cor(yhat, y_test) > 0.45


model = ModalRandomForest()

mach = machine(model, X_train, y_train) |> fit!

yhat = MLJ.predict(mach, X_test)

@test StatsBase.cor(yhat, y_test) > 0.5

# using Plots
# p = sortperm(y_test)
# scatter(y_test[p], label = "y")
# scatter!(yhat[p], label = "ŷ")
# k = 20
# plot!([mean(yhat[p][i:i+k]) for i in 1:length(yhat[p])-k], label = "ŷ, moving average")

end
