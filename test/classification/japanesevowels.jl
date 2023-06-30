@testset "japanesevowels.jl" begin

# Import packages
using MLJ
using ModalDecisionTrees
using SoleModels
using Random

# A Modal Decision Tree with ≥ 4 samples at leaf
t = ModalDecisionTree(min_samples_leaf=4)

# Load an example dataset (a temporal one)
X, y = ModalDecisionTrees.@load_japanesevowels

p = randperm(Random.MersenneTwister(2), 100)
X, y = X[:, :, p], y[p]

X = NamedTuple(zip(Symbol.(1:length(eachslice(X; dims=2))), eachslice.(eachslice(X; dims=2); dims=2)))


N = length(y)

mach = machine(t, X, y)

# Split dataset
p = randperm(Random.MersenneTwister(1), N)
train_idxs, test_idxs = p[1:round(Int, N*.8)], p[round(Int, N*.8)+1:end]

# Fit
MLJ.fit!(mach, rows=train_idxs)

# Perform predictions, compute accuracy
yhat = MLJ.predict(mach, rows=test_idxs)
accuracy = sum(yhat .== y[test_idxs])/length(yhat)

@test accuracy >= 0.6

# Access raw model
fitted_params(mach).model;
report(mach).printmodel(3);

X, y = ModalDecisionTrees.@load_japanesevowels

mach = machine(t, X, y)

# Fit
MLJ.fit!(mach)

end
