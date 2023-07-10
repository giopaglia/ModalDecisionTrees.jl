@testset "digits-regression.jl" begin

include("../data/load.jl")

X, y = load_digits()
Xcube = cat(map(r->reshape(r, (8,8,1)), eachrow(X))...; dims=4)
Xnt = NamedTuple(zip(Symbol.(1:length(eachslice(Xcube; dims=3))), eachslice.(eachslice(Xcube; dims=3); dims=3)))
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

X_traincube = cat(map(r->reshape(r, (8,8,1)), eachrow(X_train))...; dims=4)
X_trainnt = NamedTuple(zip(Symbol.(1:length(eachslice(X_traincube; dims=3))), eachslice.(eachslice(X_traincube; dims=3); dims=3)))

X_testcube = cat(map(r->reshape(r, (8,8,1)), eachrow(X_test))...; dims=4)
X_testnt = NamedTuple(zip(Symbol.(1:length(eachslice(X_testcube; dims=3))), eachslice.(eachslice(X_testcube; dims=3); dims=3)))

model = ModalDecisionTree(min_purity_increase = 0.001)

mach = machine(model, X_trainnt, y_train) |> fit!

@test StatsBase.cor(MLJ.predict_mean(mach, X_testnt), y_test) > 0.45


model = ModalRandomForest()

mach = machine(model, X_trainnt, y_train) |> fit!

@test StatsBase.cor(MLJ.predict_mean(mach, X_testnt), y_test) > 0.5


mach = machine(ModalRandomForest(;
    n_subfeatures       = 3,
    ntrees              = 10,
    sampling_fraction   = 0.7,
    max_depth           = -1,
    min_samples_leaf    = 1,
    min_samples_split   = 2,
    min_purity_increase = 0.0,
), X, y) |> m->fit!(m, rows = train_idxs)

println(StatsBase.cor(MLJ.predict_mean(mach, X_testnt), y_test))
@test StatsBase.cor(MLJ.predict_mean(mach, X_testnt), y_test) > 0.5

mach = machine(ModalRandomForest(;
    n_subfeatures       = 3,
    ntrees              = 10,
    sampling_fraction   = 0.7,
    max_depth           = -1,
    min_samples_leaf    = 1,
    min_samples_split   = 2,
    min_purity_increase = 0.0,
), Xnt, y) |> m->fit!(m, rows = train_idxs)

println(StatsBase.cor(MLJ.predict_mean(mach, X_testnt), y_test))
@test StatsBase.cor(MLJ.predict_mean(mach, X_testnt), y_test) > 0.5

# using Plots
# p = sortperm(y_test)
# scatter(y_test[p], label = "y")
# scatter!(yhat[p], label = "ŷ")
# k = 20
# plot!([mean(yhat[p][i:i+k]) for i in 1:length(yhat[p])-k], label = "ŷ, moving average")

end
