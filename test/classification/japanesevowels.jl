# Import packages
using Test
using MLJ
using ModalDecisionTrees
using SoleModels
using Random

# A Modal Decision Tree with ≥ 4 samples at leaf
t = ModalDecisionTree(;
    min_samples_split=2,
    min_samples_leaf = 4,
)

# Load an example dataset (a temporal one)
X, y = ModalDecisionTrees.@load_japanesevowels

p = randperm(Random.MersenneTwister(2), 100)
X, y = X[:, :, p], y[p]

X = NamedTuple(zip(Symbol.(1:length(eachslice(X; dims=2))), eachslice.(eachslice(X; dims=2); dims=2)))

nvars = length(X)
N = length(y)

mach = machine(t, X, y)

# Split dataset
p = randperm(Random.MersenneTwister(1), N)
train_idxs, test_idxs = p[1:round(Int, N*.8)], p[round(Int, N*.8)+1:end]

# Fit
MLJ.fit!(mach, rows=train_idxs)

# Perform predictions, compute accuracy
yhat = MLJ.predict(mach, rows=test_idxs)
acc = sum(mode.(yhat) .== y[test_idxs])/length(yhat)
yhat = MLJ.predict_mode(mach, rows=test_idxs)
acc = sum(yhat .== y[test_idxs])/length(yhat)

@test acc >= 0.8

@test_nowarn report(mach).printmodel(syntaxstring_kwargs = (; variable_names_map = [('A':('A'+nvars))], threshold_digits = 2))

@test_throws BoundsError report(mach).printmodel(syntaxstring_kwargs = (; variable_names_map = [["a", "b"]]))
@test_throws BoundsError report(mach).printmodel(syntaxstring_kwargs = (; variable_names_map = ["a", "b"]))
@test_logs (:warn,) report(mach).printmodel(syntaxstring_kwargs = (; variable_names_map = 'A':('A'+nvars)))

@test_nowarn printmodel(report(mach).solemodel)
@test_nowarn listrules(report(mach).solemodel)
@test_nowarn listrules(report(mach).solemodel; use_shortforms=true)
@test_nowarn listrules(report(mach).solemodel; use_shortforms=false)
@test_nowarn listrules(report(mach).solemodel; use_shortforms=true, use_leftmostlinearform = true)
@test_nowarn listrules(report(mach).solemodel; use_shortforms=false, use_leftmostlinearform = true)
@test_throws ErrorException listrules(report(mach).solemodel; use_shortforms=false, use_leftmostlinearform = true, force_syntaxtree = true)

# Access raw model
fitted_params(mach).model;
report(mach).printmodel(3);

MLJ.fit!(mach)

@test_nowarn feature_importances(mach)

############################################################################################
############################################################################################
############################################################################################

mach = machine(ModalDecisionTree(post_prune = true), X, y) |> MLJ.fit!
mach = machine(ModalDecisionTree(post_prune = true, max_modal_depth = 2), X, y) |> MLJ.fit!
mach = machine(ModalDecisionTree(min_samples_split=100, post_prune = true, merge_purity_threshold = 0.4), X, y) |> MLJ.fit!

############################################################################################
############################################################################################
############################################################################################

# NaNs
Xwithnans = deepcopy(X)
for i in 1:4
    rng = MersenneTwister(i)
    c = rand(rng, 1:length(Xwithnans))
    r = rand(rng, 1:length(Xwithnans[c]))
    Xwithnans[c][r][rand(1:length(Xwithnans[c][r]))] = NaN
    @test_throws ErrorException machine(ModalDecisionTree(), Xwithnans, y) |> MLJ.fit!
end

############################################################################################
############################################################################################
############################################################################################

X, y = ModalDecisionTrees.@load_japanesevowels

multilogiset, var_grouping = ModalDecisionTrees.MLJInterface.wrapdataset(X, ModalDecisionTree(; min_samples_leaf = 1))

# A Modal Decision Tree
t = ModalDecisionTree(min_samples_split=100, post_prune = true, merge_purity_threshold = true)

@test_broken mach = machine(t, multilogiset, y)


N = length(y)

p = randperm(Random.MersenneTwister(1), N)
train_idxs, test_idxs = p[1:round(Int, N*.8)], p[round(Int, N*.8)+1:end]

# Fit
MLJ.fit!(mach, rows=train_idxs)

yhat = MLJ.predict_mode(mach, rows=test_idxs)
acc = sum(yhat .== y[test_idxs])/length(yhat)
MLJ.kappa(yhat, y[test_idxs])

@test_nowarn prune(fitted_params(mach).model, simplify=true)
@test_nowarn prune(fitted_params(mach).model, simplify=true, min_samples_leaf = 20)
