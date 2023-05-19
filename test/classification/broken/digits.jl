@testset "digits.jl" begin

X, y = load_data("digits")

t = ModalDecisionTrees.build_tree(y, X)
@test length(t) == 148
@test sum(apply_tree(t, X) .== y) == length(y)


n_subfeatures       = 0
max_depth           = 6
min_samples_leaf    = 5
t = ModalDecisionTrees.build_tree(
        y, X,
        n_subfeatures, max_depth)
@test length(t) == 57

t = ModalDecisionTrees.build_tree(
        y, X,
        n_subfeatures, max_depth,
        min_samples_leaf)
@test length(t) == 50

min_samples_leaf    = 3
min_samples_split   = 5
min_purity_increase = 0.05
t = ModalDecisionTrees.build_tree(
        y, X,
        n_subfeatures, max_depth,
        min_samples_leaf,
        min_samples_split)
@test length(t) == 55

t = ModalDecisionTrees.build_tree(
        y, X,
        n_subfeatures, max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase)
@test length(t) == 54

# test that all purity decisions are based on passed-in purity function;
# if so, this should be same as previous test
entropy1000(ns, n) = ModalDecisionTrees.entropy(ns, n) * 1000
min_samples_leaf    = 3
min_samples_split   = 5
min_purity_increase = 0.05 * 1000
t = ModalDecisionTrees.build_tree(
        y, X,
        n_subfeatures, max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase,
        loss = entropy1000)
@test length(t) == 54

n_subfeatures       = 3
ntrees              = 10
partial_sampling    = 0.7
max_depth           = -1
min_samples_leaf    = 1
min_samples_split   = 2
min_purity_increase = 0.0
model = ModalDecisionTrees.build_forest(
        y, X,
        n_subfeatures,
        ntrees,
        partial_sampling,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase)
preds = apply_forest(model, X)
@test MLJBase.accuracy(y, preds) > 0.95

n_iterations        = 100
model, coeffs = ModalDecisionTrees.build_adaboost_stumps(
        y, X,
        n_iterations);
preds = apply_adaboost_stumps(model, coeffs, X);
@test MLJBase.accuracy(y, preds) > 0.8

end # @testset
