# Inspired from JuliaAI/MLJDecisionTreeInterface.jl

module MLJDecisionTreeInterface

import MLJModelInterface
using MLJModelInterface.ScientificTypesBase
import DecisionTree
import Tables

using Random
import Random.GLOBAL_RNG

const MMI = MLJModelInterface
const DT = DecisionTree
const PKG = "MLJDecisionTreeInterface"

struct TreePrinter{T}
    tree::T
end
(c::TreePrinter)(depth) = DT.print_tree(c.tree, depth)
(c::TreePrinter)() = DT.print_tree(c.tree, 5)

Base.show(stream::IO, c::TreePrinter) =
    print(stream, "TreePrinter object (call with display depth)")


# # DECISION TREE CLASSIFIER

# The following meets the MLJ standard for a `Model` docstring and is
# created without the use of interpolation so it can be used a
# template for authors of other MLJ model interfaces. The other
# doc-strings, defined later, are generated using the `doc_header`
# utility to automatically generate the header, another option.

MMI.@mlj_model mutable struct DecisionTreeClassifier <: MMI.Probabilistic
    max_depth::Int               = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int        = 1::(_ ≥ 0)
    min_samples_split::Int       = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int           = 0::(_ ≥ -1)
    post_prune::Bool             = false
    merge_purity_threshold::Float64 = 1.0::(_ ≤ 1)
    display_depth::Int           = 5::(_ ≥ 1)
    rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
end

function MMI.fit(m::DecisionTreeClassifier, verbosity::Int, X, y)
    schema = Tables.schema(X)
    Xmatrix = MMI.matrix(X)
    yplain  = MMI.int(y)

    if schema === nothing
        features = [Symbol("x$j") for j in 1:size(Xmatrix, 2)]
    else
        features = schema.names |> collect
    end

    classes_seen  = filter(in(unique(y)), MMI.classes(y[1]))
    integers_seen = MMI.int(classes_seen)

    tree = DT.build_tree(yplain, Xmatrix,
                         m.n_subfeatures,
                         m.max_depth,
                         m.min_samples_leaf,
                         m.min_samples_split,
                         m.min_purity_increase,
                         rng=m.rng)
    if m.post_prune
        tree = DT.prune_tree(tree, m.merge_purity_threshold)
    end
    verbosity < 2 || DT.print_tree(tree, m.display_depth)

    fitresult = (tree, classes_seen, integers_seen, features)

    cache  = nothing
    report = (classes_seen=classes_seen,
              print_tree=TreePrinter(tree),
              features=features)

    return fitresult, cache, report
end

function get_encoding(classes_seen)
    a_cat_element = classes_seen[1]
    return Dict(c => MMI.int(c) for c in MMI.classes(a_cat_element))
end

MMI.fitted_params(::DecisionTreeClassifier, fitresult) =
    (tree=fitresult[1],
     encoding=get_encoding(fitresult[2]),
     features=fitresult[4])

function smooth(scores, smoothing)
    iszero(smoothing) && return scores
    threshold = smoothing / size(scores, 2)
    # clip low values
    scores[scores .< threshold] .= threshold
    # normalize
    return scores ./ sum(scores, dims=2)
end

function MMI.predict(m::DecisionTreeClassifier, fitresult, Xnew)
    Xmatrix = MMI.matrix(Xnew)
    tree, classes_seen, integers_seen = fitresult
    # retrieve the predicted scores
    scores = DT.apply_tree_proba(tree, Xmatrix, integers_seen)

    # return vector of UF
    return MMI.UnivariateFinite(classes_seen, scores)
end


# # RANDOM FOREST CLASSIFIER

MMI.@mlj_model mutable struct RandomForestClassifier <: MMI.Probabilistic
    max_depth::Int               = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int        = 1::(_ ≥ 0)
    min_samples_split::Int       = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int           = (-)(1)::(_ ≥ -1)
    n_trees::Int                 = 10::(_ ≥ 2)
    sampling_fraction::Float64   = 0.7::(0 < _ ≤ 1)
    rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
end

function MMI.fit(m::RandomForestClassifier, verbosity::Int, X, y)
    Xmatrix = MMI.matrix(X)
    yplain  = MMI.int(y)

    classes_seen  = filter(in(unique(y)), MMI.classes(y[1]))
    integers_seen = MMI.int(classes_seen)

    forest = DT.build_forest(yplain, Xmatrix,
                             m.n_subfeatures,
                             m.n_trees,
                             m.sampling_fraction,
                             m.max_depth,
                             m.min_samples_leaf,
                             m.min_samples_split,
                             m.min_purity_increase;
                             rng=m.rng)
    cache  = nothing
    report = NamedTuple()
    return (forest, classes_seen, integers_seen), cache, report
end

MMI.fitted_params(::RandomForestClassifier, (forest,_)) = (forest=forest,)

function MMI.predict(m::RandomForestClassifier, fitresult, Xnew)
    Xmatrix = MMI.matrix(Xnew)
    forest, classes_seen, integers_seen = fitresult
    scores = DT.apply_forest_proba(forest, Xmatrix, integers_seen)
    return MMI.UnivariateFinite(classes_seen, scores)
end


# # ADA BOOST STUMP CLASSIFIER

MMI.@mlj_model mutable struct AdaBoostStumpClassifier <: MMI.Probabilistic
    n_iter::Int            = 10::(_ ≥ 1)
end

function MMI.fit(m::AdaBoostStumpClassifier, verbosity::Int, X, y)
    Xmatrix = MMI.matrix(X)
    yplain  = MMI.int(y)

    classes_seen  = filter(in(unique(y)), MMI.classes(y[1]))
    integers_seen = MMI.int(classes_seen)

    stumps, coefs = DT.build_adaboost_stumps(yplain, Xmatrix,
                                             m.n_iter)
    cache  = nothing
    report = NamedTuple()
    return (stumps, coefs, classes_seen, integers_seen), cache, report
end

MMI.fitted_params(::AdaBoostStumpClassifier, (stumps,coefs,_)) =
    (stumps=stumps,coefs=coefs)

function MMI.predict(m::AdaBoostStumpClassifier, fitresult, Xnew)
    Xmatrix = MMI.matrix(Xnew)
    stumps, coefs, classes_seen, integers_seen = fitresult
    scores = DT.apply_adaboost_stumps_proba(stumps, coefs,
                                            Xmatrix, integers_seen)
    return MMI.UnivariateFinite(classes_seen, scores)
end


# # DECISION TREE REGRESSOR

MMI.@mlj_model mutable struct DecisionTreeRegressor <: MMI.Deterministic
    max_depth::Int                               = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int                = 5::(_ ≥ 0)
    min_samples_split::Int               = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int                   = 0::(_ ≥ -1)
    post_prune::Bool                     = false
    merge_purity_threshold::Float64 = 1.0::(0 ≤ _ ≤ 1)
    rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
end

function MMI.fit(m::DecisionTreeRegressor, verbosity::Int, X, y)
    Xmatrix = MMI.matrix(X)
    tree    = DT.build_tree(float(y), Xmatrix,
                            m.n_subfeatures,
                            m.max_depth,
                            m.min_samples_leaf,
                            m.min_samples_split,
                            m.min_purity_increase;
                            rng=m.rng)

    if m.post_prune
        tree = DT.prune_tree(tree, m.merge_purity_threshold)
    end
    cache  = nothing
    report = NamedTuple()
    return tree, cache, report
end

MMI.fitted_params(::DecisionTreeRegressor, tree) = (tree=tree,)

function MMI.predict(::DecisionTreeRegressor, tree, Xnew)
    Xmatrix = MMI.matrix(Xnew)
    return DT.apply_tree(tree, Xmatrix)
end


# # RANDOM FOREST REGRESSOR

MMI.@mlj_model mutable struct RandomForestRegressor <: MMI.Deterministic
    max_depth::Int               = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int        = 1::(_ ≥ 0)
    min_samples_split::Int       = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int           = (-)(1)::(_ ≥ -1)
    n_trees::Int                 = 10::(_ ≥ 2)
    sampling_fraction::Float64   = 0.7::(0 < _ ≤ 1)
    rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
end

function MMI.fit(m::RandomForestRegressor, verbosity::Int, X, y)
    Xmatrix = MMI.matrix(X)
    forest  = DT.build_forest(float(y), Xmatrix,
                              m.n_subfeatures,
                              m.n_trees,
                              m.sampling_fraction,
                              m.max_depth,
                              m.min_samples_leaf,
                              m.min_samples_split,
                              m.min_purity_increase,
                              rng=m.rng)
    cache  = nothing
    report = NamedTuple()
    return forest, cache, report
end

MMI.fitted_params(::RandomForestRegressor, forest) = (forest=forest,)

function MMI.predict(::RandomForestRegressor, forest, Xnew)
    Xmatrix = MMI.matrix(Xnew)
    return DT.apply_forest(forest, Xmatrix)
end


# # METADATA (MODEL TRAITS)

# following five lines of code are redundant if using this branch of
# MLJModelInterface:
# https://github.com/JuliaAI/MLJModelInterface.jl/pull/139

# MMI.human_name(::Type{<:DecisionTreeClassifier}) = "CART decision tree classifier"
# MMI.human_name(::Type{<:RandomForestClassifier}) = "CART random forest classifier"
# MMI.human_name(::Type{<:AdaBoostStumpClassifier}) = "Ada-boosted stump classifier"
# MMI.human_name(::Type{<:DecisionTreeRegressor}) = "CART decision tree regressor"
# MMI.human_name(::Type{<:RandomForestRegressor}) = "CART random forest regressor"

MMI.metadata_pkg.(
    (DecisionTreeClassifier, DecisionTreeRegressor,
     RandomForestClassifier, RandomForestRegressor,
     AdaBoostStumpClassifier),
    name = "DecisionTree",
    package_uuid = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb",
    package_url = "https://github.com/bensadeghi/DecisionTree.jl",
    is_pure_julia = true,
    package_license = "MIT"
)

MMI.metadata_model(
    DecisionTreeClassifier,
    input_scitype = Table(Continuous, Count, OrderedFactor),
    target_scitype = AbstractVector{<:Finite},
    human_name = "CART decision tree classifier",
    load_path = "$PKG.DecisionTreeClassifier"
)

MMI.metadata_model(
    RandomForestClassifier,
    input_scitype = Table(Continuous, Count, OrderedFactor),
    target_scitype = AbstractVector{<:Finite},
    human_name = "CART random forest classifier",
    load_path = "$PKG.RandomForestClassifier"
)

MMI.metadata_model(
    AdaBoostStumpClassifier,
    input_scitype = Table(Continuous, Count, OrderedFactor),
    target_scitype = AbstractVector{<:Finite},
    human_name = "Ada-boosted stump classifier",
    load_path = "$PKG.AdaBoostStumpClassifier"
)

MMI.metadata_model(
    DecisionTreeRegressor,
    input_scitype = Table(Continuous, Count, OrderedFactor),
    target_scitype = AbstractVector{Continuous},
    human_name = "CART decision tree regressor",
    load_path = "$PKG.DecisionTreeRegressor"
)

MMI.metadata_model(
    RandomForestRegressor,
    input_scitype = Table(Continuous, Count, OrderedFactor),
    target_scitype = AbstractVector{Continuous},
    human_name = "CART random forest regressor",
    load_path = "$PKG.RandomForestRegressor")


# # DOCUMENT STRINGS

const DOC_CART = "[CART algorithm](https://en.wikipedia.org/wiki/Decision_tree_learning)"*
", originally published in Breiman, Leo; Friedman, J. H.; Olshen, R. A.; "*
"Stone, C. J. (1984): \"Classification and regression trees\". *Monterey, "*
"CA: Wadsworth & Brooks/Cole Advanced Books & Software.*"

const DOC_RANDOM_FOREST = "[Random Forest algorithm]"*
    "(https://en.wikipedia.org/wiki/Random_forest), originally published in "*
    "Breiman, L. (2001): \"Random Forests.\", *Machine Learning*, vol. 45, pp. 5–32"

"""
$(MMI.doc_header(DecisionTreeClassifier))
`DecisionTreeClassifier` implements the $DOC_CART.
# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)
where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`
Train the machine using `fit!(mach, rows=...)`.
# Hyper-parameters
- `max_depth=-1`:          max depth of the decision tree (-1=any)
- `min_samples_leaf=1`:    max number of samples each leaf needs to have
- `min_samples_split=2`:   min number of samples needed for a split
- `min_purity_increase=0`: min purity needed for a split
- `n_subfeatures=0`: number of features to select at random (0 for all,
  -1 for square root of number of features)
- `post_prune=false`:      set to `true` for post-fit pruning
- `merge_purity_threshold=1.0`: (post-pruning) merge leaves having
                           combined purity `>= merge_purity_threshold`
- `display_depth=5`:       max depth to show when displaying the tree
- `rng=Random.GLOBAL_RNG`: random number generator or seed
# Operations
- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are probabilistic, but uncalibrated.
- `predict_mode(mach, Xnew)`: instead return the mode of each
  prediction above.
# Fitted parameters
The fields of `fitted_params(mach)` are:
- `tree`: the tree or stump object returned by the core DecisionTree.jl algorithm
- `encoding`: dictionary of target classes keyed on integers used
  internally by DecisionTree.jl; needed to interpret pretty printing
  of tree (obtained by calling `fit!(mach, verbosity=2)` or from
  report - see below)
- `features`: the names of the features encountered in training, in an
  order consistent with the output of `print_tree` (see below)
# Report
The fields of `report(mach)` are:
- `classes_seen`: list of target classes actually observed in training
- `print_tree`: method to print a pretty representation of the fitted
  tree, with single argument the tree depth; interpretation requires
  internal integer-class encoding (see "Fitted parameters" above).
- `features`: the names of the features encountered in training, in an
  order consistent with the output of `print_tree` (see below)
# Examples
```
using MLJ
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree(max_depth=4, min_samples_split=3)
X, y = @load_iris
mach = machine(tree, X, y) |> fit!
Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)
yhat = predict(mach, Xnew) # probabilistic predictions
predict_mode(mach, Xnew)   # point predictions
pdf.(yhat, "virginica")    # probabilities for the "verginica" class
fitted_params(mach).tree # raw tree or stump object from DecisionTrees.jl
julia> report(mach).print_tree(3)
Feature 4, Threshold 0.8
L-> 1 : 50/50
R-> Feature 4, Threshold 1.75
    L-> Feature 3, Threshold 4.95
        L->
        R->
    R-> Feature 3, Threshold 4.85
        L->
        R-> 3 : 43/43
```
To interpret the internal class labelling:
```
julia> fitted_params(mach).encoding
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, UInt32} with 3 entries:
  "virginica"  => 0x00000003
  "setosa"     => 0x00000001
  "versicolor" => 0x00000002
```
See also
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
the unwrapped model type [`MLJDecisionTreeInterface.DecisionTree.DecisionTreeClassifier`](@ref).
"""
DecisionTreeClassifier

"""
$(MMI.doc_header(RandomForestClassifier))
`RandomForestClassifier` implements the standard $DOC_RANDOM_FOREST.
# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)
where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`
Train the machine with `fit!(mach, rows=...)`.
# Hyper-parameters
- `max_depth=-1`:          max depth of the decision tree (-1=any)
- `min_samples_leaf=1`:    min number of samples each leaf needs to have
- `min_samples_split=2`:   min number of samples needed for a split
- `min_purity_increase=0`: min purity needed for a split
- `n_subfeatures=-1`: number of features to select at random (0 for all,
  -1 for square root of number of features)
- `n_trees=10`:            number of trees to train
- `sampling_fraction=0.7`  fraction of samples to train each tree on
- `rng=Random.GLOBAL_RNG`: random number generator or seed
# Operations
- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are probabilistic, but uncalibrated.
- `predict_mode(mach, Xnew)`: instead return the mode of each
  prediction above.
# Fitted parameters
The fields of `fitted_params(mach)` are:
- `forest`: the `Ensemble` object returned by the core DecisionTree.jl algorithm
# Examples
```
using MLJ
Forest = @load RandomForestClassifier pkg=DecisionTree
forest = Forest(min_samples_split=6, n_subfeatures=3)
X, y = @load_iris
mach = machine(forest, X, y) |> fit!
Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)
yhat = predict(mach, Xnew) # probabilistic predictions
predict_mode(mach, Xnew)   # point predictions
pdf.(yhat, "virginica")    # probabilities for the "verginica" class
fitted_params(mach).forest # raw `Ensemble` object from DecisionTrees.jl
```
See also
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
the unwrapped model type
[`MLJDecisionTreeInterface.DecisionTree.RandomForestClassifier`](@ref).
"""
RandomForestClassifier

"""
$(MMI.doc_header(AdaBoostStumpClassifier))
# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)
where:
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`
Train the machine with `fit!(mach, rows=...)`.
# Hyper-parameters
- `n_iter=10`:   number of iterations of AdaBoost
# Operations
- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are probabilistic, but uncalibrated.
- `predict_mode(mach, Xnew)`: instead return the mode of each
  prediction above.
# Fitted Parameters
The fields of `fitted_params(mach)` are:
- `stumps`: the `Ensemble` object returned by the core DecisionTree.jl
  algorithm.
- `coefficients`: the stump coefficients (one per stump)
```
using MLJ
Booster = @load AdaBoostStumpClassifier pkg=DecisionTree
booster = Booster(n_iter=15)
X, y = @load_iris
mach = machine(booster, X, y) |> fit!
Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)
yhat = predict(mach, Xnew) # probabilistic predictions
predict_mode(mach, Xnew)   # point predictions
pdf.(yhat, "virginica")    # probabilities for the "verginica" class
fitted_params(mach).stumps # raw `Ensemble` object from DecisionTree.jl
fitted_params(mach).coefs  # coefficient associated with each stump
```
See also
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
the unwrapped model type
[`MLJDecisionTreeInterface.DecisionTree.AdaBoostStumpClassifier`](@ref).
"""
AdaBoostStumpClassifier

"""
$(MMI.doc_header(DecisionTreeRegressor))
`DecisionTreeRegressor` implements the $DOC_CART.
# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)
where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`
Train the machine with `fit!(mach, rows=...)`.
# Hyper-parameters
- `max_depth=-1`:          max depth of the decision tree (-1=any)
- `min_samples_leaf=1`:    max number of samples each leaf needs to have
- `min_samples_split=2`:   min number of samples needed for a split
- `min_purity_increase=0`: min purity needed for a split
- `n_subfeatures=0`: number of features to select at random (0 for all,
  -1 for square root of number of features)
- `post_prune=false`:      set to `true` for post-fit pruning
- `merge_purity_threshold=1.0`: (post-pruning) merge leaves having
                           combined purity `>= merge_purity_threshold`
- `rng=Random.GLOBAL_RNG`: random number generator or seed
# Operations
- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same scitype as `X` above.
# Fitted parameters
The fields of `fitted_params(mach)` are:
- `tree`: the tree or stump object returned by the core
  DecisionTree.jl algorithm
# Examples
```
using MLJ
Tree = @load DecisionTreeRegressor pkg=DecisionTree
tree = Tree(max_depth=4, min_samples_split=3)
X, y = make_regression(100, 2) # synthetic data
mach = machine(tree, X, y) |> fit!
Xnew, _ = make_regression(3, 2)
yhat = predict(mach, Xnew) # new predictions
fitted_params(mach).tree # raw tree or stump object from DecisionTree.jl
```
See also
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
the unwrapped model type
[`MLJDecisionTreeInterface.DecisionTree.DecisionTreeRegressor`](@ref).
"""
DecisionTreeRegressor

"""
$(MMI.doc_header(RandomForestRegressor))
`DecisionTreeRegressor` implements the standard $DOC_RANDOM_FOREST
# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)
where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`
Train the machine with `fit!(mach, rows=...)`.
# Hyper-parameters
- `max_depth=-1`:          max depth of the decision tree (-1=any)
- `min_samples_leaf=1`:    min number of samples each leaf needs to have
- `min_samples_split=2`:   min number of samples needed for a split
- `min_purity_increase=0`: min purity needed for a split
- `n_subfeatures=-1`: number of features to select at random (0 for all,
  -1 for square root of number of features)
- `n_trees=10`:            number of trees to train
- `sampling_fraction=0.7`  fraction of samples to train each tree on
- `rng=Random.GLOBAL_RNG`: random number generator or seed
# Operations
- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same scitype as `X` above.
# Fitted parameters
The fields of `fitted_params(mach)` are:
- `forest`: the `Ensemble` object returned by the core DecisionTree.jl algorithm
# Examples
```
using MLJ
Forest = @load RandomForestRegressor pkg=DecisionTree
forest = Forest(max_depth=4, min_samples_split=3)
X, y = make_regression(100, 2) # synthetic data
mach = machine(forest, X, y) |> fit!
Xnew, _ = make_regression(3, 2)
yhat = predict(mach, Xnew) # new predictions
fitted_params(mach).forest # raw `Ensemble` object from DecisionTree.jl
```
See also
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
the unwrapped model type
[`MLJDecisionTreeInterface.DecisionTree.RandomForestRegressor`](@ref).
"""
RandomForestRegressor

end # module
