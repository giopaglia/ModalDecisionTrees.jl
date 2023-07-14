
# # DOCUMENT STRINGS

descr   = "" *
    # "The model is probabilistic, symbolic model " *
    # "for classification and regression tasks with dimensional data " *
    # "(e.g., images and time-series)." *
    "The symbolic, probabilistic model is able to extract logical descriptions of the data " *
    "in terms of logical formulas (see [`SoleLogics.jl`](@ref)) on propositions that are, " *
    "scalar conditions on the variables (or features); " *
    "for example, min[V2] ≥ 10, that is, \"the minimum of variable 2 is not less than 10\"." *
    "As such, the model is suitable for tasks that involve non-scalar data, " *
    "but require some level of interpretable and transparent modeling." *
    ""

const MDT_ref = "" *
    "Manzella et al. (2021). \"Interval Temporal Random Forests with an " *
    "Application to COVID-19 Diagnosis\". 10.4230/LIPIcs.TIME.2021.7"

const DOC_RANDOM_FOREST = "[Random Forest algorithm]" *
    "(https://en.wikipedia.org/wiki/Random_forest), originally published in " *
    "Breiman, L. (2001): \"Random Forests.\", *Machine Learning*, vol. 45, pp. 5–32"

function docstring_piece_1(
    default_min_samples_leaf,
    default_min_purity_increase,
    default_max_purity_at_leaf,
)
"""
Modal C4.5. This classification and regression algorithm, originally presented in $MDT_ref,
is an extension of the CART and C4.5
[decision tree learning algorithms](https://en.wikipedia.org/wiki/Decision_tree_learning)
that leverages the expressive power of modal logics of time and space
to perform temporal/spatial reasoning on non-scalar data, such as time-series and images.

$(descr)

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)
where
- `X`: any table of input features (e.g., a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, `OrderedFactor`, or any 0-, 1-, 2-dimensional array with elements
  of these scitypes; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Multiclass`, `Continuous`, `Finite`, or `Textual`; check the scitype
  with `scitype(y)`
Train the machine with `fit!(mach)`.

# Hyper-parameters

- `max_depth=-1`:          Maximum depth of the decision tree (-1=any)
- `min_samples_leaf=$(default_min_samples_leaf)`:    Minimum number of samples required at each leaf
- `min_purity_increase=$(default_min_purity_increase)`: Minimum purity needed for a split
- `max_purity_at_leaf=$(default_max_purity_at_leaf)`: Minimum purity needed for a split
- `relations=nothing`       Relations that the model uses to "move" around the image; it can be a symbol in [:IA, :IA3, :IA7, :RCC5, :RCC8],
                            where :IA stands [Allen's Interval Algebra](https://en.wikipedia.org/wiki/Allen%27s_interval_algebra) (13 relations in 1D, 169 relations in 2D),
                            :IA3 and :IA7 are [coarser fragments with 3 and 7 relations, respectively](https://www.sciencedirect.com/science/article/pii/S0004370218305964),
                            :RCC5 and :RCC8 are [Region Connection Calculus algebras](https://en.wikipedia.org/wiki/Region_connection_calculus) with 5 and 8 topological operators, respectively.
                            Relations from :IA, :IA3, :IA7, capture directional aspects of the relative arrangement of two intervals in time (or rectangles in a 2D space),
                             while relations from :RCC5 and :RCC8 only capture topological aspects and are therefore rotation-invariant.
                            This hyper-parameter defaults to :IA for temporal features (1D), and to :RCC8 for spatial features (2D).
- `initconditions=nothing` initial conditions for evaluating modal decisions at the root; it can be a symbol in [:start_with_global, :start_at_center].
                            :start_with_global forces the first decision to be a *global* decision (e.g., `⟨G⟩ (minimum[V2] ≥ 10)`, which translates to "there exists a region where the minimum of variable 2 is higher than 10").
                            :start_at_center forces the first decision to be evaluated on the smallest central world, that is, the central value of a time-series, or the central pixel of an image.
                            This hyper-parameter defaults to :start_with_global for temporal features (1D), and to :start_at_center for spatial features (2D).
- `downsize=true` Whether to perform automatic downsizing. In fact, this algorithm has high complexity (both time and space), and can only handle small time-series (< 100 points) & small images (< 10 x 10 pixels).
"""
end

"""
$(MMI.doc_header(ModalDecisionTree))

`ModalDecisionTree` implements
$(docstring_piece_1(mlj_mdt_default_min_samples_leaf, mlj_mdt_default_min_purity_increase, mlj_mdt_default_max_purity_at_leaf))
- `display_depth=5`:       max depth to show when displaying the tree

# Operations
- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above.

# Fitted parameters
The fields of `fitted_params(mach)` are:
- `model`: the tree object, as returned by the core algorithm
- `var_grouping`: the adopted grouping of the features encountered in training, in an order consistent with the output of `printmodel`.
    The MLJ interface can currently deal with scalar, temporal and spatial features, but
    has one limitation, and one tricky procedure for handling them at the same time.
    The limitation is for temporal and spatial features to be uniform in size across the instances (the algorithm will automatically throw away features that do not satisfy this constraint).
    As for the tricky procedure: before the learning phase, features are divided into groups (referred to as `modalities`) according to each variable's `channel size`, that is, the size of the vector or matrix.
    For example, if X is multimodal, and has three temporal features :x, :y, :z with 10, 10 and 20 points, respectively,
     plus three spatial features :R, :G, :B, with the same size 5 × 5 pixels, the algorithm assumes that :x and :y share a temporal axis,
     :R, :G, :B share two spatial axis, while :z does not share any axis with any other variable. As a result,
     the model will group features into three modalities:
        - {1} [:x, :y]
        - {2} [:z]
        - {3} [:R, :G, :B]
    and `var_grouping` will be [["x", "y"], ["z"], ["R", "G", "B"]].
"R", "G", "B"]

# Report
The fields of `report(mach)` are:
- `printmodel`: method to print a pretty representation of the fitted
  model, with single argument the tree depth. The interpretation of the tree requires you
  to understand how the current MLJ interface of ModalDecisionTrees.jl handles features of different modals.
  See `var_grouping` above. Note that the split conditions (or decisions) in the tree are relativized to a specific modality, of which the number is shown.
- `var_grouping`: the adopted grouping of the features encountered in training, in an order consistent with the output of `printmodel`.
    See `var_grouping` above.
- `feature_importance_by_count`: a simple count of each of the occurrences of the features across the model, in an order consistent with `var_grouping`.
- `classes_seen`: list of target classes actually observed in training.
# Examples
```julia
using MLJ
using ModalDecisionTrees
using Random

tree = ModalDecisionTree(min_samples_leaf=4)

# Load an example dataset (a temporal one)
X, y = load_japanesevowels()
N = length(y)

mach = machine(tree, X, y)

# Split dataset
p = randperm(N)
train_idxs, test_idxs = p[1:round(Int, N*.8)], p[round(Int, N*.8)+1:end]

# Fit
fit!(mach, rows=train_idxs)

# Perform predictions, compute accuracy
yhat = predict(mach, X[test_idxs,:])
accuracy = sum(yhat .== y[test_idxs])/length(yhat)

# Access raw model
fitted_params(mach).model
report(mach).printmodel(3)

"{1} ⟨G⟩ (max(coefficient1) <= 0.883491)                 3 : 91/512 (conf = 0.1777)
✔ {1} ⟨G⟩ (max(coefficient9) <= -0.157292)                      3 : 89/287 (conf = 0.3101)
│✔ {1} ⟨L̅⟩ (max(coefficient6) <= -0.504503)                     3 : 89/209 (conf = 0.4258)
││✔ {1} ⟨A⟩ (max(coefficient3) <= 0.220312)                     3 : 81/93 (conf = 0.8710)
 [...]
││✘ {1} ⟨L̅⟩ (max(coefficient1) <= 0.493004)                     8 : 47/116 (conf = 0.4052)
 [...]
│✘ {1} ⟨A⟩ (max(coefficient2) <= -0.285645)                     7 : 41/78 (conf = 0.5256)
│ ✔ {1} min(coefficient3) >= 0.002931                   4 : 34/36 (conf = 0.9444)
 [...]
│ ✘ {1} ⟨G⟩ (min(coefficient5) >= 0.18312)                      7 : 39/42 (conf = 0.9286)
 [...]
✘ {1} ⟨G⟩ (max(coefficient3) <= 0.006087)                       5 : 51/225 (conf = 0.2267)
 ✔ {1} ⟨D⟩ (max(coefficient2) <= -0.301233)                     5 : 51/102 (conf = 0.5000)
 │✔ {1} ⟨D̅⟩ (max(coefficient3) <= -0.123654)                    5 : 51/65 (conf = 0.7846)
 [...]
 │✘ {1} ⟨G⟩ (max(coefficient9) <= -0.146962)                    7 : 16/37 (conf = 0.4324)
 [...]
 ✘ {1} ⟨G⟩ (max(coefficient9) <= -0.424346)                     1 : 47/123 (conf = 0.3821)
  ✔ {1} min(coefficient1) >= 1.181048                   6 : 39/40 (conf = 0.9750)
 [...]
  ✘ {1} ⟨G⟩ (min(coefficient4) >= -0.472485)                    1 : 47/83 (conf = 0.5663)
 [...]"
```
"""
ModalDecisionTree

"""
$(MMI.doc_header(ModalRandomForest))
`ModalRandomForest` implements the standard $DOC_RANDOM_FOREST, based on
$(docstring_piece_1(mlj_mrf_default_min_samples_leaf, mlj_mrf_default_min_purity_increase, mlj_mrf_default_max_purity_at_leaf))
- `n_subrelations=identity`            Number of relations to randomly select at any point of the tree. Must be a function of the number of the available relations. It defaults to `identity`, that is, consider all available relations.
- `n_subfeatures=x -> ceil(Int64, sqrt(x))`             Number of functions to randomly select at any point of the tree. Must be a function of the number of the available functions. It defaults to `x -> ceil(Int64, sqrt(x))`, that is, consider only about square root of the available functions.
- `ntrees=$(mlj_mrf_default_ntrees)`                   Number of trees in the forest.
- `sampling_fraction=0.7`          Fraction of samples to train each tree on.
- `rng=Random.GLOBAL_RNG`          Random number generator or seed.

# Operations
- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are probabilistic, but uncalibrated.
- `predict_mode(mach, Xnew)`: instead return the mode of each
  prediction above.

# Fitted parameters
The fields of `fitted_params(mach)` are:
- `model`: the forest object, as returned by the core algorithm
- `var_grouping`: the adopted grouping of the features encountered in training, in an order consistent with the output of `printmodel`.
    The MLJ interface can currently deal with scalar, temporal and spatial features, but
    has one limitation, and one tricky procedure for handling them at the same time.
    The limitation is for temporal and spatial features to be uniform in size across the instances (the algorithm will automatically throw away features that do not satisfy this constraint).
    As for the tricky procedure: before the learning phase, features are divided into groups (referred to as `modalities`) according to each variable's `channel size`, that is, the size of the vector or matrix.
    For example, if X is multimodal, and has three temporal features :x, :y, :z with 10, 10 and 20 points, respectively,
     plus three spatial features :R, :G, :B, with the same size 5 × 5 pixels, the algorithm assumes that :x and :y share a temporal axis,
     :R, :G, :B share two spatial axis, while :z does not share any axis with any other variable. As a result,
     the model will group features into three modalities:
        - {1} [:x, :y]
        - {2} [:z]
        - {3} [:R, :G, :B]
    and `var_grouping` will be [["x", "y"], ["z"], ["R", "G", "B"]].

# Report
The fields of `report(mach)` are:
- `printmodel`: method to print a pretty representation of the fitted
  model, with single argument the depth of the trees. The interpretation of the tree requires you
  to understand how the current MLJ interface of ModalDecisionTrees.jl handles features of different modals.
  See `var_grouping` above. Note that the split conditions (or decisions) in the tree are relativized to a specific frame, of which the number is shown.
- `var_grouping`: the adopted grouping of the features encountered in training, in an order consistent with the output of `printmodel`.
    See `var_grouping` above.
- `feature_importance_by_count`: a simple count of each of the occurrences of the features across the model, in an order consistent with `var_grouping`.
- `classes_seen`: list of target classes actually observed in training.
# Examples
```julia
using MLJ
using ModalDecisionTrees
using Random

forest = ModalRandomForest(ntrees = 50)

# Load an example dataset (a temporal one)
X, y = load_japanesevowels()
N = length(y)

mach = machine(forest, X, y)

# Split dataset
p = randperm(N)
train_idxs, test_idxs = p[1:round(Int, N*.8)], p[round(Int, N*.8)+1:end]

# Fit
fit!(mach, rows=train_idxs)

# Perform predictions, compute accuracy
Xnew = X[test_idxs,:]
yhat = predict(mach, Xnew) # probabilistic predictions
ynew = predict_mode(mach, Xnew)   # point predictions
accuracy = sum(ynew .== y[test_idxs])/length(yhat)
pdf.(yhat, "1")    # probabilities for one of the classes ("1")

# Access raw model
fitted_params(mach).model
report(mach).printmodel(3)] # Note that the output here can be quite large.
```
"""
ModalRandomForest

# # Examples
# ```
# using MLJ
# MDT = @load DecisionTreeRegressor pkg=ModalDecisionTrees
# tree = MDT(max_depth=4, min_samples_split=3)
# X, y = make_regression(100, 2) # synthetic data
# mach = machine(tree, X, y) |> fit!
# Xnew, _ = make_regression(3, 2)
# yhat = predict(mach, Xnew) # new predictions
# fitted_params(mach).model # raw tree or stump object from DecisionTree.jl
# ```
# See also
# [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
# the unwrapped model type
# [`MLJDecisionTreeInterface.DecisionTree.DecisionTreeRegressor`](@ref).
# """
# DecisionTreeRegressor

# # Examples
# ```
# using MLJ
# Forest = @load RandomForestRegressor pkg=ModalDecisionTrees
# forest = Forest(max_depth=4, min_samples_split=3)
# X, y = make_regression(100, 2) # synthetic data
# mach = machine(forest, X, y) |> fit!
# Xnew, _ = make_regression(3, 2)
# yhat = predict(mach, Xnew) # new predictions
# fitted_params(mach).forest # raw `Ensemble` object from DecisionTree.jl
# ```
# See also
# [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
# the unwrapped model type
# [`MLJDecisionTreeInterface.DecisionTree.RandomForestRegressor`](@ref).
