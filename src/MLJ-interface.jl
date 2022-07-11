# Inspired from JuliaAI/MLJDecisionTreeInterface.jl

module MLJInterface

# Reference: https://alan-turing-institute.github.io/MLJ.jl/dev/quick_start_guide_to_adding_models/#Quick-Start-Guide-to-Adding-Models
# Reference: https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/

using MLJBase
import MLJModelInterface
using MLJModelInterface.ScientificTypesBase
using ..ModalDecisionTrees
import Tables

import Base: show

using Random
using DataFrames
import Random.GLOBAL_RNG

const MMI = MLJModelInterface
const MDT = ModalDecisionTrees
const PKG = "ModalDecisionTrees"

export ModalDecisionTree

############################################################################################
############################################################################################
############################################################################################

struct ModelPrinter{T}
    model::T
    frame_grouping::Union{Nothing,AbstractVector{<:AbstractVector},AbstractVector{<:AbstractDict}}
end
(c::ModelPrinter)(; args...) = c(c.model; args...)
(c::ModelPrinter)(model; max_depth = 5) = MDT.print_model(model; attribute_names_map = c.frame_grouping, max_depth = max_depth)

Base.show(io::IO, c::ModelPrinter) =
    print(io, "ModelPrinter object (call with display depth)")

############################################################################################
############################################################################################
############################################################################################

_size = ((x)->(hasmethod(size, (typeof(x),)) ? size(x) : missing))

function separate_attributes_into_frames(X)

    columns = names(X)
    channel_sizes = [unique(_size.(X)[:,col]) for col in columns]

    # Must have common channel size across instances
    _uniform_columns = (length.(channel_sizes) .== 1)
    _nonmissing_columns = (((cs)->all((!).(ismissing.(cs)))).(channel_sizes))

    __uniform_cols = columns[(!).(_uniform_columns)]
    if length(__uniform_cols) > 0
        println("Dropping columns due to non-uniform channel size across instances: $(__uniform_cols)...")
    end
    __uniform_non_missing_cols = columns[_uniform_columns .&& (!).(_nonmissing_columns)]
    if length(__uniform_non_missing_cols) > 0
        println("Dropping columns due to missings: $(__uniform_non_missing_cols)...")
    end
    _good_columns = _uniform_columns .&& _nonmissing_columns

    _good_columns = _uniform_columns .&& _nonmissing_columns
    channel_sizes = channel_sizes[_good_columns]
    columns = columns[_good_columns]
    channel_sizes = getindex.(channel_sizes, 1)

    unique_channel_sizes = sort(unique(channel_sizes))

    frame_ids = [findfirst((ucs)->(ucs==cs), unique_channel_sizes) for cs in channel_sizes]

    frames = Dict([frame_id => [] for frame_id in unique(frame_ids)])
    for (frame_id, col) in zip(frame_ids, columns)
        push!(frames[frame_id], col)
    end
    frames = [frames[frame_id] for frame_id in unique(frame_ids)]

    # println("Frames:");
    # println()
    # display(collect(((x)->Pair(x...)).((enumerate(frames)))));

    frames
end

function DataFrame2MultiFrameModalDataset(X, frame_grouping, relations, mixed_features, init_conditions, allow_global_splits, mode)
    @assert mode in [:explicit, :implicit]

    Xs = [begin
        X_frame = X[:,frame]
        

        channel_size = unique([unique(_size.(X_frame[:, col])) for col in names(X_frame)])
        @assert length(channel_size) == 1
        @assert length(channel_size[1]) == 1
        channel_size = channel_size[1][1]
        
        # println("$(i_frame)\tchannel size: $(channel_size)\t => $(frame_grouping)")

        _X = begin
                # dataframe2cube(X_frame)
                common_type = Union{eltype.(eltype.(eachcol(X_frame)))...}
                common_type = common_type == Any ? Number : common_type
                _X = Array{common_type}(undef, channel_size..., DataFrames.ncol(X_frame), DataFrames.nrow(X_frame))
                for (i_col, col) in enumerate(eachcol(X_frame))
                    for (i_row, row) in enumerate(col)
                        _X[[(:) for i in 1:length(size(row))]...,i_col,i_row] = row
                    end
                end
                _X
            end

            channel_dim = length(channel_size)
            ontology = MDT.get_interval_ontology(channel_dim, relations)
            # println(eltype(_X))
            X = MDT.InterpretedModalDataset(_X, ontology, mixed_features)
            # println(MDT.display_structure(X))

            if mode == :implicit
                X
            else
                WorldType = MDT.world_type(ontology)
                
                compute_relation_glob =
                    WorldType != MDT.OneWorld && (
                        (allow_global_splits || init_conditions == MDT.start_without_world)
                    )
                MDT.ExplicitModalDatasetSMemo(X, compute_relation_glob = compute_relation_glob)
            end
        end for (i_frame, frame) in enumerate(frame_grouping)]
    Xs = MDT.MultiFrameModalDataset(Xs)
    # println(MDT.display_structure(Xs))
    # Xs
end

############################################################################################
############################################################################################
############################################################################################

function _check_features(features)
    good = [
        [
            map(
            (f)->(f isa CanonicalFeature || (ret = f(ch); isa(ret, Number) && typeof(ret) == eltype(ch))),
            features) for ch in [collect(1:10), collect(1.:10.)]
        ]
    ] |> Iterators.flatten |> all
    @assert good "`features` should be a vector of scalar functions accepting on object of type `AbstractVector{T}` and returning an object of type `T`."
    # println(typeof(good))
    good
end

using ModalDecisionTrees: default_min_samples_leaf
using ModalDecisionTrees: default_min_purity_increase
using ModalDecisionTrees: default_max_purity_at_leaf
using ModalDecisionTrees: Relation
using ModalDecisionTrees: CanonicalFeature
using ModalDecisionTrees: CanonicalFeatureGeq, canonical_geq
using ModalDecisionTrees: CanonicalFeatureLeq, canonical_leq
using ModalDecisionTrees: start_without_world
using ModalDecisionTrees: start_at_center

MMI.@mlj_model mutable struct ModalDecisionTree <: MMI.Deterministic
    # Pruning hyper-parameters
    max_depth              :: Union{Nothing,Int}           = nothing::(isnothing(_) || _ ≥ -1)
    min_samples_leaf       :: Int                          = default_min_samples_leaf::(_ ≥ 1)
    min_purity_increase    :: Float64                      = default_min_purity_increase
    max_purity_at_leaf     :: Float64                      = default_max_purity_at_leaf
    # Modal hyper-parameters
    relations              :: Union{Nothing,Symbol,Vector{<:Relation}} = (:IA)::(isnothing(_) || _ in [:IA, :IA3, :IA7, :RCC5, :RCC8] || _ isa AbstractVector{<:Relation})
    # TODO expand to ModalFeature
    # features               :: Vector{<:Function}           = [minimum, maximum]::(all(Iterators.flatten([(f)->(ret = f(ch); isa(ret, Number) && typeof(ret) == eltype(ch)), _) for ch in [collect(1:10), collect(1.:10.)]]))
    # features               :: Vector{<:Union{CanonicalFeature,Function}}       TODO = Vector{<:Union{CanonicalFeature,Function}}([canonical_geq, canonical_leq]) # TODO: ::(_check_features(_))
    # features               :: AbstractVector{<:CanonicalFeature}       = CanonicalFeature[canonical_geq, canonical_leq] # TODO: ::(_check_features(_))
    # features               :: Vector                       = [canonical_geq, canonical_leq] # TODO: ::(_check_features(_))
    init_conditions        :: Symbol                       = :start_with_global::(_ in [:start_with_global, :start_at_center])
    allow_global_splits    :: Bool                         = true
    # Other
    display_depth          :: Union{Nothing,Int}           = 5::(isnothing(_) || _ ≥ 0)
end

function MMI.fit(m::ModalDecisionTree, verbosity::Int, X, y, w=nothing)

    init_conditions_d = Dict([
        :start_with_global => MDT.start_without_world,
        :start_at_center   => MDT.start_at_center,
    ])

    max_depth = m.max_depth
    max_depth = isnothing(max_depth) ? typemax(Int64) : max_depth
    min_samples_leaf = m.min_samples_leaf
    min_purity_increase = m.min_purity_increase
    max_purity_at_leaf = m.max_purity_at_leaf
    relations = m.relations
    features = [canonical_geq, canonical_leq] # TODO: m.features
    init_conditions = init_conditions_d[m.init_conditions]
    allow_global_splits = m.allow_global_splits
    display_depth = m.display_depth

    frame_grouping = separate_attributes_into_frames(X)
    Xs = DataFrame2MultiFrameModalDataset(X, frame_grouping, relations, features, init_conditions, allow_global_splits, :explicit)

    # @assert y isa AbstractVector{<:CLabel} "y should be an AbstractVector{<:$(CLabel)}, got $(typeof(y)) instead"
    # original_y = y
    # println(typeof(y))
    # y  = MMI.int(y)

    model = MDT.build_tree(
        Xs,
        y,
        w,
        ##############################################################################
        loss_function        = nothing,
        max_depth            = max_depth,
        min_samples_leaf     = min_samples_leaf,
        min_purity_increase  = min_purity_increase,
        max_purity_at_leaf   = max_purity_at_leaf,
        ##############################################################################
        n_subrelations       = identity,
        n_subfeatures        = identity,
        init_conditions      = init_conditions,
        allow_global_splits  = allow_global_splits,
        ##############################################################################
        perform_consistency_check = false,
    )

    verbosity < 2 || MDT.print_model(model; max_depth = m.display_depth, attribute_names_map = frame_grouping)

    feature_importance_by_count = Dict([i_attr => frame_grouping[i_frame][i_attr] for ((i_frame, i_attr), count) in MDT.tree_attribute_countmap(model)])

    fitresult = (
        model           = model,
        frame_grouping = frame_grouping,
    )

    cache  = nothing
    report = (
        print_tree                  = ModelPrinter(model, frame_grouping),
        frame_grouping              = frame_grouping,
        feature_importance_by_count = feature_importance_by_count,
    )

    return fitresult, cache, report
end

MMI.fitted_params(::ModalDecisionTree, fitresult) =
    (
        model           = fitresult.model,
        frame_grouping  = fitresult.frame_grouping,
        # encoding        = get_encoding(fitresult.encoding),
    )

# function smooth(scores, smoothing)
#     iszero(smoothing) && return scores
#     threshold = smoothing / size(scores, 2)
#     # clip low values
#     scores[scores .< threshold] .= threshold
#     # normalize
#     return scores ./ sum(scores, dims=2)
# end

function MMI.predict(m::ModalDecisionTree, fitresult, Xnew, Ynew)
    
    relations = m.relations
    features = [canonical_geq, canonical_leq] # TODO: m.features
    init_conditions = m.init_conditions
    allow_global_splits = m.allow_global_splits
    Xs = DataFrame2MultiFrameModalDataset(Xnew, fitresult.frame_grouping, relations, features, init_conditions, allow_global_splits, :implicit)

    # MDT.apply_tree(fitresult.model, Xs)
    MDT.print_apply(fitresult.model, Xs, Ynew)
end


# # # RANDOM FOREST CLASSIFIER
# TODO
# MMI.@mlj_model mutable struct RandomForestClassifier <: MMI.Probabilistic
#     max_depth::Int               = (-)(1)::(_ ≥ -1)
#     min_samples_leaf::Int        = 1::(_ ≥ 0)
#     min_samples_split::Int       = 2::(_ ≥ 2)
#     min_purity_increase::Float64 = 0.0::(_ ≥ 0)
#     n_subfeatures::Int           = (-)(1)::(_ ≥ -1)
#     n_trees::Int                 = 10::(_ ≥ 2)
#     sampling_fraction::Float64   = 0.7::(0 < _ ≤ 1)
#     rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
# end

# function MMI.fit(m::RandomForestClassifier, verbosity::Int, X, y, w=nothing)
#     Xmatrix = MMI.matrix(X)
#     yplain  = MMI.int(y)

#     classes_seen  = filter(in(unique(y)), MMI.classes(y[1]))
#     integers_seen = MMI.int(classes_seen)

#     forest = MDT.build_forest(yplain, Xmatrix,
#                              m.n_subfeatures,
#                              m.n_trees,
#                              m.sampling_fraction,
#                              m.max_depth,
#                              m.min_samples_leaf,
#                              m.min_samples_split,
#                              m.min_purity_increase;
#                              rng=m.rng)
#     cache  = nothing
#     report = NamedTuple()
#     return (forest, classes_seen, integers_seen), cache, report
# end

# MMI.fitted_params(::RandomForestClassifier, (forest,_)) = (forest=forest,)

# function MMI.predict(m::RandomForestClassifier, fitresult, Xnew)
#     Xmatrix = MMI.matrix(Xnew)
#     forest, classes_seen, integers_seen = fitresult
#     scores = MDT.apply_forest_proba(forest, Xmatrix, integers_seen)
#     return MMI.UnivariateFinite(classes_seen, scores)
# end


# # ADA BOOST STUMP CLASSIFIER

# TODO
# MMI.@mlj_model mutable struct AdaBoostStumpClassifier <: MMI.Probabilistic
#     n_iter::Int            = 10::(_ ≥ 1)
# end

# function MMI.fit(m::AdaBoostStumpClassifier, verbosity::Int, X, y)
#     Xmatrix = MMI.matrix(X)
#     yplain  = MMI.int(y)

#     classes_seen  = filter(in(unique(y)), MMI.classes(y[1]))
#     integers_seen = MMI.int(classes_seen)

#     stumps, coefs = MDT.build_adaboost_stumps(yplain, Xmatrix,
#                                              m.n_iter)
#     cache  = nothing
#     report = NamedTuple()
#     return (stumps, coefs, classes_seen, integers_seen), cache, report
# end

# MMI.fitted_params(::AdaBoostStumpClassifier, (stumps,coefs,_)) =
#     (stumps=stumps,coefs=coefs)

# function MMI.predict(m::AdaBoostStumpClassifier, fitresult, Xnew)
#     Xmatrix = MMI.matrix(Xnew)
#     stumps, coefs, classes_seen, integers_seen = fitresult
#     scores = MDT.apply_adaboost_stumps_proba(stumps, coefs,
#                                             Xmatrix, integers_seen)
#     return MMI.UnivariateFinite(classes_seen, scores)
# end


# # # DECISION TREE REGRESSOR

# MMI.@mlj_model mutable struct DecisionTreeRegressor <: MMI.Deterministic
#     max_depth::Int                               = (-)(1)::(_ ≥ -1)
#     min_samples_leaf::Int                = 5::(_ ≥ 0)
#     min_samples_split::Int               = 2::(_ ≥ 2)
#     min_purity_increase::Float64 = 0.0::(_ ≥ 0)
#     n_subfeatures::Int                   = 0::(_ ≥ -1)
#     post_prune::Bool                     = false
#     merge_purity_threshold::Float64 = 1.0::(0 ≤ _ ≤ 1)
#     rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
# end

# function MMI.fit(m::DecisionTreeRegressor, verbosity::Int, X, y)
#     Xmatrix = MMI.matrix(X)
#     tree    = MDT.build_tree(float(y), Xmatrix,
#                             m.n_subfeatures,
#                             m.max_depth,
#                             m.min_samples_leaf,
#                             m.min_samples_split,
#                             m.min_purity_increase;
#                             rng=m.rng)

#     if m.post_prune
#         tree = MDT.prune_tree(tree, m.merge_purity_threshold)
#     end
#     cache  = nothing
#     report = NamedTuple()
#     return tree, cache, report
# end

# MMI.fitted_params(::DecisionTreeRegressor, tree) = (tree=tree,)

# function MMI.predict(::DecisionTreeRegressor, tree, Xnew)
#     Xmatrix = MMI.matrix(Xnew)
#     return MDT.apply_tree(tree, Xmatrix)
# end


# # # RANDOM FOREST REGRESSOR

# MMI.@mlj_model mutable struct RandomForestRegressor <: MMI.Deterministic
#     max_depth::Int               = (-)(1)::(_ ≥ -1)
#     min_samples_leaf::Int        = 1::(_ ≥ 0)
#     min_samples_split::Int       = 2::(_ ≥ 2)
#     min_purity_increase::Float64 = 0.0::(_ ≥ 0)
#     n_subfeatures::Int           = (-)(1)::(_ ≥ -1)
#     n_trees::Int                 = 10::(_ ≥ 2)
#     sampling_fraction::Float64   = 0.7::(0 < _ ≤ 1)
#     rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
# end

# function MMI.fit(m::RandomForestRegressor, verbosity::Int, X, y)
#     Xmatrix = MMI.matrix(X)
#     forest  = MDT.build_forest(float(y), Xmatrix,
#                               m.n_subfeatures,
#                               m.n_trees,
#                               m.sampling_fraction,
#                               m.max_depth,
#                               m.min_samples_leaf,
#                               m.min_samples_split,
#                               m.min_purity_increase,
#                               rng=m.rng)
#     cache  = nothing
#     report = NamedTuple()
#     return forest, cache, report
# end

# MMI.fitted_params(::RandomForestRegressor, forest) = (forest=forest,)

# function MMI.predict(::RandomForestRegressor, forest, Xnew)
#     Xmatrix = MMI.matrix(Xnew)
#     return MDT.apply_forest(forest, Xmatrix)
# end


# # METADATA (MODEL TRAITS)

# following five lines of code are redundant if using this branch of
# MLJModelInterface:
# https://github.com/JuliaAI/MLJModelInterface.jl/pull/139

# MMI.human_name(::Type{<:ModalDecisionTree}) = "CART decision tree classifier"
# MMI.human_name(::Type{<:RandomForestClassifier}) = "CART random forest classifier"
# MMI.human_name(::Type{<:AdaBoostStumpClassifier}) = "Ada-boosted stump classifier"
# MMI.human_name(::Type{<:DecisionTreeRegressor}) = "CART decision tree regressor"
# MMI.human_name(::Type{<:RandomForestRegressor}) = "CART random forest regressor"

MMI.metadata_pkg.(
    (
        ModalDecisionTree,
        # DecisionTreeRegressor,
        # RandomForestClassifier,
        # RandomForestRegressor,
        # AdaBoostStumpClassifier,
    ),
    name = "ModalDecisionTrees",
    package_uuid = "e54bda2e-c571-11ec-9d64-0242ac120002",
    package_url = "https://github.com/giopaglia/ModalDecisionTree.jl",
    is_pure_julia = true,
    package_license = "MIT",
    is_wrapper = false,
)

MMI.metadata_model(
    ModalDecisionTree,
    input_scitype = Table(
        Continuous,
        Count,
        OrderedFactor,
        AbstractVector{<:Continuous},
        AbstractVector{<:Count},
        AbstractVector{<:OrderedFactor},
    ),
    target_scitype = AbstractVector{<:Finite},
    human_name = "Modal Decision Tree (MDT)",
    descr   = "A Modal Decision Tree (MDT) offers high level of interpretability for classification and regression tasks with images and time-series.",
    supports_weights = true,
    load_path = "$PKG.ModalDecisionTree",
)

# MMI.metadata_model(
#     RandomForestClassifier,
#     input_scitype = Table(Continuous, Count, OrderedFactor),
#     target_scitype = AbstractVector{<:Finite},
#     human_name = "CART random forest classifier",
#     load_path = "$PKG.RandomForestClassifier"
# )

# MMI.metadata_model(
#     AdaBoostStumpClassifier,
#     input_scitype = Table(Continuous, Count, OrderedFactor),
#     target_scitype = AbstractVector{<:Finite},
#     human_name = "Ada-boosted stump classifier",
#     load_path = "$PKG.AdaBoostStumpClassifier"
# )

# MMI.metadata_model(
#     DecisionTreeRegressor,
#     input_scitype = Table(Continuous, Count, OrderedFactor),
#     target_scitype = AbstractVector{Continuous},
#     human_name = "CART decision tree regressor",
#     load_path = "$PKG.DecisionTreeRegressor"
# )

# MMI.metadata_model(
#     RandomForestRegressor,
#     input_scitype = Table(Continuous, Count, OrderedFactor),
#     target_scitype = AbstractVector{Continuous},
#     human_name = "CART random forest regressor",
#     load_path = "$PKG.RandomForestRegressor")


# # DOCUMENT STRINGS

const DOC_CART = "[CART algorithm](https://en.wikipedia.org/wiki/Decision_tree_learning)"*
", originally published in Breiman, Leo; Friedman, J. H.; Olshen, R. A.; "*
"Stone, C. J. (1984): \"Classification and regression trees\". *Monterey, "*
"CA: Wadsworth & Brooks/Cole Advanced Books & Software.*"

const DOC_RANDOM_FOREST = "[Random Forest algorithm]"*
    "(https://en.wikipedia.org/wiki/Random_forest), originally published in "*
    "Breiman, L. (2001): \"Random Forests.\", *Machine Learning*, vol. 45, pp. 5–32"

"""
$(MMI.doc_header(ModalDecisionTree))
`ModalDecisionTree` implements the $DOC_CART.
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
Tree = @load ModalDecisionTree pkg=DecisionTree
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
the unwrapped model type [`MLJDecisionTreeInterface.DecisionTree.ModalDecisionTree`](@ref).
"""
ModalDecisionTree

# """
# $(MMI.doc_header(RandomForestClassifier))
# `RandomForestClassifier` implements the standard $DOC_RANDOM_FOREST.
# # Training data
# In MLJ or MLJBase, bind an instance `model` to data with
#     mach = machine(model, X, y)
# where
# - `X`: any table of input features (eg, a `DataFrame`) whose columns
#   each have one of the following element scitypes: `Continuous`,
#   `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
# - `y`: the target, which can be any `AbstractVector` whose element
#   scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
#   with `scitype(y)`
# Train the machine with `fit!(mach, rows=...)`.
# # Hyper-parameters
# - `max_depth=-1`:          max depth of the decision tree (-1=any)
# - `min_samples_leaf=1`:    min number of samples each leaf needs to have
# - `min_samples_split=2`:   min number of samples needed for a split
# - `min_purity_increase=0`: min purity needed for a split
# - `n_subfeatures=-1`: number of features to select at random (0 for all,
#   -1 for square root of number of features)
# - `n_trees=10`:            number of trees to train
# - `sampling_fraction=0.7`  fraction of samples to train each tree on
# - `rng=Random.GLOBAL_RNG`: random number generator or seed
# # Operations
# - `predict(mach, Xnew)`: return predictions of the target given
#   features `Xnew` having the same scitype as `X` above. Predictions
#   are probabilistic, but uncalibrated.
# - `predict_mode(mach, Xnew)`: instead return the mode of each
#   prediction above.
# # Fitted parameters
# The fields of `fitted_params(mach)` are:
# - `forest`: the `Ensemble` object returned by the core DecisionTree.jl algorithm
# # Examples
# ```
# using MLJ
# Forest = @load RandomForestClassifier pkg=DecisionTree
# forest = Forest(min_samples_split=6, n_subfeatures=3)
# X, y = @load_iris
# mach = machine(forest, X, y) |> fit!
# Xnew = (sepal_length = [6.4, 7.2, 7.4],
#         sepal_width = [2.8, 3.0, 2.8],
#         petal_length = [5.6, 5.8, 6.1],
#         petal_width = [2.1, 1.6, 1.9],)
# yhat = predict(mach, Xnew) # probabilistic predictions
# predict_mode(mach, Xnew)   # point predictions
# pdf.(yhat, "virginica")    # probabilities for the "verginica" class
# fitted_params(mach).forest # raw `Ensemble` object from DecisionTrees.jl
# ```
# See also
# [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
# the unwrapped model type
# [`MLJDecisionTreeInterface.DecisionTree.RandomForestClassifier`](@ref).
# """
# # RandomForestClassifier

# """
# $(MMI.doc_header(AdaBoostStumpClassifier))
# # Training data
# In MLJ or MLJBase, bind an instance `model` to data with
#     mach = machine(model, X, y)
# where:
# - `X`: any table of input features (eg, a `DataFrame`) whose columns
#   each have one of the following element scitypes: `Continuous`,
#   `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
# - `y`: the target, which can be any `AbstractVector` whose element
#   scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
#   with `scitype(y)`
# Train the machine with `fit!(mach, rows=...)`.
# # Hyper-parameters
# - `n_iter=10`:   number of iterations of AdaBoost
# # Operations
# - `predict(mach, Xnew)`: return predictions of the target given
#   features `Xnew` having the same scitype as `X` above. Predictions
#   are probabilistic, but uncalibrated.
# - `predict_mode(mach, Xnew)`: instead return the mode of each
#   prediction above.
# # Fitted Parameters
# The fields of `fitted_params(mach)` are:
# - `stumps`: the `Ensemble` object returned by the core DecisionTree.jl
#   algorithm.
# - `coefficients`: the stump coefficients (one per stump)
# ```
# using MLJ
# Booster = @load AdaBoostStumpClassifier pkg=DecisionTree
# booster = Booster(n_iter=15)
# X, y = @load_iris
# mach = machine(booster, X, y) |> fit!
# Xnew = (sepal_length = [6.4, 7.2, 7.4],
#         sepal_width = [2.8, 3.0, 2.8],
#         petal_length = [5.6, 5.8, 6.1],
#         petal_width = [2.1, 1.6, 1.9],)
# yhat = predict(mach, Xnew) # probabilistic predictions
# predict_mode(mach, Xnew)   # point predictions
# pdf.(yhat, "virginica")    # probabilities for the "verginica" class
# fitted_params(mach).stumps # raw `Ensemble` object from DecisionTree.jl
# fitted_params(mach).coefs  # coefficient associated with each stump
# ```
# See also
# [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
# the unwrapped model type
# [`MLJDecisionTreeInterface.DecisionTree.AdaBoostStumpClassifier`](@ref).
# """
# # AdaBoostStumpClassifier

# """
# $(MMI.doc_header(DecisionTreeRegressor))
# `DecisionTreeRegressor` implements the $DOC_CART.
# # Training data
# In MLJ or MLJBase, bind an instance `model` to data with
#     mach = machine(model, X, y)
# where
# - `X`: any table of input features (eg, a `DataFrame`) whose columns
#   each have one of the following element scitypes: `Continuous`,
#   `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
# - `y`: the target, which can be any `AbstractVector` whose element
#   scitype is `Continuous`; check the scitype with `scitype(y)`
# Train the machine with `fit!(mach, rows=...)`.
# # Hyper-parameters
# - `max_depth=-1`:          max depth of the decision tree (-1=any)
# - `min_samples_leaf=1`:    max number of samples each leaf needs to have
# - `min_samples_split=2`:   min number of samples needed for a split
# - `min_purity_increase=0`: min purity needed for a split
# - `n_subfeatures=0`: number of features to select at random (0 for all,
#   -1 for square root of number of features)
# - `post_prune=false`:      set to `true` for post-fit pruning
# - `merge_purity_threshold=1.0`: (post-pruning) merge leaves having
#                            combined purity `>= merge_purity_threshold`
# - `rng=Random.GLOBAL_RNG`: random number generator or seed
# # Operations
# - `predict(mach, Xnew)`: return predictions of the target given new
#   features `Xnew` having the same scitype as `X` above.
# # Fitted parameters
# The fields of `fitted_params(mach)` are:
# - `tree`: the tree or stump object returned by the core
#   DecisionTree.jl algorithm
# # Examples
# ```
# using MLJ
# Tree = @load DecisionTreeRegressor pkg=DecisionTree
# tree = Tree(max_depth=4, min_samples_split=3)
# X, y = make_regression(100, 2) # synthetic data
# mach = machine(tree, X, y) |> fit!
# Xnew, _ = make_regression(3, 2)
# yhat = predict(mach, Xnew) # new predictions
# fitted_params(mach).tree # raw tree or stump object from DecisionTree.jl
# ```
# See also
# [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
# the unwrapped model type
# [`MLJDecisionTreeInterface.DecisionTree.DecisionTreeRegressor`](@ref).
# """
# DecisionTreeRegressor

# """
# $(MMI.doc_header(RandomForestRegressor))
# `DecisionTreeRegressor` implements the standard $DOC_RANDOM_FOREST
# # Training data
# In MLJ or MLJBase, bind an instance `model` to data with
#     mach = machine(model, X, y)
# where
# - `X`: any table of input features (eg, a `DataFrame`) whose columns
#   each have one of the following element scitypes: `Continuous`,
#   `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
# - `y`: the target, which can be any `AbstractVector` whose element
#   scitype is `Continuous`; check the scitype with `scitype(y)`
# Train the machine with `fit!(mach, rows=...)`.
# # Hyper-parameters
# - `max_depth=-1`:          max depth of the decision tree (-1=any)
# - `min_samples_leaf=1`:    min number of samples each leaf needs to have
# - `min_samples_split=2`:   min number of samples needed for a split
# - `min_purity_increase=0`: min purity needed for a split
# - `n_subfeatures=-1`: number of features to select at random (0 for all,
#   -1 for square root of number of features)
# - `n_trees=10`:            number of trees to train
# - `sampling_fraction=0.7`  fraction of samples to train each tree on
# - `rng=Random.GLOBAL_RNG`: random number generator or seed
# # Operations
# - `predict(mach, Xnew)`: return predictions of the target given new
#   features `Xnew` having the same scitype as `X` above.
# # Fitted parameters
# The fields of `fitted_params(mach)` are:
# - `forest`: the `Ensemble` object returned by the core DecisionTree.jl algorithm
# # Examples
# ```
# using MLJ
# Forest = @load RandomForestRegressor pkg=DecisionTree
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
# """
# # RandomForestRegressor

end
