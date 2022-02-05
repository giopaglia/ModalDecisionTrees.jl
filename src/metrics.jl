export overall_accuracy,
        kappa,
        GenericPerformanceType,
        macro_sensitivity,
        macro_specificity,
        macro_PPV,
        macro_NPV,
        macro_F1,
        macro_weighted_F1,
        macro_weighted_sensitivity,
        macro_weighted_specificity,
        macro_weighted_PPV,
        macro_weighted_NPV,
        safe_macro_sensitivity,
        safe_macro_specificity,
        safe_macro_PPV,
        safe_macro_NPV,
        safe_macro_F1


struct ConfusionMatrix
    classes::Vector
    matrix::Matrix{Int}
    overall_accuracy::Float64
    kappa::Float64
    mean_accuracy::Float64

    accuracies::Vector{Float64}
    F1s::Vector{Float64}
    sensitivities::Vector{Float64}
    specificities::Vector{Float64}
    PPVs::Vector{Float64}
    NPVs::Vector{Float64}
    ConfusionMatrix(classes::Vector, matrix::Matrix{Int}, overall_accuracy::Float64, kappa::Float64) = begin
        ConfusionMatrix(classes, matrix)
    end
    ConfusionMatrix(matrix::Matrix{Int}) = begin
        ConfusionMatrix(fill("", size(matrix, 1)), matrix)
    end
    ConfusionMatrix(classes::Vector, matrix::Matrix{Int}) = begin
        ALL = sum(matrix)
        T = LinearAlgebra.tr(matrix)
        F = ALL-T

        @assert size(matrix,1) == size(matrix,2) "Can't instantiate ConfusionMatrix with matrix of size ($(size(matrix))"

        n_classes = size(matrix,1)
        @assert length(classes) == n_classes "Can't instantiate ConfusionMatrix with mismatching n_classes ($(n_classes)) and classes $(classes)"
        overall_accuracy = T / ALL
        prob_chance = (sum(matrix,dims=1) * sum(matrix,dims=2))[1] / ALL^2
        kappa = (overall_accuracy - prob_chance) / (1.0 - prob_chance)

        TPs = Vector{Float64}(undef, n_classes)
        TNs = Vector{Float64}(undef, n_classes)
        FPs = Vector{Float64}(undef, n_classes)
        FNs = Vector{Float64}(undef, n_classes)

        for i in 1:n_classes
            class = i
            other_classes = [(1:i-1)..., (i+1:n_classes)...]
            TPs[i] = sum(matrix[class,class])
            TNs[i] = sum(matrix[other_classes,other_classes])
            FNs[i] = sum(matrix[class,other_classes])
            FPs[i] = sum(matrix[other_classes,class])
        end

        # https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification
        accuracies = (TPs .+ TNs)./ALL
        mean_accuracy = StatsBase.mean(accuracies)

        # https://en.wikipedia.org/wiki/F-score
        F1s           = TPs./(TPs.+.5*(FPs.+FNs))
        # https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        sensitivities = TPs./(TPs.+FNs)
        specificities = TNs./(TNs.+FPs)
        PPVs          = TPs./(TPs.+FPs)
        NPVs          = TNs./(TNs.+FNs)

        new(classes, matrix, overall_accuracy, kappa, mean_accuracy, accuracies, F1s, sensitivities, specificities, PPVs, NPVs)
    end
end

overall_accuracy(cm::ConfusionMatrix) = cm.overall_accuracy
kappa(cm::ConfusionMatrix)            = cm.kappa

class_counts(cm::ConfusionMatrix) = sum(cm.matrix,dims=2)


# Useful arcticles: 
# - https://towardsdatascience.com/a-tale-of-two-macro-f1s-8811ddcf8f04
# - https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
# - https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
# - https://www.datascienceblog.net/post/machine-learning/performance-measures-multi-class-problems/
# - https://towardsdatascience.com/multi-class-metrics-made-simple-the-kappa-score-aka-cohens-kappa-coefficient-bdea137af09c

# NOTES:
"""
macro-accuracy  = mean accuracy  = avg. accuracy (AA or MA)
macro-precision = mean precision = avg. of precisions
macro-recall    = mean recall    = avg. of recalls

macro-F1     = avg of F1 score of each class (sklearn uses this one)
micro-F1     = F1 score calculated using the global precision and global recall
Note: a second 2nd definition of macro-F1, less used = F1 score calculated using macro-precision and macro-recall (avg. of recalls)

rules:
-   micro-F1 = micro-precision = micro-recall = overall accuracy
- overall accuracy = (weighted) macro-recall (thus, if the test set is perfectly balanced: overall accuracy = macro-recall)

Note:
- The flaw of F1-score (and accuracy?) is that they give equal weight to precision and recall
- "the relative importance assigned to precision and recall should be an aspect of the problem" - David Hand
"""

# macro_F1(cm::ConfusionMatrix) = StatsBase.mean(cm.F1s)
# macro_sensitivity(cm::ConfusionMatrix) = StatsBase.mean(cm.sensitivities)
# # macro_specificity(cm::ConfusionMatrix) = StatsBase.mean(cm.specificities)
# macro_PPV(cm::ConfusionMatrix) = StatsBase.mean(cm.PPVs)
# macro_NPV(cm::ConfusionMatrix) = StatsBase.mean(cm.NPVs)

# macro_weighted_F1(cm::ConfusionMatrix) = StatsBase.sum(cm.F1s.*class_counts(cm))./sum(cm.matrix)
# macro_weighted_sensitivity(cm::ConfusionMatrix) = StatsBase.sum(cm.sensitivities.*class_counts(cm))./sum(cm.matrix)
# # macro_weighted_specificity(cm::ConfusionMatrix) = StatsBase.sum(cm.specificities.*class_counts(cm))./sum(cm.matrix)
# macro_weighted_PPV(cm::ConfusionMatrix) = StatsBase.sum(cm.PPVs.*class_counts(cm))./sum(cm.matrix)
# macro_weighted_NPV(cm::ConfusionMatrix) = StatsBase.sum(cm.NPVs.*class_counts(cm))./sum(cm.matrix)

# macro_sensitivity, also called unweighted average recall (UAR)
macro_sensitivity(cm::ConfusionMatrix) = StatsBase.mean(cm.sensitivities)
macro_specificity(cm::ConfusionMatrix) = StatsBase.mean(cm.specificities)
macro_PPV(cm::ConfusionMatrix)         = StatsBase.mean(cm.PPVs)
macro_NPV(cm::ConfusionMatrix)         = StatsBase.mean(cm.NPVs)
macro_F1(cm::ConfusionMatrix)          = StatsBase.mean(cm.F1s)

macro_weighted_F1(cm::ConfusionMatrix)  = length(cm.classes) == 2 ? throw_n_log("macro_weighted_F1 Binary case?") : StatsBase.sum(cm.F1s.*class_counts(cm))./sum(cm.matrix)
macro_weighted_sensitivity(cm::ConfusionMatrix) = length(cm.classes) == 2 ? throw_n_log("macro_weighted_sensitivity Binary case?") : StatsBase.sum(cm.sensitivities.*class_counts(cm))./sum(cm.matrix)
macro_weighted_specificity(cm::ConfusionMatrix) = length(cm.classes) == 2 ? throw_n_log("# Binary case?") : StatsBase.sum(cm.specificities.*class_counts(cm))./sum(cm.matrix)
macro_weighted_PPV(cm::ConfusionMatrix) = length(cm.classes) == 2 ? throw_n_log("macro_weighted_PPV Binary case?") : StatsBase.sum(cm.PPVs.*class_counts(cm))./sum(cm.matrix)
macro_weighted_NPV(cm::ConfusionMatrix) = length(cm.classes) == 2 ? throw_n_log("macro_weighted_NPV Binary case?") : StatsBase.sum(cm.NPVs.*class_counts(cm))./sum(cm.matrix)

safe_macro_sensitivity(cm::ConfusionMatrix) = length(cm.classes) == 2 ? cm.sensitivities[1] : macro_sensitivity(cm)
safe_macro_specificity(cm::ConfusionMatrix) = length(cm.classes) == 2 ? cm.specificities[1] : macro_specificity(cm)
safe_macro_PPV(cm::ConfusionMatrix)         = length(cm.classes) == 2 ? cm.PPVs[1]          : macro_PPV(cm)
safe_macro_NPV(cm::ConfusionMatrix)         = length(cm.classes) == 2 ? cm.NPVs[1]          : macro_NPV(cm)
safe_macro_F1(cm::ConfusionMatrix)          = length(cm.classes) == 2 ? cm.F1s[1]           : macro_F1(cm)

function show(io::IO, cm::ConfusionMatrix)
    # print(io, "classes:  ")
    # show(io, cm.classes)

    max_num_digits = maximum(length(string(val)) for val in cm.matrix)

    println("  accuracy: ", round(overall_accuracy(cm)*100, digits=2), "% kappa: ", round(cm.kappa*100, digits=2), "% ")
    for (i,(row,class)) in enumerate(zip(eachrow(cm.matrix),cm.classes))
        for val in row
            print(lpad(val,max_num_digits+1," "))
        end
        println("\t$(round(100*row[i]/sum(row), digits=2))%\t\t" * class)
    end

    # print(io, "\nmatrix:   ")
    # display(cm.matrix)
    print(io, "\noverall_acc:\t")
    show(io, overall_accuracy(cm))
    print(io, "\nκ =\t\t")
    show(io, cm.kappa)
    print(io, "\nsensitivities:\t")
    show(io, cm.sensitivities)
    print(io, "\nspecificities:\t")
    show(io, cm.specificities)
    print(io, "\nPPVs:\t\t")
    show(io, cm.PPVs)
    print(io, "\nNPVs:\t\t")
    show(io, cm.NPVs)
    print(io, "\nF1s:\t\t")
    show(io, cm.F1s)
    print(io, "\naccuracies:\t")
    show(io, cm.accuracies)
    print(io, "\nmean_accuracy:\t")
    show(io, cm.mean_accuracy)
end

function _weighted_error(actual::AbstractVector, predicted::AbstractVector, weights::AbstractVector{T}) where T <: Real
    mismatches = actual .!= predicted
    err = sum(weights[mismatches]) / sum(weights)
    return err
end

# TODO rewrite majority_vote
function majority_vote(labels::AbstractVector; suppress_parity_warning = false)
    if length(labels) == 0
        return nothing
    end
    counts = countmap(labels)
    if !suppress_parity_warning && sum(counts[argmax(counts)] .== values(counts)) > 1
        println("Warning: parity encountered in majority_vote.")
        println("Vector ($(length(labels)) elements): $(labels)")
        println("Argmax: $(argmax(counts))")
        println("Max: $(counts[argmax(counts)]) (sum = $(sum(values(counts))))")
    end
    argmax(counts)
end

function majority_vote(labels::AbstractVector{T}, weights::Union{Nothing,AbstractVector{N}}; suppress_parity_warning = false) where {T, N<:Real}
    if isnothing(weights)
        # TODO use dispatch on majority_vote ArrayOfOnes o ConstantArray
        return majority_vote(labels; suppress_parity_warning = suppress_parity_warning)
    end

    if length(labels) == 0
        return nothing
    end

    @assert length(labels) === length(weights) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."
    # @assert length(labels) != 0 "Can't compute majority_vote with 0 predictions" # TODO figure out whether we want to force this or returning nothing is fine.

    counts = Dict{T,AbstractFloat}()
    for i in 1:length(labels)
        l = labels[i]
        counts[l] = get(counts, l, 0) + weights[i]
    end

    if !suppress_parity_warning && sum(counts[argmax(counts)] .== values(counts)) > 1
        println("Warning: parity encountered in majority_vote.")
        println("Vector ($(length(labels)) elements): $(labels)")
        println("Argmax: $(argmax(counts))")
        println("Max: $(counts[argmax(counts)]) (sum = $(sum(values(counts))))")
    end
    return argmax(counts)
end

const PerformanceStruct = NamedTuple
const GenericPerformanceType = Union{ConfusionMatrix,PerformanceStruct}

### Classification ###
function confusion_matrix(actual::AbstractVector, predicted::AbstractVector, weights = nothing)
    @assert isnothing(weights) "TODO Expand code: Non-nothing weights encountered in confusion_matrix()"
    @assert length(actual) == length(predicted)
    N = length(actual)
    _actual = zeros(Int, N)
    _predicted = zeros(Int, N)

    class_labels = unique([actual; predicted])
    class_labels = sort(class_labels, lt=util.nat_sort)
    # Binary case: sort the classes with as ["YES_...", "NO_..."]
    if length(class_labels) == 2
        class_labels = reverse(class_labels)
    end
    
    N = length(class_labels)
    for i in 1:N
        _actual[actual .== class_labels[i]] .= i
        _predicted[predicted .== class_labels[i]] .= i
    end
    # CM[actual,predicted]
    CM = zeros(Int,N,N)
    for i in zip(_actual, _predicted)
        CM[i[1],i[2]] += 1
    end
    return ConfusionMatrix(class_labels, CM)
end

### Regression ###

# Coefficient of determination
function R2(actual, predicted)
  @assert length(actual) == length(predicted)
  ss_residual = sum((actual - predicted).^2)
  ss_total = sum((actual .- mean(actual)).^2)
  return 1.0 - ss_residual/ss_total
end

# TODO keep actual and predicted (both test and train?), so that I can plot the average error
function confusion_matrix(actual::AbstractVector{Float64}, predicted::AbstractVector{Float64}, weights = nothing)
    # @assert length(actual) == length(predicted)
    (
        cor   = cor(actual, predicted),
        MAE   = sum(abs.(actual .- predicted)) / length(predicted),
        # MSE   = mean((actual - predicted).^2)
        RMSE  = StatsBase.rmsd(actual, predicted),
        R2    = R2(actual, predicted),
        # TODO add MAPE, MSE
    )
end


# function _nfoldCV(classifier::Symbol, labels::AbstractVector{T}, features::AbstractMatrix{S}, args...; verbose, rng) where {S, T}
#   _rng = rng
#   nfolds = args[1]
#   if nfolds < 2
#       throw_n_log("number of folds must be greater than 1")
#   end
#   if classifier == :tree
#       pruning_purity      = args[2]
#       max_depth           = args[3]
#       min_samples_leaf    = args[4]
#       min_samples_split   = args[5]
#       min_purity_increase = args[6]
#   elseif classifier == :forest
#       n_subfeatures       = args[2]
#       n_trees             = args[3]
#       partial_sampling    = args[4]
#       max_depth           = args[5]
#       min_samples_leaf    = args[6]
#       min_samples_split   = args[7]
#       min_purity_increase = args[8]
#   elseif classifier == :stumps
#       n_iterations        = args[2]
#   end
#   N = length(labels)
#   ntest = floor(Int, N / nfolds)
#   inds = Random.randperm(_rng, N)
#   accuracy = zeros(nfolds)
#   for i in 1:nfolds
#       test_inds = falses(N)
#       test_inds[(i - 1) * ntest + 1 : i * ntest] .= true
#       train_inds = (!).(test_inds)
#       test_features = features[inds[test_inds],:]
#       test_labels = labels[inds[test_inds]]
#       train_features = features[inds[train_inds],:]
#       train_labels = labels[inds[train_inds]]

#       if classifier == :tree
#           n_subfeatures = 0
#           model = build_tree(train_labels, train_features,
#                  n_subfeatures,
#                  max_depth,
#                  min_samples_leaf,
#                  min_samples_split,
#                  min_purity_increase;
#                  rng = rng)
#           if pruning_purity < 1.0
#               model = prune_tree(model, pruning_purity)
#           end
#           predictions = apply_tree(model, test_features)
#       elseif classifier == :forest
#           model = build_forest(
#                       train_labels, train_features,
#                       n_subfeatures,
#                       n_trees,
#                       partial_sampling,
#                       max_depth,
#                       min_samples_leaf,
#                       min_samples_split,
#                       min_purity_increase;
#                       rng = rng)
#           predictions = apply_forest(model, test_features)
#       elseif classifier == :stumps
#           model, coeffs = build_adaboost_stumps(
#               train_labels, train_features, n_iterations)
#           predictions = apply_adaboost_stumps(model, coeffs, test_features)
#       end
#       cm = confusion_matrix(test_labels, predictions)
#       accuracy[i] = cm.accuracy
#       if verbose
#           println("\nFold ", i)
#           println(cm)
#       end
#   end
#   println("\nMean Accuracy: ", mean(accuracy))
#   return accuracy
# end

# function nfoldCV_tree(
#       labels              :: AbstractVector{T},
#       features            :: AbstractMatrix{S},
#       n_folds             :: Integer,
#       pruning_purity      :: Float64 = 1.0,
#       max_depth           :: Integer = -1,
#       min_samples_leaf    :: Integer = 1,
#       min_samples_split   :: Integer = 2,
#       min_purity_increase :: Float64 = 0.0;
#       verbose             :: Bool = true,
#       rng                 = Random.GLOBAL_RNG) where {S, T}
#   _nfoldCV(:tree, labels, features, n_folds, pruning_purity, max_depth,
#               min_samples_leaf, min_samples_split, min_purity_increase; verbose=verbose, rng=rng)
# end
# function nfoldCV_forest(
#       labels              :: AbstractVector{T},
#       features            :: AbstractMatrix{S},
#       n_folds             :: Integer,
#       n_subfeatures       :: Integer = -1,
#       n_trees             :: Integer = 10,
#       partial_sampling    :: Float64 = 0.7,
#       max_depth           :: Integer = -1,
#       min_samples_leaf    :: Integer = 1,
#       min_samples_split   :: Integer = 2,
#       min_purity_increase :: Float64 = 0.0;
#       verbose             :: Bool = true,
#       rng                 = Random.GLOBAL_RNG) where {S, T}
#   _nfoldCV(:forest, labels, features, n_folds, n_subfeatures, n_trees, partial_sampling,
#               max_depth, min_samples_leaf, min_samples_split, min_purity_increase; verbose=verbose, rng=rng)
# end
# function nfoldCV_stumps(
#       labels       ::AbstractVector{T},
#       features     ::AbstractMatrix{S},
#       n_folds      ::Integer,
#       n_iterations ::Integer = 10;
#       verbose             :: Bool = true,
#       rng          = Random.GLOBAL_RNG) where {S, T}
#   _nfoldCV(:stumps, labels, features, n_folds, n_iterations; verbose=verbose, rng=rng)
# end

# ### Regression ###

# function _nfoldCV(regressor::Symbol, labels::AbstractVector{T}, features::AbstractMatrix, args...; verbose, rng) where T <: Float64
#   _rng = rng
#   nfolds = args[1]
#   if nfolds < 2
#       throw_n_log("number of folds must be greater than 1")
#   end
#   if regressor == :tree
#       pruning_purity      = args[2]
#       max_depth           = args[3]
#       min_samples_leaf    = args[4]
#       min_samples_split   = args[5]
#       min_purity_increase = args[6]
#   elseif regressor == :forest
#       n_subfeatures       = args[2]
#       n_trees             = args[3]
#       partial_sampling    = args[4]
#       max_depth           = args[5]
#       min_samples_leaf    = args[6]
#       min_samples_split   = args[7]
#       min_purity_increase = args[8]
#   end
#   N = length(labels)
#   ntest = floor(Int, N / nfolds)
#   inds = Random.randperm(_rng, N)
#   R2s = zeros(nfolds)
#   for i in 1:nfolds
#       test_inds = falses(N)
#       test_inds[(i - 1) * ntest + 1 : i * ntest] .= true
#       train_inds = (!).(test_inds)
#       test_features = features[inds[test_inds],:]
#       test_labels = labels[inds[test_inds]]
#       train_features = features[inds[train_inds],:]
#       train_labels = labels[inds[train_inds]]
#       if regressor == :tree
#           n_subfeatures = 0
#           model = build_tree(train_labels, train_features,
#                  n_subfeatures,
#                  max_depth,
#                  min_samples_leaf,
#                  min_samples_split,
#                  min_purity_increase;
#                  rng = rng)
#           if pruning_purity < 1.0
#               model = prune_tree(model, pruning_purity)
#           end
#           predictions = apply_tree(model, test_features)
#       elseif regressor == :forest
#           model = build_forest(
#                       train_labels, train_features,
#                       n_subfeatures,
#                       n_trees,
#                       partial_sampling,
#                       max_depth,
#                       min_samples_leaf,
#                       min_samples_split,
#                       min_purity_increase;
#                       rng = rng)
#           predictions = apply_forest(model, test_features)
#       end
#       err = mean_squared_error(test_labels, predictions)
#       corr = cor(test_labels, predictions)
#       r2 = R2(test_labels, predictions)
#       R2s[i] = r2
#       if verbose
#           println("\nFold ", i)
#           println("Mean Squared Error:     ", err)
#           println("Correlation Coeff:      ", corr)
#           println("Coeff of Determination: ", r2)
#       end
#   end
#   println("\nMean Coeff of Determination: ", mean(R2s))
#   return R2s
# end

# function nfoldCV_tree(
#   labels              :: AbstractVector{T},
#   features            :: AbstractMatrix{S},
#   n_folds             :: Integer,
#   pruning_purity      :: Float64 = 1.0,
#   max_depth           :: Integer = -1,
#   min_samples_leaf    :: Integer = 5,
#   min_samples_split   :: Integer = 2,
#   min_purity_increase :: Float64 = 0.0;
#   verbose             :: Bool = true,
#   rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}
# _nfoldCV(:tree, labels, features, n_folds, pruning_purity, max_depth,
#           min_samples_leaf, min_samples_split, min_purity_increase; verbose=verbose, rng=rng)
# end
# function nfoldCV_forest(
#   labels              :: AbstractVector{T},
#   features            :: AbstractMatrix{S},
#   n_folds             :: Integer,
#   n_subfeatures       :: Integer = -1,
#   n_trees             :: Integer = 10,
#   partial_sampling    :: Float64 = 0.7,
#   max_depth           :: Integer = -1,
#   min_samples_leaf    :: Integer = 5,
#   min_samples_split   :: Integer = 2,
#   min_purity_increase :: Float64 = 0.0;
#   verbose             :: Bool = true,
#   rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}
# _nfoldCV(:forest, labels, features, n_folds, n_subfeatures, n_trees, partial_sampling,
#           max_depth, min_samples_leaf, min_samples_split, min_purity_increase; verbose=verbose, rng=rng)
# end
