export overall_accuracy,
        kappa,
        PerformanceStruct,
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

const PerformanceStruct = NamedTuple

################################################################################
################################################################################
################################################################################

struct ConfusionMatrix
    ############################################################################
    class_names::Vector{String}
    matrix::Matrix{Int}
    ############################################################################
    overall_accuracy::Float64
    kappa::Float64
    mean_accuracy::Float64
    ############################################################################
    accuracies::Vector{Float64}
    F1s::Vector{Float64}
    sensitivities::Vector{Float64}
    specificities::Vector{Float64}
    PPVs::Vector{Float64}
    NPVs::Vector{Float64}
    ############################################################################
    
    function ConfusionMatrix(matrix::Matrix)
        ConfusionMatrix(string.(1:size(matrix, 1)), matrix)
    end
    function ConfusionMatrix(class_names::Vector, matrix::Matrix)
        
        @assert size(matrix,1) == size(matrix,2) "Can't instantiate ConfusionMatrix with matrix of size ($(size(matrix))"
        n_classes = size(matrix,1)
        @assert length(class_names) == n_classes "Can't instantiate ConfusionMatrix with mismatching n_classes ($(n_classes)) and class_names $(class_names)"

        ALL = sum(matrix)
        T = LinearAlgebra.tr(matrix)
        F = ALL-T

        overall_accuracy = T / ALL
        prob_chance = (sum(matrix,dims=1) * sum(matrix,dims=2))[1] / ALL^2
        kappa = (overall_accuracy - prob_chance) / (1.0 - prob_chance)

        ########################################################################
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
        ########################################################################

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

        new(class_names,
            matrix,
            overall_accuracy,
            kappa,
            mean_accuracy,
            accuracies,
            F1s,
            sensitivities,
            specificities,
            PPVs,
            NPVs,
        )
    end

    function ConfusionMatrix(
            actual::AbstractVector{L},
            predicted::AbstractVector{L},
            weights = nothing,
        ) where {L<:CLabel}
        @assert length(actual) == length(predicted) "Can't compute ConfusionMatrixs with uneven number of actual $(length(actual)) and predicted $(length(predicted)) labels."
        
        if isnothing(weights)
            weights = UniformVector{Int64}(1,length(actual))
        end
        @assert length(actual) == length(weights)   "Can't compute ConfusionMatrixs with uneven number of actual $(length(actual)) and weights $(length(weights)) labels."

        class_labels = begin
            class_labels = unique([actual; predicted])
            class_labels = sort(class_labels, lt=util.nat_sort)
            # Binary case: retain order of classes YES/NO
            if length(class_labels) == 2 &&
                    startswith(class_labels[1], "YES") &&
                    startswith(class_labels[2], "NO")
                class_labels = reverse(class_labels)
            end
            class_labels
        end

        n_instances = length(actual)
        _actual    = zeros(Int, n_instances)
        _predicted = zeros(Int, n_instances)

        n_classes = length(class_labels)
        for i in 1:n_classes
            _actual[actual .== class_labels[i]] .= i
            _predicted[predicted .== class_labels[i]] .= i
        end

        matrix = zeros(Int,n_classes,n_classes)
        for (act,pred,w) in zip(_actual, _predicted, weights)
            matrix[act,pred] += w
        end
        ConfusionMatrix(class_labels, matrix)
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
macro_F1(cm::ConfusionMatrix)          = StatsBase.mean(cm.F1s)
macro_sensitivity(cm::ConfusionMatrix) = StatsBase.mean(cm.sensitivities)
macro_specificity(cm::ConfusionMatrix) = StatsBase.mean(cm.specificities)
macro_PPV(cm::ConfusionMatrix)         = StatsBase.mean(cm.PPVs)
macro_NPV(cm::ConfusionMatrix)         = StatsBase.mean(cm.NPVs)

macro_weighted_F1(cm::ConfusionMatrix)  = length(cm.class_names) == 2 ? throw_n_log("macro_weighted_F1 Binary case?") : StatsBase.sum(cm.F1s.*class_counts(cm))./sum(cm.matrix)
macro_weighted_sensitivity(cm::ConfusionMatrix) = length(cm.class_names) == 2 ? throw_n_log("macro_weighted_sensitivity Binary case?") : StatsBase.sum(cm.sensitivities.*class_counts(cm))./sum(cm.matrix)
macro_weighted_specificity(cm::ConfusionMatrix) = length(cm.class_names) == 2 ? throw_n_log("# Binary case?") : StatsBase.sum(cm.specificities.*class_counts(cm))./sum(cm.matrix)
macro_weighted_PPV(cm::ConfusionMatrix) = length(cm.class_names) == 2 ? throw_n_log("macro_weighted_PPV Binary case?") : StatsBase.sum(cm.PPVs.*class_counts(cm))./sum(cm.matrix)
macro_weighted_NPV(cm::ConfusionMatrix) = length(cm.class_names) == 2 ? throw_n_log("macro_weighted_NPV Binary case?") : StatsBase.sum(cm.NPVs.*class_counts(cm))./sum(cm.matrix)

safe_macro_F1(cm::ConfusionMatrix)          = length(cm.class_names) == 2 ? cm.F1s[1]           : macro_F1(cm)
safe_macro_sensitivity(cm::ConfusionMatrix) = length(cm.class_names) == 2 ? cm.sensitivities[1] : macro_sensitivity(cm)
safe_macro_specificity(cm::ConfusionMatrix) = length(cm.class_names) == 2 ? cm.specificities[1] : macro_specificity(cm)
safe_macro_PPV(cm::ConfusionMatrix)         = length(cm.class_names) == 2 ? cm.PPVs[1]          : macro_PPV(cm)
safe_macro_NPV(cm::ConfusionMatrix)         = length(cm.class_names) == 2 ? cm.NPVs[1]          : macro_NPV(cm)

function Base.show(io::IO, cm::ConfusionMatrix)
    
    max_num_digits = maximum(length(string(val)) for val in cm.matrix)

    println(io, "Confusion Matrix ($(length(cm.class_names)) classes):")
    for (i,(row,class_name,sensitivity)) in enumerate(zip(eachrow(cm.matrix),cm.class_names,cm.sensitivities))
        for val in row
            print(io, lpad(val,max_num_digits+1," "))
        end
        println(io, "\t\t\t$(round(100*sensitivity, digits=2))%\t\t" * class_name)
    end

    ############################################################################
    println(io, "accuracy =\t\t$(round(overall_accuracy(cm), digits=4))")
    println(io, "κ =\t\t\t\t$(round(cm.kappa, digits=4))")
    ############################################################################
    println(io, "sensitivities:\t$(round.(cm.sensitivities, digits=4))")
    println(io, "specificities:\t$(round.(cm.specificities, digits=4))")
    println(io, "PPVs:\t\t$(round.(cm.PPVs, digits=4))")
    println(io, "NPVs:\t\t$(round.(cm.NPVs, digits=4))")
    print(io,   "F1s:\t\t$(round.(cm.F1s, digits=4))")
    println(io, "\t\tmean_F1:\t$(round(cm.mean_accuracy, digits=4))")
    print(io,   "accuracies:\t\t$(round.(cm.accuracies, digits=4))")
    println(io, "\tmean_accuracy:\t$(round(cm.mean_accuracy, digits=4))")
end

################################################################################
################################################################################
################################################################################

### Classification ###

function compute_metrics(
        actual::AbstractVector{L},
        predicted::AbstractVector{L},
        weights = nothing,
    ) where {L<:CLabel}
    @assert length(actual) == length(predicted) "Can't compute_metrics with uneven number of actual $(length(actual)) and predicted $(length(predicted)) labels."
    (;
        # n_inst = length(actual),
        # 
        # actual = actual,
        # predicted = predicted,
        # weights = weights,
        # 
        cm = ConfusionMatrix(actual, predicted, weights),
    )
end
confusion_matrix = compute_metrics # TODO remove

### Regression ###

function compute_metrics(
        actual::AbstractVector{L},
        predicted::AbstractVector{L},
        weights = nothing,
    ) where {L<:RLabel}
    @assert isnothing(weights) "TODO Expand code: Non-nothing weights encountered in compute_metrics()"
    @assert length(actual) == length(predicted) "Can't compute_metrics with uneven number of actual $(length(actual)) and predicted $(length(predicted)) labels."
    (;
        # n_inst = length(actual),
        # 
        # actual = actual,
        # predicted = predicted,
        # weights = weights,
        # 
        cor   = cor(actual, predicted),
        MAE   = sum(abs.(actual .- predicted)) / length(predicted),
        # MSE   = mean((actual - predicted).^2)
        RMSE  = StatsBase.rmsd(actual, predicted),
        R2    = R2(actual, predicted),
        # TODO add MAPE, MSE
    )
end

################################################################################
################################################################################
################################################################################
