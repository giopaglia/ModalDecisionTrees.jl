module ModalDecisionTrees

############################################################################################

import Base: show, length

using FillArrays # TODO remove?
using FunctionWrappers: FunctionWrapper
using Logging: LogLevel, @logmsg
using Printf
using ProgressMeter
using Random
using ReTest
using StatsBase

using SoleBase
using SoleBase: LogOverview, LogDebug, LogDetail, throw_n_log
using SoleBase: spawn_rng, nat_sort
using SoleModels
using SoleModels: CLabel, RLabel, Label, _CLabel, _Label, get_categorical_form
using SoleModels: ConfusionMatrix, overall_accuracy, kappa, class_counts, macro_F1, macro_sensitivity, macro_specificity, macro_PPV, macro_NPV, macro_weighted_F1, macro_weighted_sensitivity, macro_weighted_specificity, macro_weighted_PPV, macro_weighted_NPV, safe_macro_F1, safe_macro_sensitivity, safe_macro_specificity, safe_macro_PPV, safe_macro_NPV
using SoleModels: best_guess, default_weights, slice_weights
using SoleData: max_channel_size,
                nattributes,
                get_instance,
                slice_dataset,
                _slice_dataset,
                instance_channel_size

using SoleModels: AbstractConditionalDataset

import SoleData: nsamples

import SoleModels: feature, test_operator, threshold

import AbstractTrees: print_tree

# Data structures
using SoleModels.ModalLogic

############################################################################################

export slice_dataset, 
       nframes, nsamples, nattributes, max_channel_size

export DTree,                   # Decision tree
        DForest,                # Decision forest
        #
        nnodes, height, modal_height

############################################################################################

# Utility functions
include("utils.jl")

# Decisions at the tree's internal nodes
include("decisions.jl")

# TODO fix
include("interpret-one-step-decisions.jl")

# Loss functions
include("entropy-measures.jl")

# Purity helpers
include("purity.jl")

# Definitions for Decision Leaf, Internal, Node, Tree & Forest
include("base.jl")
include("print.jl")

# Default parameter values
include("default-parameters.jl")

# Metrics for assessing the goodness of a decision leaf/rule
include("leaf-metrics.jl")

# Build a decision tree/forest from a dataset
include("build.jl")

# Perform post-hoc manipulation/analysis on a decision tree/forest (e.g., pruning)
include("posthoc.jl")

# Apply decision tree/forest to a dataset
include("apply.jl")

# Interfaces
include("interfaces/SoleModels.jl")
include("interfaces/MLJ.jl")
include("interfaces/AbstractTrees.jl")

# Experimental features
include("experimentals/main.jl")

# Example datasets
include("other/example-datasets.jl")

end # module
