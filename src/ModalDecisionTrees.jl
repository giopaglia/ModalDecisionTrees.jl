__precompile__()

module ModalDecisionTrees

############################################################################################

import Base: show, length

using FillArrays # TODO remove?
using FunctionWrappers: FunctionWrapper
using Logging: LogLevel, @logmsg
using Printf
using ProgressMeter
using Random
using StatsBase

using SoleBase
using SoleBase: LogOverview, LogDebug, LogDetail, throw_n_log
using SoleBase: spawn_rng, nat_sort
using SoleModels
using SoleModels: CLabel, RLabel, Label, _CLabel, _Label, get_categorical_form

using SoleModels: bestguess, default_weights, slice_weights
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
        RootLevelNeuroSymbolicHybrid,                # Root-level neurosymbolic hybrid model
        #
        nnodes, height, modalheight

FrameId = Int

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
