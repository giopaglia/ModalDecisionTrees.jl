__precompile__()

module ModalDecisionTrees

############################################################################################

import Base: show, length

using FunctionWrappers: FunctionWrapper
using Logging: LogLevel, @logmsg
using Printf
using ProgressMeter
using Random
using Reexport
using StatsBase

using SoleBase
using SoleBase: LogOverview, LogDebug, LogDetail
using SoleBase: spawn, nat_sort

using SoleData
using SoleData: maxchannelsize,
                nvariables,
                get_instance,
                slicedataset

using SoleModels
using SoleModels: AbstractLogiset
using SoleModels: CLabel, RLabel, Label, _CLabel, _Label, get_categorical_form

using SoleModels: bestguess, default_weights, slice_weights

import SoleModels: feature, test_operator, threshold

import AbstractTrees: print_tree

# Data structures
@reexport using SoleModels.DimensionalDatasets
using SoleModels: MultiLogiset
using SoleModels: WorldSet, GenericDataset

using SoleModels: nfeatures, nrelations,
                            nmodalities, eachmodality, modality,
                            displaystructure,
                            #
                            relations,
                            #
                            GenericDataset,
                            MultiLogiset,
                            AbstractLogiset,
                            SupportedLogiset

using SoleModels: AbstractWorld, AbstractRelation
using SoleModels: AbstractWorldSet, WorldSet

using SoleModels: worldtype

using SoleModels: OneWorld

using SoleModels: Interval, Interval2D

using SoleModels: IARelations, IA2DRelations

using SoleLogics: FullDimensionalFrame

using SoleModels: existential_aggregator, universal_aggregator, aggregator_bottom

import SoleModels: nnodes
import SoleModels: nleaves
import SoleModels: height

############################################################################################

export slicedataset,
       nmodalities, ninstances, nvariables, maxchannelsize

export DTree,                         # Decision tree
        DForest,                      # Decision forest
        RootLevelNeuroSymbolicHybrid, # Root-level neurosymbolic hybrid model
        #
        nnodes, height, modalheight

############################################################################################

ModalityId = Int

# Utility functions
include("utils.jl")

# Loss functions
include("entropy-measures.jl")

# Purity helpers
include("purity.jl")


export SimpleDecision,
       ScalarExistentialFormula,
       displaydecision

# Definitions for Decision Leaf, Internal, Node, Tree & Forest
include("base.jl")

include("print.jl")

# # Default parameter values
# include("default-parameters.jl")

# Metrics for assessing the goodness of a decision leaf/rule
include("leaf-metrics.jl")

# One-step decisions
include("interpret-onestep-decisions.jl")

# Build a decision tree/forest from a dataset
include("build.jl")

# Perform post-hoc manipulation/analysis on a decision tree/forest (e.g., pruning)
include("posthoc.jl")

# Apply decision tree/forest to a dataset
include("apply.jl")

export ModalDecisionTree, ModalRandomForest

# Interfaces
include("interfaces/SoleModels.jl")
include("interfaces/MLJ.jl")
include("interfaces/AbstractTrees.jl")

# Experimental features
include("experimentals/main.jl")

# Example datasets
include("other/example-datasets.jl")

end # module
