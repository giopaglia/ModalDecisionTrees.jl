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
using SoleBase: LogOverview, LogDebug, LogDetail, throw_n_log
using SoleBase: spawn, nat_sort

using SoleData
using SoleData: maxchannelsize,
                nvariables,
                get_instance,
                slicedataset,
                instances

using SoleModels
using SoleModels: AbstractLogiset
using SoleModels: CLabel, RLabel, Label, _CLabel, _Label, get_categorical_form

using SoleModels: bestguess, default_weights, slice_weights

import SoleData: ninstances

import SoleModels: feature, test_operator, threshold

import AbstractTrees: print_tree

# Data structures
@reexport using SoleModels.DimensionalDatasets
using SoleModels: MultiLogiset
using SoleModels: WorldSet, GenericDataset

using SoleModels: nfeatures, nrelations,
                            nmodalities, frames, frame,
                            displaystructure,
                            #
                            relations,
                            #
                            GenericDataset,
                            MultiLogiset,
                            AbstractLogiset,
                            DimensionalLogiset,
                            Logiset,
                            SupportedScalarLogiset

using SoleModels: AbstractWorld, AbstractRelation
using SoleModels: AbstractWorldSet, WorldSet
using SoleModels: FullDimensionalFrame

using SoleModels: Ontology, worldtype

import SoleModels: worldtypes

using SoleModels: get_ontology,
                            get_interval_ontology

using SoleModels: OneWorld, OneWorldOntology

using SoleModels: Interval, Interval2D

using SoleModels: IARelations

using SoleModels: existential_aggregator, universal_aggregator, aggregator_bottom

############################################################################################

export slicedataset,
       nmodalities, ninstances, nvariables, maxchannelsize

export DTree,                   # Decision tree
        DForest,                # Decision forest
        RootLevelNeuroSymbolicHybrid,                # Root-level neurosymbolic hybrid model
        #
        nnodes, height, modalheight

ModalityId = Int

############################################################################################

# Utility functions
include("utils.jl")

# TODO fix
include("interpret-onestep-decisions.jl")

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

export build_stump, build_tree, build_forest

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
