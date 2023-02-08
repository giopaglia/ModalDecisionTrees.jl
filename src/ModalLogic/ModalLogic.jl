module ModalLogic

using ..ModalDecisionTrees

using ..ModalDecisionTrees: DimensionalDataset, AbstractDimensionalChannel, AbstractDimensionalInstance, UniformDimensionalDataset, DimensionalChannel, DimensionalInstance

using ..ModalDecisionTrees: LogOverview, LogDebug, LogDetail

using BenchmarkTools
using ComputedFieldTypes
using DataStructures
using Logging: @logmsg
using ResumableFunctions

using SoleLogics
using SoleLogics: AbstractRelation, AbstractWorld

using SoleModels

import Base: size, show, getindex, iterate, length, push!

# Reexport from SoleLogics:
export AbstractWorld, AbstractRelation
export AbstractWorldSet, WorldSet
export RelationGlob, RelationId

export Ontology, worldtype, world_types

using SoleLogics: AbstractFrame, FullDimensionalFrame
using SoleModels: ActiveConditionalDataset, FeatCondition

import SoleLogics: accessibles, allworlds
import SoleModels: representatives, FeatMetaCondition
import SoleModels.utils: minify

using SoleModels: AbstractMultiModalFrame

using SoleData: _isnan
import SoleData: hasnans

import SoleLogics: goeswith
import SoleLogics: initialworldset
using SoleLogics: InitCondition
import SoleLogics: worldtype

# Concrete type for ontologies
include("ontology.jl")

############################################################################################
# Dataset structures
############################################################################################
import ..ModalDecisionTrees: concat_datasets,
       nsamples, nattributes, max_channel_size, get_instance,
       instance_channel_size


export nfeatures, nrelations,
       nframes, frames, get_frame,
       display_structure,
       #
       relations,
       #
       ModalDataset,
       GenericModalDataset,
       ActiveMultiFrameModalDataset,
       MultiFrameModalDataset,
       ActiveModalDataset,
       InterpretedModalDataset,
       ExplicitModalDataset,
       ExplicitModalDatasetS

# A modal dataset can be *active* or *passive*.
# 
# A passive modal dataset is one that you can interpret decisions on, but cannot necessarily
#  enumerate decisions for, as it doesn't have objects for storing the logic (relations, features, etc.).
# Dimensional datasets are passive.
include("datasets/dimensional-dataset.jl")
# 
const PassiveModalDataset{T} = Union{DimensionalDataset{T}}
# 
# Active datasets comprehend structures for representing relation sets, features, enumerating worlds,
#  etc. While learning a model can be done only with active modal datasets, testing a model
#  can be done with both active and passive modal datasets.
# 
abstract type ActiveModalDataset{T<:Number,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}} <: ActiveConditionalDataset{W,FeatCondition,Bool,FR} end

# By default an active modal dataset cannot be miniaturized
isminifiable(::ActiveModalDataset) = false
#
const ModalDataset{T} = Union{PassiveModalDataset{T},ActiveModalDataset{T}}
#
include("interpret-one-step-decisions.jl")
#
include("datasets/main.jl")
#
include("gamma-access.jl")
#
# Define the multi-modal version of modal datasets (basically, a vector of datasets with the
#  same number of instances)
#
include("multi-frame-dataset.jl")
#
# TODO figure out which convert function works best: convert(::Type{<:MultiFrameModalDataset{T}}, X::MD) where {T,MD<:ModalDataset{T}} = MultiFrameModalDataset{MD}([X])
# convert(::Type{<:MultiFrameModalDataset}, X::ModalDataset) = MultiFrameModalDataset([X])
#
const ActiveMultiFrameModalDataset{T} = MultiFrameModalDataset{<:ActiveModalDataset{<:T}}
#
const GenericModalDataset = Union{ModalDataset,MultiFrameModalDataset}
#

# Dimensional Ontologies
include("dimensional-ontologies.jl")

############################################################################################

end # module
