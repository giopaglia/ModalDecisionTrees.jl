module ModalLogic

using ..ModalDecisionTrees

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

using SoleLogics: AbstractFrame, AbstractDimensionalFrame, FullDimensionalFrame
using SoleModels: AbstractConditionalDataset, AbstractCondition, FeatMetaCondition, FeatCondition

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

import ..ModalDecisionTrees: nsamples, nattributes, max_channel_size, get_instance,
       instance_channel_size


export nfeatures, nrelations,
       nframes, frames, get_frame,
       display_structure,
       #
       relations,
       #
       GenericModalDataset,
       ActiveMultiFrameModalDataset,
       MultiFrameModalDataset,
       ActiveFeaturedDataset,
       DimensionalFeaturedDataset,
       FeaturedDataset,
       SupportedFeaturedDataset

# Concrete type for ontologies
include("ontology.jl")

# Dataset structures
# 
include("datasets/main.jl")
# 
include("interpret-one-step-decisions.jl")
# 
include("gamma-access.jl")
# 
# Define the multi-modal version of modal datasets (basically, a vector of datasets with the
#  same number of instances)
# 
include("multi-frame-dataset.jl")
# 
# TODO figure out which convert function works best: convert(::Type{<:MultiFrameModalDataset{T}}, X::MD) where {T,MD<:AbstractConditionalDataset{T}} = MultiFrameModalDataset{MD}([X])
# convert(::Type{<:MultiFrameModalDataset}, X::AbstractConditionalDataset) = MultiFrameModalDataset([X])
# 
const ActiveMultiFrameModalDataset{T} = MultiFrameModalDataset{<:ActiveFeaturedDataset{<:T}}
#
const GenericModalDataset = Union{AbstractDimensionalDataset,AbstractConditionalDataset,MultiFrameModalDataset}
# 

# Dimensional Ontologies
include("dimensional-ontologies.jl")

############################################################################################

end # module
