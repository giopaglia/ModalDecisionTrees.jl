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
import SoleModels: representatives, allworlds_aggr, FeatMetaCondition
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
       get_gamma, test_decision,
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


Base.@propagate_inbounds @inline function get_modal_gamma(emd::ActiveModalDataset{T,W}, i_sample::Integer, w::W, r::AbstractRelation, f::AbstractFeature, test_operator::TestOperatorFun) where {T,W<:AbstractWorld}
    aggr = existential_aggregator(test_operator)
    _get_modal_gamma(emd, i_sample, w, r, f, aggr)
end

Base.@propagate_inbounds @inline function get_global_gamma(emd::ActiveModalDataset{T,W}, i_sample::Integer, f::AbstractFeature, test_operator::TestOperatorFun) where {T,W<:AbstractWorld}
    aggr = existential_aggregator(test_operator)
    _get_global_gamma(emd, i_sample, f, aggr)
end


# TODO maybe remove
allworlds_aggr(X::ActiveModalDataset, i_sample, args...) = representatives(X, i_sample, RelationGlob, args...)

# By default an active modal dataset cannot be miniaturized
isminifiable(::ActiveModalDataset) = false
#
const ModalDataset{T} = Union{PassiveModalDataset{T},ActiveModalDataset{T}}
#
include("decisions-for-modal-datasets.jl")
#
include("datasets/main.jl")
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
