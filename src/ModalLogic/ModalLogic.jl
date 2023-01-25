module ModalLogic

using ..ModalDecisionTrees
using ..ModalDecisionTrees: util

using ..ModalDecisionTrees: DimensionalDataset, AbstractDimensionalChannel, AbstractDimensionalInstance, UniformDimensionalDataset, DimensionalChannel, DimensionalInstance

using ..ModalDecisionTrees: LogOverview, LogDebug, LogDetail

using BenchmarkTools
using ComputedFieldTypes
using DataStructures
using IterTools
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

export Ontology, world_type, world_types

# Fix (not needed from Julia 1.7, see https://github.com/JuliaLang/julia/issues/34674 )
if length(methods(Base.keys, (Base.Generator,))) == 0
    Base.keys(g::Base.Generator) = g.iter
end

# Accessibles base
include("base.jl")

# Concrete type for ontologies
include("ontology.jl")

############################################################################################
# Dataset structures
############################################################################################
# TODO sort these
import ..ModalDecisionTrees: slice_dataset, concat_datasets,
       nsamples, nattributes, max_channel_size, get_instance,
       instance_channel_size


export nfeatures, nrelations,
       nframes, frames, get_frame,
       display_structure,
       get_gamma, test_decision,
       #
       relations,
       init_world_sets_fun,
       #
       ModalDataset,
       GenericModalDataset,
       ActiveMultiFrameModalDataset,
       MultiFrameModalDataset,
       ActiveModalDataset,
       InterpretedModalDataset,
       ExplicitModalDataset,
       ExplicitModalDatasetS,
       ExplicitModalDatasetSMemo

_isnan(n::Number) = isnan(n)
_isnan(n::Nothing) = false
hasnans(n::Number) = _isnan(n)
hasnans(n::AbstractArray{<:Union{Nothing, Number}}) = any(_isnan.(n))

# A modal dataset can be *active* or *passive*.
#
# A passive modal dataset is one that you can interpret decisions on, but cannot necessarily
#  enumerate decisions for, as it doesn't have objects for storing the logic (relations, features, etc.).
# Dimensional datasets are passive.
include("dimensional-dataset-bindings.jl")
#
const PassiveModalDataset{T} = Union{DimensionalDataset{T}}
#
# Active datasets comprehend structures for representing relation sets, features, enumerating worlds,
#  etc. While learning a model can be done only with active modal datasets, testing a model
#  can be done with both active and passive modal datasets.
#
abstract type ActiveModalDataset{T<:Number,WorldType<:AbstractWorld} end
#
# Active modal datasets hold the WorldType, and thus can initialize world sets with a lighter interface
#
init_world_sets_fun(imd::ActiveModalDataset{T, WorldType},  i_sample::Integer, ::Type{WorldType}) where {T, WorldType} =
    init_world_sets_fun(imd, i_sample)
#
# By default an active modal dataset cannot be miniaturized
isminifiable(::ActiveModalDataset) = false
#
const ModalDataset{T} = Union{PassiveModalDataset{T},ActiveModalDataset{T}}
#
include("active-modal-datasets.jl")
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

# World-specific featured world datasets and supports
include("world-specific-fwds.jl")

############################################################################################

end # module
