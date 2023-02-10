using ProgressMeter

using SoleModels: CanonicalFeatureGeq, CanonicalFeatureGeqSoft, CanonicalFeatureLeq, CanonicalFeatureLeqSoft
using SoleModels: evaluate_thresh_decision, existential_aggregator, aggregator_bottom, aggregator_to_binary

import SoleData: get_instance, instance, max_channel_size, channel_size, nattributes, nsamples, slice_dataset, _slice_dataset
import SoleData: dimensionality

import Base: eltype

############################################################################################

# Convenience function
function grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops::AbstractVector{<:AbstractVector{<:TestOperatorFun}})::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
    grouped_featsaggrsnops = Dict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}[]
    for (i_feature, test_operators) in enumerate(grouped_featsnops)
        aggrsnops = Dict{Aggregator,AbstractVector{<:TestOperatorFun}}()
        for test_operator in test_operators
            aggregator = existential_aggregator(test_operator)
            if (!haskey(aggrsnops, aggregator))
                aggrsnops[aggregator] = TestOperatorFun[]
            end
            push!(aggrsnops[aggregator], test_operator)
        end
        push!(grouped_featsaggrsnops, aggrsnops)
    end
    grouped_featsaggrsnops
end

function features_grouped_featsaggrsnops2featsnaggrs_grouped_featsnaggrs(features, grouped_featsaggrsnops)
    featsnaggrs = Tuple{<:AbstractFeature,<:Aggregator}[]
    grouped_featsnaggrs = AbstractVector{Tuple{<:Integer,<:Aggregator}}[]
    i_featsnaggr = 1
    for (feat,aggrsnops) in zip(features, grouped_featsaggrsnops)
        aggrs = []
        for aggr in keys(aggrsnops)
            push!(featsnaggrs, (feat,aggr))
            push!(aggrs, (i_featsnaggr,aggr))
            i_featsnaggr += 1
        end
        push!(grouped_featsnaggrs, aggrs)
    end
    featsnaggrs, grouped_featsnaggrs
end

function features_grouped_featsaggrsnops2featsnaggrs(features, grouped_featsaggrsnops)
    featsnaggrs = Tuple{<:AbstractFeature,<:Aggregator}[]
    i_featsnaggr = 1
    for (feat,aggrsnops) in zip(features, grouped_featsaggrsnops)
        for aggr in keys(aggrsnops)
            push!(featsnaggrs, (feat,aggr))
            i_featsnaggr += 1
        end
    end
    featsnaggrs
end

function features_grouped_featsaggrsnops2grouped_featsnaggrs(features, grouped_featsaggrsnops)
    grouped_featsnaggrs = AbstractVector{Tuple{<:Integer,<:Aggregator}}[]
    i_featsnaggr = 1
    for (feat,aggrsnops) in zip(features, grouped_featsaggrsnops)
        aggrs = []
        for aggr in keys(aggrsnops)
            push!(aggrs, (i_featsnaggr,aggr))
            i_featsnaggr += 1
        end
        push!(grouped_featsnaggrs, aggrs)
    end
    grouped_featsnaggrs
end

############################################################################################
# Active datasets comprehend structures for representing relation sets, features, enumerating worlds,
#  etc. While learning a model can be done only with active modal datasets, testing a model
#  can be done with both active and passive modal datasets.
# 
abstract type ActiveFeaturedDataset{V<:Number,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},FT<:AbstractFeature{V}} <: AbstractConditionalDataset{W,AbstractCondition,Bool,FR} end

import SoleModels: featvaltype
import SoleModels: frame

featvaltype(::Type{<:ActiveFeaturedDataset{V}}) where {V} = V
featvaltype(d::ActiveFeaturedDataset) = featvaltype(typeof(d))

featuretype(::Type{<:ActiveFeaturedDataset{V,W,FR,FT}}) where {V,W,FR,FT} = FT
featuretype(d::ActiveFeaturedDataset) = featuretype(typeof(d))

nsamples(X::ActiveFeaturedDataset) = error("Please, provide method nsamples(::$(typeof(X))).")
Base.length(X::ActiveFeaturedDataset) = nsamples(X)
Base.iterate(X::ActiveFeaturedDataset, state=1) = state > nsamples(X) ? nothing : (get_instance(X, state), state+1)

# By default an active modal dataset cannot be minified
isminifiable(::ActiveFeaturedDataset) = false

include("passive-dimensional-dataset.jl")

include("dimensional-featured-dataset.jl")
include("featured-dataset.jl")


abstract type SupportingDataset{W<:AbstractWorld,FR<:AbstractFrame{W,Bool}} end

isminifiable(X::SupportingDataset) = false

worldtype(X::SupportingDataset{W}) where {W} = W

function display_structure(X::SupportingDataset; indent_str = "")
    out = "$(typeof(X))\t$((Base.summarysize(X)) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
    out *= " ($(round(nmemoizedvalues(X))) values)\n"
    out
end

abstract type FeaturedSupportingDataset{V<:Number,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}} <: SupportingDataset{W,FR} end


include("supported-featured-dataset.jl")

include("one-step-featured-supporting-dataset/main.jl")
include("generic-supporting-datasets.jl")

############################################################################################
