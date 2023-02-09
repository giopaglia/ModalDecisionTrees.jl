using ProgressMeter

using SoleModels: CanonicalFeatureGeq, CanonicalFeatureGeqSoft, CanonicalFeatureLeq, CanonicalFeatureLeqSoft
using SoleModels: evaluate_thresh_decision, existential_aggregator, aggregator_bottom, aggregator_to_binary

using SoleLogics: TruthValue

import SoleData: get_instance, instance, max_channel_size, channel_size, nattributes, nsamples, slice_dataset, _slice_dataset

import SoleModels: featvaltype

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
abstract type ActiveModalDataset{T<:Number,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},U,FT<:AbstractFeature{U}} <: AbstractConditionalDataset{W,FeatCondition{U},Bool,FR} end

featvaltype(::Type{<:ActiveModalDataset{T,W,FR,U,FT}}) where {T,W,FR,U,FT} = U
featvaltype(d::ActiveModalDataset) = featvaltype(typeof(d))

featuretype(::Type{<:ActiveModalDataset{T,W,FR,U,FT}}) where {T,W,FR,U,FT} = FT
featuretype(d::ActiveModalDataset) = featuretype(typeof(d))

nsamples(X::ActiveModalDataset) = error("Please, provide method nsamples(::$(typeof(X))).")
Base.length(X::ActiveModalDataset) = nsamples(X)
Base.iterate(X::ActiveModalDataset, state=1) = state > nsamples(X) ? nothing : (get_instance(X, state), state+1)

# By default an active modal dataset cannot be miniaturized
isminifiable(::ActiveModalDataset) = false

include("passive-dimensional-dataset.jl")
include("interpreted-modal-dataset.jl")
include("explicit-modal-dataset.jl")
include("explicit-modal-dataset-with-supports.jl")

include("one-step-supporting-dataset/main.jl")

############################################################################################
