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
using SoleModels: average_label, majority_vote, default_weights, slice_weights
using SoleData: AbstractDimensionalInstance, DimensionalDataset, AbstractDimensionalChannel, UniformDimensionalDataset, DimensionalChannel, DimensionalInstance, max_channel_size, nsamples, nattributes, get_instance, slice_dataset, concat_datasets, instance_channel_size

export slice_dataset, concat_datasets,
       nframes, nsamples, nattributes, max_channel_size

############################################################################################

export DTree,                   # Decision tree
        DForest,                # Decision forest
        #
        num_nodes, height, modal_height

############################################################################################
# Basics
############################################################################################

include("util.jl")

############################################################################################
# Modal Logic structures
############################################################################################

include("ModalLogic/ModalLogic.jl")

using .ModalLogic
import .ModalLogic: nsamples, display_decision

############################################################################################
# Initial world conditions
############################################################################################

abstract type InitCondition end
struct StartWithoutWorld               <: InitCondition end; const start_without_world  = StartWithoutWorld();
struct StartAtCenter                   <: InitCondition end; const start_at_center      = StartAtCenter();
struct StartAtWorld{WT<:AbstractWorld} <: InitCondition w::WT end;

init_world_set(init_conditions::AbstractVector{<:InitCondition}, world_types::AbstractVector{<:Type#={<:AbstractWorld}=#}, args...) =
    [init_world_set(iC, WT, args...) for (iC, WT) in zip(init_conditions, Vector{Type{<:AbstractWorld}}(world_types))]

init_world_set(iC::StartWithoutWorld, ::Type{WorldType}, args...) where {WorldType<:AbstractWorld} =
    WorldSet{WorldType}([WorldType(ModalLogic.EmptyWorld())])

init_world_set(iC::StartAtCenter, ::Type{WorldType}, args...) where {WorldType<:AbstractWorld} =
    WorldSet{WorldType}([WorldType(ModalLogic.CenteredWorld(), args...)])

init_world_set(iC::StartAtWorld{WorldType}, ::Type{WorldType}, args...) where {WorldType<:AbstractWorld} =
    WorldSet{WorldType}([WorldType(iC.w)])

init_world_sets(Xs::MultiFrameModalDataset, init_conditions::AbstractVector{<:InitCondition}) = begin
    Ss = Vector{Vector{WST} where {WorldType,WST<:WorldSet{WorldType}}}(undef, nframes(Xs))
    for (i_frame,X) in enumerate(frames(Xs))
        WT = world_type(X)
        Ss[i_frame] = WorldSet{WT}[init_world_sets_fun(X, i_sample, world_type(Xs, i_frame))(init_conditions[i_frame]) for i_sample in 1:nsamples(Xs)]
        # Ss[i_frame] = WorldSet{WT}[[ModalLogic.Interval(1,2)] for i_sample in 1:nsamples(Xs)]
    end
    Ss
end

############################################################################################
# Loss & purity functions
############################################################################################

include("entropy-measures.jl")

default_loss_function(::Type{<:CLabel}) = entropy
default_loss_function(::Type{<:RLabel}) = variance

function dishonor_min_purity_increase(::Type{L}, min_purity_increase, purity, best_purity_times_nt, nt) where {L<:CLabel}
    (best_purity_times_nt/nt - purity < min_purity_increase)
end
function dishonor_min_purity_increase(::Type{L}, min_purity_increase, purity, best_purity_times_nt, nt) where {L<:RLabel}
    # (best_purity_times_nt - tsum * label <= min_purity_increase * nt) # ORIGINAL
    (best_purity_times_nt/nt - purity < min_purity_increase * nt)
end

# TODO fix
# function _compute_purity( # faster_version assuming L<:Integer and labels going from 1:n_classes
#     labels           ::AbstractVector{L},
#     n_classes        ::Int,
#     weights          ::AbstractVector{U} = default_weights(length(labels));
#     loss_function    ::Union{Nothing,Function} = default_loss_function(L),
# ) where {L<:CLabel, L<:Integer, U}
#     nc = fill(zero(U), n_classes)
#     @simd for i in 1:max(length(labels),length(weights))
#         nc[labels[i]] += weights[i]
#     end
#     nt = sum(nc)
#     return loss_function(nc, nt)::Float64
# end
function compute_purity(
    labels           ::AbstractVector{L},
    weights          ::AbstractVector{U} = default_weights(length(labels));
    loss_function    ::Union{Nothing,Function} = default_loss_function(L),
) where {L<:CLabel, U}
    nc = Dict{L, U}()
    @simd for i in 1:max(length(labels),length(weights))
        nc[labels[i]] = get(nc, labels[i], 0) + weights[i]
    end
    nc = collect(values(nc))
    nt = sum(nc)
    return loss_function(nc, nt)::Float64
end
# function _compute_purity(
#     labels           ::AbstractVector{L},
#     weights          ::AbstractVector{U} = default_weights(length(labels));
#     loss_function    ::Union{Nothing,Function} = default_loss_function(L),
# ) where {L<:RLabel, U}
#     sums = labels .* weights
#     nt = sum(weights)
#     return -(loss_function(sums, nt))::Float64
# end
function compute_purity(
    labels           ::AbstractVector{L},
    weights          ::AbstractVector{U} = default_weights(length(labels));
    loss_function    ::Union{Nothing,Function} = default_loss_function(L),
) where {L<:RLabel, U}
    _compute_purity = _compute_purity(labels, weights = weights; loss_function = loss_function)
end

############################################################################################
# Decision Leaf, Internal, Node, Tree & RF
############################################################################################

include("base.jl")

############################################################################################
# Includes
############################################################################################

default_max_depth = typemax(Int64)
default_min_samples_leaf = 1
default_min_purity_increase = -Inf
default_max_purity_at_leaf = Inf
default_n_trees = typemax(Int64)

# function parametrization_is_going_to_prune(pruning_params)
#     (haskey(pruning_params, :max_depth)           && pruning_params.max_depth            < default_max_depth) ||
#     # (haskey(pruning_params, :min_samples_leaf)    && pruning_params.min_samples_leaf     > default_min_samples_leaf) ||
#     (haskey(pruning_params, :min_purity_increase) && pruning_params.min_purity_increase  > default_min_purity_increase) ||
#     (haskey(pruning_params, :max_purity_at_leaf)  && pruning_params.max_purity_at_leaf   < default_max_purity_at_leaf) ||
#     (haskey(pruning_params, :n_trees)             && pruning_params.n_trees              < default_n_trees)
# end

include("leaf-metrics.jl")
include("build.jl")
include("apply.jl")
include("posthoc.jl")
include("print.jl")
include("decisionpath.jl")

include("MLJ-interface.jl")
include("AbstractTrees-interface.jl")

include("translation.jl")

include("experimentals.jl")
using .experimentals

end # module
