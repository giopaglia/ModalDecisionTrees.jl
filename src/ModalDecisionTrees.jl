module ModalDecisionTrees

############################################################################################

import Base: show, length

using FillArrays
using FunctionWrappers: FunctionWrapper
using LinearAlgebra
using Logging: LogLevel, @logmsg
using Printf
using ProgressMeter
using Random
using ReTest
using StatsBase

############################################################################################

export Decision,                # Decision (e.g., (e.g., ⟨L⟩ (minimum(A2) ≤ 10) )
        DTLeaf, NSDTLeaf,       # Decision leaf (simple or complex)
        DTInternal,             # Internal decision node
        DTNode,                 # Decision node (leaf or internal)
        DTree,                  # Decision tree
        DForest,                # Decision forest
        #
        num_nodes, height, modal_height


############################################################################################
############################################################################################
############################################################################################

# Log overview info (e.g., splits performed in decision tree building)
const DTOverview = LogLevel(-500)
# Log debug info
const DTDebug = LogLevel(-1000)
# Log more detailed debug info
const DTDetail = LogLevel(-1500)

# Log string with @error and throw error
function throw_n_log(str::AbstractString, err_type = ErrorException)
    @error str
    throw(err_type(str))
end

############################################################################################
# Basics
############################################################################################

# Classification and regression labels
CLabel  = Union{String,Integer}
RLabel  = AbstractFloat
Label   = Union{CLabel,RLabel}
# Raw labels
_CLabel = Integer # (classification labels are internally represented as integers)
_Label  = Union{_CLabel,RLabel}

include("util.jl")
include("metrics.jl")

############################################################################################
# Simple dataset structure (basically, an hypercube)
############################################################################################

export slice_dataset, concat_datasets,
       n_samples, n_attributes, max_channel_size

include("dimensional-dataset.jl")

############################################################################################
# Modal features
############################################################################################

include("modal-features.jl")

############################################################################################
# Test operators
############################################################################################

include("test-operators.jl")

############################################################################################
# Modal Logic structures
############################################################################################

include("ModalLogic/ModalLogic.jl")

using .ModalLogic
import .ModalLogic: n_samples, display_decision

############################################################################################
# Initial world conditions
############################################################################################

export startWithRelationGlob, startAtCenter, startAtWorld

abstract type _initCondition end
struct _startWithRelationGlob           <: _initCondition end; const startWithRelationGlob  = _startWithRelationGlob();
struct _startAtCenter                   <: _initCondition end; const startAtCenter          = _startAtCenter();
struct _startAtWorld{wT<:AbstractWorld} <: _initCondition w::wT end;

init_world_set(initConditions::AbstractVector{<:_initCondition}, worldTypes::AbstractVector{<:Type#={<:AbstractWorld}=#}, args...) =
    [init_world_set(iC, WT, args...) for (iC, WT) in zip(initConditions, Vector{Type{<:AbstractWorld}}(worldTypes))]

init_world_set(initCondition::_startWithRelationGlob, ::Type{WorldType}, args...) where {WorldType<:AbstractWorld} =
    WorldSet{WorldType}([WorldType(ModalLogic._emptyWorld())])

init_world_set(initCondition::_startAtCenter, ::Type{WorldType}, args...) where {WorldType<:AbstractWorld} =
    WorldSet{WorldType}([WorldType(ModalLogic._centeredWorld(), args...)])

init_world_set(initCondition::_startAtWorld{WorldType}, ::Type{WorldType}, args...) where {WorldType<:AbstractWorld} =
    WorldSet{WorldType}([WorldType(initCondition.w)])

init_world_sets(Xs::MultiFrameModalDataset, initConditions::AbstractVector{<:_initCondition}) = begin
    Ss = Vector{Vector{WST} where {WorldType,WST<:WorldSet{WorldType}}}(undef, n_frames(Xs))
    for (i_frame,X) in enumerate(frames(Xs))
        WT = world_type(X)
        Ss[i_frame] = WorldSet{WT}[init_world_sets_fun(X, i_sample, world_type(Xs, i_frame))(initConditions[i_frame]) for i_sample in 1:n_samples(Xs)]
        # Ss[i_frame] = WorldSet{WT}[[ModalLogic.Interval(1,2)] for i_sample in 1:n_samples(Xs)]
    end
    Ss
end

############################################################################################
# Loss & purity functions
############################################################################################

default_loss_function(::Type{<:CLabel}) = util.entropy
default_loss_function(::Type{<:RLabel}) = util.variance

average_label(labels::AbstractVector{<:CLabel}) = majority_vote(labels; suppress_parity_warning = false) # argmax(countmap(labels))
average_label(labels::AbstractVector{<:RLabel}) = majority_vote(labels; suppress_parity_warning = false) # StatsBase.mean(labels)

function majority_vote(
        labels::AbstractVector{L},
        weights::Union{Nothing,AbstractVector} = nothing;
        suppress_parity_warning = false,
    ) where {L<:CLabel}
    
    if length(labels) == 0
        return nothing
    end

    counts = begin
        if isnothing(weights)
            countmap(labels)
        else
            @assert length(labels) === length(weights) "Can't compute majority_vote with uneven number of votes $(length(labels)) and weights $(length(weights))."
            countmap(labels, weights)
        end
    end

    if !suppress_parity_warning && sum(counts[argmax(counts)] .== values(counts)) > 1
        println("Warning: parity encountered in majority_vote.")
        println("Counts ($(length(labels)) elements): $(counts)")
        println("Argmax: $(argmax(counts))")
        println("Max: $(counts[argmax(counts)]) (sum = $(sum(values(counts))))")
    end
    argmax(counts)
end

function majority_vote(
        labels::AbstractVector{L},
        weights::Union{Nothing,AbstractVector} = nothing;
        suppress_parity_warning = false,
    ) where {L<:RLabel}
    if length(labels) == 0
        return nothing
    end

    (isnothing(weights) ? mean(labels) : sum(labels .* weights)/sum(weights))
end

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
############################################################################################
############################################################################################

# Default weights are optimized using FillArrays
function default_weights(n::Integer)
    Ones{Int64}(n)
end
function default_weights_rebalance(Y::AbstractVector{L}) where {L<:Label}
    class_counts_dict = countmap(Y)
    if length(unique(values(class_counts)_dict)) == 1 # balanced case
        default_weights(length(Y))
    else
        # Assign weights in such a way that the dataset becomes balanced 
        tot = sum(values(class_counts_dict))
        balanced_tot_per_class = tot/length(class_counts_dict)
        weights_map = Dict{L,Float64}([class => (balanced_tot_per_class/n_instances) for (class,n_instances) in class_counts_dict])
        W = [weights_map[y] for y in Y]
        W ./ sum(W)
    end
end
_slice_weights(W::Ones{Int64}, inds::AbstractVector) = default_weights(length(inds))
_slice_weights(W::Any,         inds::AbstractVector) = @view W[inds]
_slice_weights(W::Ones{Int64}, i::Integer) = 1
_slice_weights(W::Any,         i::Integer) = W[i]


############################################################################################
# Decision Leaf, Internal, Node, Tree & RF
############################################################################################

abstract type AbstractDecisionLeaf{L<:Label} end

# Decision leaf node, holding an output (prediction)
struct DTLeaf{L<:Label} <: AbstractDecisionLeaf{L}
    # prediction
    prediction         :: L
    # supporting (e.g., training) instances labels
    supp_labels   :: Vector{L}

    # create leaf
    DTLeaf{L}(prediction, supp_labels::AbstractVector) where {L<:Label} = new{L}(prediction, supp_labels)
    DTLeaf(prediction::L, supp_labels::AbstractVector) where {L<:Label} = DTLeaf{L}(prediction, supp_labels)

    # create leaf without supporting labels
    DTLeaf{L}(prediction) where {L<:Label} = DTLeaf{L}(prediction, L[])
    DTLeaf(prediction::L) where {L<:Label} = DTLeaf{L}(prediction, L[])

    # create leaf from supporting labels
    DTLeaf{L}(supp_labels::AbstractVector) where {L<:Label} = DTLeaf{L}(average_label(L.(supp_labels)), supp_labels)
    function DTLeaf(supp_labels::AbstractVector)
        prediction = average_label(supp_labels)
        DTLeaf(prediction, supp_labels)
    end
end

prediction(leaf::DTLeaf) = leaf.prediction

function supp_labels(leaf::DTLeaf; train_or_valid = true)
    @assert train_or_valid == true
    leaf.supp_labels
end
function predictions(leaf::DTLeaf; train_or_valid = true)
    @assert train_or_valid == true
    fill(prediction(leaf), length(supp_labels(leaf; train_or_valid = train_or_valid)))
end

############################################################################################

struct PredictingFunction{L<:Label}
    # f::FunctionWrapper{Vector{L},Tuple{MultiFrameModalDataset}} # TODO restore!!!
    f::FunctionWrapper{Any,Tuple{MultiFrameModalDataset}}

    function PredictingFunction{L}(f::Any) where {L<:Label}
        # new{L}(FunctionWrapper{Vector{L},Tuple{MultiFrameModalDataset}}(f)) # TODO restore!!!
        new{L}(FunctionWrapper{Any,Tuple{MultiFrameModalDataset}}(f))
    end
end
(pf::PredictingFunction{L})(args...; kwargs...) where {L} = pf.f(args...; kwargs...)::Vector{L}

# const ModalInstance = Union{AbstractArray,Any}
# const LFun{L} = FunctionWrapper{L,Tuple{ModalInstance}}
# TODO maybe join DTLeaf and NSDTLeaf Union{L,LFun{L}}
# Decision leaf node, holding an output predicting function
struct NSDTLeaf{L<:Label} <: AbstractDecisionLeaf{L}
    # predicting function
    predicting_function         :: PredictingFunction{L}
    
    # supporting labels
    supp_train_labels        :: Vector{L}
    supp_valid_labels        :: Vector{L}

    # supporting predictions
    supp_train_predictions   :: Vector{L}
    supp_valid_predictions   :: Vector{L}

    # create leaf
    # NSDTLeaf{L}(predicting_function, supp_labels::AbstractVector) where {L<:Label} = new{L}(predicting_function, supp_labels)
    # NSDTLeaf(predicting_function::PredictingFunction{L}, supp_labels::AbstractVector) where {L<:Label} = NSDTLeaf{L}(predicting_function, supp_labels)

    # create leaf without supporting labels
    function NSDTLeaf{L}(
        predicting_function      :: PredictingFunction{L},
        supp_train_labels        :: Vector{L},
        supp_valid_labels        :: Vector{L},
        supp_train_predictions   :: Vector{L},
        supp_valid_predictions   :: Vector{L},
    ) where {L<:Label}
        new{L}(
            predicting_function,
            supp_train_labels,
            supp_valid_labels,
            supp_train_predictions,
            supp_valid_predictions,
        )
    end
    function NSDTLeaf(
        predicting_function      :: PredictingFunction{L},
        supp_train_labels        :: Vector{L},
        supp_valid_labels        :: Vector{L},
        supp_train_predictions   :: Vector{L},
        supp_valid_predictions   :: Vector{L},
    ) where {L<:Label}
        NSDTLeaf{L}(
            predicting_function,
            supp_train_labels,
            supp_valid_labels,
            supp_train_predictions,
            supp_valid_predictions,
        )
    end

    function NSDTLeaf{L}(f::Any, args...; kwargs...) where {L<:Label}
        NSDTLeaf{L}(PredictingFunction{L}(f), args...; kwargs...)
    end

    # create leaf from supporting labels
    # NSDTLeaf{L}(supp_labels::AbstractVector) where {L<:Label} = NSDTLeaf{L}(average_label(supp_labels), supp_labels)
    # function NSDTLeaf(supp_labels::AbstractVector)
    #     predicting_function = average_label(supp_labels)
    #     NSDTLeaf(predicting_function, supp_labels)
    # end
end

supp_labels(leaf::NSDTLeaf; train_or_valid = true) = (train_or_valid ? leaf.supp_train_labels      : leaf.supp_valid_labels)
predictions(leaf::NSDTLeaf; train_or_valid = true) = (train_or_valid ? leaf.supp_train_predictions : leaf.supp_valid_predictions)


############################################################################################

# Internal decision node, holding a split-decision and a frame index
struct DTInternal{T, L<:Label}
    # frame index + split-decision
    i_frame       :: Int64
    decision      :: Decision{T}
    # representative leaf for the current node
    this          :: AbstractDecisionLeaf{<:L}
    # child nodes
    left          :: Union{AbstractDecisionLeaf{<:L}, DTInternal{T, L}}
    right         :: Union{AbstractDecisionLeaf{<:L}, DTInternal{T, L}}

    # create node
    function DTInternal{T, L}(
        i_frame          :: Int64,
        decision         :: Decision,
        this             :: AbstractDecisionLeaf,
        left             :: Union{AbstractDecisionLeaf, DTInternal},
        right            :: Union{AbstractDecisionLeaf, DTInternal}) where {T, L<:Label}
        new{T, L}(i_frame, decision, this, left, right)
    end
    function DTInternal(
        i_frame          :: Int64,
        decision         :: Decision{T},
        this             :: AbstractDecisionLeaf{<:L},
        left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{T, L}},
        right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{T, L}}) where {T, L<:Label}
        DTInternal{T, L}(i_frame, decision, this, left, right)
    end

    # create node without local decision
    # function DTInternal{T, L}(
    #     i_frame          :: Int64,
    #     decision         :: Decision,
    #     left             :: Union{AbstractDecisionLeaf, DTInternal},
    #     right            :: Union{AbstractDecisionLeaf, DTInternal}) where {T, L<:Label}
    #     this = AbstractDecisionLeaf{<:L} (NOPE) (L[(supp_labels(left; supp_labels = supp_labels?))..., (supp_labels(right; supp_labels = supp_labels?))...])
    #     new{T, L}(i_frame, decision, this, left, right)
    # end
    # function DTInternal(
    #     i_frame          :: Int64,
    #     decision         :: Decision{T},
    #     left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{T, L}},
    #     right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{T, L}}) where {T, L<:Label}
    #     DTInternal{T, L}(i_frame, decision, left, right)
    # end

    # create node without frame
    # function DTInternal{T, L}(
    #     decision         :: Decision,
    #     this             :: AbstractDecisionLeaf,
    #     left             :: Union{AbstractDecisionLeaf, DTInternal},
    #     right            :: Union{AbstractDecisionLeaf, DTInternal}) where {T, L<:Label}
    #     i_frame = 1
    #     DTInternal{T, L}(i_frame, decision, this, left, right)
    # end
    # function DTInternal(
    #     decision         :: Decision{T},
    #     this             :: AbstractDecisionLeaf,
    #     left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{T, L}},
    #     right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{T, L}}) where {T, L<:Label}
    #     DTInternal{T, L}(decision, this, left, right)
    # end
    
    # # create node without frame nor local decision
    # function DTInternal{T, L}(
    #     decision         :: Decision,
    #     left             :: Union{AbstractDecisionLeaf, DTInternal},
    #     right            :: Union{AbstractDecisionLeaf, DTInternal}) where {T, L<:Label}
    #     i_frame = 1
    #     DTInternal{T, L}(i_frame, decision, left, right)
    # end
    # function DTInternal(
    #     decision         :: Decision{T},
    #     left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{T, L}},
    #     right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{T, L}}) where {T, L<:Label}
    #     DTInternal{T, L}(decision, left, right)
    # end
end

function supp_labels(node::DTInternal; train_or_valid = true)
    @assert train_or_valid == true
    supp_labels(node.this; train_or_valid = train_or_valid)
end

############################################################################################

# Decision Node (Leaf or Internal)
const DTNode{T, L} = Union{<:AbstractDecisionLeaf{<:L}, DTInternal{T, L}}

############################################################################################

# Decision Tree
struct DTree{L<:Label}
    # root node
    root           :: DTNode{T, L} where T
    # worldTypes (one per frame)
    worldTypes     :: Vector{Type{<:AbstractWorld}}
    # initial world conditions (one per frame)
    initConditions :: Vector{<:_initCondition}

    function DTree{L}(
        root           :: DTNode,
        worldTypes     :: AbstractVector{<:Type},
        initConditions :: AbstractVector{<:_initCondition},
    ) where {L<:Label}
        new{L}(root, collect(worldTypes), collect(initConditions))
    end

    function DTree(
        root           :: DTNode{T, L},
        worldTypes     :: AbstractVector{<:Type},
        initConditions :: AbstractVector{<:_initCondition},
    ) where {T, L<:Label}
        DTree{L}(root, worldTypes, initConditions)
    end
end

############################################################################################

# Decision Forest (i.e., ensable of trees via bagging)
struct DForest{L<:Label}
    # trees
    trees       :: Vector{<:DTree{L}}
    # metrics
    metrics     :: NamedTuple

    # create forest from vector of trees
    function DForest{L}(
        trees     :: AbstractVector{<:DTree},
    ) where {L<:Label}
        new{L}(collect(trees), (;))
    end
    function DForest(
        trees     :: AbstractVector{<:DTree{L}},
    ) where {L<:Label}
        DForest{L}(trees)
    end

    # create forest from vector of trees, with attached metrics
    function DForest{L}(
        trees     :: AbstractVector{<:DTree},
        metrics   :: NamedTuple,
    ) where {L<:Label}
        new{L}(collect(trees), metrics)
    end
    function DForest(
        trees     :: AbstractVector{<:DTree{L}},
        metrics   :: NamedTuple,
    ) where {L<:Label}
        DForest{L}(trees, metrics)
    end

end

############################################################################################
# Methods
############################################################################################

# Number of leaves
num_leaves(leaf::AbstractDecisionLeaf)     = 1
num_leaves(node::DTInternal) = num_leaves(node.left) + num_leaves(node.right)
num_leaves(tree::DTree)      = num_leaves(tree.root)

# Number of nodes
num_nodes(leaf::AbstractDecisionLeaf)     = 1
num_nodes(node::DTInternal) = 1 + num_nodes(node.left) + num_nodes(node.right)
num_nodes(tree::DTree)   = num_nodes(tree.root)
num_nodes(f::DForest) = sum(num_nodes.(f.trees))

# Number of trees
num_trees(f::DForest) = length(f.trees)
Base.length(f::DForest)    = num_trees(f)

# Height
height(leaf::AbstractDecisionLeaf)     = 0
height(node::DTInternal) = 1 + max(height(node.left), height(node.right))
height(tree::DTree)      = height(tree.root)

# Modal height
modal_height(leaf::AbstractDecisionLeaf)     = 0
modal_height(node::DTInternal) = Int(is_modal_node(node)) + max(modal_height(node.left), modal_height(node.right))
modal_height(tree::DTree)      = modal_height(tree.root)

# Number of supporting instances 
n_samples(leaf::AbstractDecisionLeaf; train_or_valid = true) = length(supp_labels(leaf; train_or_valid = train_or_valid))
n_samples(node::DTInternal;           train_or_valid = true) = n_samples(node.left; train_or_valid = train_or_valid) + n_samples(node.right; train_or_valid = train_or_valid)
n_samples(tree::DTree;                train_or_valid = true) = n_samples(tree.root; train_or_valid = train_or_valid)

# TODO remove deprecated use num_leaves
Base.length(leaf::AbstractDecisionLeaf)     = num_leaves(leaf)    
Base.length(node::DTInternal) = num_leaves(node)
Base.length(tree::DTree)      = num_leaves(tree)        

############################################################################################
############################################################################################

is_leaf_node(leaf::AbstractDecisionLeaf)     = true
is_leaf_node(node::DTInternal) = false
is_leaf_node(tree::DTree)      = is_leaf_node(tree.root)

is_modal_node(node::DTInternal) = (!is_leaf_node(node) && !is_propositional_decision(node.decision))
is_modal_node(tree::DTree)      = is_modal_node(tree.root)

############################################################################################
############################################################################################

display_decision(node::DTInternal; threshold_display_method::Function = x -> x) =
    display_decision(node.i_frame, node.decision; threshold_display_method = threshold_display_method)
display_decision_inverse(node::DTInternal; threshold_display_method::Function = x -> x) =
    display_decision_inverse(node.i_frame, node.decision; threshold_display_method = threshold_display_method)

############################################################################################
############################################################################################

function Base.show(io::IO, leaf::DTLeaf{L}) where {L<:CLabel}
    println(io, "Classification Decision Leaf{$(L)}(")
    println(io, "\tlabel: $(prediction(leaf))")
    println(io, "\tsupporting labels:  $(supp_labels(leaf))")
    println(io, "\tsupporting labels countmap:  $(StatsBase.countmap(supp_labels(leaf)))")
    println(io, "\tmetrics: $(get_metrics(leaf))")
    println(io, ")")
end
function Base.show(io::IO, leaf::DTLeaf{L}) where {L<:RLabel}
    println(io, "Regression Decision Leaf{$(L)}(")
    println(io, "\tlabel: $(prediction(leaf))")
    println(io, "\tsupporting labels:  $(supp_labels(leaf))")
    println(io, "\tmetrics: $(get_metrics(leaf))")
    println(io, ")")
end

function Base.show(io::IO, leaf::NSDTLeaf{L}) where {L<:CLabel}
    println(io, "Classification Functional Decision Leaf{$(L)}(")
    println(io, "\tpredicting_function: $(leaf.predicting_function)")
    println(io, "\tsupporting labels (train):  $(leaf.supp_train_labels)")
    println(io, "\tsupporting labels (valid):  $(leaf.supp_valid_labels)")
    println(io, "\tsupporting predictions (train):  $(leaf.supp_train_predictions)")
    println(io, "\tsupporting predictions (valid):  $(leaf.supp_valid_predictions)")
    println(io, "\tsupporting labels countmap (train):  $(StatsBase.countmap(leaf.supp_train_labels))")
    println(io, "\tsupporting labels countmap (valid):  $(StatsBase.countmap(leaf.supp_valid_labels))")
    println(io, "\tsupporting predictions countmap (train):  $(StatsBase.countmap(leaf.supp_train_predictions))")
    println(io, "\tsupporting predictions countmap (valid):  $(StatsBase.countmap(leaf.supp_valid_predictions))")
    println(io, "\tmetrics (train): $(get_metrics(leaf; train_or_valid = true))")
    println(io, "\tmetrics (valid): $(get_metrics(leaf; train_or_valid = false))")
    println(io, ")")
end
function Base.show(io::IO, leaf::NSDTLeaf{L}) where {L<:RLabel}
    println(io, "Regression Functional Decision Leaf{$(L)}(")
    println(io, "\tpredicting_function: $(leaf.predicting_function)")
    println(io, "\tsupporting labels (train):  $(leaf.supp_train_labels)")
    println(io, "\tsupporting labels (valid):  $(leaf.supp_valid_labels)")
    println(io, "\tsupporting predictions (train):  $(leaf.supp_train_predictions)")
    println(io, "\tsupporting predictions (valid):  $(leaf.supp_valid_predictions)")
    println(io, "\tmetrics (train): $(get_metrics(leaf; train_or_valid = true))")
    println(io, "\tmetrics (valid): $(get_metrics(leaf; train_or_valid = false))")
    println(io, ")")
end

function Base.show(io::IO, node::DTInternal{T,L}) where {T,L}
    println(io, "Decision Node{$(T),$(L)}(")
    Base.show(io, node.this)
    println(io, "\t###########################################################")
    println(io, "\ti_frame: $(node.i_frame)")
    print(io, "\tdecision: $(node.decision)")
    println(io, "\t###########################################################")
    println(io, "\tsub-tree leaves: $(num_leaves(node))")
    println(io, "\tsub-tree nodes: $(num_nodes(node))")
    println(io, "\tsub-tree height: $(height(node))")
    println(io, "\tsub-tree modal height:  $(modal_height(node))")
    println(io, ")")
end

function Base.show(io::IO, tree::DTree{L}) where {L}
    println(io, "Decision Tree{$(L)}(")
    println(io, "\tworldTypes:     $(tree.worldTypes)")
    println(io, "\tinitConditions: $(tree.initConditions)")
    println(io, "\t###########################################################")
    println(io, "\tsub-tree leaves: $(num_leaves(tree))")
    println(io, "\tsub-tree nodes: $(num_nodes(tree))")
    println(io, "\tsub-tree height: $(height(tree))")
    println(io, "\tsub-tree modal height:  $(modal_height(tree))")
    println(io, "\t###########################################################")
    println(io, "\ttree:")
    print_model(io, tree)
    println(io, ")")
end

function Base.show(io::IO, forest::DForest{L}) where {L}
    println(io, "Decision Forest{$(L)}(")
    println(io, "\t# trees: $(length(forest))")
    println(io, "\tmetrics: $(forest.metrics)")
    println(io, "\ttrees:")
    print_model(io, forest)
    println(io, ")")
end

############################################################################################
# Includes
############################################################################################

default_max_depth = typemax(Int64)
default_min_samples_leaf = 1
default_min_purity_increase = -Inf
default_max_purity_at_leaf = Inf
default_n_trees = typemax(Int64)

function parametrization_is_going_to_prune(pruning_params)
    (haskey(pruning_params, :max_depth)           && pruning_params.max_depth            < default_max_depth) ||
    # (haskey(pruning_params, :min_samples_leaf)    && pruning_params.min_samples_leaf     > default_min_samples_leaf) ||
    (haskey(pruning_params, :min_purity_increase) && pruning_params.min_purity_increase  > default_min_purity_increase) ||
    (haskey(pruning_params, :max_purity_at_leaf)  && pruning_params.max_purity_at_leaf   < default_max_purity_at_leaf) ||
    (haskey(pruning_params, :n_trees)             && pruning_params.n_trees              < default_n_trees)
end

include("leaf-metrics.jl")
include("build.jl")
include("predict.jl")
include("posthoc.jl")
include("print.jl")
include("print-latex.jl")
include("decisionpath.jl")

############################################################################################
# Tests
############################################################################################

# https://stackoverflow.com/questions/66801702/deriving-equality-for-julia-structs-with-mutable-members
import Base.==
function ==(a::S, b::S) where {S<:AbstractDecisionLeaf}
    for name in fieldnames(S)
        if getfield(a, name) != getfield(b, name)
            return false
        end
    end
    return true
end

@testset "Creation of decision leaves, nodes, decision trees, forests" begin

    @testset "Decision leaves (DTLeaf)" begin

        # Construct a leaf from a label
        @test DTLeaf(1)        == DTLeaf{Int64}(1, Int64[])
        @test DTLeaf{Int64}(1) == DTLeaf{Int64}(1, Int64[])
        
        @test DTLeaf("Class_1")           == DTLeaf{String}("Class_1", String[])
        @test DTLeaf{String}("Class_1")   == DTLeaf{String}("Class_1", String[])

        # Construct a leaf from a label & supporting labels
        @test DTLeaf(1, [])               == DTLeaf{Int64}(1, Int64[])
        @test DTLeaf{Int64}(1, [1.0])     == DTLeaf{Int64}(1, Int64[1])

        @test DTLeaf(1.0, [1.0])   == DTLeaf{Float64}(1.0, [1.0])
        @test_nowarn DTLeaf{Float32}(1, [1])
        @test_nowarn DTLeaf{Float32}(1.0, [1.5])

        @test_throws MethodError DTLeaf(1, ["Class1"])
        @test_throws InexactError DTLeaf(1, [1.5])

        @test_nowarn DTLeaf{String}("1.0", ["0.5", "1.5"])

        # Inferring the label from supporting labels
        @test prediction(DTLeaf{String}(["Class_1", "Class_1", "Class_2"])) == "Class_1"
        
        @test_nowarn DTLeaf(["1.5"])
        @test_throws MethodError DTLeaf([1.0,"Class_1"])

        # Check robustness
        @test_nowarn DTLeaf{Int64}(1, 1:10)
        @test_nowarn DTLeaf{Int64}(1, 1.0:10.0)
        @test_nowarn DTLeaf{Float32}(1, 1:10)

        # @test prediction(DTLeaf(1:10)) == 5
        @test prediction(DTLeaf{Float64}(1:10)) == 5.5
        @test prediction(DTLeaf{Float32}(1:10)) == 5.5f0
        @test prediction(DTLeaf{Float64}(1:11)) == 6

        # Check edge parity case (aggregation biased towards the first class)
        @test prediction(DTLeaf{String}(["Class_1", "Class_2"])) == "Class_1"
        @test prediction(DTLeaf(["Class_1", "Class_2"])) == "Class_1"

    end

    # TODO test NSDT Leaves

    @testset "Decision internal node (DTInternal) + Decision Tree & Forest (DTree & DForest)" begin

        decision = Decision(ModalLogic.RelationGlob, SingleAttributeMin(1), >=, 10)

        reg_leaf, cls_leaf = DTLeaf([1.0,2.0]), DTLeaf([1,2])

        # # create node
        # # cls_node = @test_nowarn DTInternal(decision, cls_leaf, cls_leaf, cls_leaf)
        # # cls_node = @test_nowarn DTInternal(2, decision, cls_leaf, cls_leaf, cls_leaf)
        # # create node without local decision
        # cls_node = @test_nowarn DTInternal(2, decision, cls_leaf, cls_leaf)
        # @test_throws MethodError DTInternal(2, decision, reg_leaf, cls_leaf)
        # # create node without frame
        # # @test_nowarn DTInternal(decision, reg_leaf, reg_leaf, reg_leaf)
        # # create node without frame nor local decision
        # cls_node = @test_nowarn DTInternal(decision, cls_node, cls_leaf)

        # cls_tree = @test_nowarn DTree(cls_node, [ModalLogic.Interval], [startWithRelationGlob])
        # cls_forest = @test_nowarn DForest([cls_tree, cls_tree, cls_tree])
    end
    
end

end # module
