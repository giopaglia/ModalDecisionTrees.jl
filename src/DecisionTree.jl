# __precompile__()

module DecisionTree

################################################################################

import Base: length, show

using LinearAlgebra
using Logging: LogLevel, @logmsg
using Printf
using Random
using StatsBase

################################################################################

# TODO update these
export Decision, DTNode, DTLeaf, DTInternal, DTree, DForest,
        is_leaf_node, is_modal_node,
        num_nodes, height, modal_height,
        ConfusionMatrix, confusion_matrix, mean_squared_error, R2, load_data,
        initWorldSet

################################################################################
# TODO fix
################################################################################

export  DTOverview, DTDebug, DTDetail,
        throw_n_log

# Log single algorithm overview (e.g. splits performed in decision tree building)
const DTOverview = LogLevel(-500)
# Log debug info
const DTDebug = LogLevel(-1000)
# Log more detailed debug info
const DTDetail = LogLevel(-1500)

# TODO Use RegressionLabel instead of Float64 and ClassificationLabel instead of String where it is appropriate!
# Actually, the distinction must be made between String and Numbers (Categorical and Numeral (Numeral Continuous like Float64 or Numeral Discrete like Integers))
# But this can be dangerous because if not coded right Categorical cases indexed by Integers end up being considered Numerical cases
# TODO remove treeclassifier.jl, treeregressor.jl, and these two definitions
const ClassificationLabel = String
const RegressionLabel = Float64 # AbstractFloat

function throw_n_log(str::AbstractString, err_type = ErrorException)
    @error str
    throw(err_type(str))
end

# # ScikitLearn API
# export DecisionTreeClassifier,
# #        DecisionTreeRegressor, RandomForestClassifier,
# #        RandomForestRegressor, AdaBoostStumpClassifier,
# #        # Should we export these functions? They have a conflict with
# #        # DataFrames/RDataset over fit!, and users can always
# #        # `using ScikitLearnBase`.
#             predict,
#             # predict_proba,
#             fit!, get_classes

################################################################################
################################################################################
################################################################################

include("util.jl")
include("metrics.jl")

################################################################################
# Data Feature types
################################################################################

include("featureTypes.jl")

################################################################################
# Modal Logic structures
################################################################################

include("ModalLogic/ModalLogic.jl")
using .ModalLogic
import .ModalLogic: n_samples, display_decision

################################################################################
# Initial world conditions
################################################################################

export startWithRelationGlob, startAtCenter, startAtWorld

abstract type _initCondition end
struct _startWithRelationGlob           <: _initCondition end; const startWithRelationGlob  = _startWithRelationGlob();
struct _startAtCenter                   <: _initCondition end; const startAtCenter          = _startAtCenter();
struct _startAtWorld{wT<:AbstractWorld} <: _initCondition w::wT end;

initWorldSet(initConditions::AbstractVector{<:_initCondition}, worldTypes::AbstractVector{<:Type#={<:AbstractWorld}=#}, args::Vararg) =
    [initWorldSet(iC, WT, args...) for (iC, WT) in zip(initConditions, Vector{Type{<:AbstractWorld}}(worldTypes))]

initWorldSet(initCondition::_startWithRelationGlob, ::Type{WorldType}, channel_size::NTuple{N,Integer} where N) where {WorldType<:AbstractWorld} =
    WorldSet{WorldType}([WorldType(ModalLogic.emptyWorld)])

initWorldSet(initCondition::_startAtCenter, ::Type{WorldType}, channel_size::NTuple{N,Integer} where N) where {WorldType<:AbstractWorld} =
    WorldSet{WorldType}([WorldType(ModalLogic.centeredWorld, channel_size...)])

initWorldSet(initCondition::_startAtWorld{WorldType}, ::Type{WorldType}, channel_size::NTuple{N,Integer} where N) where {WorldType<:AbstractWorld} =
    WorldSet{WorldType}([WorldType(initCondition.w)])


################################################################################
# Decision Leaf, Internal, Node, Tree & RF
################################################################################

# Decision leaf node, holding an output label
struct DTLeaf{L}
    # output label
    label         :: L
    # supporting (e.g., training) instances labels
    supp_labels   :: Vector{L}
end

# Decision inner node, holding a split-decision and a frame index
struct DTInternal{T, L}
    # frame index + split-decision
    i_frame       :: Int64
    decision      :: Decision{T}
    # child nodes
    left          :: Union{DTLeaf{L}, DTInternal{T, L}}
    right         :: Union{DTLeaf{L}, DTInternal{T, L}}
end

# Decision Node (Leaf or Internal)
const DTNode{T, L} = Union{DTLeaf{L}, DTInternal{T, L}}

# Decision Tree
struct DTree{L}
    # root node
    root           :: DTNode{T, L} where T
    # worldTypes (one per frame)
    worldTypes     :: Vector{Type{<:AbstractWorld}}
    # initial world conditions (one per frame)
    initConditions :: Vector{<:_initCondition}

    function DTree(
        root           :: DTNode{T, L},
        worldTypes     :: AbstractVector{<:Type},
        initConditions :: AbstractVector{<:_initCondition},
    ) where {T, L}
        return new{L}(root, collect(worldTypes), collect(initConditions))
    end
end

# Decision Forest (i.e., ensable of trees via bagging)
struct DForest{L}
    # trees
    trees       :: Vector{<:DTree{L}}
    # out-of-bag confusion matrices
    cm          :: Union{Vector{ConfusionMatrix},Vector{PerformanceStruct}}
    # out-of-bag error
    oob_error   :: Float64
end

################################################################################
# Methods
################################################################################

# Number of leaves
num_leaves(leaf::DTLeaf)     = 1
num_leaves(node::DTInternal) = num_leaves(node.left) + num_leaves(node.right)
num_leaves(t::DTree)         = num_leaves(t.root)

length(f::DForest)    = num_trees(f.trees)

# Number of nodes
num_nodes(leaf::DTLeaf)     = 1
num_nodes(node::DTInternal) = 1 + num_nodes(node.left) + num_nodes(node.right)
num_nodes(t::DTree)   = num_nodes(t.root)
num_nodes(f::DForest) = sum(num_nodes.(f.trees))

# Number of trees
num_trees(f::DForest) = length(f.trees)
length(f::DForest)    = num_trees(f.trees)

# Height
height(leaf::DTLeaf)     = 0
height(node::DTInternal) = 1 + max(height(node.left), height(node.right))
height(t::DTree)         = height(t.root)

# Modal height
modal_height(leaf::DTLeaf)     = 0
modal_height(node::DTInternal) = Int(is_modal_node(node)) + max(modal_height(node.left), modal_height(node.right))
modal_height(t::DTree)         = modal_height(t.root)

# Number of supporting instances 
n_samples(leaf::DTLeaf)     = length(leaf.supp_labels)
n_samples(node::DTInternal) = n_samples(node.left) + n_samples(node.right)
n_samples(tree::DTree)      = n_samples(tree.root)

# TODO remove deprecated use num_leaves
length(leaf::DTLeaf)     = num_leaves(leaf::DTLeaf)    
length(node::DTInternal) = num_leaves(node::DTInternal)
length(t::DTree)         = num_leaves(t::DTree)        

################################################################################
################################################################################

is_leaf_node(l::DTLeaf)     = true
is_leaf_node(n::DTInternal) = false
is_leaf_node(t::DTree)      = is_leaf_node(t.root)

is_modal_node(n::DTInternal) = (!is_leaf_node(n) && is_modal_decision(n.decision))
is_modal_node(t::DTree)      = is_modal_node(t.root)

################################################################################
################################################################################

display_decision(node::DTInternal; threshold_display_method::Function = x -> x) =
    display_decision(node.i_frame, node.decision; threshold_display_method = threshold_display_method)
display_decision_neg(node::DTInternal; threshold_display_method::Function = x -> x) =
    display_decision_neg(node.i_frame, node.decision; threshold_display_method = threshold_display_method)

################################################################################
################################################################################

function show(io::IO, leaf::DTLeaf)
    println(io, "Decision Leaf")
    println(io, "Label: $(leaf.label)")
    println(io, "Samples:  $(length(leaf.supp_labels))")
    print_tree(io, leaf)
end

function show(io::IO, node::DTInternal)
    println(io, "Decision Node")
    println(io, display_decision(tree))
    println(io, "Leaves: $(length(tree))")
    println(io, "Tot nodes: $(num_nodes(tree))")
    println(io, "Height: $(height(tree))")
    println(io, "Modal height:  $(modal_height(tree))")
    print_tree(io, tree)
end

function show(io::IO, tree::DTree)
    println(io, "Decision Tree")
    println(io, "Leaves: $(length(tree))")
    println(io, "Tot nodes: $(num_nodes(tree))")
    println(io, "Height: $(height(tree))")
    println(io, "Modal height:  $(modal_height(tree))")
    print_tree(io, tree)
end

function show(io::IO, forest::DForest)
    println(io, "DForest")
    println(io, "Num trees: $(length(forest))")
    println(io, "Out-Of-Bag Error: $(forest.oob_error)")
    println(io, "ConfusionMatrix: $(forest.cm)")
    println(io, "Trees:")
    print_forest(io, forest)
end

################################################################################
# Includes
################################################################################

include("build.jl")
include("predict.jl")
include("posthoc.jl")
include("print.jl")
include("print-latex.jl")
include("decisionpath.jl")


end # module
