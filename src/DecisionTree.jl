# __precompile__()

module DecisionTree

################################################################################

import Base: length, show

using StructuredArrays: UniformVector # , FillArrays # TODO choose one

using LinearAlgebra
using Logging: LogLevel, @logmsg
using Printf
using Random
using StatsBase
using ReTest

using Dagger

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

################################################################################
################################################################################
################################################################################
# TODO MOVE
# TODO Label could also be Nothing! Think about it
CLabel  = Union{String,Integer}
RLabel  = AbstractFloat
Label   = Union{CLabel,RLabel}
# Raw labels
_CLabel = Integer
_Label  = Union{_CLabel,RLabel}

default_loss_function(::Type{<:CLabel}) = util.entropy
default_loss_function(::Type{<:RLabel}) = util.variance

# TODO handle parity!
average_label(labels::AbstractVector{<:CLabel}) = argmax(countmap(labels))
average_label(labels::AbstractVector{<:RLabel}) = StatsBase.mean(labels)

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
#     weights          ::AbstractVector{U} = UniformVector{Int}(1,length(labels));
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
    weights          ::AbstractVector{U} = UniformVector{Int}(1,length(labels));
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
#     weights          ::AbstractVector{U} = UniformVector{Int}(1,length(labels));
#     loss_function    ::Union{Nothing,Function} = default_loss_function(L),
# ) where {L<:RLabel, U}
#     sums = labels .* weights
#     nt = sum(weights)
#     return -(loss_function(sums, nt))::Float64
# end
function compute_purity(
    labels           ::AbstractVector{L},
    weights          ::AbstractVector{U} = UniformVector{Int}(1,length(labels));
    loss_function    ::Union{Nothing,Function} = default_loss_function(L),
) where {L<:RLabel, U}
    _compute_purity = _compute_purity(labels, weights = weights; loss_function = loss_function)
end


################################################################################
################################################################################
################################################################################

# Decision leaf node, holding an output label
struct DTLeaf{L<:Label}
    # output label
    label         :: L
    # supporting (e.g., training) instances labels
    supp_labels   :: Vector{L}

    # create leaf
    DTLeaf{L}(label, supp_labels::AbstractVector) where {L<:Label} = new{L}(label, supp_labels)
    DTLeaf(label::L, supp_labels::AbstractVector) where {L<:Label} = DTLeaf{L}(label, supp_labels)

    # create leaf without supporting labels
    DTLeaf{L}(label) where {L<:Label} = DTLeaf{L}(label, L[])
    DTLeaf(label::L) where {L<:Label} = DTLeaf{L}(label, L[])

    # create leaf from supporting labels
    DTLeaf{L}(supp_labels::AbstractVector) where {L<:Label} = DTLeaf{L}(average_label(supp_labels), supp_labels)
    # DTLeaf(supp_labels::AbstractVector) where {L<:Label} = DTLeaf{L}(average_label(supp_labels), supp_labels)
    function DTLeaf(supp_labels::AbstractVector)
        label = average_label(supp_labels)
        DTLeaf(label, supp_labels)
    end
end

# Decision inner node, holding a split-decision and a frame index
struct DTInternal{T, L<:Label}
    # frame index + split-decision
    i_frame       :: Int64
    decision      :: Decision{T}
    # representative leaf for the current node
    this          :: DTLeaf{L}
    # child nodes
    left          :: Union{DTLeaf{L}, DTInternal{T, L}}
    right         :: Union{DTLeaf{L}, DTInternal{T, L}}

    # create node
    function DTInternal{T, L}(
        i_frame          :: Int64,
        decision         :: Decision,
        this             :: DTLeaf,
        left             :: Union{DTLeaf, DTInternal},
        right            :: Union{DTLeaf, DTInternal}) where {T, L<:Label}
        new{T, L}(i_frame, decision, this, left, right)
    end
    function DTInternal(
        i_frame          :: Int64,
        decision         :: Decision{T},
        this             :: DTLeaf,
        left             :: Union{DTLeaf{L}, DTInternal{T, L}},
        right            :: Union{DTLeaf{L}, DTInternal{T, L}}) where {T, L<:Label}
        DTInternal{T, L}(i_frame, decision, this, left, right)
    end

    # create node without local decision
    function DTInternal{T, L}(
        i_frame          :: Int64,
        decision         :: Decision,
        left             :: Union{DTLeaf, DTInternal},
        right            :: Union{DTLeaf, DTInternal}) where {T, L<:Label}
        supp_labels = L[(left.this.labels)..., (right.this.labels)...]
        this = DTLeaf{L}(supp_labels)
        DTInternal{T, L}(i_frame, decision, this, left, right)
    end
    function DTInternal(
        i_frame          :: Int64,
        decision         :: Decision{T},
        left             :: Union{DTLeaf{L}, DTInternal{T, L}},
        right            :: Union{DTLeaf{L}, DTInternal{T, L}}) where {T, L<:Label}
        DTInternal{T, L}(i_frame, decision, left, right)
    end

    # create node without frame
    function DTInternal{T, L}(
        decision         :: Decision,
        this             :: DTLeaf,
        left             :: Union{DTLeaf, DTInternal},
        right            :: Union{DTLeaf, DTInternal}) where {T, L<:Label}
        i_frame = 1
        DTInternal{T, L}(i_frame, decision, this, left, right)
    end
    function DTInternal(
        decision         :: Decision{T},
        this             :: DTLeaf,
        left             :: Union{DTLeaf{L}, DTInternal{T, L}},
        right            :: Union{DTLeaf{L}, DTInternal{T, L}}) where {T, L<:Label}
        DTInternal{T, L}(decision, this, left, right)
    end
    
    # create node without frame nor local decision
    function DTInternal{T, L}(
        decision         :: Decision,
        left             :: Union{DTLeaf, DTInternal},
        right            :: Union{DTLeaf, DTInternal}) where {T, L<:Label}
        i_frame = 1
        DTInternal{T, L}(i_frame, decision, left, right)
    end
    function DTInternal(
        decision         :: Decision{T},
        left             :: Union{DTLeaf{L}, DTInternal{T, L}},
        right            :: Union{DTLeaf{L}, DTInternal{T, L}}) where {T, L<:Label}
        DTInternal{T, L}(decision, left, right)
    end

# Decision Node (Leaf or Internal)
const DTNode{T, L} = Union{DTLeaf{L}, DTInternal{T, L}}

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
        return new{L}(root, collect(worldTypes), collect(initConditions))
    end

    function DTree(
        root           :: DTNode{T, L},
        worldTypes     :: AbstractVector{<:Type},
        initConditions :: AbstractVector{<:_initCondition},
    ) where {T, L<:Label}
        return DTree{L}(root, collect(worldTypes), collect(initConditions))
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
num_leaves(tree::DTree)      = num_leaves(tree.root)

# Number of nodes
num_nodes(leaf::DTLeaf)     = 1
num_nodes(node::DTInternal) = 1 + num_nodes(node.left) + num_nodes(node.right)
num_nodes(tree::DTree)   = num_nodes(tree.root)
num_nodes(f::DForest) = sum(num_nodes.(f.trees))

# Number of trees
num_trees(f::DForest) = length(f.trees)
length(f::DForest)    = num_trees(f.trees)

# Height
height(leaf::DTLeaf)     = 0
height(node::DTInternal) = 1 + max(height(node.left), height(node.right))
height(tree::DTree)      = height(tree.root)

# Modal height
modal_height(leaf::DTLeaf)     = 0
modal_height(node::DTInternal) = Int(is_modal_node(node)) + max(modal_height(node.left), modal_height(node.right))
modal_height(tree::DTree)      = modal_height(tree.root)

# Number of supporting instances 
n_samples(leaf::DTLeaf)     = length(leaf.supp_labels)
n_samples(node::DTInternal) = n_samples(node.left) + n_samples(node.right)
n_samples(tree::DTree)      = n_samples(tree.root)

# TODO remove deprecated use num_leaves
length(leaf::DTLeaf)     = num_leaves(leaf)    
length(node::DTInternal) = num_leaves(node)
length(tree::DTree)      = num_leaves(tree)        

################################################################################
################################################################################

is_leaf_node(leaf::DTLeaf)     = true
is_leaf_node(node::DTInternal) = false
is_leaf_node(tree::DTree)      = is_leaf_node(tree.root)

is_modal_node(node::DTInternal) = (!is_leaf_node(node) && is_modal_decision(node.decision))
is_modal_node(tree::DTree)      = is_modal_node(tree.root)

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
    println(io, "Supporting labels:  $(leaf.supp_labels)")
    print_tree(io, leaf)
end

function show(io::IO, node::DTInternal)
    println(io, "Decision Node")
    println(io, display_decision(node))
    println(io, "Leaves: $(length(node))")
    println(io, "Tot nodes: $(num_nodes(node))")
    println(io, "Height: $(height(node))")
    println(io, "Modal height:  $(modal_height(node))")
    println(io, "Sub-tree:")
    print_tree(io, node)
end

function show(io::IO, tree::DTree)
    println(io, "Decision Tree")
    println(io, "Leaves: $(length(tree))")
    println(io, "Tot nodes: $(num_nodes(tree))")
    println(io, "Height: $(height(tree))")
    println(io, "Modal height:  $(modal_height(tree))")
    println(io, "worldTypes:     $(tree.worldTypes)")
    println(io, "initConditions: $(tree.initConditions)")
    println(io, "Tree:")
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

################################################################################
# Tests
################################################################################

# https://stackoverflow.com/questions/66801702/deriving-equality-for-julia-structs-with-mutable-members
import Base.==
function ==(a::S, b::S) where {S<:DTLeaf}
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
        @test DTLeaf{String}(["Class_1", "Class_1", "Class_2"]).label == "Class_1"
        
        @test_nowarn DTLeaf(["1.5"])
        @test_throws MethodError DTLeaf([1.0,"Class_1"])

        # Check robustness
        @test_nowarn DTLeaf{Int64}(1, 1:10)
        @test_nowarn DTLeaf{Int64}(1, 1.0:10.0)
        @test_nowarn DTLeaf{Float32}(1, 1:10)

        @test DTLeaf(1:10).label == 5.5
        @test_throws InexactError DTLeaf{Int64}(1:10)
        @test DTLeaf{Float32}(1:10).label == 5.5f0
        @test DTLeaf{Int64}(1:11).label == 6

        # Check edge parity case (aggregation biased towards the first class)
        @test DTLeaf{String}(["Class_1", "Class_2"]).label == "Class_1"
        @test DTLeaf(["Class_1", "Class_2"]).label == "Class_1"

    end
end


end # module
