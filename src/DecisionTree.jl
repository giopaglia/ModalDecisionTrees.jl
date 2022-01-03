# __precompile__()

module DecisionTree

import Base: length, show
using LinearAlgebra
using Printf
using Random
using Statistics
using StatsBase

using Logging: LogLevel, @logmsg

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

# TODO update these
export DTNode, DTLeaf, DTInternal, DTree, DForest,
        is_leaf_node, is_modal_node,
        num_nodes, height, modal_height,
        ConfusionMatrix, confusion_matrix, mean_squared_error, R2, load_data,
        #
        startWithRelationGlob, startAtCenter,
        DTOverview, DTDebug, DTDetail,
        #
        initWorldSet,
        #
        throw_n_log

function throw_n_log(str::AbstractString, err_type = ErrorException)
    @error str
    throw(err_type(str))
end


# make a Random Number Generator object
mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(seed::T) where T <: Integer = Random.MersenneTwister(seed)

# Generate a new rng from a random pick from a given one.
spawn_rng(rng) = Random.MersenneTwister(abs(rand(rng, Int)))


# ScikitLearn API
export DecisionTreeClassifier,
#        DecisionTreeRegressor, RandomForestClassifier,
#        RandomForestRegressor, AdaBoostStumpClassifier,
#        # Should we export these functions? They have a conflict with
#        # DataFrames/RDataset over fit!, and users can always
#        # `using ScikitLearnBase`.
            predict,
            # predict_proba,
            fit!, get_classes

include("ModalLogic/ModalLogic.jl")
using .ModalLogic

import .ModalLogic: n_samples # TODO make this a DecisionTree.n_samples export, and make ModalLogic import it?

include("measures.jl")
include("util.jl")

################################################################################
# Initial world conditions
################################################################################

abstract type _initCondition end
struct _startWithRelationGlob  <: _initCondition end; const startWithRelationGlob  = _startWithRelationGlob();
struct _startAtCenter          <: _initCondition end; const startAtCenter          = _startAtCenter();
struct _startAtWorld{wT<:AbstractWorld} <: _initCondition w::wT end;

initWorldSet(initConditions::AbstractVector{<:_initCondition}, worldTypes::AbstractVector{<:Type#={<:AbstractWorld}=#}, args::Vararg) =
    [initWorldSet(iC, WT, args...) for (iC, WT) in zip(initConditions, worldTypes)]

initWorldSet(initCondition::_startWithRelationGlob, ::Type{WorldType}, channel_size::NTuple{N,Integer} where N) where {WorldType<:AbstractWorld} =
    WorldSet{WorldType}([WorldType(ModalLogic.emptyWorld)])

initWorldSet(initCondition::_startAtCenter, ::Type{WorldType}, channel_size::NTuple{N,Integer} where N) where {WorldType<:AbstractWorld} =
    WorldSet{WorldType}([WorldType(ModalLogic.centeredWorld, channel_size...)])

initWorldSet(initCondition::_startAtWorld{WorldType}, ::Type{WorldType}, channel_size::NTuple{N,Integer} where N) where {WorldType<:AbstractWorld} =
    WorldSet{WorldType}([WorldType(initCondition.w)])

################################################################################
# Decision Leaf, Internal, Node, Tree & RF
################################################################################

# Leaf node, holding the output decision
struct DTLeaf{S}
    # Majority class/value (output)
    majority :: S
    # Training support
    values   :: Vector{S}
end

# Inner node, holding the output decision
struct DTInternal{T, S}
    # Split label
    i_frame       :: Integer
    relation      :: AbstractRelation
    feature       :: ModalLogic.FeatureTypeFun # TODO move FeatureTypeFun out of ModalLogic?
    test_operator :: TestOperatorFun # Test operator (e.g. <=, ==)
    threshold     :: T
    # Child nodes
    left          :: Union{DTLeaf{S}, DTInternal{T, S}}
    right         :: Union{DTLeaf{S}, DTInternal{T, S}}
end

display_decision(tree::DTInternal; threshold_display_method::Function = x -> x) = ModalLogic.display_decision(tree.i_frame, tree.relation, tree.feature, tree.test_operator, tree.threshold; threshold_display_method = threshold_display_method)
display_decision_inverse(tree::DTInternal; threshold_display_method::Function = x -> x) = ModalLogic.display_decision_inverse(tree.i_frame, tree.relation, tree.feature, tree.test_operator, tree.threshold; threshold_display_method = threshold_display_method)

# Decision node/tree # TODO figure out, maybe this has to be abstract and to supertype DTLeaf and DTInternal
const DTNode{T, S} = Union{DTLeaf{S}, DTInternal{T, S}}

# TODO attach info about training (e.g. algorithm used + full namedtuple of training arguments) to these models
struct DTree{S}
    root           :: DTNode{T, S} where T
    worldTypes     :: AbstractVector{<:Type} # <:Type{<:AbstractWorld}}
    initConditions :: AbstractVector{<:_initCondition}
    function DTree{S}(
        root           :: DTNode{T, S},
        worldTypes     :: AbstractVector{<:Type}, # <:Type{<:AbstractWorld}},
        initConditions :: AbstractVector{<:_initCondition},
    ) where {T, S}
    new{S}(root, worldTypes, initConditions)
    end
    function DTree(
        root           :: DTNode{T, S},
        args...,
    ) where {T, S}
    DTree{S}(root, args...)
    end
end

struct DForest{S}
    trees       :: AbstractVector{<:DTree{S}}
    cm          :: Union{
        AbstractVector{<:ConfusionMatrix},
        AbstractVector{<:PerformanceStruct}, #where {N},
    }
    oob_error   :: AbstractFloat

    function DForest{S}(
        trees       :: AbstractVector{<:DTree{S}},
        cm          :: Union{
            AbstractVector{<:ConfusionMatrix},
            AbstractVector{<:PerformanceStruct}, #where {N},
        },
        oob_error   :: AbstractFloat,
    ) where {S}
    new{S}(trees, cm, oob_error)
    end
    function DForest(
        trees       :: AbstractVector{<:DTree{S}},
        args...,
    ) where {S}
    DForest{S}(root, args...)
    end
end

is_leaf_node(l::DTLeaf)     = true
is_leaf_node(n::DTInternal) = false
is_leaf_node(t::DTree)      = is_leaf_node(t.root)

is_modal_node(n::DTInternal) = (!is_leaf_node(n) && n.relation != RelationId)
is_modal_node(t::DTree) = is_modal_node(t.root)

################################################################################
# Methods
################################################################################

# Length (total # of nodes)
num_nodes(leaf::DTLeaf) = 1
num_nodes(tree::DTInternal) = 1 + num_nodes(tree.left) + num_nodes(tree.right)
num_nodes(t::DTree) = num_nodes(t.root)
num_nodes(f::DForest) = sum(num_nodes.(f.trees))

length(leaf::DTLeaf) = 1
length(tree::DTInternal) = length(tree.left) + length(tree.right)
length(t::DTree) = length(t.root)
length(f::DForest) = length(f.trees)

# Height
height(leaf::DTLeaf) = 0
height(tree::DTInternal) = 1 + max(height(tree.left), height(tree.right))
height(t::DTree) = height(t.root)

# Modal height
modal_height(leaf::DTLeaf) = 0
modal_height(tree::DTInternal) = (is_modal_node(tree) ? 1 : 0) + max(modal_height(tree.left), modal_height(tree.right))
modal_height(t::DTree) = modal_height(t.root)

################################################################################
################################################################################
################################################################################

function show(io::IO, leaf::DTLeaf)
    println(io, "Decision Leaf")
    println(io, "Majority: $(leaf.majority)")
    println(io, "Samples:  $(length(leaf.values))")
    print_tree(io, leaf)
end

function show(io::IO, tree::DTInternal)
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
include("print-latex.jl")
include("decisionpath.jl")


end # module
