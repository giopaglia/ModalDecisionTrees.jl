
abstract type AbstractDecisionLeaf{L<:Label} end

# Decision leaf node, holding an output (prediction)
struct DTLeaf{L<:Label} <: AbstractDecisionLeaf{L}
    # prediction
    prediction    :: L
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
struct DTInternal{D<:AbstractDecision, L<:Label}
    # frame index + split-decision
    i_frame       :: Int64
    decision      :: D
    # representative leaf for the current node
    this          :: AbstractDecisionLeaf{<:L}
    # child nodes
    left          :: Union{AbstractDecisionLeaf{<:L}, DTInternal{D, L}}
    right         :: Union{AbstractDecisionLeaf{<:L}, DTInternal{D, L}}

    # semantics-specific miscellanoeus info
    miscellaneous :: NamedTuple

    # create node
    function DTInternal{D, L}(
        i_frame          :: Int64,
        decision         :: D,
        this             :: AbstractDecisionLeaf,
        left             :: Union{AbstractDecisionLeaf, DTInternal},
        right            :: Union{AbstractDecisionLeaf, DTInternal},
        miscellaneous    :: NamedTuple = (;),
        ) where {D<:AbstractDecision, L<:Label}
        new{D, L}(i_frame, decision, this, left, right, miscellaneous)
    end
    function DTInternal(
        i_frame          :: Int64,
        decision         :: D,
        this             :: AbstractDecisionLeaf{<:L},
        left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{D, L}},
        right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{D, L}},
        miscellaneous    :: NamedTuple = (;),
        ) where {D<:AbstractDecision, L<:Label}
        DTInternal{D, L}(i_frame, decision, this, left, right, miscellaneous)
    end

    # create node without local decision
    function DTInternal{D, L}(
        i_frame          :: Int64,
        decision         :: D,
        left             :: Union{AbstractDecisionLeaf, DTInternal},
        right            :: Union{AbstractDecisionLeaf, DTInternal},
        miscellaneous    :: NamedTuple = (;),
        ) where {D<:AbstractDecision, L<:Label}
        # this = merge_into_leaf(Vector{<:Union{AbstractDecisionLeaf,DTInternal}}([left, right]))
        this = merge_into_leaf(Union{<:AbstractDecisionLeaf,<:DTInternal}[left, right])
        new{D, L}(i_frame, decision, this, left, right, miscellaneous)
    end
    function DTInternal(
        i_frame          :: Int64,
        decision         :: D,
        left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{D, L}},
        right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{D, L}},
        miscellaneous    :: NamedTuple = (;),
        ) where {D<:AbstractDecision, L<:Label}
        DTInternal{D, L}(i_frame, decision, left, right, miscellaneous)
    end

    # create node without frame
    # function DTInternal{T, L}(
    #     decision         :: AbstractDecision,
    #     this             :: AbstractDecisionLeaf,
    #     left             :: Union{AbstractDecisionLeaf, DTInternal},
    #     right            :: Union{AbstractDecisionLeaf, DTInternal}) where {T, L<:Label}
    #     i_frame = 1
    #     DTInternal{T, L}(i_frame, decision, this, left, right)
    # end
    # function DTInternal(
    #     decision         :: AbstractDecision{T},
    #     this             :: AbstractDecisionLeaf,
    #     left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{T, L}},
    #     right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{T, L}}) where {T, L<:Label}
    #     DTInternal{T, L}(decision, this, left, right)
    # end

    # # create node without frame nor local decision
    # function DTInternal{T, L}(
    #     decision         :: AbstractDecision,
    #     left             :: Union{AbstractDecisionLeaf, DTInternal},
    #     right            :: Union{AbstractDecisionLeaf, DTInternal}) where {T, L<:Label}
    #     i_frame = 1
    #     DTInternal{T, L}(i_frame, decision, left, right)
    # end
    # function DTInternal(
    #     decision         :: AbstractDecision{T},
    #     left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{T, L}},
    #     right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{T, L}}) where {T, L<:Label}
    #     DTInternal{T, L}(decision, left, right)
    # end
end

i_frame(node::DTInternal) = node.i_frame
decision(node::DTInternal) = node.decision
this(node::DTInternal) = node.this
left(node::DTInternal) = node.left
right(node::DTInternal) = node.right
miscellaneous(node::DTInternal) = node.miscellaneous

function supp_labels(node::DTInternal; train_or_valid = true)
    @assert train_or_valid == true
    supp_labels(this(node); train_or_valid = train_or_valid)
end

############################################################################################

# Decision Node (Leaf or Internal)
const DTNode{T, L} = Union{<:AbstractDecisionLeaf{<:L}, DTInternal{T, L}}

############################################################################################

abstract type SymbolicModel{T} end

# Decision Tree
struct DTree{L<:Label} <: SymbolicModel{L}
    # root node
    root            :: DTNode{T, L} where T
    # world types (one per frame)
    world_types     :: Vector{Type{<:AbstractWorld}}
    # initial world conditions (one per frame)
    init_conditions :: Vector{<:InitCondition}

    function DTree{L}(
        root            :: DTNode,
        world_types     :: AbstractVector{<:Type},
        init_conditions :: AbstractVector{<:InitCondition},
    ) where {L<:Label}
        new{L}(root, collect(world_types), collect(init_conditions))
    end

    function DTree(
        root            :: DTNode{T, L},
        world_types     :: AbstractVector{<:Type},
        init_conditions :: AbstractVector{<:InitCondition},
    ) where {T, L<:Label}
        DTree{L}(root, world_types, init_conditions)
    end
end

root(tree::DTree) = tree.root
world_types(tree::DTree) = tree.world_types
init_conditions(tree::DTree) = tree.init_conditions

############################################################################################

# Decision Forest (i.e., ensable of trees via bagging)
struct DForest{L<:Label} <: SymbolicModel{L}
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

trees(forest::DForest) = forest.trees
metrics(forest::DForest) = forest.metrics

############################################################################################
# Methods
############################################################################################

# Number of leaves
num_leaves(leaf::AbstractDecisionLeaf)     = 1
num_leaves(node::DTInternal) = num_leaves(left(node)) + num_leaves(right(node))
num_leaves(tree::DTree)      = num_leaves(root(tree))

# Number of nodes
num_nodes(leaf::AbstractDecisionLeaf)     = 1
num_nodes(node::DTInternal) = 1 + num_nodes(left(node)) + num_nodes(right(node))
num_nodes(tree::DTree)   = num_nodes(root(tree))
num_nodes(f::DForest) = sum(num_nodes.(trees(f)))

# Number of trees
num_trees(f::DForest) = length(trees(f))
Base.length(f::DForest)    = num_trees(f)

# Height
height(leaf::AbstractDecisionLeaf)     = 0
height(node::DTInternal) = 1 + max(height(left(node)), height(right(node)))
height(tree::DTree)      = height(root(tree))

# Modal height
modal_height(leaf::AbstractDecisionLeaf)     = 0
modal_height(node::DTInternal) = Int(is_modal_node(node)) + max(modal_height(left(node)), modal_height(right(node)))
modal_height(tree::DTree)      = modal_height(root(tree))

# Number of supporting instances
nsamples(leaf::AbstractDecisionLeaf; train_or_valid = true) = length(supp_labels(leaf; train_or_valid = train_or_valid))
nsamples(node::DTInternal;           train_or_valid = true) = nsamples(left(node); train_or_valid = train_or_valid) + nsamples(right(node); train_or_valid = train_or_valid)
nsamples(tree::DTree;                train_or_valid = true) = nsamples(root(tree); train_or_valid = train_or_valid)

# TODO remove deprecated use num_leaves
Base.length(leaf::AbstractDecisionLeaf)     = num_leaves(leaf)
Base.length(node::DTInternal) = num_leaves(node)
Base.length(tree::DTree)      = num_leaves(tree)

############################################################################################
############################################################################################

is_leaf_node(leaf::AbstractDecisionLeaf)     = true
is_leaf_node(node::DTInternal) = false
is_leaf_node(tree::DTree)      = is_leaf_node(root(tree))

is_modal_node(node::DTInternal) = (!is_leaf_node(node) && !is_propositional_decision(decision(node)))
is_modal_node(tree::DTree)      = is_modal_node(root(tree))

############################################################################################
############################################################################################

display_decision(node::DTInternal, args...; kwargs...) =
    ModalLogic.display_decision(i_frame(node), decision(node), args...; kwargs...)
display_decision_inverse(node::DTInternal, args...; kwargs...) =
    ModalLogic.display_decision_inverse(i_frame(node), decision(node), args...; kwargs...)

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
    Base.show(io, this(node))
    println(io, "\t###########################################################")
    println(io, "\ti_frame: $(i_frame(node))")
    print(io, "\tdecision: $(decision(node))")
    print(io, "\tmiscellaneous: $(miscellaneous(node))")
    println(io, "\t###########################################################")
    println(io, "\tsub-tree leaves: $(num_leaves(node))")
    println(io, "\tsub-tree nodes: $(num_nodes(node))")
    println(io, "\tsub-tree height: $(height(node))")
    println(io, "\tsub-tree modal height:  $(modal_height(node))")
    println(io, ")")
end

function Base.show(io::IO, tree::DTree{L}) where {L}
    println(io, "Decision Tree{$(L)}(")
    println(io, "\tworld_types:    $(world_types(tree))")
    println(io, "\tinitConditions: $(init_conditions(tree))")
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
    println(io, "\tmetrics: $(metrics(forest))")
    println(io, "\ttrees:")
    print_model(io, forest)
    println(io, ")")
end


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

        decision = ExistentialDimensionalDecision(ModalLogic.RelationGlob, SingleAttributeMin(1), >=, 10)

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

        # cls_tree = @test_nowarn DTree(cls_node, [ModalLogic.Interval], [start_without_world])
        # cls_forest = @test_nowarn DForest([cls_tree, cls_tree, cls_tree])
    end

end
