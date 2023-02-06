using SoleLogics: AbstractMultiModalFrame

############################################################################################
# Initial world conditions
############################################################################################
using SoleLogics: InitCondition
import SoleLogics: initialworldset

struct StartWithoutWorld               <: InitCondition end; const start_without_world  = StartWithoutWorld();
struct StartAtCenter                   <: InitCondition end; const start_at_center      = StartAtCenter();
struct StartAtWorld{WT<:AbstractWorld} <: InitCondition w::WT end;

initialworldset(frs::AbstractVector{<:AbstractMultiModalFrame}, iCs::AbstractVector{<:InitCondition}) =
    [initialworldset(fr, iC) for (fr, iC) in zip(frs, iCs)]

initialworldset(::AbstractMultiModalFrame{W}, iC::StartWithoutWorld) where {W<:AbstractWorld} =
    WorldSet{W}([W(ModalLogic.EmptyWorld())])

initialworldset(::AbstractMultiModalFrame{W}, iC::StartAtCenter) where {W<:AbstractWorld} =
    WorldSet{W}([W(ModalLogic.CenteredWorld(), args...)])

initialworldset(::AbstractMultiModalFrame{W}, iC::StartAtWorld{W}) where {W<:AbstractWorld} =
    WorldSet{W}([W(iC.w)])

initialworldsets(Xs::MultiFrameModalDataset, iCs::AbstractVector{<:InitCondition}) = begin
    Ss = Vector{Vector{WST} where {W,WST<:WorldSet{W}}}(undef, nframes(Xs))
    for (i_frame,X) in enumerate(frames(Xs))
        WT = world_type(X)
        Ss[i_frame] = WorldSet{WT}[initialworldset(X, i_sample, iCs[i_frame]) for i_sample in 1:nsamples(Xs)]
        # Ss[i_frame] = WorldSet{WT}[[ModalLogic.Interval(1,2)] for i_sample in 1:nsamples(Xs)]
    end
    Ss
end

############################################################################################

abstract type AbstractNode{L<:Label} end
abstract type AbstractDecisionLeaf{L<:Label} <: AbstractNode{L} end
abstract type AbstractDecisionInternal{L<:Label,D<:AbstractDecision} <: AbstractNode{L} end

mutable struct DoubleEdgedDecision{D<:SimpleDecision} <: AbstractDecision
    decision :: D
    _back     :: Base.RefValue{N} where N<:AbstractNode # {L,DoubleEdgedDecision}
    _forth    :: Base.RefValue{N} where N<:AbstractNode # {L,DoubleEdgedDecision}

    function DoubleEdgedDecision{D}(decision::D) where {D<:SimpleDecision}
        ded = new{D}()
        ded.decision = decision
        ded
    end

    function DoubleEdgedDecision(decision::D) where {D<:SimpleDecision}
        DoubleEdgedDecision{D}(decision)
    end
end

decision(ded::DoubleEdgedDecision) = ded.decision
back(ded::DoubleEdgedDecision) = isdefined(ded, :_back) ? ded._back[] : nothing
forth(ded::DoubleEdgedDecision) = isdefined(ded, :_forth) ? ded._forth[] : nothing
_back(ded::DoubleEdgedDecision) = isdefined(ded, :_back) ? ded._back : nothing
_forth(ded::DoubleEdgedDecision) = isdefined(ded, :_forth) ? ded._forth : nothing

decision!(ded::DoubleEdgedDecision, decision) = (ded.decision = decision)
_back!(ded::DoubleEdgedDecision, _back) = (ded._back = _back)
_forth!(ded::DoubleEdgedDecision, _forth) = (ded._forth = _forth)

# TODO remove?
is_propositional_decision(ded::DoubleEdgedDecision) = is_propositional_decision(decision(ded))
is_global_decision(ded::DoubleEdgedDecision) = is_global_decision(decision(ded))

function display_decision(ded::DoubleEdgedDecision, args...; kwargs...)
    outstr = ""
    outstr *= "DoubleEdgedDecision("
    outstr *= display_decision(decision(ded))
    outstr *= ", " * (isnothing(_back(ded)) ? "-" : "$(typeof(_back(ded)))")
    outstr *= ", " * (isnothing(_forth(ded)) ? "-" : "$(typeof(_forth(ded)))")
    outstr *= ")"
    # outstr *= "DoubleEdgedDecision(\n\t"
    # outstr *= display_decision(decision(ded))
    # # outstr *= "\n\tback: " * (isnothing(back(ded)) ? "-" : display_model(back(ded), args...; kwargs...))
    # # outstr *= "\n\tforth: " * (isnothing(forth(ded)) ? "-" : display_model(forth(ded), args...; kwargs...))
    # outstr *= "\n\tback: " * (isnothing(_back(ded)) ? "-" : "$(typeof(_back(ded)))")
    # outstr *= "\n\tforth: " * (isnothing(_forth(ded)) ? "-" : "$(typeof(_forth(ded)))")
    # outstr *= "\n)"
    outstr
end

############################################################################################
############################################################################################
############################################################################################

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

predicting_function(leaf::NSDTLeaf) = leaf.predicting_function
supp_labels(leaf::NSDTLeaf; train_or_valid = true) = (train_or_valid ? leaf.supp_train_labels      : leaf.supp_valid_labels)
predictions(leaf::NSDTLeaf; train_or_valid = true) = (train_or_valid ? leaf.supp_train_predictions : leaf.supp_valid_predictions)

############################################################################################

# Internal decision node, holding a split-decision and a frame index
struct DTInternal{L<:Label,D<:AbstractDecision} <: AbstractDecisionInternal{L,D}
    # frame index + split-decision
    i_frame       :: Int64
    decision      :: D
    # representative leaf for the current node
    this          :: AbstractDecisionLeaf{<:L}
    # child nodes
    left          :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}}
    right         :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}}

    # semantics-specific miscellanoeus info
    miscellaneous :: NamedTuple

    # create node
    function DTInternal{L,D}(
        i_frame          :: Int64,
        decision         :: D,
        this             :: AbstractDecisionLeaf,
        left             :: Union{AbstractDecisionLeaf,DTInternal},
        right            :: Union{AbstractDecisionLeaf,DTInternal},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:AbstractDecision,L<:Label}
        new{L,D}(i_frame, decision, this, left, right, miscellaneous)
    end
    function DTInternal{L}(
        i_frame          :: Int64,
        decision         :: D,
        this             :: AbstractDecisionLeaf{<:L},
        left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:AbstractDecision,L<:Label}
        DTInternal{L,D}(i_frame, decision, this, left, right, miscellaneous)
    end
    function DTInternal(
        i_frame          :: Int64,
        decision         :: D,
        this             :: AbstractDecisionLeaf{<:L},
        left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:AbstractDecision,L<:Label}
        DTInternal{L,D}(i_frame, decision, this, left, right, miscellaneous)
    end

    # create node without local decision
    function DTInternal{L,D}(
        i_frame          :: Int64,
        decision         :: D,
        left             :: Union{AbstractDecisionLeaf,DTInternal},
        right            :: Union{AbstractDecisionLeaf,DTInternal},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:AbstractDecision,L<:Label}
        # this = merge_into_leaf(Vector{<:Union{AbstractDecisionLeaf,DTInternal}}([left, right]))
        this = merge_into_leaf(Union{<:AbstractDecisionLeaf,<:DTInternal}[left, right])
        new{L,D}(i_frame, decision, this, left, right, miscellaneous)
    end
    function DTInternal{L}(
        i_frame          :: Int64,
        decision         :: D,
        left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:AbstractDecision,L<:Label}
        DTInternal{L,D}(i_frame, decision, left, right, miscellaneous)
    end
    function DTInternal(
        i_frame          :: Int64,
        decision         :: D,
        left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:AbstractDecision,L<:Label}
        DTInternal{L,D}(i_frame, decision, left, right, miscellaneous)
    end

    # create node without frame
    # function DTInternal{L,D}(
    #     decision         :: AbstractDecision,
    #     this             :: AbstractDecisionLeaf,
    #     left             :: Union{AbstractDecisionLeaf,DTInternal},
    #     right            :: Union{AbstractDecisionLeaf,DTInternal}) where {T,L<:Label}
    #     i_frame = 1
    #     DTInternal{L,D}(i_frame, decision, this, left, right)
    # end
    # function DTInternal(
    #     decision         :: AbstractDecision{T},
    #     this             :: AbstractDecisionLeaf,
    #     left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
    #     right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}}) where {T,L<:Label}
    #     DTInternal{L,D}(decision, this, left, right)
    # end

    # # create node without frame nor local decision
    # function DTInternal{L,D}(
    #     decision         :: AbstractDecision,
    #     left             :: Union{AbstractDecisionLeaf,DTInternal},
    #     right            :: Union{AbstractDecisionLeaf,DTInternal}) where {T,L<:Label}
    #     i_frame = 1
    #     DTInternal{L,D}(i_frame, decision, left, right)
    # end
    # function DTInternal(
    #     decision         :: AbstractDecision{T},
    #     left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
    #     right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}}) where {T,L<:Label}
    #     DTInternal{L,D}(decision, left, right)
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
const DTNode{L,D} = Union{<:AbstractDecisionLeaf{<:L},<:AbstractDecisionInternal{L,D}}

############################################################################################

abstract type SymbolicModel{L} end

# Decision Tree
struct DTree{L<:Label} <: SymbolicModel{L}
    # root node
    root            :: DTNode{L}
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
        root            :: DTNode{L,D},
        world_types     :: AbstractVector{<:Type},
        init_conditions :: AbstractVector{<:InitCondition},
    ) where {L<:Label,D<:AbstractDecision}
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
nleaves(leaf::AbstractDecisionLeaf)     = 1
nleaves(node::DTInternal) = nleaves(left(node)) + nleaves(right(node))
nleaves(tree::DTree)      = nleaves(root(tree))

# Number of nodes
nnodes(leaf::AbstractDecisionLeaf)     = 1
nnodes(node::DTInternal) = 1 + nnodes(left(node)) + nnodes(right(node))
nnodes(tree::DTree)   = nnodes(root(tree))
nnodes(f::DForest) = sum(nnodes.(trees(f)))

# Number of trees
ntrees(f::DForest) = length(trees(f))
Base.length(f::DForest)    = ntrees(f)

# Height
height(leaf::AbstractDecisionLeaf)     = 0
height(node::DTInternal) = 1 + max(height(left(node)), height(right(node)))
height(tree::DTree)      = height(root(tree))

# Modal height
modal_height(leaf::AbstractDecisionLeaf)     = 0
modal_height(node::DTInternal) = Int(ismodalnode(node)) + max(modal_height(left(node)), modal_height(right(node)))
modal_height(tree::DTree)      = modal_height(root(tree))

# Number of supporting instances
nsamples(leaf::AbstractDecisionLeaf; train_or_valid = true) = length(supp_labels(leaf; train_or_valid = train_or_valid))
nsamples(node::DTInternal;           train_or_valid = true) = nsamples(left(node); train_or_valid = train_or_valid) + nsamples(right(node); train_or_valid = train_or_valid)
nsamples(tree::DTree;                train_or_valid = true) = nsamples(root(tree); train_or_valid = train_or_valid)

############################################################################################
############################################################################################

isleafnode(leaf::AbstractDecisionLeaf)     = true
isleafnode(node::DTInternal) = false
isleafnode(tree::DTree)      = isleafnode(root(tree))

ismodalnode(node::DTInternal) = (!isleafnode(node) && !is_propositional_decision(decision(node)))
ismodalnode(tree::DTree)      = ismodalnode(root(tree))

############################################################################################
############################################################################################

display_decision(node::DTInternal, args...; kwargs...) =
    ModalLogic.display_decision(i_frame(node), decision(node), args...; kwargs...)
display_decision_inverse(node::DTInternal, args...; kwargs...) =
    ModalLogic.display_decision_inverse(i_frame(node), decision(node), args...; kwargs...)

############################################################################################
############################################################################################

Base.show(io::IO, a::Union{DTNode,DTree,DForest}) = println(io, display(a))

function display(leaf::DTLeaf{L}) where {L<:CLabel}
    return """
Classification Decision Leaf{$(L)}(
    label: $(prediction(leaf))
    supporting labels:  $(supp_labels(leaf))
    supporting labels countmap:  $(StatsBase.countmap(supp_labels(leaf)))
    metrics: $(get_metrics(leaf))
)
"""
end
function display(leaf::DTLeaf{L}) where {L<:RLabel}
    return """
Regression Decision Leaf{$(L)}(
    label: $(prediction(leaf))
    supporting labels:  $(supp_labels(leaf))
    metrics: $(get_metrics(leaf))
)
"""
end

function display(leaf::NSDTLeaf{L}) where {L<:CLabel}
    return """
Classification Functional Decision Leaf{$(L)}(
    predicting_function: $(leaf.predicting_function)
    supporting labels (train):  $(leaf.supp_train_labels)
    supporting labels (valid):  $(leaf.supp_valid_labels)
    supporting predictions (train):  $(leaf.supp_train_predictions)
    supporting predictions (valid):  $(leaf.supp_valid_predictions)
    supporting labels countmap (train):  $(StatsBase.countmap(leaf.supp_train_labels))
    supporting labels countmap (valid):  $(StatsBase.countmap(leaf.supp_valid_labels))
    supporting predictions countmap (train):  $(StatsBase.countmap(leaf.supp_train_predictions))
    supporting predictions countmap (valid):  $(StatsBase.countmap(leaf.supp_valid_predictions))
    metrics (train): $(get_metrics(leaf; train_or_valid = true))
    metrics (valid): $(get_metrics(leaf; train_or_valid = false))
)
"""
end
function display(leaf::NSDTLeaf{L}) where {L<:RLabel}
    return """
Regression Functional Decision Leaf{$(L)}(
    predicting_function: $(leaf.predicting_function)
    supporting labels (train):  $(leaf.supp_train_labels)
    supporting labels (valid):  $(leaf.supp_valid_labels)
    supporting predictions (train):  $(leaf.supp_train_predictions)
    supporting predictions (valid):  $(leaf.supp_valid_predictions)
    metrics (train): $(get_metrics(leaf; train_or_valid = true))
    metrics (valid): $(get_metrics(leaf; train_or_valid = false))
)
"""
end

function display(node::DTInternal{L,D}) where {L,D}
    return """
Decision Node{$(L),$(D)}(
$(display(this(node)))
    ###########################################################
    i_frame: $(i_frame(node))
    decision: $(decision(node))
    miscellaneous: $(miscellaneous(node))
    ###########################################################
    sub-tree leaves: $(nleaves(node))
    sub-tree nodes: $(nnodes(node))
    sub-tree height: $(height(node))
    sub-tree modal height:  $(modal_height(node))
)
"""
end

function display(tree::DTree{L}) where {L}
    return """
Decision Tree{$(L)}(
    world_types:    $(world_types(tree))
    initConditions: $(init_conditions(tree))
    ###########################################################
    sub-tree leaves: $(nleaves(tree))
    sub-tree nodes: $(nnodes(tree))
    sub-tree height: $(height(tree))
    sub-tree modal height:  $(modal_height(tree))
    ###########################################################
    tree:
$(display_model(tree))
)
"""
end

function display(forest::DForest{L}) where {L}
    return """
Decision Forest{$(L)}(
    # trees: $(length(forest))
    metrics: $(metrics(forest))
    forest:
$(display_model(forest))
)
"""
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

        decision = ExistentialDimensionalDecision(RelationGlob, SingleAttributeMin(1), >=, 10)

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
