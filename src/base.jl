using SoleLogics: AbstractMultiModalFrame
import SoleModels: printmodel, displaymodel
import SoleModels.DimensionalDatasets: worldtypes

############################################################################################
# Initial world conditions
############################################################################################
using SoleLogics: InitCondition
import SoleLogics: initialworldset

struct StartWithoutWorld               <: InitCondition end; const start_without_world  = StartWithoutWorld();
struct StartAtCenter                   <: InitCondition end; const start_at_center      = StartAtCenter();
struct StartAtWorld{W<:AbstractWorld}  <: InitCondition w::W end;

function initialworldset(frs::AbstractVector{<:AbstractMultiModalFrame}, iCs::AbstractVector{<:InitCondition})
    [initialworldset(fr, iC) for (fr, iC) in zip(frs, iCs)]
end

function initialworldset(fr::AbstractMultiModalFrame{W}, iC::StartWithoutWorld) where {W<:AbstractWorld}
    WorldSet{W}([SoleLogics.emptyworld(fr)])
end

function initialworldset(fr::AbstractMultiModalFrame{W}, iC::StartAtCenter) where {W<:AbstractWorld}
    WorldSet{W}([SoleLogics.centeredworld(fr)])
end

function initialworldset(::AbstractMultiModalFrame{W}, iC::StartAtWorld{W}) where {W<:AbstractWorld}
    WorldSet{W}([iC.w])
end

function initialworldsets(Xs::MultiLogiset, iCs::AbstractVector{<:InitCondition})
    Ss = Vector{Vector{WST} where {W,WST<:WorldSet{W}}}(undef, nmodalities(Xs)) # Fix
    for (i_modality,X) in enumerate(modalities(Xs))
        W = worldtype(X)
        Ss[i_modality] = WorldSet{W}[initialworldset(X, i_instance, iCs[i_modality]) for i_instance in 1:ninstances(Xs)]
        # Ss[i_modality] = WorldSet{W}[[Interval(1,2)] for i_instance in 1:ninstances(Xs)]
    end
    Ss
end

initialworldset(X::AbstractLogiset, i_instance::Integer, args...) = initialworldset(SoleModels.frame(X, i_instance), args...)

############################################################################################

abstract type AbstractDecision end

abstract type AbstractNode{L<:Label} end
abstract type AbstractDecisionLeaf{L<:Label} <: AbstractNode{L} end
abstract type AbstractDecisionInternal{L<:Label,D<:AbstractDecision} <: AbstractNode{L} end

# Decision Node (Leaf or Internal)
const DTNode{L<:Label,D<:AbstractDecision} = Union{<:AbstractDecisionLeaf{<:L},<:AbstractDecisionInternal{L,D}}

isleftchild(node::DTNode, parent::DTNode) = (left(parent) == node)
isrightchild(node::DTNode, parent::DTNode) = (right(parent) == node)

# TODO maybe one day?
# abstract type AbstractNode{L<:Label} end
# abstract type DTNode{L<:Label,D<:AbstractDecision} <: AbstractNode{L} end
# abstract type AbstractDecisionLeaf{L<:Label} <: DTNode{L,D} where D<:AbstractDecision end
# abstract type AbstractDecisionInternal{L<:Label,D<:AbstractDecision} <: DTNode{L,D} end

############################################################################################
TODO...
using SoleLogics: identityrel, globalrel
using SoleModels: syntaxstring
using SoleModels.DimensionalDatasets: alpha

export ExistentialScalarDecision,
       #
       relation, feature, test_operator, threshold,
       is_propositional_decision,
       is_global_decision,
       #
       displaydecision, displaydecision_inverse

is_propositional_decision(d::ScalarOneStepFormula) = (relation(d) == identityrel)
is_global_decision(d::ScalarOneStepFormula) = (relation(d) == globalrel)

function displaydecision(
    decision::Union{ScalarExistentialFormula,ScalarUniversalFormula};
    threshold_display_method::Function = x -> x,
    variable_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing,
    use_feature_abbreviations::Bool = false,
)
    prop_decision_str = syntaxstring(
        decision.p;
        threshold_display_method = threshold_display_method,
        variable_names_map = variable_names_map,
        use_feature_abbreviations = use_feature_abbreviations,
    )
    if !is_propositional_decision(decision)
        TODO
        rel_display_fun = (decision isa ScalarExistentialFormula ? display_existential : display_universal)
        "$(rel_display_fun(relation(decision))) ($prop_decision_str)"
    else
        "$prop_decision_str"
    end
end



mutable struct SimpleDecision{D<:AbstractTemplatedFormula} <: AbstractDecision
    decision  :: D
end

TODO

mutable struct DoubleEdgedDecision{D<:AbstractTemplatedFormula} <: AbstractDecision
    decision  :: D
    _back     :: Base.RefValue{N} where N<:AbstractNode # {L,DoubleEdgedDecision}
    _forth    :: Base.RefValue{N} where N<:AbstractNode # {L,DoubleEdgedDecision}

    function DoubleEdgedDecision{D}(decision::D) where {D<:AbstractTemplatedFormula}
        ded = new{D}()
        ded.decision = decision
        ded
    end

    function DoubleEdgedDecision(decision::D) where {D<:AbstractTemplatedFormula}
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

function displaydecision(ded::DoubleEdgedDecision, args...; kwargs...)
    outstr = ""
    outstr *= "DoubleEdgedDecision("
    outstr *= displaydecision(decision(ded))
    outstr *= ", " * (isnothing(_back(ded)) ? "-" : "$(typeof(_back(ded)))")
    outstr *= ", " * (isnothing(_forth(ded)) ? "-" : "$(typeof(_forth(ded)))")
    outstr *= ")"
    # outstr *= "DoubleEdgedDecision(\n\t"
    # outstr *= displaydecision(decision(ded))
    # # outstr *= "\n\tback: " * (isnothing(back(ded)) ? "-" : displaymodel(back(ded), args...; kwargs...))
    # # outstr *= "\n\tforth: " * (isnothing(forth(ded)) ? "-" : displaymodel(forth(ded), args...; kwargs...))
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
    DTLeaf{L}(supp_labels::AbstractVector) where {L<:Label} = DTLeaf{L}(bestguess(L.(supp_labels)), supp_labels)
    function DTLeaf(supp_labels::AbstractVector)
        prediction = bestguess(supp_labels)
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
    # f::FunctionWrapper{Vector{L},Tuple{MultiLogiset}} # TODO restore!!!
    f::FunctionWrapper{Any,Tuple{MultiLogiset}}

    function PredictingFunction{L}(f::Any) where {L<:Label}
        # new{L}(FunctionWrapper{Vector{L},Tuple{MultiLogiset}}(f)) # TODO restore!!!
        new{L}(FunctionWrapper{Any,Tuple{MultiLogiset}}(f))
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
    # NSDTLeaf{L}(supp_labels::AbstractVector) where {L<:Label} = NSDTLeaf{L}(bestguess(supp_labels), supp_labels)
    # function NSDTLeaf(supp_labels::AbstractVector)
    #     predicting_function = bestguess(supp_labels)
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
    frameid       :: FrameId
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
        frameid          :: FrameId,
        decision         :: D,
        this             :: AbstractDecisionLeaf,
        left             :: Union{AbstractDecisionLeaf,DTInternal},
        right            :: Union{AbstractDecisionLeaf,DTInternal},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:AbstractDecision,L<:Label}
        new{L,D}(frameid, decision, this, left, right, miscellaneous)
    end
    function DTInternal{L}(
        frameid          :: FrameId,
        decision         :: D,
        this             :: AbstractDecisionLeaf{<:L},
        left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:AbstractDecision,L<:Label}
        DTInternal{L,D}(frameid, decision, this, left, right, miscellaneous)
    end
    function DTInternal(
        frameid          :: FrameId,
        decision         :: D,
        this             :: AbstractDecisionLeaf{<:L},
        left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:AbstractDecision,L<:Label}
        DTInternal{L,D}(frameid, decision, this, left, right, miscellaneous)
    end

    # create node without local decision
    function DTInternal{L,D}(
        frameid          :: FrameId,
        decision         :: D,
        left             :: Union{AbstractDecisionLeaf,DTInternal},
        right            :: Union{AbstractDecisionLeaf,DTInternal},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:Union{AbstractDecision,AbstractTemplatedFormula},L<:Label}
        if decision isa AbstractTemplatedFormula
            decision = SimpleDecision(decision)
        end
        # this = merge_into_leaf(Vector{<:Union{AbstractDecisionLeaf,DTInternal}}([left, right]))
        this = merge_into_leaf(Union{<:AbstractDecisionLeaf,<:DTInternal}[left, right])
        new{L,D}(frameid, decision, this, left, right, miscellaneous)
    end
    function DTInternal{L}(
        frameid          :: FrameId,
        decision         :: D,
        left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:AbstractDecision,L<:Label}
        DTInternal{L,D}(frameid, decision, left, right, miscellaneous)
    end
    function DTInternal(
        frameid          :: FrameId,
        decision         :: D,
        left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:Union{AbstractDecision,AbstractTemplatedFormula},L<:Label}
        if decision isa AbstractTemplatedFormula
            decision = SimpleDecision(decision)
        end
        DTInternal{L,D}(frameid, decision, left, right, miscellaneous)
    end

    # create node without frame
    # function DTInternal{L,D}(
    #     decision         :: AbstractDecision,
    #     this             :: AbstractDecisionLeaf,
    #     left             :: Union{AbstractDecisionLeaf,DTInternal},
    #     right            :: Union{AbstractDecisionLeaf,DTInternal}) where {T,L<:Label}
    #     frameid = 1
    #     DTInternal{L,D}(frameid, decision, this, left, right)
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
    #     frameid = 1
    #     DTInternal{L,D}(frameid, decision, left, right)
    # end
    # function DTInternal(
    #     decision         :: AbstractDecision{T},
    #     left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}},
    #     right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{L,D}}) where {T,L<:Label}
    #     DTInternal{L,D}(decision, left, right)
    # end
end

frameid(node::DTInternal) = node.frameid
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

abstract type SymbolicModel{L} end

# Decision Tree
struct DTree{L<:Label} <: SymbolicModel{L}
    # root node
    root            :: DTNode{L}
    # world types (one per frame)
    worldtypes     :: Vector{Type{<:AbstractWorld}}
    # initial world conditions (one per frame)
    init_conditions :: Vector{<:InitCondition}

    function DTree{L}(
        root            :: DTNode,
        worldtypes     :: AbstractVector{<:Type},
        init_conditions :: AbstractVector{<:InitCondition},
    ) where {L<:Label}
        new{L}(root, collect(worldtypes), collect(init_conditions))
    end

    function DTree(
        root            :: DTNode{L,D},
        worldtypes     :: AbstractVector{<:Type},
        init_conditions :: AbstractVector{<:InitCondition},
    ) where {L<:Label,D<:AbstractDecision}
        DTree{L}(root, worldtypes, init_conditions)
    end
end

root(tree::DTree) = tree.root
worldtypes(tree::DTree) = tree.worldtypes
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

# Ensamble of decision trees weighted by softmax autoencoder
struct RootLevelNeuroSymbolicHybrid{F<:Any,L<:Label} <: SymbolicModel{L}
    feature_function :: F
    # trees
    trees       :: Vector{<:DTree{L}}
    # metrics
    metrics     :: NamedTuple

    function RootLevelNeuroSymbolicHybrid{F,L}(
        feature_function :: F,
        trees     :: AbstractVector{<:DTree},
        metrics   :: NamedTuple = (;),
    ) where {F<:Any,L<:Label}
        new{F,L}(feature_function, collect(trees), metrics)
    end
    function RootLevelNeuroSymbolicHybrid(
        feature_function :: F,
        trees     :: AbstractVector{<:DTree{L}},
        metrics   :: NamedTuple = (;),
    ) where {F<:Any,L<:Label}
        RootLevelNeuroSymbolicHybrid{F,L}(feature_function, trees, metrics)
    end
end

trees(nsdt::RootLevelNeuroSymbolicHybrid) = nsdt.trees
metrics(nsdt::RootLevelNeuroSymbolicHybrid) = nsdt.metrics

############################################################################################
# Methods
############################################################################################

# Number of leaves
nleaves(leaf::AbstractDecisionLeaf)     = 1
nleaves(node::DTInternal) = nleaves(left(node)) + nleaves(right(node))
nleaves(tree::DTree)      = nleaves(root(tree))
nleaves(nsdt::RootLevelNeuroSymbolicHybrid)      = sum(nleaves.(trees(nsdt)))

# Number of nodes
nnodes(leaf::AbstractDecisionLeaf)     = 1
nnodes(node::DTInternal) = 1 + nnodes(left(node)) + nnodes(right(node))
nnodes(tree::DTree)   = nnodes(root(tree))
nnodes(f::DForest) = sum(nnodes.(trees(f)))
nnodes(nsdt::RootLevelNeuroSymbolicHybrid)      = sum(nnodes.(trees(nsdt)))

# Number of trees
ntrees(f::DForest) = length(trees(f))
Base.length(f::DForest)    = ntrees(f)
ntrees(nsdt::RootLevelNeuroSymbolicHybrid) = length(trees(nsdt))
Base.length(nsdt::RootLevelNeuroSymbolicHybrid)    = ntrees(nsdt)

# Height
height(leaf::AbstractDecisionLeaf)     = 0
height(node::DTInternal) = 1 + max(height(left(node)), height(right(node)))
height(tree::DTree)      = height(root(tree))
height(f::DForest)      = maximum(height.(trees(f)))
height(nsdt::RootLevelNeuroSymbolicHybrid)      = maximum(height.(trees(nsdt)))

# Modal height
modalheight(leaf::AbstractDecisionLeaf)     = 0
modalheight(node::DTInternal) = Int(ismodalnode(node)) + max(modalheight(left(node)), modalheight(right(node)))
modalheight(tree::DTree)      = modalheight(root(tree))
modalheight(f::DForest)      = maximum(modalheight.(trees(f)))
modalheight(nsdt::RootLevelNeuroSymbolicHybrid)      = maximum(modalheight.(trees(nsdt)))

# Number of supporting instances
ninstances(leaf::AbstractDecisionLeaf; train_or_valid = true) = length(supp_labels(leaf; train_or_valid = train_or_valid))
ninstances(node::DTInternal;           train_or_valid = true) = ninstances(left(node); train_or_valid = train_or_valid) + ninstances(right(node); train_or_valid = train_or_valid)
ninstances(tree::DTree;                train_or_valid = true) = ninstances(root(tree); train_or_valid = train_or_valid)
ninstances(f::DForest;                 train_or_valid = true) = maximum(map(t->ninstances(t; train_or_valid = train_or_valid), trees(f))) # TODO actually wrong
ninstances(nsdt::RootLevelNeuroSymbolicHybrid;                 train_or_valid = true) = maximum(map(t->ninstances(t; train_or_valid = train_or_valid), trees(nsdt))) # TODO actually wrong

############################################################################################
############################################################################################

isleafnode(leaf::AbstractDecisionLeaf)     = true
isleafnode(node::DTInternal) = false
isleafnode(tree::DTree)      = isleafnode(root(tree))

ismodalnode(node::DTInternal) = (!isleafnode(node) && !is_propositional_decision(decision(node)))
ismodalnode(tree::DTree)      = ismodalnode(root(tree))

############################################################################################
############################################################################################

displaydecision(node::DTInternal, args...; kwargs...) =
    displaydecision(frameid(node), decision(node), args...; kwargs...)
displaydecision_inverse(node::DTInternal, args...; kwargs...) =
    displaydecision_inverse(frameid(node), decision(node), args...; kwargs...)

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
    frameid: $(frameid(node))
    decision: $(decision(node))
    miscellaneous: $(miscellaneous(node))
    ###########################################################
    sub-tree leaves: $(nleaves(node))
    sub-tree nodes: $(nnodes(node))
    sub-tree height: $(height(node))
    sub-tree modal height:  $(modalheight(node))
)
"""
end

function display(tree::DTree{L}) where {L}
    return """
Decision Tree{$(L)}(
    worldtypes:    $(worldtypes(tree))
    init_conditions: $(init_conditions(tree))
    ###########################################################
    sub-tree leaves: $(nleaves(tree))
    sub-tree nodes: $(nnodes(tree))
    sub-tree height: $(height(tree))
    sub-tree modal height:  $(modalheight(tree))
    ###########################################################
    tree:
$(displaymodel(tree))
)
"""
end

function display(forest::DForest{L}) where {L}
    return """
Decision Forest{$(L)}(
    # trees: $(ntrees(forest))
    metrics: $(metrics(forest))
    forest:
$(displaymodel(forest))
)
"""
end


function display(nsdt::RootLevelNeuroSymbolicHybrid{F,L}) where {F,L}
    return """
Root-Level Neuro-Symbolic Decision Tree Hybrid{$(F),$(L)}(
    # trees: $(ntrees(nsdt))
    metrics: $(metrics(nsdt))
    nsdt:
$(displaymodel(nsdt))
)
"""
end


############################################################################################
# Tests
############################################################################################

# # https://stackoverflow.com/questions/66801702/deriving-equality-for-julia-structs-with-mutable-members
# import Base: == # TODO isequal...?
# function (Base).==(a::S, b::S) where {S<:AbstractDecisionLeaf}
#     for name in fieldnames(S)
#         if getfield(a, name) != getfield(b, name)
#             return false
#         end
#     end
#     return true
# end
