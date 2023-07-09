using SoleModels: AbstractLogiset
import SoleModels: printmodel, displaymodel
import SoleModels: ninstances, height, nnodes


############################################################################################
# Initial conditions
############################################################################################

using SoleLogics: AbstractMultiModalFrame
using SoleLogics: AbstractSyntaxStructure

abstract type InitialCondition end

struct StartWithoutWorld               <: InitialCondition end;
const start_without_world  = StartWithoutWorld();

struct StartAtCenter                   <: InitialCondition end;
const start_at_center      = StartAtCenter();

struct StartAtWorld{W<:AbstractWorld}  <: InitialCondition w::W end;

function initialworldset(fr::AbstractMultiModalFrame{W}, initcond::StartWithoutWorld) where {W<:AbstractWorld}
    WorldSet{W}([SoleLogics.emptyworld(fr)])
end

function initialworldset(fr::AbstractMultiModalFrame{W}, initcond::StartAtCenter) where {W<:AbstractWorld}
    WorldSet{W}([SoleLogics.centralworld(fr)])
end

function initialworldset(::AbstractMultiModalFrame{W}, initcond::StartAtWorld{W}) where {W<:AbstractWorld}
    WorldSet{W}([initcond.w])
end

anchor(φ::AbstractSyntaxStructure, ::StartWithoutWorld) = φ
anchor(φ::AbstractSyntaxStructure, ::StartAtCenter) = DiamondRelationalOperator(SoleLogics.tocenterrel)(φ)
anchor(φ::AbstractSyntaxStructure, cm::StartAtWorld) = error("TODO expand code")

function initialworldset(
    X,
    i_instance::Integer,
    args...
)
    initialworldset(frame(X, i_instance), args...)
end

function initialworldsets(Xs::MultiLogiset, initconds::AbstractVector{<:InitialCondition})
    Ss = Vector{Vector{WST} where {W,WST<:WorldSet{W}}}(undef, nmodalities(Xs)) # Fix
    for (i_modality,X) in enumerate(eachmodality(Xs))
        W = worldtype(X)
        Ss[i_modality] = WorldSet{W}[initialworldset(X, i_instance, initconds[i_modality]) for i_instance in 1:ninstances(Xs)]
        # Ss[i_modality] = WorldSet{W}[[Interval(1,2)] for i_instance in 1:ninstances(Xs)]
    end
    Ss
end

############################################################################################

"""
A decision is an object that is placed at an internal decision node,
and influences on how the instances are routed to its left or right child.
"""
abstract type AbstractDecision end

"""
Abstract type for nodes in a decision tree.
"""
abstract type AbstractNode{L<:Label} end

predictiontype(::AbstractNode{L}) where {L} = L

"""
Abstract type for leaves in a decision tree.
"""
abstract type AbstractDecisionLeaf{L<:Label} <: AbstractNode{L} end

"""
Abstract type for internal decision nodes of a decision tree.
"""
abstract type AbstractDecisionInternal{L<:Label,D<:AbstractDecision} <: AbstractNode{L} end

"""
Union type for internal and decision nodes of a decision tree.
"""
const DTNode{L<:Label,D<:AbstractDecision} = Union{<:AbstractDecisionLeaf{<:L},<:AbstractDecisionInternal{L,D}}

isleftchild(node::DTNode, parent::AbstractDecisionInternal) = (left(parent) == node)
isrightchild(node::DTNode, parent::AbstractDecisionInternal) = (right(parent) == node)

isinleftsubtree(node::DTNode, parent::AbstractDecisionInternal) = isleftchild(node, parent) || isinsubtree(node, left(parent))
isinrightsubtree(node::DTNode, parent::AbstractDecisionInternal) = isrightchild(node, parent) || isinsubtree(node, right(parent))
isinsubtree(node::DTNode, parent::DTNode) = (node == parent) || (isinleftsubtree(node, parent) || isinrightsubtree(node, parent))

isleftchild(node::DTNode, parent::AbstractDecisionLeaf) = false
isrightchild(node::DTNode, parent::AbstractDecisionLeaf) = false
isinleftsubtree(node::DTNode, parent::AbstractDecisionLeaf) = false
isinrightsubtree(node::DTNode, parent::AbstractDecisionLeaf) = false

############################################################################################
# Decisions
############################################################################################
using SoleLogics
using SoleLogics: identityrel, globalrel
using SoleModels.DimensionalDatasets: alpha
using SoleModels: AbstractTemplatedFormula,
                  ScalarOneStepFormula,
                  ScalarExistentialFormula,
                  ScalarUniversalFormula


function displaydecision(
    i_modality::ModalityId,
    decision::AbstractDecision;
    variable_names_map::Union{Nothing,AbstractVector{<:AbstractVector},AbstractVector{<:AbstractDict}} = nothing,
    kwargs...,
)
    _variable_names_map = isnothing(variable_names_map) ? nothing : variable_names_map[i_modality]
    "{$i_modality} $(displaydecision(decision; variable_names_map = _variable_names_map, kwargs...))"
end

function displaydecision_inverse(decision::AbstractDecision, kwargs...; args...)
    syntaxstring(negation(decision), kwargs...; args...)
end

function displaydecision_inverse(i_modality::ModalityId, decision::AbstractDecision, kwargs...; args...)
    displaydecision(i_modality, negation(decision), kwargs...; args...)
end


is_propositional_decision(d::ScalarOneStepFormula) = (SoleModels.relation(d) == identityrel)
is_global_decision(d::ScalarOneStepFormula) = (SoleModels.relation(d) == globalrel)

import SoleModels: relation, proposition, metacond, feature, test_operator, threshold

struct SimpleDecision{F<:ScalarExistentialFormula} <: AbstractDecision
    formula  :: F
end

formula(d::SimpleDecision) = d.formula

relation(d::SimpleDecision) = relation(formula(d))
proposition(d::SimpleDecision) = proposition(formula(d))
metacond(d::SimpleDecision) = metacond(formula(d))
feature(d::SimpleDecision) = feature(formula(d))
test_operator(d::SimpleDecision) = test_operator(formula(d))
threshold(d::SimpleDecision) = threshold(formula(d))

is_propositional_decision(d::SimpleDecision) = is_propositional_decision(formula(d))
is_global_decision(d::SimpleDecision) = is_global_decision(formula(d))

function displaydecision(d::SimpleDecision; kwargs...)
    outstr = ""
    outstr *= "SimpleDecision("
    outstr *= syntaxstring(formula(d); kwargs...)
    outstr *= ")"
    outstr
end

function SimpleDecision(
    d::SimpleDecision{<:ScalarExistentialFormula},
    threshold_backmap::Function
)
    f = formula(d)
    p = proposition(f)
    cond = atom(p)
    newcond = ScalarCondition(metacond(cond), threshold_backmap(threshold(cond)))
    SimpleDecision(ScalarExistentialFormula(relation(f), newcond))
end

mutable struct DoubleEdgedDecision{F<:AbstractTemplatedFormula} <: AbstractDecision
    formula   :: F
    _back     :: Base.RefValue{N} where N<:AbstractNode # {L,DoubleEdgedDecision}
    _forth    :: Base.RefValue{N} where N<:AbstractNode # {L,DoubleEdgedDecision}

    function DoubleEdgedDecision{F}(formula::F) where {F<:AbstractTemplatedFormula}
        ded = new{F}()
        ded.formula = formula
        ded
    end

    function DoubleEdgedDecision(formula::F) where {F<:AbstractTemplatedFormula}
        DoubleEdgedDecision{F}(formula)
    end
end

formula(ded::DoubleEdgedDecision) = ded.formula
back(ded::DoubleEdgedDecision) = isdefined(ded, :_back) ? ded._back[] : nothing
forth(ded::DoubleEdgedDecision) = isdefined(ded, :_forth) ? ded._forth[] : nothing
_back(ded::DoubleEdgedDecision) = isdefined(ded, :_back) ? ded._back : nothing
_forth(ded::DoubleEdgedDecision) = isdefined(ded, :_forth) ? ded._forth : nothing

formula!(ded::DoubleEdgedDecision, formula) = (ded.formula = formula)
_back!(ded::DoubleEdgedDecision, _back) = (ded._back = _back)
_forth!(ded::DoubleEdgedDecision, _forth) = (ded._forth = _forth)


# TODO remove?
is_propositional_decision(ded::DoubleEdgedDecision) = is_propositional_decision(formula(ded))
is_global_decision(ded::DoubleEdgedDecision) = is_global_decision(formula(ded))

function displaydecision(ded::DoubleEdgedDecision; kwargs...)
    outstr = ""
    outstr *= "DoubleEdgedDecision("
    outstr *= syntaxstring(formula(ded); kwargs...)
    outstr *= ", " * (isnothing(_back(ded)) ? "-" : "$(typeof(_back(ded)))")
    outstr *= ", " * (isnothing(_forth(ded)) ? "-" : "$(typeof(_forth(ded)))")
    outstr *= ")"
    # outstr *= "DoubleEdgedDecision(\n\t"
    # outstr *= syntaxstring(formula(ded))
    # # outstr *= "\n\tback: " * (isnothing(back(ded)) ? "-" : displaymodel(back(ded), args...; kwargs...))
    # # outstr *= "\n\tforth: " * (isnothing(forth(ded)) ? "-" : displaymodel(forth(ded), args...; kwargs...))
    # outstr *= "\n\tback: " * (isnothing(_back(ded)) ? "-" : "$(typeof(_back(ded)))")
    # outstr *= "\n\tforth: " * (isnothing(_forth(ded)) ? "-" : "$(typeof(_forth(ded)))")
    # outstr *= "\n)"
    outstr
end

function DoubleEdgedDecision(
    d::DoubleEdgedDecision,
    threshold_backmap::Function
)
    return error("TODO implement")
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

# Internal decision node, holding a split-decision and a modality index
struct DTInternal{L<:Label,D<:AbstractDecision} <: AbstractDecisionInternal{L,D}
    # modality index + split-decision
    i_modality    :: ModalityId
    decision      :: D
    # representative leaf for the current node
    this          :: AbstractDecisionLeaf{<:L}
    # child nodes
    left          :: Union{AbstractDecisionLeaf{<:L}, DTInternal{<:L,D}}
    right         :: Union{AbstractDecisionLeaf{<:L}, DTInternal{<:L,D}}

    # semantics-specific miscellanoeus info
    miscellaneous :: NamedTuple

    # create node
    function DTInternal{L,D}(
        i_modality       :: ModalityId,
        decision         :: D,
        this             :: AbstractDecisionLeaf,
        left             :: Union{AbstractDecisionLeaf,DTInternal},
        right            :: Union{AbstractDecisionLeaf,DTInternal},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:AbstractDecision,L<:Label}
        new{L,D}(i_modality, decision, this, left, right, miscellaneous)
    end
    function DTInternal{L}(
        i_modality       :: ModalityId,
        decision         :: D,
        this             :: AbstractDecisionLeaf{<:L},
        left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{<:L,D}},
        right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{<:L,D}},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:AbstractDecision,L<:Label}
        DTInternal{L,D}(i_modality, decision, this, left, right, miscellaneous)
    end
    function DTInternal(
        i_modality       :: ModalityId,
        decision         :: D,
        this             :: AbstractDecisionLeaf{L0},
        left             :: Union{AbstractDecisionLeaf{L1}, DTInternal{L1,D}},
        right            :: Union{AbstractDecisionLeaf{L2}, DTInternal{L2,D}},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:AbstractDecision,L0<:Label,L1<:Label,L2<:Label}
        L = Union{L0,L1,L2}
        DTInternal{L,D}(i_modality, decision, this, left, right, miscellaneous)
    end

    # create node without local leaf
    function DTInternal{L,D}(
        i_modality       :: ModalityId,
        decision         :: D,
        left             :: Union{AbstractDecisionLeaf,DTInternal},
        right            :: Union{AbstractDecisionLeaf,DTInternal},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:Union{AbstractDecision,ScalarExistentialFormula},L<:Label}
        if decision isa ScalarExistentialFormula
            decision = SimpleDecision(decision)
        end
        this = squashtoleaf(Union{<:AbstractDecisionLeaf,<:DTInternal}[left, right])
        DTInternal{L,D}(i_modality, decision, this, left, right, miscellaneous)
    end
    function DTInternal{L}(
        i_modality       :: ModalityId,
        decision         :: D,
        left             :: Union{AbstractDecisionLeaf{<:L}, DTInternal{<:L,D}},
        right            :: Union{AbstractDecisionLeaf{<:L}, DTInternal{<:L,D}},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:AbstractDecision,L<:Label}
        DTInternal{L,D}(i_modality, decision, left, right, miscellaneous)
    end
    function DTInternal(
        i_modality       :: ModalityId,
        decision         :: D,
        left             :: Union{AbstractDecisionLeaf{L1}, DTInternal{L1,D}},
        right            :: Union{AbstractDecisionLeaf{L2}, DTInternal{L2,D}},
        miscellaneous    :: NamedTuple = (;),
    ) where {D<:Union{AbstractDecision,ScalarExistentialFormula},L1<:Label,L2<:Label}
        if decision isa ScalarExistentialFormula
            decision = SimpleDecision(decision)
        end
        L = Union{L1,L2}
        DTInternal{L,D}(i_modality, decision, left, right, miscellaneous)
    end
end

i_modality(node::DTInternal) = node.i_modality
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
    root           :: DTNode{L}
    # world types (one per modality)
    worldtypes     :: Vector{<:Type}
    # initial world conditions (one per modality)
    initconditions :: Vector{InitialCondition}

    function DTree{L}(
        root           :: DTNode,
        worldtypes     :: AbstractVector{<:Type},
        initconditions :: AbstractVector{<:InitialCondition},
    ) where {L<:Label}
        new{L}(root, collect(worldtypes), Vector{InitialCondition}(collect(initconditions)))
    end

    function DTree(
        root           :: DTNode{L,D},
        worldtypes     :: AbstractVector{<:Type},
        initconditions :: AbstractVector{<:InitialCondition},
    ) where {L<:Label,D<:AbstractDecision}
        DTree{L}(root, worldtypes, initconditions)
    end
end

root(tree::DTree) = tree.root
worldtypes(tree::DTree) = tree.worldtypes
initconditions(tree::DTree) = tree.initconditions

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
    displaydecision(i_modality(node), decision(node), args...; kwargs...)

displaydecision_inverse(node::DTInternal, args...; kwargs...) =
    displaydecision_inverse(i_modality(node), decision(node), args...; kwargs...)

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
    i_modality: $(i_modality(node))
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
    initconditions: $(initconditions(tree))
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
