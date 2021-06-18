# __precompile__()

module DecisionTree

import Base: length, show, convert, promote_rule, zero
using DelimitedFiles
using LinearAlgebra
import Random
using Statistics

using Logging
using Logging: @logmsg
# Log single algorithm overview (e.g. splits performed in decision tree building)
const DTOverview = Logging.LogLevel(-500)
# Log debug info
const DTDebug = Logging.LogLevel(-1000)
# Log more detailed debug info
const DTDetail = Logging.LogLevel(-1500)

# TODO update these
export DTNode, DTLeaf, DTInternal,
				is_leaf, is_modal_node,
				num_nodes, height, modal_height,
				build_stump, build_tree,
				build_forest, apply_forest,
				print_tree, prune_tree, apply_tree, print_forest,
				ConfusionMatrix, confusion_matrix, mean_squared_error, R2, load_data,
				#
				startWithRelationAll, startAtCenter,
				DTOverview, DTDebug, DTDetail,
				#
				GammaType, GammaSliceType, spawn_rng,
				#
				initWorldSet


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

export print_apply_tree

include("ModalLogic/ModalLogic.jl")
using .ModalLogic
include("gammas.jl")
# include("modalDataset.jl")
include("measures.jl")

###########################
########## Types ##########

abstract type _initCondition end
struct _startWithRelationAll  <: _initCondition end; const startWithRelationAll  = _startWithRelationAll();
struct _startAtCenter         <: _initCondition end; const startAtCenter         = _startAtCenter();
struct _startAtWorld{wT<:AbstractWorld} <: _initCondition w::wT end;

initWorldSet(initConditions::AbstractVector{<:_initCondition}, worldTypes::AbstractVector{<:Type{<:AbstractWorld}}, args::Vararg) =
	[initWorldSet(iC, WT, args...) for (iC, WT) in zip(initConditions, worldTypes)]

initWorldSet(initCondition::_startWithRelationAll, ::Type{WorldType}, channel_size::NTuple{N,Integer} where N) where {WorldType<:AbstractWorld} =
	WorldSet{WorldType}([WorldType(ModalLogic.emptyWorld)])

initWorldSet(initCondition::_startAtCenter, ::Type{WorldType}, channel_size::NTuple{N,Integer} where N) where {WorldType<:AbstractWorld} =
	WorldSet{WorldType}([WorldType(ModalLogic.centeredWorld, channel_size...)])

initWorldSet(initCondition::_startAtWorld{WorldType}, ::Type{WorldType}, channel_size::NTuple{N,Integer} where N) where {WorldType<:AbstractWorld} =
	WorldSet{WorldType}([WorldType(initCondition.w)])

# Leaf node, holding the output decision
struct DTLeaf{S} # TODO specify output type: Number, Label, String, Union{Number,Label,String}?
	# Majority class/value (output)
	majority :: S
	# Training support
	values   :: Vector{S}

	function DTLeaf{S}(
		majority :: S,
		values   :: Vector{S},
	) where {S}
		new{S}(majority, values)
	end
	function DTLeaf(
		majority :: S,
		values   :: Vector{S},
	) where {S}
		new{S}(majority, values)
	end
end

# Inner node, holding the output decision
struct DTInternal{T, S}
	# Split label
	modality      :: AbstractRelation
	feature       :: ModalLogic.FeatureTypeFun # TODO move FeatureTypeFun out of ModalLogic?
	test_operator :: TestOperatorFun # Test operator (e.g. <=, ==)
	threshold     :: T
	# Child nodes
	left          :: Union{DTLeaf{S}, DTInternal{T, S}}
	right         :: Union{DTLeaf{S}, DTInternal{T, S}}

	function DTInternal{T, S}(
		modality      :: AbstractRelation,
		feature       :: ModalLogic.FeatureTypeFun,
		test_operator :: TestOperatorFun,
		threshold     :: T,
		left          :: Union{DTLeaf{S}, DTInternal{T, S}},
		right         :: Union{DTLeaf{S}, DTInternal{T, S}},
	) where {T, S}
		new{T, S}(modality, feature, test_operator, threshold, left, right)
	end
	function DTInternal(
		modality      :: AbstractRelation,
		feature       :: ModalLogic.FeatureTypeFun,
		test_operator :: TestOperatorFun,
		threshold     :: T,
		left          :: Union{DTLeaf{S}, DTInternal{T, S}},
		right         :: Union{DTLeaf{S}, DTInternal{T, S}},
	) where {T, S}
		new{T, S}(modality, feature, test_operator, threshold, left, right)
	end
end

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

struct Forest{S}
	trees       :: AbstractVector{<:DTree{S}}
	cm          :: AbstractVector{<:ConfusionMatrix}
	oob_error   :: AbstractFloat
	function Forest{S}(
		trees       :: AbstractVector{<:DTree{S}},
		cm          :: AbstractVector{<:ConfusionMatrix},
		oob_error   :: AbstractFloat,
	) where {S}
	new{S}(trees, cm, oob_error)
	end
	function Forest(
		trees       :: AbstractVector{<:DTree{S}},
		args...,
	) where {S}
	Forest{S}(root, args...)
	end
end

is_leaf(l::DTLeaf) = true
is_leaf(n::DTInternal) = false
is_leaf(t::DTree) = is_leaf(t.root)

is_modal_node(n::DTInternal) = (!is_leaf(n) && n.modality != RelationId)
is_modal_node(t::DTree) = is_modal_node(t.root)

zero(String) = ""

# convert(::Type{DTInternal{T, S}}, lf::DTLeaf{S}) where {S, T} = DTInternal{T, S}(RelationNone, 0, :(nothing), zero(S), lf, DTLeaf(zero(T), [zero(T)]))

promote_rule(::Type{DTInternal{T, S}}, ::Type{DTLeaf{S}}) where {T, S} = DTInternal{T, S}

# make a Random Number Generator object
mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(seed::T) where T <: Integer = Random.MersenneTwister(seed)

# Generate a new rng from a random pick from a given one.
spawn_rng(rng) = Random.MersenneTwister(abs(rand(rng, Int)))

##############################
########## Includes ##########

include("load_data.jl")
include("util.jl")
include("modal-classification/main.jl")
# TODO: include("ModalscikitlearnAPI.jl")

#############################
########## Methods ##########

# Length (total # of nodes)
num_nodes(leaf::DTLeaf) = 1
num_nodes(tree::DTInternal) = 1 + num_nodes(tree.left) + num_nodes(tree.right)
num_nodes(t::DTree) = num_nodes(t.root)

length(leaf::DTLeaf) = 1
length(tree::DTInternal) = length(tree.left) + length(tree.right)
length(t::DTree) = length(t.root)
length(forest::Forest) = length(forest.trees)

# Height
height(leaf::DTLeaf) = 0
height(tree::DTInternal) = 1 + max(height(tree.left), height(tree.right))
height(t::DTree) = height(t.root)

# Modal height
modal_height(leaf::DTLeaf) = 0
modal_height(tree::DTInternal) = (is_modal_node(tree) ? 1 : 0) + max(modal_height(tree.left), modal_height(tree.right))
modal_height(t::DTree) = modal_height(t.root)

function print_tree(leaf::DTLeaf, depth=-1, indent=0, indent_guides=[]; n_tot_inst = false)
	matches = findall(leaf.values .== leaf.majority)

	n_correct =length(matches)
	n_inst = length(leaf.values)

	confidence = n_correct/n_inst
	
	metrics = "conf: $(confidence)"
	
	if n_tot_inst != false
		support = n_inst/n_tot_inst
		metrics *= ", supp = $(support)"
		# lift = ...
		# metrics *= ", lift = $(lift)"
		# conv = ...
		# metrics *= ", conv = $(conv)"
	end

	println("$(leaf.majority) : $(n_correct)/$(n_inst) ($(metrics))") # TODO print purity?
end

function print_tree(tree::DTInternal, depth=-1, indent=0, indent_guides=[]; n_tot_inst = false)
	if depth == indent
		println()
		return
	end

	println(display_modal_decision(tree.modality, tree.test_operator, tree.feature, tree.threshold)) # TODO print purity?
	# indent_str = " " ^ indent
	indent_str = reduce(*, [i == 1 ? "│" : " " for i in indent_guides])
	# print(indent_str * "╭✔")
	print(indent_str * "✔ ")
	print_tree(tree.left, depth, indent + 1, [indent_guides..., 1], n_tot_inst = n_tot_inst)
	# print(indent_str * "╰✘")
	print(indent_str * "✘ ")
	print_tree(tree.right, depth, indent + 1, [indent_guides..., 0], n_tot_inst = n_tot_inst)
end

function print_tree(tree::DTree; n_tot_inst = false)
	println("worldTypes: $(tree.worldTypes)")
	println("initConditions: $(tree.initConditions)")
	print_tree(tree.root, n_tot_inst = n_tot_inst)
end

function show(io::IO, leaf::DTLeaf)
	println(io, "Decision Leaf")
	println(io, "Majority: $(leaf.majority)")
	println(io, "Samples:  $(length(leaf.values))")
	print_tree(leaf)
end

function show(io::IO, tree::DTInternal)
	println(io, "Decision Node")
	println(io, "Leaves: $(length(tree))")
	println(io, "Tot nodes: $(num_nodes(tree))")
	println(io, "Height: $(height(tree))")
	println(io, "Modal height:  $(modal_height(tree))")
	print_tree(tree)
end

function show(io::IO, tree::DTree)
	println(io, "Decision Tree")
	println(io, "Leaves: $(length(tree))")
	println(io, "Tot nodes: $(num_nodes(tree))")
	println(io, "Height: $(height(tree))")
	println(io, "Modal height:  $(modal_height(tree))")
	print_tree(tree)
end


function print_forest(forest::Forest)
	n_trees = length(forest)
	for i in 1:n_trees
		println("Tree $(i) / $(n_trees)")
		print_tree(forest.trees[i])
	end
end

function show(io::IO, forest::Forest)
	println(io, "Forest")
	println(io, "Num trees: $(length(forest))")
	println(io, "Out-Of-Bag Error: $(forest.oob_error)")
	println(io, "ConfusionMatrix: $(forest.cm)")
	println(io, "Trees:")
	print_forest(forest)
end

end # module
