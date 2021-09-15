# __precompile__()

module DecisionTree

import Base: length, show, convert, promote_rule, zero
using DelimitedFiles
using LinearAlgebra
import Random
using Statistics
using StatsBase
using Printf

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
				DTree, Forest,
				is_leaf, is_modal_node,
				num_nodes, height, modal_height,
				build_stump, build_tree,
				build_forest, apply_forest, apply_trees,
				apply_model, print_model, print_apply_model,
				print_tree, prune_tree, apply_tree, print_forest,
				print_apply_tree,
				tree_walk_metrics,
				ConfusionMatrix, confusion_matrix, mean_squared_error, R2, load_data,
				#
				startWithRelationGlob, startAtCenter,
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

include("ModalLogic/ModalLogic.jl")
using .ModalLogic

import .ModalLogic: n_samples

include("gammas.jl")
# include("modalDataset.jl")
include("measures.jl")

###########################
########## Types ##########

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
		DTLeaf{S}(majority, values)
	end
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

	function DTInternal{T, S}(
		i_frame       :: Integer,
		relation      :: AbstractRelation,
		feature       :: ModalLogic.FeatureTypeFun,
		test_operator :: TestOperatorFun,
		threshold     :: T,
		left          :: Union{DTLeaf{S}, DTInternal{T, S}},
		right         :: Union{DTLeaf{S}, DTInternal{T, S}},
	) where {T, S}
		new{T, S}(i_frame, relation, feature, test_operator, threshold, left, right)
	end
	function DTInternal(
		i_frame       :: Integer,
		relation      :: AbstractRelation,
		feature       :: ModalLogic.FeatureTypeFun,
		test_operator :: TestOperatorFun,
		threshold     :: T,
		left          :: Union{DTLeaf{S}, DTInternal{T, S}},
		right         :: Union{DTLeaf{S}, DTInternal{T, S}},
	) where {T, S}
		DTInternal{T, S}(i_frame, relation, feature, test_operator, threshold, left, right)
	end
end

display_decision(tree::DTInternal) = ModalLogic.display_decision(tree.i_frame, tree.relation, tree.feature, tree.test_operator, tree.threshold)

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

is_modal_node(n::DTInternal) = (!is_leaf(n) && n.relation != RelationId)
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
num_nodes(f::Forest) = sum(num_nodes.(f.trees))

length(leaf::DTLeaf) = 1
length(tree::DTInternal) = length(tree.left) + length(tree.right)
length(t::DTree) = length(t.root)
length(f::Forest) = length(f.trees)

# Height
height(leaf::DTLeaf) = 0
height(tree::DTInternal) = 1 + max(height(tree.left), height(tree.right))
height(t::DTree) = height(t.root)

# Modal height
modal_height(leaf::DTLeaf) = 0
modal_height(tree::DTInternal) = (is_modal_node(tree) ? 1 : 0) + max(modal_height(tree.left), modal_height(tree.right))
modal_height(t::DTree) = modal_height(t.root)

print_model(tree::DTree;    kwargs...) = print_tree(tree;     kwargs...)
print_model(forest::Forest; kwargs...) = print_forest(forest; kwargs...)

apply_model(tree::DTree,    args...; kwargs...) = apply_tree(tree,     args...; kwargs...)
apply_model(forest::Forest, args...; kwargs...) = apply_forest(forest, args...; kwargs...)

print_apply_model(tree::DTree, args...; kwargs...) = print_apply_tree(tree, args...; kwargs...)

function print_tree(leaf::DTLeaf, depth=-1, indent=0, indent_guides=[]; n_tot_inst = nothing, rel_confidence_class_counts = nothing)
	matches = findall(leaf.values .== leaf.majority)

	n_correct = length(matches)
	n_inst = length(leaf.values)
	confidence = n_correct/n_inst

	metrics_str = "conf: $(@sprintf "%.4f" confidence)"
	
	if !isnothing(rel_confidence_class_counts)
		if !isnothing(n_tot_inst)
			@assert n_tot_inst == sum(values(rel_confidence_class_counts)) "n_tot_inst != sum(values(rel_confidence_class_counts)): $(n_tot_inst) $(sum(values(rel_confidence_class_counts))) sum($(values(rel_confidence_class_counts)))"
		else
			n_tot_inst = sum(values(rel_confidence_class_counts))
		end
	end


	if !isnothing(rel_confidence_class_counts)
		cur_class_counts = countmap(leaf.values)
		# println(cur_class_counts)
		# println(rel_confidence_class_counts)
		rel_tot_inst = sum([(haskey(cur_class_counts, class) ? cur_class_counts[class] : 0)/(haskey(rel_confidence_class_counts, class) ? rel_confidence_class_counts[class] : 0) for class in keys(rel_confidence_class_counts)])
		# "rel_conf: $(n_correct/rel_confidence_class_counts[leaf.majority])"
		class = leaf.majority

		if !isnothing(n_tot_inst)
			class_support = (haskey(rel_confidence_class_counts, class) ? rel_confidence_class_counts[class] : 0)/n_tot_inst
			lift = confidence/class_support
			metrics_str *= ", lift: $(@sprintf "%.2f" lift)"
		end
		rel_conf = ((haskey(cur_class_counts, class) ? cur_class_counts[class] : 0)/(haskey(rel_confidence_class_counts, class) ? rel_confidence_class_counts[class] : 0))/rel_tot_inst
		metrics_str *= ", rel_conf: $(@sprintf "%.4f" rel_conf)"
	end

	if !isnothing(n_tot_inst)
		support = n_inst/n_tot_inst
		metrics_str *= ", supp = $(@sprintf "%.4f" support)"
	end

	if !isnothing(rel_confidence_class_counts) && !isnothing(n_tot_inst)
		conv = (1-class_support)/(1-confidence)
		metrics_str *= ", conv: $(@sprintf "%.4f" conv)"
	end

	println("$(leaf.majority) : $(n_correct)/$(n_inst) ($(metrics_str))")
end

function print_tree(tree::DTInternal, depth=-1, indent=0, indent_guides=[]; n_tot_inst = nothing, rel_confidence_class_counts = nothing)
	if depth == indent
		println()
		return
	end

	if !isnothing(rel_confidence_class_counts)
		if !isnothing(n_tot_inst)
			@assert n_tot_inst == sum(values(rel_confidence_class_counts)) "n_tot_inst != sum(values(rel_confidence_class_counts)): $(n_tot_inst) $(sum(values(rel_confidence_class_counts))) sum($(values(rel_confidence_class_counts)))"
		else
			n_tot_inst = sum(values(rel_confidence_class_counts))
		end
	end
	
	println(display_decision(tree))
	# indent_str = " " ^ indent
	indent_str = reduce(*, [i == 1 ? "│" : " " for i in indent_guides])
	# print(indent_str * "╭✔")
	print(indent_str * "✔ ")
	print_tree(tree.left, depth, indent + 1, [indent_guides..., 1]; n_tot_inst = n_tot_inst, rel_confidence_class_counts)
	# print(indent_str * "╰✘")
	print(indent_str * "✘ ")
	print_tree(tree.right, depth, indent + 1, [indent_guides..., 0]; n_tot_inst = n_tot_inst, rel_confidence_class_counts)
end

function print_tree(tree::DTree; n_tot_inst = nothing, rel_confidence_class_counts = nothing)
	println("worldTypes: $(tree.worldTypes)")
	println("initConditions: $(tree.initConditions)")

	if !isnothing(rel_confidence_class_counts)
		if !isnothing(n_tot_inst)
			@assert n_tot_inst == sum(values(rel_confidence_class_counts)) "n_tot_inst != sum(values(rel_confidence_class_counts)): $(n_tot_inst) $(sum(values(rel_confidence_class_counts))) sum($(values(rel_confidence_class_counts)))"
		else
			n_tot_inst = sum(values(rel_confidence_class_counts))
		end
	end
	
	print_tree(tree.root, n_tot_inst = n_tot_inst, rel_confidence_class_counts = rel_confidence_class_counts)
end

function show(io::IO, leaf::DTLeaf)
	println(io, "Decision Leaf")
	println(io, "Majority: $(leaf.majority)")
	println(io, "Samples:  $(length(leaf.values))")
	print_tree(leaf)
end

function show(io::IO, tree::DTInternal)
	println(io, "Decision Node")
	println(io, display_decision(tree))
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


# TODO fix this using specified purity
function prune_tree(tree::DTNode, max_purity_threshold::AbstractFloat = 1.0)
	if max_purity_threshold >= 1.0
		return tree
	end
	# Prune the tree once TODO make more efficient (avoid copying so many nodes.)
	function _prune_run(tree::DTNode)
		N = length(tree)
		if N == 1        ## a DTLeaf
			return tree
		elseif N == 2    ## a stump
			all_labels = [tree.left.values; tree.right.values]
			majority = majority_vote(all_labels)
			matches = findall(all_labels .== majority)
			purity = length(matches) / length(all_labels)
			if purity >= max_purity_threshold
				return DTLeaf(majority, all_labels)
			else
				return tree
			end
		else
			# TODO also associate an Internal node with values and majority (all_labels, majority)
			return DTInternal(tree.i_frame, tree.relation, tree.feature, tree.test_operator, tree.threshold,
						_prune_run(tree.left),
						_prune_run(tree.right))
		end
	end

	# Keep pruning until "convergence"
	pruned = _prune_run(tree)
	while true
		length(pruned) < length(tree) || break
		pruned = _prune_run(tree)
		tree = pruned
	end
	return pruned
end

function prune_tree(tree::DTree, max_purity_threshold::AbstractFloat = 1.0)
	DTree(prune_tree(tree.root), tree.worldTypes, tree.initConditions)
end


################################################################################
# Apply tree: predict labels for a new dataset of instances
################################################################################

# TODO avoid these fallbacks?
inst_init_world_sets(X::SingleFrameGenericDataset, tree::DTree, i_instance::Integer) = 
	inst_init_world_sets(MultiFrameModalDataset(X), tree, i_instance)
print_apply_tree(tree::DTree{S}, X::SingleFrameGenericDataset, Y::Vector{S}; reset_leaves = true, update_majority = false) where {S} = 
	print_apply_tree(tree, MultiFrameModalDataset(X), Y; reset_leaves = reset_leaves, update_majority = update_majority)

inst_init_world_sets(Xs::MultiFrameModalDataset, tree::DTree, i_instance::Integer) = begin
	Ss = Vector{WorldSet}(undef, n_frames(Xs))
	for (i_frame,X) in enumerate(ModalLogic.frames(Xs))
		Ss[i_frame] = initws_function(X, i_instance)(tree.initConditions[i_frame])
	end
	Ss
end
 
apply_tree(leaf::DTLeaf, X::Any, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}) = leaf.majority

function apply_tree(tree::DTInternal, X::MultiFrameModalDataset, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet})
	@logmsg DTDetail "applying branch..."
	satisfied = true
	@logmsg DTDetail " worlds" worlds
	(satisfied,new_worlds) =
		ModalLogic.modal_step(
						get_frame(X, tree.i_frame),
						i_instance,
						worlds[tree.i_frame],
						tree.relation,
						tree.feature,
						tree.test_operator,
						tree.threshold)

	worlds[tree.i_frame] = new_worlds
	@logmsg DTDetail " ->(satisfied,worlds')" satisfied worlds
	apply_tree((satisfied ? tree.left : tree.right), X, i_instance, worlds)
end

# Apply tree with initialConditions to a dimensional dataset in matricial form
function apply_tree(tree::DTree{S}, X::GenericDataset) where {S}
	@logmsg DTDetail "apply_tree..."
	n_instances = n_samples(X)
	predictions = Vector{S}(undef, n_instances)
	
	for i_instance in 1:n_instances
		@logmsg DTDetail " instance $i_instance/$n_instances"
		# TODO figure out: is it better to interpret the whole dataset at once, or instance-by-instance? The first one enables reusing training code

		worlds = inst_init_world_sets(X, tree, i_instance)

		predictions[i_instance] = apply_tree(tree.root, X, i_instance, worlds)
	end
	predictions
	# return (if S <: Float64 # TODO remove
	# 		Float64.(predictions)
	# 	else
	# 		predictions
	# 	end)
end

# Apply tree to a dimensional dataset in matricial form
# function apply_tree(tree::DTNode, d::MatricialDataset{T,D}) where {T, D}
# 	apply_tree(DTree(tree, [world_type(ModalLogic.getIntervalOntologyOfDim(Val(D-2)))], [startWithRelationGlob]), d)
# end

################################################################################
# Apply tree: predict labels for a new dataset of instances
################################################################################

function _empty_tree_leaves(leaf::DTLeaf{S}) where {S}
		DTLeaf(leaf.majority, S[])
end

function _empty_tree_leaves(tree::DTInternal)
	return DTInternal(
		tree.i_frame,
		tree.relation,
		tree.feature,
		tree.test_operator,
		tree.threshold,
		_empty_tree_leaves(tree.left),
		_empty_tree_leaves(tree.right)
	)
end

function _empty_tree_leaves(tree::DTree)
	return DTree(
		_empty_tree_leaves(tree.root),
		tree.worldTypes,
		tree.initConditions
	)
end

function print_apply_tree(leaf::DTLeaf{S}, X::Any, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, class::S; update_majority = false) where {S}
	vals = S[ leaf.values..., class ] # Note: this works when leaves are reset

	majority = 
	if update_majority

		# TODO optimize this code
		occur = Dict{S,Int}(v => 0 for v in unique(vals))
		for v in vals
			occur[v] += 1
		end
		cur_maj = vals[1]
		cur_max = occur[vals[1]]
		for v in vals
			if occur[v] > cur_max
				cur_max = occur[v]
				cur_maj = v
			end
		end
		cur_maj
	else
		leaf.majority
	end

	return DTLeaf(majority, vals)
end

function print_apply_tree(tree::DTInternal{T, S}, X::MultiFrameModalDataset, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, class::S; update_majority = false) where {T, S}
	
	(satisfied,new_worlds) = ModalLogic.modal_step(get_frame(X, tree.i_frame), i_instance, worlds[tree.i_frame], tree.relation, tree.feature, tree.test_operator, tree.threshold)
	
	# if satisfied
	# 	println("new_worlds: $(new_worlds)")
	# end

	worlds[tree.i_frame] = new_worlds

	DTInternal(
		tree.i_frame,
		tree.relation,
		tree.feature,
		tree.test_operator,
		tree.threshold,
		  satisfied  ? print_apply_tree(tree.left,  X, i_instance, worlds, class, update_majority = update_majority) : tree.left,
		(!satisfied) ? print_apply_tree(tree.right, X, i_instance, worlds, class, update_majority = update_majority) : tree.right,
	)
end

function print_apply_tree(tree::DTree{S}, X::GenericDataset, Y::Vector{S}; reset_leaves = true, update_majority = false, do_print = true, print_relative_confidence = false) where {S}
	# Reset 
	tree = (reset_leaves ? _empty_tree_leaves(tree) : tree)

	# Propagate instances down the tree
	for i_instance in 1:n_samples(X)

		worlds = inst_init_world_sets(X, tree, i_instance)

		tree = DTree(
			print_apply_tree(tree.root, X, i_instance, worlds, Y[i_instance], update_majority = update_majority),
			tree.worldTypes,
			tree.initConditions
		)
	end
	if do_print
		if print_relative_confidence
			print_tree(tree; rel_confidence_class_counts = countmap(Y))
		else
			print_tree(tree)
		end
		# print(tree)
	end
	return tree
end

# function print_apply_tree(tree::DTNode{T, S}, X::MatricialDataset{T,D}, Y::Vector{S}; reset_leaves = true, update_majority = false) where {S, T, D}
# 	return print_apply_tree(DTree(tree, [world_type(ModalLogic.getIntervalOntologyOfDim(Val(D-2)))], [startWithRelationGlob]), X, Y, reset_leaves = reset_leaves, update_majority = update_majority)
# end


n_samples(leaf::DTLeaf) = length(leaf.values)
n_samples(tree::DTInternal) = n_samples(tree.left) + n_samples(tree.right)
n_samples(tree::DTree) = n_samples(tree.root)

function tree_walk_metrics(leaf::DTLeaf; n_tot_inst = nothing, best_rule_params = [])
	if isnothing(n_tot_inst)
		n_tot_inst = n_samples(leaf)
	end
	
	matches = findall(leaf.values .== leaf.majority)

	n_correct = length(matches)
	n_inst = length(leaf.values)

	metrics = Dict()
	confidence = n_correct/n_inst
	
	metrics["n_instances"] = n_inst
	metrics["n_correct"] = n_correct
	metrics["avg_confidence"] = confidence
	metrics["best_confidence"] = confidence
	
	if !isnothing(n_tot_inst)
		support = n_inst/n_tot_inst
		metrics["avg_support"] = support
		metrics["support"] = support
		metrics["best_support"] = support

		for best_rule_p in best_rule_params
			if (haskey(best_rule_p, :min_confidence) && best_rule_p.min_confidence > metrics["best_confidence"]) ||
				(haskey(best_rule_p, :min_support) && best_rule_p.min_support > metrics["best_support"])
				metrics["best_rule_t=$(best_rule_p)"] = -Inf
			else
				metrics["best_rule_t=$(best_rule_p)"] = metrics["best_confidence"] * best_rule_p.t + metrics["best_support"] * (1-best_rule_p.t)
			end
		end
	end


	metrics
end

function tree_walk_metrics(tree::DTInternal; n_tot_inst = nothing, best_rule_params = [])
	if isnothing(n_tot_inst)
		n_tot_inst = n_samples(tree)
	end
	metrics_l = tree_walk_metrics(tree.left;  n_tot_inst = n_tot_inst, best_rule_params = best_rule_params)
	metrics_r = tree_walk_metrics(tree.right; n_tot_inst = n_tot_inst, best_rule_params = best_rule_params)

	metrics = Dict()

	# Number of instances passing through the node
	metrics["n_instances"] =
		metrics_l["n_instances"] + metrics_r["n_instances"]

	# Number of correct instances passing through the node
	metrics["n_correct"] =
		metrics_l["n_correct"] + metrics_r["n_correct"]
	
	# Average confidence of the subtree
	metrics["avg_confidence"] =
		(metrics_l["n_instances"] * metrics_l["avg_confidence"] +
		metrics_r["n_instances"] * metrics_r["avg_confidence"]) /
			(metrics_l["n_instances"] + metrics_r["n_instances"])
	
	# Average support of the subtree (Note to self: weird...?)
	metrics["avg_support"] =
		(metrics_l["n_instances"] * metrics_l["avg_support"] +
		metrics_r["n_instances"] * metrics_r["avg_support"]) /
			(metrics_l["n_instances"] + metrics_r["n_instances"])
	
	# Best confidence of the best-confidence path passing through the node
	metrics["best_confidence"] = max(metrics_l["best_confidence"], metrics_r["best_confidence"])
	
	# Support of the current node
	if !isnothing(n_tot_inst)
		metrics["support"] = (metrics_l["n_instances"] + metrics_r["n_instances"])/n_tot_inst
	
		# Best support of the best-support path passing through the node
		metrics["best_support"] = max(metrics_l["best_support"], metrics_r["best_support"])
		
		# Best rule (confidence and support) passing through the node
		for best_rule_p in best_rule_params
			metrics["best_rule_t=$(best_rule_p)"] = max(metrics_l["best_rule_t=$(best_rule_p)"], metrics_r["best_rule_t=$(best_rule_p)"])
		end
	end

	metrics
end

tree_walk_metrics(tree::DTree; kwargs...) = tree_walk_metrics(tree.root; kwargs...)


#=
TODO

# Returns a dict ("Label1" => 1, "Label2" => 2, "Label3" => 3, ...)
label_index(labels::AbstractVector{Label}) = Dict(v => k for (k, v) in enumerate(labels))

## Helper function. Counts the votes.
## Returns a vector of probabilities (eg. [0.2, 0.6, 0.2]) which is in the same
## order as get_labels(classifier) (eg. ["versicolor", "setosa", "virginica"])
function compute_probabilities(labels::AbstractVector{Label}, votes::AbstractVector{Label}, weights=1.0)
	label2ind = label_index(labels)
	counts = zeros(Float64, length(label2ind))
	for (i, label) in enumerate(votes)
		if isa(weights, Real)
			counts[label2ind[label]] += weights
		else
			counts[label2ind[label]] += weights[i]
		end
	end
	return counts / sum(counts) # normalize to get probabilities
end

# Applies `row_fun(X_row)::AbstractVector` to each row in X
# and returns a matrix containing the resulting vectors, stacked vertically
function stack_function_results(row_fun::Function, X::AbstractMatrix)
	N = size(X, 1)
	N_cols = length(row_fun(X[1, :])) # gets the number of columns
	out = Array{Float64}(undef, N, N_cols)
	for i in 1:N
		out[i, :] = row_fun(X[i, :])
	end
	return out
end

"""    apply_tree_proba(::Node, features, col_labels::AbstractVector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
apply_tree_proba(leaf::DTLeaf, features::AbstractVector, labels) where =
	compute_probabilities(labels, leaf.values)

function apply_tree_proba(tree::DTInternal{S, T}, features::AbstractVector{S}, labels) where {S, T}
	if tree.threshold === nothing
		return apply_tree_proba(tree.left, features, labels)
	elseif eval(Expr(:call, tree.test_operator, tree.feature ... , tree.threshold))
		return apply_tree_proba(tree.left, features, labels)
	else
		return apply_tree_proba(tree.right, features, labels)
	end
end

apply_tree_proba(tree::DTNode, features::AbstractMatrix{S}, labels) =
	stack_function_results(row->apply_tree_proba(tree, row, labels), features)

=#

# use an array of trees to test features
function apply_trees(trees::AbstractVector{DTree{S}}, X::GenericDataset; tree_weights::Union{AbstractVector{Z},Nothing} = nothing) where {S, Z<:Real}
	@logmsg DTDetail "apply_trees..."
	n_trees = length(trees)
	n_instances = n_samples(X)

	if !isnothing(tree_weights)
		@assert length(trees) === length(tree_weights) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."
	end

	votes = Matrix{S}(undef, n_trees, n_instances)
	for i_tree in 1:n_trees
		votes[i_tree,:] = apply_tree(trees[i_tree], X)
	end

	predictions = Vector{S}(undef, n_instances)
	Threads.@threads for i in 1:n_instances
		if S <: Float64
			@error "apply_trees need a code expansion. The case is S = $(S) <: Float64"
			if isnothing(tree_weights)
				predictions[i] = mean(votes[:,i])
			else
				weighted_votes = Vector{N}()
				for j in 1:length(votes[:,i])
					weighted_votes = votes[j,i] * tree_weights[j]
				end
				predictions[i] = mean(weighted_votes)
			end
		else
			predictions[i] = best_score(votes[:,i], tree_weights)
		end
	end

	return predictions
end

# use a proper forest to test features
function apply_forest(forest::Forest, X::GenericDataset; weight_trees_by::Bool = false)
	if weight_trees_by == false
		apply_trees(forest.trees, X)
	elseif weight_trees_by == :accuracy
		# TODO: choose HOW to weight a tree... overall_accuracy is just an example (maybe can be parameterized)
		apply_trees(forest.trees, X, tree_weights = map(cm -> overall_accuracy(cm), forest.cm))
	else
		@error "Unexpected value for weight_trees_by: $(weight_trees_by)"
	end
end


end # module
