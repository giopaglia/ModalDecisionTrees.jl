# __precompile__()

module DecisionTree

import Base: length, show, convert, promote_rule, zero
using DelimitedFiles
using LinearAlgebra
import Random
using Statistics
using StatsBase
using Printf
using Catch22

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
				initWorldSet,
				#
				throw_n_log

function throw_n_log(str::AbstractString, err_type = ErrorException)
	@error str
	throw(err_type(str))
end

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

struct Forest{S}
	trees       :: AbstractVector{<:DTree{S}}
	cm          :: Union{
		AbstractVector{<:ConfusionMatrix},
		AbstractVector{<:PerformanceStruct}, #where {N},
	}
	oob_error   :: AbstractFloat

	function Forest{S}(
		trees       :: AbstractVector{<:DTree{S}},
		cm          :: Union{
			AbstractVector{<:ConfusionMatrix},
			AbstractVector{<:PerformanceStruct}, #where {N},
		},
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
include("main.jl")
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

function print_tree(io::IO, leaf::DTLeaf{String}, depth=-1, indent=0, indent_guides=[]; n_tot_inst = nothing, rel_confidence_class_counts = nothing)
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
		# println(io, cur_class_counts)
		# println(io, rel_confidence_class_counts)
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

	println(io, "$(leaf.majority) : $(n_correct)/$(n_inst) ($(metrics_str))")
end
function print_tree(io::IO, leaf::DTLeaf{<:Float64}, depth=-1, indent=0, indent_guides=[]; n_tot_inst = nothing, rel_confidence_class_counts = nothing)
	
	n_inst = length(leaf.values)
	
	mae = sum(abs.(leaf.values .- leaf.majority)) / n_inst
	rmse = StatsBase.rmsd(leaf.values, [leaf.majority for i in 1:length(leaf.values)])
	var = StatsBase.var(leaf.values)
	
	metrics_str = ""
	# metrics_str *= "$(leaf.values) "
	metrics_str *= "var: $(@sprintf "%.4f" mae)"
	metrics_str *= ", mae: $(@sprintf "%.4f" mae)"
	metrics_str *= ", rmse: $(@sprintf "%.4f" rmse)"
	
	if !isnothing(n_tot_inst)
		support = n_inst/n_tot_inst
		metrics_str *= ", supp = $(@sprintf "%.4f" support)"
	end

	println(io, "$(leaf.majority) : $(n_inst) ($(metrics_str))")
end
print_tree(leaf::DTLeaf, depth=-1, indent=0, indent_guides=[]; kwargs...) = print_tree(stdout, leaf, depth, indent, indent_guides; kwargs...)

function print_tree(io::IO, tree::DTInternal, depth=-1, indent=0, indent_guides=[]; n_tot_inst = nothing, rel_confidence_class_counts = nothing)
	if depth == indent
		println(io, "")
		return
	end

	if !isnothing(rel_confidence_class_counts)
		if !isnothing(n_tot_inst)
			@assert n_tot_inst == sum(values(rel_confidence_class_counts)) "n_tot_inst != sum(values(rel_confidence_class_counts)): $(n_tot_inst) $(sum(values(rel_confidence_class_counts))) sum($(values(rel_confidence_class_counts)))"
		else
			n_tot_inst = sum(values(rel_confidence_class_counts))
		end
	end
	
	println(io, display_decision(tree))
	# indent_str = " " ^ indent
	indent_str = reduce(*, [i == 1 ? "│" : " " for i in indent_guides])
	# print(io, indent_str * "╭✔")
	print(io, indent_str * "✔ ")
	print_tree(io, tree.left, depth, indent + 1, [indent_guides..., 1]; n_tot_inst = n_tot_inst, rel_confidence_class_counts)
	# print(io, indent_str * "╰✘")
	print(io, indent_str * "✘ ")
	print_tree(io, tree.right, depth, indent + 1, [indent_guides..., 0]; n_tot_inst = n_tot_inst, rel_confidence_class_counts)
end
print_tree(tree::DTInternal, depth=-1, indent=0, indent_guides=[]; kwargs...) = print_tree(stdout, tree, depth, indent, indent_guides; kwargs...)

function print_tree(io::IO, tree::DTree; n_tot_inst = nothing, rel_confidence_class_counts = nothing)
	println(io, "worldTypes: $(tree.worldTypes)")
	println(io, "initConditions: $(tree.initConditions)")

	if !isnothing(rel_confidence_class_counts)
		if !isnothing(n_tot_inst)
			@assert n_tot_inst == sum(values(rel_confidence_class_counts)) "n_tot_inst != sum(values(rel_confidence_class_counts)): $(n_tot_inst) $(sum(values(rel_confidence_class_counts))) sum($(values(rel_confidence_class_counts)))"
		else
			n_tot_inst = sum(values(rel_confidence_class_counts))
		end
	end
	
	print_tree(io, tree.root, n_tot_inst = n_tot_inst, rel_confidence_class_counts = rel_confidence_class_counts)
end
print_tree(tree::DTree; kwargs...) = print_tree(stdout, tree; kwargs...)

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


function print_forest(io::IO, forest::Forest)
	n_trees = length(forest)
	for i in 1:n_trees
		println(io, "Tree $(i) / $(n_trees)")
		print_tree(io, forest.trees[i])
	end
end
print_forest(forest::Forest) = print_forest(stdout, forest::Forest)

function show(io::IO, forest::Forest)
	println(io, "Forest")
	println(io, "Num trees: $(length(forest))")
	println(io, "Out-Of-Bag Error: $(forest.oob_error)")
	println(io, "ConfusionMatrix: $(forest.cm)")
	println(io, "Trees:")
	print_forest(io, forest)
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


function print_apply_tree(leaf::DTLeaf{<:Float64}, X::Any, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, class::S; update_majority = false) where {S}
	vals = [leaf.values..., class]

	majority = 
	if update_majority
		StatsBase.mean(leaf.values)
	else
		leaf.majority
	end

	return DTLeaf(majority, vals)
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

function print_apply_tree(io::IO, tree::DTree{S}, X::GenericDataset, Y::Vector{S}; reset_leaves = true, update_majority = false, do_print = true, print_relative_confidence = false) where {S}
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
			print_tree(io, tree; rel_confidence_class_counts = countmap(Y))
		else
			print_tree(io, tree)
		end
		# print(tree)
	end
	return tree
end
print_apply_tree(tree::DTree{S}, X::GenericDataset, Y::Vector{S}; kwargs...) where {S} = print_apply_tree(stdout, tree, X, Y; kwargs...)

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
function apply_trees(trees::AbstractVector{DTree{S}}, X::GenericDataset; suppress_parity_warning = false, tree_weights::Union{AbstractVector{Z},Nothing} = nothing) where {S, Z<:Real}
	@logmsg DTDetail "apply_trees..."
	n_trees = length(trees)
	n_instances = n_samples(X)

	if !isnothing(tree_weights)
		@assert length(trees) === length(tree_weights) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."
	end

	# apply each tree to the whole dataset
	votes = Matrix{S}(undef, n_trees, n_instances)
	for i_tree in 1:n_trees
		votes[i_tree,:] = apply_tree(trees[i_tree], X)
	end

	# for each instance, aggregate the votes
	predictions = Vector{S}(undef, n_instances)
	Threads.@threads for i in 1:n_instances
		predictions[i] = begin
			if S <: Float64
				# @error "apply_trees need a code expansion. The case is S = $(S) <: Float64"
				if isnothing(tree_weights)
					mean(votes[:,i])
				else
					n_trees = length(votes[:,i])
					sum([votes[j,i] * tree_weights[j] for j in 1:n_trees]) / sum(tree_weights)
				end
			else
				best_score(votes[:,i], tree_weights; suppress_parity_warning = suppress_parity_warning)
			end
		end
	end

	return predictions
end

# use a proper forest to test features
function apply_forest(forest::Forest, X::GenericDataset; weight_trees_by::Union{Bool,Symbol,AbstractVector} = false)
	if weight_trees_by == false
		apply_trees(forest.trees, X)
	elseif isa(weight_trees_by, AbstractVector)
		apply_trees(forest.trees, X; tree_weights = weight_trees_by)
	elseif weight_trees_by == :accuracy
		# TODO: choose HOW to weight a tree... overall_accuracy is just an example (maybe can be parameterized)
		apply_trees(forest.trees, X; tree_weights = map(cm -> overall_accuracy(cm), forest.cm))
	else
		@error "Unexpected value for weight_trees_by: $(weight_trees_by)"
	end
end

################################################################################
# Print tree latex
################################################################################

default_conversion_dict_latex = Dict{String, String}(
    "τ" => "\\tau ",
    "⫹" => "\\leq ",
    "⫺" => "\\geq ",
    "⪳" => "\\preceqq ",
    "⪴" => "\\succeqq ",
    "⪵" => "\\precneqq ",
    "⪶" => "\\succneqq ",
    "⟨" => "\\langle ",
    "⟩" => "\\rangle ",
    "A̅" => "\\overline{A}",
    "L̅" => "\\overline{L}",
    "B̅" => "\\overline{B}",
    "E̅" => "\\overline{E}",
    "D̅" => "\\overline{D}",
    "O̅" => "\\overline{O}",
)

const NodeCoord = Tuple{Number,Number}

import Base: +, -

+(coord1::NodeCoord, coord2::NodeCoord)::NodeCoord = (coord1[1] + coord2[1], coord1[2] + coord2[2])
-(coord1::NodeCoord, coord2::NodeCoord)::NodeCoord = (coord1[1] - coord2[1], coord1[2] - coord2[2])

function _attr_to_latex(str::String)::String
	matched = match(r"\bA[0-9]+\b", str)
	if !isnothing(matched)
		str = replace(str, matched.match => "A_{" * replace(matched.match, "A" => "") * "}")
	end
	str
end

function _latex_string(
		obj::Any;
		conversion_dict::Dict{String,String} = default_conversion_dict_latex,
		add_dollars::Bool = true,
		show_test_operator_alpha::Bool = true,
		show_frame_number::Bool = true
	)::String

	subscript_replace = Dict{String,String}(
		"₀" => "0",
		"₁" => "1",
		"₂" => "2",
		"₃" => "3",
		"₄" => "4",
		"₅" => "5",
		"₆" => "6",
		"₇" => "7",
		"₈" => "8",
		"₉" => "9",
		"ₑ" => "e",
		"․" => ".",
		"․" => ".",
		"₋" => "-"
	)

	result = string(obj)
	if !show_test_operator_alpha
		for k in keys(subscript_replace)
			result = replace(result, k => "")
		end
	end

	# WARN: assumption: Global relation is actually Later
	result = replace(result, "G" => "L")

	if show_frame_number
		# Escape {frame}
		result = replace(result, "{" => "\\{", count = 1)
		result = replace(result, "}" => "\\}", count = 1)
	else
		# Remove {frame}
		matched = match(r"^\{[0-9]+\}", result)
		if !isnothing(matched)
			result = replace(result, matched.match => "", count = 1)
		end
	end

	result = _attr_to_latex(result)

	subscript_num_regex = Regex("\\b[" * join(keys(subscript_replace)) * "]+\\b")
	matched = match(subscript_num_regex, result)
	while !isnothing(matched)
		m = matched.match
		for (k, v) in subscript_replace
			m = replace(m, k => v)
		end
		result = replace(result, matched.match => "_{" * m * "}")
		matched = match(subscript_num_regex, result)
	end

    for (k, v) in conversion_dict
        result = replace(result, k => v)
    end

	if add_dollars
        result = "\$" * result * "\$"
		if result == "\$\$"
			result = ""
		end
    end

    result
end

_node_content(leaf::DTLeaf; kwargs...)::String = _latex_string(leaf.majority; kwargs...)
_node_content(node::DTInternal; kwargs...)::String = ""


function _print_tree_latex(
    leaf                      :: DTLeaf,
    previous_node_index       :: String,
    previous_node_position    :: NodeCoord,
    space_unit                :: NodeCoord,
		nodes_margin              :: NodeCoord,
    conversion_dict           :: Dict{String,String},
    add_dollars               :: Bool,
		print_test_operator_alpha :: Bool,
		show_frame_number         :: Bool,
		t_display_func            :: Function,
		nodes_script_size         :: Symbol,
		edges_script_size         :: Symbol
    )::String
    ""
end
function _print_tree_latex(
		node                      :: DTInternal,
		previous_node_index       :: String,
		previous_node_position    :: NodeCoord,
		space_unit                :: NodeCoord,
		nodes_margin              :: NodeCoord,
		conversion_dict           :: Dict{String,String},
		add_dollars               :: Bool,
		print_test_operator_alpha :: Bool,
		show_frame_number         :: Bool,
		t_display_func            :: Function,
		nodes_script_size         :: Symbol,
		edges_script_size         :: Symbol
    )::String

    # use tree height to determine the horizontal-spacing between the nodes
    h = height(node)

    # TODO: calculate proper position
    left_node_pos = previous_node_position + (-abs(space_unit[1])*h, -abs(space_unit[2])) + (-abs(nodes_margin[1]), -abs(nodes_margin[2]))
    right_node_pos = previous_node_position + (abs(space_unit[1])*h, -abs(space_unit[2])) + (abs(nodes_margin[1]), -abs(nodes_margin[2]))

	result = "\\" * string(nodes_script_size) * "\n"
    # add left node
    result *= "\\node ($(previous_node_index)0) at $left_node_pos {$(_node_content(node.left; conversion_dict = conversion_dict, add_dollars = add_dollars))};\n"
    # add right node
    result *= "\\node ($(previous_node_index)1) at $right_node_pos {$(_node_content(node.right; conversion_dict = conversion_dict, add_dollars = add_dollars))};\n"

	result *= "\\" * string(edges_script_size) * "\n"
    # add left edge
    result *= "\\path ($previous_node_index) edge[sloped,above] node {$(_latex_string(display_decision(node; threshold_display_method = t_display_func); conversion_dict = conversion_dict, add_dollars = add_dollars, show_test_operator_alpha = print_test_operator_alpha, show_frame_number = show_frame_number))} ($(previous_node_index)0);\n"
    # add right edge
    result *= "\\path ($previous_node_index) edge[sloped,above] node {$(_latex_string(display_decision_inverse(node; threshold_display_method = t_display_func); conversion_dict = conversion_dict, add_dollars = add_dollars, show_test_operator_alpha = print_test_operator_alpha, show_frame_number = show_frame_number))} ($(previous_node_index)1);\n"
    # recursive calls
    result *= _print_tree_latex(node.left, previous_node_index * "0", left_node_pos, space_unit, nodes_margin, conversion_dict, add_dollars, print_test_operator_alpha, show_frame_number, t_display_func, nodes_script_size, edges_script_size)
    result *= _print_tree_latex(node.right, previous_node_index * "1", right_node_pos, space_unit, nodes_margin, conversion_dict, add_dollars, print_test_operator_alpha, show_frame_number, t_display_func, nodes_script_size, edges_script_size)

    result
end
# :Huge         = \Huge
# :huge         = \huge
# :LARGE        = \LARGE
# :Large        = \Large
# :large        = \large
# :normalsize   = \normalsize
# :small        = \small
# :footnotesize = \footnotesize
# :scriptsize   = \scriptsize
# :tiny         = \tiny
function print_tree_latex(
		tree                               :: DTree;
		tree_name                          :: String                             = "τ",
		conversion_dict                    :: Union{Nothing,Dict{String,String}} = nothing,
		first_node_idx                     :: String                             = "0",
		first_node_position                :: NodeCoord                          = (0, 0),
		space_unit                         :: NodeCoord                          = (0.5, 2.0),
		nodes_margin                       :: NodeCoord                          = (1.8, 0),
		merge_conversion_dict_with_default :: Bool                               = true,
		wrap_in_tikzpicture_block          :: Bool                               = true,
		add_dollars                        :: Bool                               = true,
		print_test_operator_alpha          :: Bool                               = true,
		show_frame_number                  :: Bool                               = true,

		threshold_scale_factor             :: Integer                            = 0,
		threshold_show_decimals            :: Union{Symbol,Integer}              = :all,

		tree_name_script_size              :: Symbol                             = :large,
		nodes_script_size                  :: Symbol                             = :normalsize,
		edges_script_size                  :: Symbol                             = :footnotesize
    )::String

	function threshold_display_func(threshold::Number, scale_factor::Integer, show_decimals::Union{Symbol,Integer})::Number
		result = threshold * (10^scale_factor)
		if isa(show_decimals, Integer)
			result = round(result, digits = show_decimals)
		end
		result
	end

    if merge_conversion_dict_with_default
        if isnothing(conversion_dict)
            conversion_dict = deepcopy(default_conversion_dict_latex)
        else
            merge!(conversion_dict, default_conversion_dict_latex)
        end
    else
        if isnothing(conversion_dict)
            conversion_dict = Dict{String,String}()
        end
    end

	print_tree_comment = replace(string(tree), "\n" => "\n% ")

		result = "\$\$\n"
		result *= "% packages needed: tikz, amssymb, newtxmath\n"
		result *= "% " * tree_name * "\n"
		result *= "% " * print_tree_comment * "\n"
		result *= wrap_in_tikzpicture_block ? "\\begin{tikzpicture}\n" : ""
		result *= "\\" * string(tree_name_script_size) * "\n"
    result *= "\\node ($first_node_idx) at $first_node_position [above] {$(_latex_string(tree_name; conversion_dict = conversion_dict, add_dollars = add_dollars))};\n"
    result *= _print_tree_latex(
		tree.root,
		first_node_idx,
		first_node_position,
		space_unit,
		nodes_margin,
		conversion_dict,
		add_dollars,
		print_test_operator_alpha,
		show_frame_number,
		x -> threshold_display_func(x, threshold_scale_factor, threshold_show_decimals),
		nodes_script_size,
		edges_script_size
	)
    result *= wrap_in_tikzpicture_block ? "\\end{tikzpicture}\n" : ""
	result *= "\$\$\n"

	result
end

export print_tree_latex


##############################################################
##############################################################
##############################################################


struct DecisionPathNode
    taken         :: Bool
    feature       :: ModalLogic.FeatureTypeFun
    test_operator :: TestOperatorFun
    threshold     :: T where T
    worlds        :: AbstractWorldSet
end

const DecisionPath = Vector{DecisionPathNode}

_get_path_in_tree(leaf::DTLeaf, X::Any, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, i_frame::Integer, paths::Vector{DecisionPath})::AbstractWorldSet = return worlds[i_frame]
function _get_path_in_tree(tree::DTInternal, X::MultiFrameModalDataset, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, i_frame::Integer, paths::Vector{DecisionPath})::AbstractWorldSet
    satisfied = true
	(satisfied,new_worlds,worlds_map) =
		ModalLogic.modal_step(
						get_frame(X, tree.i_frame),
						i_instance,
						worlds[tree.i_frame],
						tree.relation,
						tree.feature,
						tree.test_operator,
						tree.threshold,
						Val(true)
					)

	worlds[tree.i_frame] = new_worlds
	survivors = _get_path_in_tree((satisfied ? tree.left : tree.right), X, i_instance, worlds, tree.i_frame, paths)

	# if survivors of next step are in the list of worlds viewed by one
	# of the just accumulated "new_worlds" then that world is a survivor
	# for this step
	new_survivors::AbstractWorldSet = Vector{AbstractWorld}()
	for curr_w in keys(worlds_map)
		if length(intersect(worlds_map[curr_w], survivors)) > 0
			push!(new_survivors, curr_w)
		end
	end

	pushfirst!(paths[i_instance], DecisionPathNode(satisfied, tree.feature, tree.test_operator, tree.threshold, deepcopy(new_survivors)))

	return new_survivors
end
function get_path_in_tree(tree::DTree{S}, X::GenericDataset)::Vector{DecisionPath} where {S}
	n_instances = n_samples(X)
	paths::Vector{DecisionPath} = [ DecisionPath() for i in 1:n_instances ]
	for i_instance in 1:n_instances
		worlds = DecisionTree.inst_init_world_sets(X, tree, i_instance)
		_get_path_in_tree(tree.root, X, i_instance, worlds, 1, paths)
	end
	paths
end

function get_internalnode_dirname(node::DTInternal)::String
    replace(DecisionTree.display_decision(node), " " => "_")
end

mk_tree_path(leaf::DTLeaf; path::String) = touch(path * "/" * string(leaf.majority) * ".txt")
function mk_tree_path(node::DTInternal; path::String)
    dir_name = get_internalnode_dirname(node)
    mkpath(path * "/Y_" * dir_name)
    mkpath(path * "/N_" * dir_name)
    mk_tree_path(node.left; path = path * "/Y_" * dir_name)
    mk_tree_path(node.right; path = path * "/N_" * dir_name)
end
function mk_tree_path(tree_hash::String, tree::DTree; path::String)
    mkpath(path * "/" * tree_hash)
    mk_tree_path(tree.root; path = path * "/" * tree_hash)
end

function get_tree_path_as_dirpath(tree_hash::String, tree::DTree, decpath::DecisionPath; path::String)::String
    current = tree.root
    result = path * "/" * tree_hash
    for node in decpath
        if current isa DTLeaf break end
        result *= "/" * (node.taken ? "Y" : "N") * "_" * get_internalnode_dirname(current)
        current = node.taken ? current.left : current.right
    end
    result
end

export DecisionPathNode, DecisionPath,
			get_path_in_tree, get_internalnode_dirname,
			mk_tree_path, get_tree_path_as_dirpath

end # module
