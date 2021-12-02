Pkg.activate("..")
using Revise
using DecisionTree
using DecisionTree.ModalLogic
using DecisionTree.util
using NPZ
# using FileSystem
# TODO prova variance anziché quello che c'è ora

################################################################################
################################################################################
################################################################################

test_operators = [TestOpGeq_80, TestOpLeq_80]

X, Y = begin

	pw = pwd()
	cd("/home/gio/Desktop/SpatialDecisionTree/NovartisDatathon/")

	include("windowing.jl")

	X_t = npzread("dataset6-train-temp.npy")
	X_s = npzread("dataset6-train-static.npy")

	index_b1, index_b2 = 1, 2
	# pwd(pw)

	windows = windowing(X_t, X_s, (index_b1, index_b2))
	
	X_t, X_s, Y_b1, Y_b2 = windows

	X_s = reshape(X_s, (1, size(X_s)...))

	# [X_t, X_s], Y_b1
	[X_t, X_s], Y_b2
end

# X, Y = begin

# 	pw = pwd()
# 	cd("/home/gio/Desktop/SpatialDecisionTree/NovartisDatathon/")

# 	include("build_matricial_dataset.jl")

# 	pwd(pw)

# 	X_t, X_s, Y_b1, Y_b2 = windows

# 	X_s = reshape(X_s, (1, size(X_s)...))

# 	[X_t, X_s], Y_b2
# end

# X, Y = begin
# 	n_points = 3
# 	n_attrs = 2
# 	n_samps = 10

# 	X = Array{Int64,1+2}(undef, n_points, n_attrs, n_samps)
# 	Y = Vector{Float64}()

# 	for i in 1:n_samps
# 		if i > div(n_samps, 2)
# 			X[:,:,i] = reshape([1,1,1, 2,2,2], (n_points, n_attrs))
# 			push!(Y, abs(randn()))
# 		else
# 			X[:,:,i] = reshape([2,2,2, 2,2,2], (n_points, n_attrs))
# 			push!(Y, -abs(randn()))
# 		end
# 	end
	
# 	[X], Y
# end

################################################################################
################################################################################
################################################################################

X_all = X

Xs = MultiFrameModalDataset([
	begin
		features = FeatureTypeFun[]

		for i_attr in 1:ModalLogic.n_attributes(X)
			for test_operator in test_operators
				if test_operator == TestOpGeq
					push!(features, ModalLogic.AttributeMinimumFeatureType(i_attr))
				elseif test_operator == TestOpLeq
					push!(features, ModalLogic.AttributeMaximumFeatureType(i_attr))
				elseif test_operator isa _TestOpGeqSoft
					push!(features, ModalLogic.AttributeSoftMinimumFeatureType(i_attr, test_operator.alpha))
				elseif test_operator isa _TestOpLeqSoft
					push!(features, ModalLogic.AttributeSoftMaximumFeatureType(i_attr, test_operator.alpha))
				else
					throw_n_log("Unknown test_operator type: $(test_operator), $(typeof(test_operator))")
				end
			end
		end

		featsnops = Vector{<:TestOperatorFun}[
			if any(map(t->isa(feature,t), [AttributeMinimumFeatureType, AttributeSoftMinimumFeatureType]))
				[≥]
			elseif any(map(t->isa(feature,t), [AttributeMaximumFeatureType, AttributeSoftMaximumFeatureType]))
				[≤]
			else
				throw_n_log("Unknown feature type: $(feature), $(typeof(feature))")
				[≥, ≤]
			end for feature in features
		]

	X = OntologicalDataset(X, ModalLogic.getIntervalOntologyOfDim(Val(1)), features, featsnops)
	Xs = ModalLogic.StumpFeatModalDatasetWithMemoization(X, computeRelationGlob = true);

	end for (i_frame, X) in enumerate(X_all)
]);

################################################################################
################################################################################
################################################################################

id_f = x->x
half_f = x->ceil(Int, x/2)
sqrt_f = x->ceil(Int, sqrt(x))

T = build_tree(
	Xs,
	Y,
	nothing;
	##############################################################################
	loss_function       = util.variance,
	max_depth           = typemax(Int),
	min_samples_leaf    = 2,
	min_purity_increase = 0.01,
	max_purity_at_leaf  = 0.4,
	##############################################################################
	n_subrelations      = sqrt_f,
	n_subfeatures       = sqrt_f,
	initConditions      = DecisionTree.startWithRelationGlob,
	useRelationGlob     = true,
	##############################################################################
	perform_consistency_check = true
)

DecisionTree.print_tree(T; n_tot_inst = 10)
print_apply_tree(T, Xs, Y)

preds = apply_tree(T, Xs)
performances = confusion_matrix(Y, preds)


F = build_forest(
	Xs,
	Y;
	##############################################################################
	# Forest logic-agnostic parameters
	n_trees             = 10,
	partial_sampling    = 0.7,      # portion of instances sampled (without replacement) by each tree
	##############################################################################
	# Tree logic-agnostic parameters
	loss_function       = util.entropy,
	max_depth           = typemax(Int),
	min_samples_leaf    = 1,
	min_purity_increase = 0.0,
	max_purity_at_leaf  = -Inf,
	##############################################################################
	# Modal parameters
	n_subrelations      = sqrt_f,
	n_subfeatures       = sqrt_f,
	initConditions      = DecisionTree.startWithRelationGlob,
	useRelationGlob     = true,
	##############################################################################
	perform_consistency_check = true);


preds = apply_forest(F, Xs)
performances = confusion_matrix(Y, preds)
