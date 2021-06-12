# runner.jl

include("runner.jl")
global_logger(ConsoleLogger(stderr, DecisionTree.DTOverview))
# global_logger(ConsoleLogger(stderr, DecisionTree.DTDebug))
# global_logger(ConsoleLogger(stderr, DecisionTree.DTDetail))
X = OntologicalDataset{Int64,1}(Ontology{ModalLogic.Interval}(ModalLogic.IARelations),
	reshape(1:(15*2*10) |> collect,15,2,10))

################################################################################
################################################################################

# Generate features + operators
features_n_operators = Tuple{<:FeatureTypeFun,<:TestOperatorFun}[]

for i_attr in 1:n_attributes(X)
	# push!(features_n_operators, (ModalLogic.AttributeMinimumFeatureType(i_attr), ≥))
	# push!(features_n_operators, (ModalLogic.AttributeMinimumFeatureType(i_attr), ≤)) # "Anormal" feature
	# push!(features_n_operators, (ModalLogic.AttributeMaximumFeatureType(i_attr), ≤))
	# TODO
	push!(features_n_operators, (AttributeSoftMinimumFeatureType(i_attr, .8), ≥))
	push!(features_n_operators, (AttributeSoftMaximumFeatureType(i_attr, .8), ≤))
end

features_n_operators

################################################################################
################################################################################

# Group feats_n_operators by polarity
(features, grouped_featnaggrs, flattened_featnaggrs) = DecisionTree.prepare_featnaggrs(features_n_operators)

################################################################################
################################################################################

computeRelationAll = true
timing_mode = :none
stumpModalDataset = stumpModalDataset(X, features, grouped_featnaggrs, flattened_featnaggrs, computeRelationAll = computeRelationAll, timing_mode = timing_mode);

modalDatasetP = stumpModalDataset.modalDatasetP
modalDatasetM = stumpModalDataset.modalDatasetM
modalDatasetG = stumpModalDataset.modalDatasetG
