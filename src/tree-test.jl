using DecisionTree
using DecisionTree.ModalLogic
using ReTest

using Random
using BenchmarkTools
using StructuredArrays
rng = MersenneTwister(1)

n_insts, n_insts_h = 10, 5
n_attrs = 2
n_feats = n_attrs*2
n_pts = 5

features  = FeatureTypeFun[]
featsnops = Vector{TestOperatorFun}[]
for i_attr in 1:n_attrs
    push!(features, AttributeMinimumFeatureType(i_attr))
    push!(featsnops, [≥])
    push!(features, AttributeMaximumFeatureType(i_attr))
    push!(featsnops, [≤])
end

Xs = MultiFrameModalDataset([
    StumpFeatModalDataset(
        OntologicalDataset(randn(n_pts, n_attrs, n_insts), ModalLogic.getIntervalOntologyOfDim(Val(1)), features, featsnops),
        computeRelationGlob = true,
    )
]);

W = UniformVector{Int}(1,n_insts)


kwargs = (;
loss_function              = DecisionTree.util.entropy,
max_depth                  = typemax(Int),
min_samples_leaf           = 4,
min_purity_increase        = -Inf,
max_purity_at_leaf         = 0.2,
n_subrelations             = Function[identity],
n_subfeatures              = Int64[n_feats],
allowRelationGlob          = [true],
)
perform_consistency_check  = true

# @code_warntype n_samples(Xs)

initConditions             = [startWithRelationGlob]

################################################################################
# fit
################################################################################

Y  = String[fill("0", n_insts_h)..., fill("1", n_insts_h)...]
DecisionTree.fit(Xs, Y, initConditions; perform_consistency_check = perform_consistency_check, kwargs...)
@code_warntype DecisionTree.fit(Xs, Y, initConditions; perform_consistency_check = perform_consistency_check, kwargs...)
@inferred DecisionTree.fit(Xs, Y, initConditions; perform_consistency_check = perform_consistency_check, kwargs...)

Y  = Int64[fill(3, n_insts_h)..., fill(1, n_insts_h)...]
DecisionTree.fit(Xs, Y, initConditions; perform_consistency_check = perform_consistency_check, kwargs...)
@code_warntype DecisionTree.fit(Xs, Y, initConditions; perform_consistency_check = perform_consistency_check, kwargs...)
@inferred DecisionTree.fit(Xs, Y, initConditions; perform_consistency_check = perform_consistency_check, kwargs...)

Y  = Float64[fill(0.0, n_insts_h)..., fill(1.0, n_insts_h)...]
DecisionTree.fit(Xs, Y, initConditions; perform_consistency_check = perform_consistency_check, kwargs...)
@code_warntype DecisionTree.fit(Xs, Y, initConditions; perform_consistency_check = perform_consistency_check, kwargs...)
@inferred DecisionTree.fit(Xs, Y, initConditions; perform_consistency_check = perform_consistency_check, kwargs...)

################################################################################
# _fit
################################################################################

Y  = Int64[fill(1, n_insts_h)..., fill(2, n_insts_h)...]

DecisionTree._fit(Xs, Y, initConditions, W;
    n_classes = 2,
    _is_classification = Val(true),
    _perform_consistency_check = Val(perform_consistency_check), kwargs...)
@code_warntype DecisionTree._fit(Xs, Y, initConditions, W;
    n_classes = 2,
    _is_classification = Val(true),
    _perform_consistency_check = Val(perform_consistency_check), kwargs...)


Y  = Float64[fill(0.0, n_insts_h)..., fill(1.0, n_insts_h)...]
DecisionTree._fit(Xs, Y, initConditions, W;
    n_classes = 0,
    _is_classification = Val(false),
    _perform_consistency_check = Val(perform_consistency_check), kwargs...)
@code_warntype DecisionTree._fit(Xs, Y, initConditions, W;
    n_classes = 0,
    _is_classification = Val(false),
    _perform_consistency_check = Val(perform_consistency_check), kwargs...)

################################################################################
# split_node!
################################################################################

Y  = Int64[fill(1, n_insts_h)..., fill(2, n_insts_h)...]

idxs = collect(1:n_insts)
Ss = DecisionTree.init_world_sets(Xs, initConditions)

onlyallowRelationGlob = [(iC == startWithRelationGlob) for iC in initConditions]
node = DecisionTree.NodeMeta{Float64,Int64}(1:n_insts, 0, 0, onlyallowRelationGlob)


@code_warntype DecisionTree.split_node!(node, Xs, Ss, Y, initConditions, W;
    idxs                       = idxs,
    rng                        = rng,
    n_classes = 2,
    _is_classification         = Val(true),
    _perform_consistency_check = Val(perform_consistency_check),
    kwargs...,
)

# https://docs.julialang.org/en/v1/manual/performance-tips/#Be-aware-of-when-Julia-avoids-specializing
# @which f(...)).specializations

# TODO regression case
