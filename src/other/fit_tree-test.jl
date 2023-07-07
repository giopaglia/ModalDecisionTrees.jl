using Revise
using ModalDecisionTrees
using SoleModels.DimensionalDatasets

using Random
using BenchmarkTools
rng = MersenneTwister(1)

_ninstances, _ninstances_h = 10, 5
n_vars = 2
n_feats = n_vars*2
n_pts = 5

using SoleModels
using SoleModels: UnivariateMin, UnivariateMax

features  = []
featsnops = []
for i_var in 1:n_vars
    push!(features, UnivariateMin(i_var))
    push!(featsnops, [≥])
    push!(features, UnivariateMax(i_var))
    push!(featsnops, [≤])
end

Xs = MultiLogiset([
    SupportedLogiset(
        scalarlogiset(randn(n_pts, n_vars, _ninstances), get_interval_ontology(1), features, featsnops);
        precompute_relmemoset = true,
        precompute_globmemoset = true,
    )
]);

W = ModalDecisionTrees.default_weights(_ninstances)


kwargs = (;
loss_function              = ModalDecisionTrees.entropy,
max_depth                  = typemax(Int),
min_samples_leaf           = 4,
min_purity_increase        = -Inf,
max_purity_at_leaf         = 0.2,
n_subrelations             = Function[identity],
n_subfeatures              = Int64[n_feats],
allow_global_splits        = [true],
use_minification = false,
)
perform_consistency_check  = true

# @code_warntype ninstances(Xs)

initconditions              = [ModalDecisionTrees.start_without_world]

################################################################################
# fit
################################################################################

Y  = String[fill("0", _ninstances_h)..., fill("1", _ninstances_h)...]
ModalDecisionTrees.fit_tree(Xs, Y, initconditions, W; perform_consistency_check = perform_consistency_check, kwargs...)
@code_warntype ModalDecisionTrees.fit_tree(Xs, Y, initconditions, W; perform_consistency_check = perform_consistency_check, kwargs...)
@inferred ModalDecisionTrees.fit_tree(Xs, Y, initconditions, W; perform_consistency_check = perform_consistency_check, kwargs...)

Y  = Int64[fill(3, _ninstances_h)..., fill(1, _ninstances_h)...]
ModalDecisionTrees.fit_tree(Xs, Y, initconditions, W; perform_consistency_check = perform_consistency_check, kwargs...)
@code_warntype ModalDecisionTrees.fit_tree(Xs, Y, initconditions, W; perform_consistency_check = perform_consistency_check, kwargs...)
@inferred ModalDecisionTrees.fit_tree(Xs, Y, initconditions, W; perform_consistency_check = perform_consistency_check, kwargs...)

Y  = Float64[fill(0.0, _ninstances_h)..., fill(1.0, _ninstances_h)...]
ModalDecisionTrees.fit_tree(Xs, Y, initconditions, W; perform_consistency_check = perform_consistency_check, kwargs...)
@code_warntype ModalDecisionTrees.fit_tree(Xs, Y, initconditions, W; perform_consistency_check = perform_consistency_check, kwargs...)
@inferred ModalDecisionTrees.fit_tree(Xs, Y, initconditions, W; perform_consistency_check = perform_consistency_check, kwargs...)

################################################################################
# _fit
################################################################################

Y  = Int64[fill(1, _ninstances_h)..., fill(2, _ninstances_h)...]

ModalDecisionTrees._fit_tree(Xs, Y, initconditions, W;
    n_classes = 2,
    _is_classification = Val(true),
    _perform_consistency_check = Val(perform_consistency_check), kwargs...)
@code_warntype ModalDecisionTrees._fit_tree(Xs, Y, initconditions, W;
    n_classes = 2,
    _is_classification = Val(true),
    _perform_consistency_check = Val(perform_consistency_check), kwargs...)


Y  = Float64[fill(0.0, _ninstances_h)..., fill(1.0, _ninstances_h)...]
ModalDecisionTrees._fit_tree(Xs, Y, initconditions, W;
    n_classes = 0,
    _is_classification = Val(false),
    _perform_consistency_check = Val(perform_consistency_check), kwargs...)
@code_warntype ModalDecisionTrees._fit_tree(Xs, Y, initconditions, W;
    n_classes = 0,
    _is_classification = Val(false),
    _perform_consistency_check = Val(perform_consistency_check), kwargs...)

################################################################################
# split_node!
################################################################################

Y  = Int64[fill(1, _ninstances_h)..., fill(2, _ninstances_h)...]

idxs = collect(1:_ninstances)
Ss = ModalDecisionTrees.initialworldsets(Xs, initconditions)

onlyallowglobal = [(initcond == ModalDecisionTrees.start_without_world) for initcond in initconditions]
node = ModalDecisionTrees.NodeMeta{Float64,Int64}(1:_ninstances, 0, 0, onlyallowglobal)


@code_warntype ModalDecisionTrees.split_node!(node, Xs, Ss, Y, initconditions, W;
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
