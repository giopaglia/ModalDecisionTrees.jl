using ModalDecisionTrees
using MLJ
using DataFrames
using SoleData
using Logging
using SoleLogics
using Random

N = 5
y = vcat(fill(true, div(N,2)+1), fill(false, div(N,2)))

# Split dataset
p = randperm(Random.MersenneTwister(1), N)
train_idxs, test_idxs = p[1:round(Int, N*.8)], p[round(Int, N*.8)+1:end]


_size = ((x)->(hasmethod(size, (typeof(x),)) ? size(x) : missing))

X_static = DataFrame(
    ID = 1:N,
    a = randn(N),
    b = [-2.0, 1.0, 2.0, missing, 3.0],
    c = [1, 2, 3, 4, 5],
    d = [0, 1, 0, 1, 0],
    e = ['M', 'F', missing, 'M', 'F'],
)
_size.(X_static)

@test_throws AssertionError MLJ.fit!(machine(ModalDecisionTree(;), X_static, y), rows=train_idxs)

X_static = DataFrame(
    ID = 1:N,
    # a = randn(N),
    b = [-2.0, -1.0, 2.0, 2.0, 3.0],
    c = [1, 2, 3, 4, 5],
    d = [0, 1, 0, 1, 0],
)
_size.(X_static)

@test_throws AssertionError MLJ.fit!(machine(ModalDecisionTree(;), X_static, y), rows=train_idxs)
mach = MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 2), Float64.(X_static[:,Not(:ID)]), y), rows=train_idxs)

@test depth(fitted_params(mach).tree) > 0

X_multi1 = DataFrame(
    ID = 1:N,
    t1 = [randn(2), randn(2), randn(2), randn(2), randn(2)], # good
    t2 = [randn(2), randn(2), randn(2), randn(2), randn(2)], # good
)
_size.(X_multi1)

MLJ.fit!(machine(ModalDecisionTree(;), X_multi1, y), rows=train_idxs)

X_multi2 = DataFrame(
    ID = 1:N,
    t3 = [randn(2), randn(2), randn(2), randn(2), randn(2)], # good
    twrong1 = [randn(2), randn(2), randn(5), randn(2), randn(4)], # good but actually TODO
)
_size.(X_multi2)

MLJ.fit!(machine(ModalDecisionTree(;), X_multi2, y), rows=train_idxs)

X_images1 = DataFrame(
    ID = 1:N,
    R1 = [randn(2,2), randn(2,2), randn(2,2), randn(2,2), randn(2,2)], # good
)
_size.(X_images1)

MLJ.fit!(machine(ModalDecisionTree(;), X_images1, y), rows=train_idxs)

X_images1 = DataFrame(
    ID = 1:N,
    R1 = [randn(2,3), randn(2,3), randn(2,3), randn(2,3), randn(2,3)], # good
)
_size.(X_images1)

logiset = scalarlogiset(X_images1[:,Not(:ID)]; use_onestep_memoization=true, conditions = [
    ScalarMetaCondition(UnivariateMax{Float64}(1), ≥),
    ScalarMetaCondition(UnivariateMax{Float64}(1), <),
    ScalarMetaCondition(UnivariateMin{Float64}(1), ≥),
    ScalarMetaCondition(UnivariateMin{Float64}(1), <),
], relations = [globalrel])
ModalDecisionTrees.build_tree(logiset, y)
ModalDecisionTrees.build_tree(logiset, y;
    max_depth           = nothing,
    min_samples_leaf    = ModalDecisionTrees.BOTTOM_MIN_SAMPLES_LEAF,
    min_purity_increase = ModalDecisionTrees.BOTTOM_MIN_PURITY_INCREASE,
    max_purity_at_leaf  = ModalDecisionTrees.BOTTOM_MAX_PURITY_AT_LEAF,
)
ModalDecisionTrees.build_tree(MultiLogiset(logiset), y)

multilogiset, _ = ModalDecisionTrees.wrapdataset(X_images1[:,Not(:ID)], ModalDecisionTree(; min_samples_leaf = 1))

kwargs = (loss_function = nothing, max_depth = nothing, min_samples_leaf = 1, min_purity_increase = 0.002, max_purity_at_leaf = Inf, max_modal_depth = nothing, n_subrelations = identity, n_subfeatures = identity, initconditions = ModalDecisionTrees.StartAtCenter(), allow_global_splits = true, use_minification = false, perform_consistency_check = false, rng = Random.GLOBAL_RNG, print_progress = false)


ModalDecisionTrees.build_tree(multilogiset, y;
    kwargs...
)
MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 1, relations = (d)->[globalrel]), X_images1[:,Not(:ID)], y), rows=train_idxs, verbosity=2)
MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 1, relations = (d)->[globalrel]), X_images1[:,Not(:ID)], y), verbosity=2)
@test_throws CompositeException MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 1, initconditions = :start_at_center, relations = (d)->SoleLogics.AbstractRelation[]), X_images1[:,Not(:ID)], y), verbosity=2)

X_images1 = DataFrame(
    ID = 1:N,
    R1 = [randn(2,3), randn(2,3), randn(2,3), randn(2,3), randn(2,3)], # good
    G1 = [randn(3,3), randn(3,3), randn(3,3), randn(3,3), randn(3,3)], # good
    B1 = [randn(3,3), randn(3,3), randn(3,3), randn(3,3), randn(3,3)], # good
)
_size.(X_images1)

MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 2), X_images1[:,Not(:ID)], y), rows=train_idxs)

X_images2 = DataFrame(
    ID = 1:N,
    R2 = [ones(5,5),  ones(5,5),  ones(5,5),  zeros(5,5), zeros(5,5)], # good
    G2 = [randn(5,5), randn(5,5), randn(5,5), randn(5,5), randn(5,5)], # good
    B2 = [randn(5,5), randn(5,5), randn(5,5), randn(5,5), randn(5,5)], # good
)
_size.(X_images2)

X_all = innerjoin([Float64.(X_static), X_multi1, X_multi2, X_images1, X_images2]... , on = :ID)[:, Not(:ID)]
_size.(X_all)

MLJ.fit!(machine(ModalDecisionTree(;), X_all, y), rows=train_idxs)

X_all = innerjoin([X_multi1, X_images2]... , on = :ID)[:, Not(:ID)]
mach = MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 1), X_all, y), rows=train_idxs)

multilogiset, var_grouping = ModalDecisionTrees.wrapdataset(X_all, ModalDecisionTree(; min_samples_leaf = 1))


ModalDecisionTrees.build_tree(multilogiset, y;
    kwargs...
)

############################################################################################
############################################################################################
############################################################################################

# Multimodal tree:
X_all = DataFrame(
    mode0 = [1.0, 0.0, 0.0, 0.0, 0.0],
    mode1 = [zeros(5), ones(5), zeros(5), zeros(5), zeros(5)],
    mode2 = [zeros(5,5), zeros(5,5), ones(5,5), zeros(5,5), zeros(5,5)],
)
mach = MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 1), X_all, y), rows=train_idxs)

report(mach).printmodel(1000; threshold_digits = 2);

printmodel.(listrules(report(mach).solemodel; use_shortforms=true, use_leftmostlinearform = true))

printmodel.(joinrules(listrules(report(mach).solemodel; use_shortforms=true, use_leftmostlinearform = true)))

model = ModalDecisionTree(min_purity_increase = 0.001)

@test_logs min_level=Logging.Error machine(model, X_multi1, y) |> fit!
@test_logs min_level=Logging.Error machine(model, X_multi2, y) |> fit!
@test_logs min_level=Logging.Error machine(model, X_images1, y) |> fit!
@test_logs min_level=Logging.Error machine(model, X_images2, y) |> fit!
machine(model, X_all, y) |> fit!
# @test_throws AssertionError machine(model, X_all, y) |> fit!

############################################################################################
############################################################################################
############################################################################################

using SoleData

# Multimodal tree:
X_all = DataFrame(
    mode0 = [1.0, 0.0, 0.0, 0.0, 0.0],
    mode1 = [zeros(5), ones(5), zeros(5), zeros(5), zeros(5)],
    mode2 = [zeros(5,5), zeros(5,5), ones(5,5), zeros(5,5), zeros(5,5)],
)

X_all = MultiModalDataset(X_all)
@test_broken mach = MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 1), X_all, y), rows=train_idxs)
