using ModalDecisionTrees
using MLJ
using DataFrames
using SoleData
using Random

N = 5
y = vcat(fill(true, div(N,2)+1), fill(false, div(N,2)))

# Split dataset
p = randperm(Random.MersenneTwister(1), N)
train_idxs, test_idxs = p[1:round(Int, N*.8)], p[round(Int, N*.8)+1:end]


_size = ((x)->(hasmethod(size, (typeof(x),)) ? size(x) : missing))

# Very multimodal tree:
N = 100
X_all = DataFrame(
    mode0 = [rand() for i in 1:N],
    mode1 = [rand(5) for i in 1:N],
    mode2 = [rand(2,2) for i in 1:N],
)
y = string.(rand(1:10, N))
mach = MLJ.fit!(machine(ModalDecisionTree(; n_subfeatures = 0.2, min_samples_leaf = 1), X_all, y))

report(mach).printmodel(1000; threshold_digits = 2);

redundant_ruleset = (listrules(report(mach).solemodel; use_shortforms=false))
succinct_ruleset = (listrules(report(mach).solemodel; use_shortforms=true))

multilogiset, var_grouping = ModalDecisionTrees.wrapdataset(X_all, ModalDecisionTree(; min_samples_leaf = 1))

redundant_y = [SoleModels.apply(r, multilogiset) for r in redundant_ruleset]
succinct_y = [SoleModels.apply(r, multilogiset) for r in succinct_ruleset]

_redundant_y = [map(r->SoleModels.apply(r, multilogiset, i_instance), redundant_ruleset) for i_instance in 1:ninstances(multilogiset)]
_succinct_y = [map(r->SoleModels.apply(r, multilogiset, i_instance), succinct_ruleset) for i_instance in 1:ninstances(multilogiset)]

preds = string.(predict_mode(mach, X_all))

@test_broken redundant_y == succinct_y
@test_broken _redundant_y == _succinct_y
@test eachcol(hcat(_redundant_y...)) == eachrow(hcat(redundant_y...))
@test eachcol(hcat(_succinct_y...)) == eachrow(hcat(succinct_y...))

m1 = hcat(redundant_y...)
m2 = hcat(succinct_y...)

@test_broken map(r->count(!isnothing, r) == 1, eachrow(m1))
@test_broken map(r->count(!isnothing, r) == 1, eachrow(m2))

@test_broken map(r->count(!isnothing, r) == 1, eachcol(m1))
@test_broken map(r->count(!isnothing, r) == 1, eachcol(m2))

preds
filter.(!isnothing, eachrow(hcat(redundant_y...)))
filter.(!isnothing, eachcol(hcat(redundant_y...)))
filter.(!isnothing, eachrow(hcat(succinct_y...)))
filter.(!isnothing, eachcol(hcat(succinct_y...)))

@test_broken preds2 = filter.(!isnothing, eachrow(hcat(succinct_y...)))
@test_broken preds2 = first.(preds2)
@test_broken preds == preds2

printmodel.(listrules(report(mach).solemodel; use_shortforms=true));
printmodel.(listrules(report(mach).solemodel; use_shortforms=true, use_leftmostlinearform = true));
printmodel.(joinrules(redundant_ruleset); show_metrics = true);
printmodel.(joinrules(succinct_ruleset); show_metrics = true);

printmodel.(joinrules(listrules(report(mach).solemodel)));


@test_nowarn printmodel.(listrules(report(mach).solemodel; use_shortforms=true, use_leftmostlinearform = true))
@test_nowarn printmodel.(joinrules(listrules(report(mach).solemodel; use_shortforms=true, use_leftmostlinearform = true)))
