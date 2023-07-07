using Random

include("../data/load.jl")

X, y = load_digits()
Xcube = cat(map(r->reshape(r, (8,8,1)), eachrow(X))...; dims=4)
Xnt = NamedTuple(zip(Symbol.(1:length(eachslice(Xcube; dims=3))), eachslice.(eachslice(Xcube; dims=3); dims=3)))
y = string.(y.-1)

N = length(y)
p = randperm(Random.MersenneTwister(1), N)
train_idxs, test_idxs = p[1:round(Int, N*.4)], p[round(Int, N*.4)+1:end]

############################################################################################

# Full training
mach = machine(ModalDecisionTree(;), X, y) |> fit!
@test nnodes(fitted_params(mach).model) == 191
@test sum(predict_mode(mach, X) .== y) / length(y) > 0.92

############################################################################################

mach = machine(ModalDecisionTree(;), X, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).model) == 115
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.78


mach = machine(ModalDecisionTree(;
    n_subfeatures       = 0,
    max_depth           = 6,
    min_samples_leaf    = 5,
), X, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).model) == 77
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.75


mach = machine(ModalDecisionForest(;
    n_subfeatures       = 3,
    ntrees              = 10,
    partial_sampling    = 0.7,
    max_depth           = -1,
    min_samples_leaf    = 1,
    min_samples_split   = 2,
    min_purity_increase = 0.0,
), X, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).model) == 77
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.75


############################################################################################

# NamedTuple dataset
mach = mach = machine(ModalDecisionTree(;), Xnt, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).model) == 137
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.68

mach = machine(ModalDecisionTree(;
    relations = :IA7,
    conditions = [minimum],
    initconditions = :start_at_center,
), Xnt, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).model) == 167
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.67

mach = machine(ModalDecisionTree(;
    relations = :IA7,
    conditions = [minimum, maximum],
    # initconditions = :start_at_center,
    featvaltype = Float32,
), selectrows(Xnt, train_idxs), selectrows(y, train_idxs)) |> m->fit!(m)
@test nnodes(fitted_params(mach).model) == 71
@test sum(predict_mode(mach, selectrows(Xnt, test_idxs)) .== y[test_idxs]) / length(y[test_idxs]) > 0.73


using ImageFiltering
using StatsBase

kernel = [1 0 -1;
          2 0 -2;
          1 0 -1]
im = imfilter(rand(10,10), kernel)
im = imfilter(rand(2,2), kernel)

recvedge(x) = (imfilter(x, [1;; -1]))
rechedge(x) = (imfilter(x, [1;; -1]'))
recvsobel(x) = (imfilter(x, [1 0 -1; 2 0 -2; 1 0 -1]))
rechsobel(x) = (imfilter(x, [1 0 -1; 2 0 -2; 1 0 -1]'))

vedge(x) = StatsBase.mean(recvedge(x)) # prod(size(x)) == 1 ? Inf : StatsBase.mean(recvedge(x))
hedge(x) = StatsBase.mean(rechedge(x)) # prod(size(x)) == 1 ? Inf : StatsBase.mean(rechedge(x))
vsobel(x) = StatsBase.mean(recvsobel(x)) # prod(size(x)) == 1 ? Inf : StatsBase.mean(recvsobel(x))
hsobel(x) = StatsBase.mean(rechsobel(x)) # prod(size(x)) == 1 ? Inf : StatsBase.mean(rechsobel(x))

svedge(x) = StatsBase.sum(recvedge(x)) # prod(size(x)) == 1 ? Inf : StatsBase.sum(recvedge(x))
shedge(x) = StatsBase.sum(rechedge(x)) # prod(size(x)) == 1 ? Inf : StatsBase.sum(rechedge(x))
svsobel(x) = StatsBase.sum(recvsobel(x)) # prod(size(x)) == 1 ? Inf : StatsBase.sum(recvsobel(x))
shsobel(x) = StatsBase.sum(rechsobel(x)) # prod(size(x)) == 1 ? Inf : StatsBase.sum(rechsobel(x))

train_idxs, test_idxs = p[1:round(Int, N*.2)], p[round(Int, N*.2)+1:end]

# train_idxs = train_idxs[1:10]
mach = machine(ModalDecisionTree(;
    relations = :IA7,
    conditions = [hedge, vedge],
    # initconditions = :start_at_center,
    featvaltype = Float32,
), selectrows(Xnt, train_idxs), selectrows(y, train_idxs)) |> m->fit!(m)
@test nnodes(fitted_params(mach).model) == 71
@test sum(predict_mode(mach, selectrows(Xnt, test_idxs)) .== y[test_idxs]) / length(y[test_idxs]) > 0.73

preds, tree2 = report(mach).printapply(selectrows(Xnt, test_idxs), selectrows(y, test_idxs));

@test MLJBase.accuracy(preds, selectrows(y, test_idxs)) > 0.75

# printmodel.(joinrules(listrules(report(mach).solemodel)); show_metrics = true, threshold_digits = 2);
printmodel.(joinrules(listrules(ModalDecisionTrees.translate(tree2))); show_metrics = true, threshold_digits = 2);
readmetrics.(joinrules(listrules(ModalDecisionTrees.translate(tree2))))

# train_idxs = train_idxs[1:10]
mach = machine(ModalDecisionTree(;
    relations = :IA7,
    conditions = [shedge, svedge],
    # initconditions = :start_at_center,
    featvaltype = Float32,
), selectrows(Xnt, train_idxs), selectrows(y, train_idxs)) |> m->fit!(m)
@test nnodes(fitted_params(mach).model) == 79
@test sum(predict_mode(mach, selectrows(Xnt, test_idxs)) .== y[test_idxs]) / length(y[test_idxs]) > 0.73

preds, tree2 = report(mach).printapply(selectrows(Xnt, test_idxs), selectrows(y, test_idxs));

@test MLJBase.accuracy(preds, selectrows(y, test_idxs)) > 0.75

# printmodel.(joinrules(listrules(report(mach).solemodel)); show_metrics = true, threshold_digits = 2);
printmodel.(joinrules(listrules(ModalDecisionTrees.translate(tree2))); show_metrics = true, threshold_digits = 2);
readmetrics.(joinrules(listrules(ModalDecisionTrees.translate(tree2))))

mach = machine(ModalDecisionTree(;), Xnt, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).model) == 137
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.70

mach = machine(ModalDecisionTree(;
    n_subfeatures       = 0,
    max_depth           = 6,
    min_samples_leaf    = 5,
), Xnt, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).model) == 77
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.75


mach = machine(ModalDecisionForest(;
    n_subfeatures       = 3,
    ntrees              = 10,
    partial_sampling    = 0.7,
    max_depth           = -1,
    min_samples_leaf    = 1,
    min_samples_split   = 2,
    min_purity_increase = 0.0,
), Xnt, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).model) == 77
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.75
