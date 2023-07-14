using MLJ
using ModalDecisionTrees
using MLDatasets
using SoleData
using SoleModels
using Test

Xcube, y = CIFAR10(:test)[:]
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
y = map(_y-> class_names[_y+1], y)

N = length(y)

n_test = 1000
n_train = 1000
p = 1:n_test
p_test = n_test .+ (1:n_train)

############################################################################################
############################################################################################
############################################################################################

X = SoleData.cube2dataframe(Xcube, ["R", "G", "B"])

X_train, y_train = X[p,:], y[p]
X_test, y_test = X[p_test,:], y[p_test]

model = ModalDecisionTree(;
    relations = :RCC8,
    conditions = [minimum],
    # initconditions = :start_at_center,
    featvaltype = Float32,
    downsize = (10,10), # (x)->ModalDecisionTrees.moving_average(x, (10,10))
    # conditions = [minimum, maximum, UnivariateFeature{Float64}(recheight), UnivariateFeature{Float64}(recwidth)],
    # conditions = [minimum, maximum, UnivariateFeature{Float32}(1, recheight), UnivariateFeature{Float32}(1, recwidth)],
    print_progress = true,
)

mach = machine(model, X_train, y_train) |> fit!

report(mach).printmodel(1000; threshold_digits = 2);

printmodel(report(mach).solemodel; show_metrics = true);
printmodel.(listrules(report(mach).solemodel); show_metrics = true);

yhat_test = MLJ.predict_mode(mach, X_test)

@test MLJ.accuracy(y_test, yhat_test) > 0.15
@test_broken MLJ.accuracy(y_test, yhat_test) > 0.5

yhat_test2, tree2 = report(mach).sprinkle(X_test, y_test);

@test yhat_test2 == yhat_test

soletree2 = ModalDecisionTrees.translate(tree2)
printmodel(soletree2; show_metrics = true);
printmodel.(listrules(soletree2); show_metrics = true);

SoleModels.info.(listrules(soletree2), :supporting_labels);
leaves = consequent.(listrules(soletree2))
SoleModels.readmetrics.(leaves)
zip(SoleModels.readmetrics.(leaves),leaves) |> collect |> sort


@test MLJ.accuracy(y_test, yhat_test) > 0.4

############################################################################################
############################################################################################
############################################################################################

using Images
using ImageFiltering
using StatsBase

Xcube

# img = eachslice(Xcube, dims=4)[1]
Xcubergb = mapslices(c->RGB(c...), Xcube, dims=3)
Xcubehsv = HSV.(Xcubergb)
# Xcubergb = mapslices(c->(@show c), Xcubehsv, dims=3)
Xcubehsv = mapslices(c->[first(c).h, first(c).s, first(c).v], Xcubehsv, dims=3)
# Xcubergb = mapslices(c->[c.h, c.s, c.v], Xcubehsv, dims=[1,2,4])
X = SoleData.cube2dataframe(Xcube, ["H", "S", "V"])

X_train, y_train = X[p,:], y[p]
X_test, y_test = X[p_test,:], y[p_test]

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


model = ModalDecisionTree(;
    relations = :RCC8,
    min_samples_leaf = 8,
    conditions = [svsobel, shsobel],
    # initconditions = :start_at_center,
    initconditions = :start_with_global,
    featvaltype = Float32,
    downsize = (8,8),
    # conditions = [minimum, maximum, UnivariateFeature{Float64}(recheight), UnivariateFeature{Float64}(recwidth)],
    # conditions = [minimum, maximum, UnivariateFeature{Float32}(1, recheight), UnivariateFeature{Float32}(1, recwidth)],
    print_progress = true,
)

mach = machine(model, X_train, y_train) |> fit!

report(mach).printmodel(1000; threshold_digits = 2);

printmodel(report(mach).solemodel; show_metrics = true);
printmodel.(listrules(report(mach).solemodel); show_metrics = true);

yhat_test = MLJ.predict_mode(mach, X_test)

MLJ.accuracy(y_test, yhat_test)

@test MLJ.accuracy(y_test, yhat_test) > 0.15
@test_broken MLJ.accuracy(y_test, yhat_test) > 0.5

model = ModalDecisionTree(;
    relations = :RCC8,
    conditions = [svedge, shedge],
    # initconditions = :start_at_center,
    featvaltype = Float32,
    downsize = (5,5),
    # conditions = [minimum, maximum, UnivariateFeature{Float64}(recheight), UnivariateFeature{Float64}(recwidth)],
    # conditions = [minimum, maximum, UnivariateFeature{Float32}(1, recheight), UnivariateFeature{Float32}(1, recwidth)],
    print_progress = true,
)
