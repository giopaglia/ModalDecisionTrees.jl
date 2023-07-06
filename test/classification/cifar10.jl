using MLJ
using ModalDecisionTrees
using MLDatasets

Xcube, y = CIFAR10(:test)[:]
y = string.(y)

N = length(y)

p = 1:100
p_test = 101:1000 # N

_s = collect(size(Xcube))
insert!(_s, length(_s), 1)
Xcube = reshape(Xcube, _s...)
X = SoleData.cube2dataframe(Xcube, ["white"])

X_train, y_train = X[p,:], y[p]
X_test, y_test = X[p_test,:], y[p_test]

model = ModalDecisionTree(;
    relations = :IA7,
    conditions = [minimum],
    initconditions = :start_at_center,
    featvaltype = Float32,
    downsize = (x)->ModalDecisionTrees.MLJInterface.moving_average(x, (10,10))
    # conditions = [minimum, maximum, UnivariateFeature{Float64}(recheight), UnivariateFeature{Float64}(recwidth)],
    # conditions = [minimum, maximum, UnivariateFeature{Float32}(1, recheight), UnivariateFeature{Float32}(1, recwidth)],
)

mach = machine(model, X_train, y_train) |> fit!

report(mach).printmodel(1000; threshold_digits = 2);

yhat_test = MLJ.predict_mode(mach, X_test)

MLJ.accuracy(y_test, yhat_test)

@test yhat_test2 == yhat_test

yhat_test2, tree2 = report(mach).printapply(X_test, y_test);

soletree2 = ModalDecisionTrees.translate(tree2)
printmodel(soletree2; show_metrics = true);
printmodel.(listrules(soletree2); show_metrics = true);

SoleModels.info.(listrules(soletree2), :supporting_labels);
leaves = consequent.(listrules(soletree2))
SoleModels.leafmetrics.(leaves)
zip(SoleModels.leafmetrics.(leaves),leaves) |> collect |> sort


@test MLJ.accuracy(y_test, yhat_test) > 0.4
