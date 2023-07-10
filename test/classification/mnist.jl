using MLJ
using ModalDecisionTrees
using MLDatasets

trainset = MNIST(:train)

Xcube, y = trainset[:]
y = string.(y)

N = length(y)

p = 1:100
p_test = 101:1000 # N

begin
    X = SoleData.cube2dataframe(Xcube)
    model = ModalDecisionTree()

    mach = machine(model, X_train, y_train) |> fit!

    report(mach).printmodel(1000; threshold_digits = 2);


    yhat_test = MLJ.predict_mode(mach, X_test)

    @test MLJ.accuracy(y_test, yhat_test) > 0.2
end


_s = collect(size(Xcube))
insert!(_s, length(_s), 1)
Xcube = reshape(Xcube, _s...)
X = SoleData.cube2dataframe(Xcube, ["white"])

X_train, y_train = X[p,:], y[p]
X_test, y_test = X[p_test,:], y[p_test]

begin
    model = ModalDecisionTree()

    mach = machine(model, X_train, y_train) |> fit!

    report(mach).printmodel(1000; threshold_digits = 2);


    yhat_test = MLJ.predict_mode(mach, X_test)
    MLJ.accuracy(y_test, yhat_test)
    @test MLJ.accuracy(y_test, yhat_test) > 0.2
end


begin
    model = ModalDecisionTree(; relations = :IA7)

    mach = machine(model, X_train, y_train) |> fit!

    report(mach).printmodel(1000; threshold_digits = 2);


    yhat_test = MLJ.predict_mode(mach, X_test)
    MLJ.accuracy(y_test, yhat_test)
    @test MLJ.accuracy(y_test, yhat_test) > 0.2
end


begin
    recheight(x) = Float32(size(x, 1))
    recwidth(x) = Float32(size(x, 2))
    model = ModalDecisionTree(;
        relations = :IA7,
        conditions = [minimum, maximum, recheight, recwidth],
        featvaltype = Float32,
    )

    mach1 = machine(model, X_train, y_train) |> fit!

    model = ModalDecisionTree(;
        relations = :IA7,
        conditions = [recheight, recwidth, minimum, maximum],
        featvaltype = Float32,
    )

    mach2 = machine(model, X_train, y_train) |> fit!

    report(mach1).printmodel(1000; threshold_digits = 2);
    report(mach2).printmodel(1000; threshold_digits = 2);
    @test fitted_params(mach1).soletree == fitted_params(mach2).soletree


    yhat_test = MLJ.predict_mode(mach, X_test)
    MLJ.accuracy(y_test, yhat_test)
    @test MLJ.accuracy(y_test, yhat_test) > 0.2
end

begin
    recheight(x) = Float32(size(x, 1))
    recwidth(x) = Float32(size(x, 2))
    model = ModalDecisionTree(;
        relations = :IA7,
        conditions = [minimum, maximum, recheight, recwidth],
        initconditions = :start_at_center,
        featvaltype = Float32,
        # conditions = [minimum, maximum, UnivariateFeature{Float64}(recheight), UnivariateFeature{Float64}(recwidth)],
        # conditions = [minimum, maximum, UnivariateFeature{Float32}(1, recheight), UnivariateFeature{Float32}(1, recwidth)],
    )

    mach = machine(model, X_train, y_train) |> fit!

    report(mach).printmodel(1000; threshold_digits = 2);


    yhat_test = MLJ.predict_mode(mach, X_test)
    MLJ.accuracy(y_test, yhat_test)
    @test MLJ.accuracy(y_test, yhat_test) > 0.3
end

begin
    model = ModalDecisionTree(;
        relations = :IA3,
        conditions = [minimum],
        initconditions = :start_at_center,
        featvaltype = Float32,
        # conditions = [minimum, maximum, UnivariateFeature{Float64}(recheight), UnivariateFeature{Float64}(recwidth)],
        # conditions = [minimum, maximum, UnivariateFeature{Float32}(1, recheight), UnivariateFeature{Float32}(1, recwidth)],
    )

    mach = machine(model, X_train, y_train) |> fit!

    report(mach).printmodel(1000; threshold_digits = 2);


    yhat_test = MLJ.predict_mode(mach, X_test)
    MLJ.accuracy(y_test, yhat_test)
    @test MLJ.accuracy(y_test, yhat_test) > 0.3
end


begin
    model = ModalDecisionTree(;
        relations = :IA7,
        conditions = [minimum],
        initconditions = :start_at_center,
        downsize = (x)->ModalDecisionTrees.moving_average(x, (10,10)),
    )

    mach = machine(model, X_train, y_train) |> fit!

    model = ModalDecisionTree(;
        relations = :IA7,
        conditions = [minimum],
        initconditions = :start_at_center,
        downsize = (10,10),
        # conditions = [minimum, maximum, UnivariateFeature{Float64}(recheight), UnivariateFeature{Float64}(recwidth)],
        # conditions = [minimum, maximum, UnivariateFeature{Float32}(1, recheight), UnivariateFeature{Float32}(1, recwidth)],
    )

    mach = machine(model, X_train, y_train) |> fit!

    report(mach).printmodel(1000; threshold_digits = 2);

    yhat_test = MLJ.predict_mode(mach, X_test)

    MLJ.accuracy(y_test, yhat_test)

    @test yhat_test2 == yhat_test

    yhat_test2, tree2 = report(mach).sprinkle(X_test, y_test);

    soletree2 = ModalDecisionTrees.translate(tree2)
    printmodel(soletree2; show_metrics = true);
    printmodel.(listrules(soletree2); show_metrics = true, threshold_digits = 2);
    printmodel.(joinrules(listrules(soletree2)); show_metrics = true, threshold_digits = 2);


    SoleModels.info.(listrules(soletree2), :supporting_labels);
    leaves = consequent.(listrules(soletree2))
    SoleModels.readmetrics.(leaves)
    zip(SoleModels.readmetrics.(leaves),leaves) |> collect |> sort


    @test MLJ.accuracy(y_test, yhat_test) > 0.4
end
