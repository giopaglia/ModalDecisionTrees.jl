function benchmark_classification(build::Function, apply::Function)
    println("\nRunning benchmarks ...")
    ########## benchmarks suite ##########
    suite                       = BenchmarkGroup()
    suite["BUILD"]              = BenchmarkGroup()
    suite["BUILD"]["DIGITS"]    = BenchmarkGroup()
    suite["BUILD"]["ADULT"]     = BenchmarkGroup()
    suite["APPLY"]              = BenchmarkGroup()
    suite["APPLY"]["DIGITS"]    = BenchmarkGroup()
    suite["APPLY"]["ADULT"]     = BenchmarkGroup()

    # using DIGITS dataset
    X, Y = load_digits()

    m, n = size(X)
    X_Any = Array{Any}(undef, m, n)
    Y_Any = Array{Any}(undef, m)
    X_Any[:,:] = X
    Y_Any[:]   = Y
    X_Any :: Matrix{Any}
    Y_Any :: Vector{Any}
    model = build(Y_Any, X_Any)
    preds = apply(model, X_Any)
    suite["BUILD"]["DIGITS"][pad("Y::Any X::Any")] = @benchmarkable $build($Y_Any, $X_Any)
    suite["APPLY"]["DIGITS"][pad("Y::Any X::Any")] = @benchmarkable $apply($model, $X_Any)

    X_Any         :: Matrix{Any}
    Y = Int64.(Y) :: Vector{Int64}
    model = build(Y, X_Any)
    preds = apply(model, X_Any)
    suite["BUILD"]["DIGITS"][pad("Y::Int64 X::Any")] = @benchmarkable $build($Y, $X_Any)
    suite["APPLY"]["DIGITS"][pad("Y::Int64 X::Any")] = @benchmarkable $apply($model, $X_Any)

    X = Int64.(X) :: Matrix{Int64}
    Y_Any         :: Vector{Any}
    model = build(Y_Any, X)
    preds = apply(model, X)
    suite["BUILD"]["DIGITS"][pad("Y::Any X::Int64")] = @benchmarkable $build($Y_Any, $X)
    suite["APPLY"]["DIGITS"][pad("Y::Any X::Int64")] = @benchmarkable $apply($model, $X)


    Y = Int32.(Y)
    X = Float32.(X)
    model = build(Y, X)
    preds = apply(model, X)
    suite["BUILD"]["DIGITS"][pad("Y::Int32 X::Float32")] = @benchmarkable $build($Y, $X)
    suite["APPLY"]["DIGITS"][pad("Y::Int32 X::Float32")] = @benchmarkable $apply($model, $X)

    Y = Int64.(Y)
    X = Float64.(X)
    model = build(Y, X)
    preds = apply(model, X)
    suite["BUILD"]["DIGITS"][pad("Y::Int64 X::Float64")] = @benchmarkable $build($Y, $X)
    suite["APPLY"]["DIGITS"][pad("Y::Int64 X::Float64")] = @benchmarkable $apply($model, $X)

    Y = Int64.(Y)
    X = Int64.(X)
    model = build(Y, X)
    preds = apply(model, X)
    suite["BUILD"]["DIGITS"][pad("Y::Int64 X::Int64")] = @benchmarkable $build($Y, $X)
    suite["APPLY"]["DIGITS"][pad("Y::Int64 X::Int64")] = @benchmarkable $apply($model, $X)

    Y = string.(Y) :: Vector{String}
    X = Float64.(X)
    model = build(Y, X)
    preds = apply(model, X)
    suite["BUILD"]["DIGITS"][pad("Y::String X::Float64")] = @benchmarkable $build($Y, $X)
    suite["APPLY"]["DIGITS"][pad("Y::String X::Float64")] = @benchmarkable $apply($model, $X)


    # using ADULT dataset
    X_Any, Y_Any = load_adult()

    Y_Any :: Vector{Any}
    X_Any :: Matrix{Any}
    model = build(Y_Any, X_Any)
    preds = apply(model, X_Any)
    suite["BUILD"]["ADULT"][pad("Y::Any X::Any")] = @benchmarkable $build($Y_Any, $X_Any)
    suite["APPLY"]["ADULT"][pad("Y::Any X::Any")] = @benchmarkable $apply($model, $X_Any)

    Y = String.(Y_Any) :: Vector{String}
    X_Any              :: Matrix{Any}
    model = build(Y, X_Any)
    preds = apply(model, X_Any)
    suite["BUILD"]["ADULT"][pad("Y::String X::Any")] = @benchmarkable $build($Y, $X_Any)
    suite["APPLY"]["ADULT"][pad("Y::String X::Any")] = @benchmarkable $apply($model, $X_Any)

    Y_Any              :: Vector{Any}
    X = string.(X_Any) :: Matrix{AbstractString}
    model = build(Y_Any, X)
    preds = apply(model, X)
    suite["BUILD"]["ADULT"][pad("Y::Any X::AbsString")] = @benchmarkable $build($Y_Any, $X)
    suite["APPLY"]["ADULT"][pad("Y::Any X::AbsString")] = @benchmarkable $apply($model, $X)

    Y = String.(Y) :: Vector{String}
    X = String.(X) :: Matrix{String}
    model = build(Y, X)
    preds = apply(model, X)
    suite["BUILD"]["ADULT"][pad("Y::String X::String")] = @benchmarkable $build($Y, $X)
    suite["APPLY"]["ADULT"][pad("Y::String X::String")] = @benchmarkable $apply($model, $X)


    ########## run suite ##########
    tune!(suite)
    results = run(suite, verbose = true)
    return results
end



function build_adaboost(labels, features)
    n_iterations = 10
    model, coeffs = build_adaboost_stumps(labels, features, n_iterations)
    return model
end
function apply_adaboost(model, features)
    n = length(model)
    return apply_adaboost_stumps(model, ones(n), features)
end
