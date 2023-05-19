using ModalDecisionTrees
using MLJ
using DataFrames

N = 5

relations = :IA
mixed_features = [minimum, maximum]

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

X_series1 = DataFrame(
    ID = 1:N,
    t1 = [randn(2), randn(2), randn(2), randn(2), randn(2)], # good
    t2 = [randn(2), randn(2), randn(2), randn(2), randn(2)], # good
)
_size.(X_series1)

X_series2 = DataFrame(
    ID = 1:N,
    t3 = [randn(2), randn(2), randn(2), randn(2), randn(2)], # good
    twrong1 = [randn(2), randn(2), randn(5), randn(2), randn(4)], # good but actually TODO
)
_size.(X_series2)

X_images1 = DataFrame(
    ID = 1:N,
    R1 = [randn(2,3), randn(2,3), randn(2,3), randn(2,3), randn(2,3)], # good
    G1 = [randn(3,3), randn(3,3), randn(3,3), randn(3,3), randn(3,3)], # good
    B1 = [randn(3,3), randn(3,3), randn(3,3), randn(3,3), randn(3,3)], # good
)
_size.(X_images1)

X_images2 = DataFrame(
    ID = 1:N,
    R2 = [randn(5,5), randn(5,5), randn(5,5), randn(5,5), randn(5,5)], # good
    G2 = [randn(5,5), randn(5,5), randn(5,5), randn(5,5), randn(5,5)], # good
    B2 = [randn(5,5), randn(5,5), randn(5,5), randn(5,5), randn(5,5)], # good
)
_size.(X_images2)

X_all = innerjoin([
    X_static,
    X_series1,
    X_series2,
    X_images1,
    X_images2]...
, on = :ID)
X_all = X_all[:, Not(:ID)]

_size.(X_all)


y = rand(["A", "B"], 5)

model = ModalDecisionTree(min_purity_increase = 0.001)

@test_throws AssertionError machine(model, X_static, y) |> fit!
@test_logs (:info,) machine(model, X_series1, y) |> fit!
@test_logs (:info,) machine(model, X_series2, y) |> fit!
@test_logs (:info,) machine(model, X_images1, y) |> fit!
@test_logs (:info,) machine(model, X_images2, y) |> fit!
@test_throws AssertionError machine(model, X_all, y) |> fit!
