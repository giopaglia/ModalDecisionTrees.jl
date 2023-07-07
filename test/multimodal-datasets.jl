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
    R1 = [randn(2,3), randn(2,3), randn(2,3), randn(2,3), randn(2,3)], # good
    G1 = [randn(3,3), randn(3,3), randn(3,3), randn(3,3), randn(3,3)], # good
    B1 = [randn(3,3), randn(3,3), randn(3,3), randn(3,3), randn(3,3)], # good
)
_size.(X_images1)

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

X_all = innerjoin([X_multi1, X_images1, X_images2]... , on = :ID)[:, Not(:ID)]
MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 2), X_all, y), rows=train_idxs)

model = ModalDecisionTree(min_purity_increase = 0.001)

@test_logs (:info,) machine(model, X_multi1, y) |> fit!
@test_logs (:info,) machine(model, X_multi2, y) |> fit!
@test_logs (:info,) machine(model, X_images1, y) |> fit!
@test_logs (:info,) machine(model, X_images2, y) |> fit!
@test_throws AssertionError machine(model, X_all, y) |> fit!
