using MLJModels
X, y = make_regression(100, 2)


model = ModalDecisionTree(min_purity_increase = 0.001)

mach = machine(model, X, y)

fit!(mach, rows=1:length(y))
predict(mach, rows=length(y)+1:100)
