@testset "demo-juliacon2022.jl" begin

# Import ModalDecisionTrees.jl & MLJ
using ModalDecisionTrees
using ModalDecisionTrees: ConfusionMatrix
using MLJ

include("demo-juliacon2022-utils.jl");

################################################################################

# Obtain dataset (official) split
dataset_train, dataset_test = load_arff_dataset("NATOPS");

# Unpack split
X_train, Y_train = dataset_train;
X_test,  Y_test  = dataset_test;

# X_train[1,:"Elbow left, X coordinate"] = begin x = X_train[1,:"Elbow left, X coordinate"]; x[1] = NaN; x end

# X_train[:,:"Elbow left, X coordinatex"] = [
# begin x = Vector{Union{Float64,Missing}}(y); x[1] = missing; x end
# for y in X_train[:,:"Elbow left, X coordinate"]]

# names(X_train[:,[end]])

# X_train = moving_average.(X_train, 10, 10)
X_train = ((x)->x[1:3]).(X_train)
# X_test = moving_average.(X_test, 10, 10)
X_test = ((x)->x[1:3]).(X_test)

# X_train[:,:new] = [randn(2,2) for i in 1:nrow(X_train)]
# X_test[:, :new] = [randn(2,2) for i in 1:nrow(X_test)]
# X_train = X_train[:,[1,end]]
# X_test  = X_test[:,[1, end]]

w = abs.(randn(nrow(X_train)))

# Instantiate model with standard pruning conditions
model = ModalDecisionTree()
# model = ModalDecisionTree(; relations = :RCC8)


################################################################################

# Train model & ring a bell :D
@time mach = machine(model, X_train, Y_train, w) |> fit!
# run(`paplay /usr/share/sounds/freedesktop/stereo/complete.oga`);

# Print model
mach.report.printmodel()

# Test on the hold-out set &
#  inspect the distribution of test instances across the leaves
Y_test_preds = MLJ.predict(mach, X_test);

# predict(args...) = MLJ.predict(ModalDecisionTree(), args...);
# Y_test_preds, test_tree = MLJ.predict(mach, X_test, Y_test);

# Inspect confusion matrix
cm = ConfusionMatrix(Y_test, Y_test_preds; force_class_order=["I have command", "All clear", "Not clear", "Spread wings", "Fold wings", "Lock wings",]);

@test overall_accuracy(cm) > 0.6

# Render model in LaTeX
# show_latex(mach.fitresult.model; attribute_names = [attribute_names_latex], silent = true);

end
