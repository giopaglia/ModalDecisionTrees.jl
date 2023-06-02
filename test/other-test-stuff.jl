using Pkg

# Pkg.activate(".")

# Pkg.add("Revise")
# Pkg.add("MLJ")
# Pkg.add("MLJBase")
# Pkg.add(url = "https://github.com/giopaglia/ModalDecisionTrees.jl", rev = "dev-v0.8")
# Pkg.add("ScientificTypes")
# Pkg.add("DataFrames")
# Pkg.add("Tables")
# Pkg.add("ARFFFiles#main")

using Revise
using ModalDecisionTrees
using MLJ
using MLJBase
using ScientificTypes
using DataFrames
using Tables
using StatsBase
using MLJModelInterface

MMI = MLJModelInterface
MDT = ModalDecisionTrees

include("utils.jl")

model = ModalDecisionTree(;
    min_samples_leaf = 4,
    min_purity_increase = 0.002,
)

using ARFFFiles, DataFrames

dataset_name = "NATOPS"
# dataset_name = "RacketSports"
# dataset_name = "Libras"




fitresult = MMI.fit(model, 0, X_train, Y_train);

Y_test_preds, test_tree = MMI.predict(model, fitresult[1], X_test, Y_test);

tree = fitresult[1].model

fitresult[3].print_tree()

fitresult[3].print_tree(test_tree)

println(tree)
println(test_tree)

# MLJ.ConfusionMatrix()(Y_test_preds, Y_test);
# SoleModels.ConfusionMatrix(Y_test_preds, Y_test)


# tree = fitresult.model
# println(tree)
# println(test_tree)
# fitreport.print_tree()
# fitreport.print_tree(test_tree)


# using AbstractTrees
# using GraphRecipes
# using Plots
# default(size=(1000, 1000))

# plot(TreePlot(tree.root), method=:tree, fontsize=10)


# show_latex(tree, "train")
# show_latex(test_tree, "test")



show_latex(tree, "train", [variable_names_latex])
show_latex(test_tree, "test", [variable_names_latex])


# function apply_static_descriptor(X::DataFrame, f::Function)
#     variable_names = names(X)
#     rename(f.(X), Dict([Symbol(a) => Symbol("$(f)($(a))") for a in variable_names]))
# end

# function apply_static_descriptor(X::DataFrame, fs::AbstractVector{<:Function})
#     hcat([apply_static_descriptor(X, f) for f in fs]...)
# end

# for fs in [mean, var, minimum, maximum, [minimum, maximum], [mean, minimum, maximum], [var, mean, minimum, maximum]]
#     X_static_train = apply_static_descriptor(X_train, fs)
#     X_static_test  = apply_static_descriptor(X_test, fs)


#     fitresult = MMI.fit(model, 0, X_static_train, Y_train);

#     Y_test_preds, test_tree = MMI.predict(model, fitresult[1], X_static_test, Y_test);

#     tree = fitresult[1].model

#     fitresult[3].print_tree()

#     fitresult[3].print_tree(test_tree)

#     # println(tree)
#     # println(test_tree)

#     # MLJ.ConfusionMatrix()(Y_test_preds, Y_test)
#     println(fs)
#     println(SoleModels.ConfusionMatrix(Y_test_preds, Y_test))
#     readline()
# end
