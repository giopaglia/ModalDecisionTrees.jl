# Modal Decision Trees & Forests

[![DOI](https://zenodo.org/badge/323867446.svg)](https://zenodo.org/badge/latestdoi/323867446)

### Interpretable models for native time-series & image classification!

This package provides algorithms for learning *decision trees* and *decision forests* with enhanced abilities.
Leveraging the express power of Modal Logic, these models can extract *temporal/spatial patterns*, and can natively handle *time series* and *images* (without any data preprocessing). Currently available via [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl) and [*Sole.jl*](https://github.com/aclai-lab/Sole.jl).

#### Features & differences with [DecisionTree.jl](https://github.com/JuliaAI/DecisionTree.jl):
- Ability to handle attributes that are `AbstractVector{<:Real}` or `AbstractMatrix{<:Real}`;
- Supports *multimodal* learning (i.e., learning from *combinations* of scalars, time series and images);
- Fully optimized implementation (fancy data structures, multithreading, memoization, minification, Pareto-based pruning optimizations, etc);
- A unique algorithm that extends CART and C4.5;
- Slightly different set of hyperparameters (e.g., no `min_samples_split`, `post_prune` & `merge_purity_threshold`).
<!-- - TODO -->
<!-- - Four pruning conditions: max_depth, min_samples_leaf, min_purity_increase, max_purity_at_leaf -->
<!-- TODO - Top-down pre-pruning & post-pruning -->
<!-- - Bagging (Random Forests) TODO dillo meglio -->

#### *Current* limitations (also see [TODOs](#todos)):
- Only supports numeric features;
- Only supports classification tasks;
<!-- - Only available via [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl); -->
- Does not support `missing` or `NaN` values.

#### Checkout the 8-minute [lightning talk](https://www.youtube.com/watch?v=8F1vZsl8Zvg) at JuliaCon 2022!

<!-- 
## Installation

Simply type the following commands in Julia's REPL:

```julia
using Pkg; Pkg.add(url="https://github.com/giopaglia/ModalDecisionTrees.jl")
```
-->

## Installation & Usage

At the moment, the package depends on unregistered packages, which installation is
automated via the `install.jl` script.
```shell
# Clone package
cd ~/Desktop
git clone https://github.com/giopaglia/ModalDecisionTrees.jl
# Install package and its dependencies
cd ModalDecisionTrees
julia -it8 install.jl
julia -it8 -e 'Pkg.test()'
```

```julia
# Install packages
Pkg.add(path="~/Desktop/ModalDecisionTrees")
Pkg.add("MLJ")

# Import packages
using MLJ
using ModalDecisionTrees
using Random

# Load an example dataset (a temporal one)
X, y = @load_japanesevowels
N = length(y)

# Instantiate an MLJ machine based on a Modal Decision Tree with ≥ 4 samples at leaf
mach = machine(ModalDecisionTree(min_samples_leaf=4), X, y)

# Split dataset
p = randperm(N)
train_idxs, test_idxs = p[1:round(Int, N*.8)], p[round(Int, N*.8)+1:end]

# Fit
fit!(mach, rows=train_idxs)

# Perform predictions, compute accuracy
yhat = predict(mach, X[test_idxs,:])
accuracy = sum(yhat .== y[test_idxs])/length(yhat)

# Print model
report(mach).printmodel(3)

# Access raw model
model = fitted_params(mach).model
```


<!--
# TODO
# Render raw model
Pkg.add("GraphRecipes"); Pkg.add("Plots")

using GraphRecipes
using Plots

#wrapped_model = ModalDecisionTrees.wrap(model.root, (attribute_names_map = report(mach).frame_grouping,))
# for _method in [:spectral, :sfdp, :circular, :shell, :stress, :spring, :tree, :buchheim, :arcdiagram, :chorddiagram]
wrapped_model = ModalDecisionTrees.wrap(model.root, (; threshold_display_method = x->round(x, digits=2)), use_feature_abbreviations = true)
for _method in [:tree, :buchheim]
	for _nodeshape in [:rect] # , [:rect, :ellipse]
		display(plot(
 		TreePlot(wrapped_model), 
 		method = _method,
 		nodeshape = _nodeshape,
 		# nodesize = (3,10),
 		# root = :left,
 		curves = false,
		fontsize = 10,
		size=(860, 640),
		title = "$(_method)"
		))
	end
end
-->

<!-- TODO (`Y isa Vector{<:{Integer,String}}`) -->

<!--
Detailed usage instructions are available for each model using the doc method. For example:

```julia
using MLJ
doc("DecisionTreeClassifier", pkg="ModalDecisionTrees")
```

Available models are: AdaBoostStumpClassifier, DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor.


-->
<!-- 
## Visualization

A DecisionTree model can be visualized using the print_tree-function of its native interface (for an example see above in section 'Classification Example'). -->

## TODOs

- [x]  Enable loss functions different from Shannon's entropy (*untested*)
- [x]  Enable regression (*untested*)
- [ ]  Proper test suite
- [ ]  Visualizations of modal rules/patterns
<!-- - [x]  AbstractTrees interface -->

## Theoretical foundations

Most of the works in *symbolic learning* are based either on Propositional Logics (PLs) or First-order Logics (FOLs); PLs are the simplest kind of logic and can only handle *tabular data*, while FOLs can express complex entity-relation concepts. Machine Learning with FOLs enables handling data with complex topologies, such as time series, images, or videos; however, these logics are computationally challenging. Instead, Modal Logics (e.g. [Interval Logic](https://en.wikipedia.org/wiki/Interval_temporal_logic)) represent a perfect trade-off in terms of computational tractability and expressive power, and naturally lend themselves for expressing some forms of *temporal/spatial reasoning*.

Recently, symbolic learning techniques such as Decision Trees, Random Forests and Rule-Based models have been extended to the use of Modal Logics of time and space. *Modal Decision Trees* and *Modal Random Forests* have been applied to classification tasks, showing statistical performances that are often comparable to those of functional methods (e.g., neural networks), while providing, at the same time, highly-interpretable classification models. Examples of these tasks are COVID-19 diagnosis from cough/breath audio [[1]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4102488), [[2]](https://drops.dagstuhl.de/opus/volltexte/2021/14783/pdf/LIPIcs-TIME-2021-7.pdf), land cover classification from aereal images [[3]](https://arxiv.org/abs/2109.08325), EEG-related tasks [[4]](https://link.springer.com/chapter/10.1007/978-3-031-06242-1_53), and gas turbine trip prediction.
This technology also offers a natural extension for *multimodal* learning [[5]](http://ceur-ws.org/Vol-2987/paper7.pdf).

## Credits

The package is developed by Giovanni Pagliarini ([@giopaglia](https://giopaglia.github.io/)) and Federico Manzella ([@ferdiu](https://ferdiu.github.io/)).

Thanks to [ACLAI Lab](https://aclai.unife.it/en/) @ University of Ferrara.

Thanks to Ben Sadeghi ([@bensadeghi](https://github.com/bensadeghi/)), author of [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl),
which inspired the construction of this package.

<!-- TODO add citation and CITATION.bib file -->
