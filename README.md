# Modal Decision Trees & Forests

### Interpretable models for native time-series classification!

This package provides algorithms for learning *decision trees* and *decision forests* with enhanced abilities.
Leveraging the express power of Modal Logic, these models can extract *temporal/spatial patterns*, and can natively handle data such as *time series* and *images* (without any data preprocessing).

#### Features & differences with [DecisionTree.jl](https://github.com/JuliaAI/DecisionTree.jl):
- Ability to handle attributes that are `AbstractVector{<:Real}` or `AbstractMatrix{<:Real}`;
- Enables interpretable *multimodal* learning
- Fully optimized implementation (fancy data structures, multithreading, memoization, minification, Pareto-based pruning optimizations, etc).
- A unique algorithm that extends CART and C4.5;
<!-- - TODO -->
<!-- - Four pruning conditions: max_depth, min_samples_leaf, min_purity_increase, max_purity_at_leaf -->
<!-- TODO - Top-down pre-pruning & post-pruning -->
<!-- - Bagging (Random Forests) TODO dillo meglio -->

#### *Current* limitations (also see [TODO](#todo)):
- Only supports numeric features;
- Only supports classification tasks;
- Only available via [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl);
- Does not support missing values.

<!-- 
## Installation

Simply type the following commands in Julia's REPL:

```julia
Pkg.add(url="https://github.com/giopaglia/ModalDecisionTrees.jl")
```
-->

<!--
## Usage

```julia
# Install package
using Pkg;
Pkg.add(url="https://github.com/giopaglia/ModalDecisionTrees.jl")
Pkg.add("MLJ")

using MLJ
using ModalDecisionTrees

TODO load dummy dataset
TODO perform learning

...

TODO show tree and explain how to interpret the results
print_tree(tree)
```

TODO explain parameters
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

## TODO

- [x]  Enable choosing a loss functions different from Shannon's entropy (*untested*)
- [x]  Enable use of weights (*untested*)
- [x]  Enable regression (*untested*)
- [x]  Enable support for images (*untested*)
- [x]  Enable multimodal learning (learning with scalars, time-series and images together)
- [ ]  Proper test suite
- [ ]  Visualizations of modal rules/patterns

## Theoretical foundations

Most of the works in *symbolic learning* are based either on Propositional Logics (PLs) or First-order Logics (FOLs); PLs are the simplest kind of logic and can only handle *tabular data*, while FOLs can express complex entity-relation concepts. Machine Learning with FOLs enables handling data with complex topologies, such as time series, images, or videos; however, these logics are computationally challenging. Instead, Modal Logics (e.g. [Interval Logic](https://en.wikipedia.org/wiki/Interval_temporal_logic)) represent a perfect trade-off in terms of computational tractability and expressive power, and naturally lend themselves for expressing some forms of *temporal/spatial reasoning*.

Recently, symbolic learning techniques such as Decision Trees, Random Forests and Rule-Based models have been extended to the use of Modal Logics of time and space. *Modal Decision Trees* and *Modal Random Forests* have been applied to classification tasks, showing statistical performances that are often comparable to those of functional methods (e.g., neural networks), while providing, at the same time, highly-interpretable classification models. Examples of these tasks are COVID-19 diagnosis from cough/breath audio [[1]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4102488), [[2]](https://drops.dagstuhl.de/opus/volltexte/2021/14783/pdf/LIPIcs-TIME-2021-7.pdf), land cover classification from aereal images [[3]](https://arxiv.org/abs/2109.08325), EEG-related tasks [[4]](https://link.springer.com/chapter/10.1007/978-3-031-06242-1_53), and gas turbine trip prediction.
This technology also offers a natural extension for *multimodal* learning [[5]](http://ceur-ws.org/Vol-2987/paper7.pdf).

## Credits

The package is developed by Giovanni Pagliarini ([@giopaglia](https://giopaglia.github.io/)) and Federico Manzella ([@ferdiu](https://ferdiu.github.io/)).

Thanks to [ACLAI Lab](https://aclai.unife.it/index.php/en/home-page/) @ University of Ferrara.

Thanks to Ben Sadeghi (@bensadeghi), original author of [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl).

<!-- TODO add citation and CITATION.bib file -->
