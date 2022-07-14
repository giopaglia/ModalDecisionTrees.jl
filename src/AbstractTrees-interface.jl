"""
Adapted from https://github.com/JuliaAI/DecisionTree.jl/blob/dev/src/abstract_trees.jl
"""

using AbstractTrees

MDT = ModalDecisionTrees

"""
Implementation of the `AbstractTrees.jl`-interface 
(see: [AbstractTrees.jl](https://github.com/JuliaCollections/AbstractTrees.jl)).
The functions `children` and `printnode` make up the interface traits of `AbstractTrees.jl` 
(see below for details).
The goal of this implementation is to wrap a `ModalDecisionTree` in this abstract layer, 
so that a plot recipe for visualization of the tree can be created that doesn't rely 
on any implementation details of `ModalDecisionTrees.jl`. That opens the possibility to create
a plot recipe which can be used by a variety of tree-like models. 
For a more detailed explanation of this concept have a look at the follwing article 
in "Towards Data Science": 
["If things are not ready to use"](https://towardsdatascience.com/part-iii-if-things-are-not-ready-to-use-59d2db378bec)
"""


"""
    InfoNode{S, T}
    InfoLeaf{T}
These types are introduced so that additional information currently not present in 
a `ModalDecisionTree`-structure -- for example, the names of the variables -- 
can be used for visualization. This additional information is stored in the attribute `info` of 
these types. It is a `NamedTuple`. So it can be used to store arbitraty information, 
apart from the two points mentioned.
In analogy to the type definitions of `ModalDecisionTree`, the generic type `S` is 
the type of the variable values used within a node as a threshold for the splits
between its children and `T` is the type of the output given (basically, a Number or a String).
"""
struct InfoNode{S, T}
    node    :: MDT.DTInternal{S, T}
    info    :: NamedTuple
end

struct InfoLeaf{T}
    leaf    :: MDT.AbstractDecisionLeaf{T}
    info    :: NamedTuple
end

"""
    wrap(node::MDT.DTInternal, info = NamedTuple())
    wrap(leaf::MDT.AbstractDecisionLeaf, info = NamedTuple())
Add to each `node` (or `leaf`) the additional information `info` 
and wrap both in an `InfoNode`/`InfoLeaf`.
Typically a `node` or a `leaf` is obtained by creating a decision tree using either
the native interface of `ModalDecisionTrees.jl` or via other interfaces which are available
for this package (e.g., `MLJ`, see their docs for further details).
Using the function `build_tree` of the native interface returns such an object. 
To use a ModalDecisionTree `mdt` (obtained this way) with the abstraction layer 
provided by the `AbstractTrees`-interface implemented here
and optionally add variable names (`variable_names`, an arrays of arrays of strings)
 use the following syntax:
1.  `wdc = wrap(mdt)` 
2.  `wdc = wrap(mdt, (variable_names = variable_names, ))`
In the first case `mdt` gets just wrapped, no information is added. No. 2 adds variable names.
Note that the trailing comma is needed, in order to create a NamedTuple.
"""
wrap(node::MDT.DTInternal,           info::NamedTuple = NamedTuple()) = InfoNode(node, info)
wrap(leaf::MDT.AbstractDecisionLeaf, info::NamedTuple = NamedTuple()) = InfoLeaf(leaf, info)

"""
    children(node::InfoNode)
Return for each `node` given, its children. 
    
In case of a `ModalDecisionTree` there are always exactly two children, because 
the model produces binary trees where all nodes have exactly one left and 
one right child. `children` is used for tree traversal.
The additional information `info` is carried over from `node` to its children.
"""
AbstractTrees.children(node::InfoNode) = (
    wrap(node.node.left,  node.info),
    wrap(node.node.right, node.info),
)
AbstractTrees.children(leaf::InfoLeaf) = ()

"""
    printnode(io::IO, node::InfoNode)
    printnode(io::IO, leaf::InfoLeaf)
Write a printable representation of `node` or `leaf` to output-stream `io`.
If `node.info`/`leaf.info` have a field called
- `variable_names` it is expected to be an array of arrays of variable names corresponding 
  to the variable names used in the tree nodes; note that there are two layers of reference
  because variables are grouped into `frames` (see MLJ's docs for ModalDecisionTree: @doc ModalDecisionTree)
  They will be used for printing instead of the ids.
Note that the left subtree of any split node represents the 'yes-branch', while the right subtree
 the 'no-branch', respectively. `AbstractTrees.print_tree` outputs the left subtree first
and then below the right subtree.
"""
function AbstractTrees.printnode(io::IO, node::InfoNode)
    dt_node = node.node
    if :variable_names ∈ keys(node.info)
        print(io, display_decision(dt_node; attribute_names_map = node.info.variable_names))
    else
	    print(io, display_decision(dt_node))
    end
end

function AbstractTrees.printnode(io::IO, leaf::InfoLeaf)
    dt_leaf = leaf.leaf
    metrics = get_metrics(dt_leaf)

    if :classlabels ∈ keys(leaf.info)
        print(io, leaf.info.classlabels[MDT.brief_prediction_str(dt_leaf)], " ($(metrics.n_correct)/$(metrics.n_inst))")
    else
	    print(io, MDT.brief_prediction_str(dt_leaf), " ($(metrics.n_correct)/$(metrics.n_inst))")
    end
end
