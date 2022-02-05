export print_model, print_tree, print_forest

print_model(tree::DTree;     kwargs...) = print_tree(tree;     kwargs...)
print_model(forest::DForest; kwargs...) = print_forest(forest; kwargs...)

print_model(io::IO, tree::DTree;     kwargs...) = print_tree(io, tree;     kwargs...)
print_model(io::IO, forest::DForest; kwargs...) = print_forest(io, forest; kwargs...)

print_forest(forest::DForest, args...; kwargs...) = print_forest(stdout, forest, args...; kwargs...)
print_tree(tree::Union{DTree,DTNode}, args...; kwargs...) = print_tree(stdout, tree, args...; kwargs...)

function print_forest(
    io::IO,
    forest::DForest,
    args...;
    kwargs...,
)
    n_trees = length(forest)
    for i in 1:n_trees
        println(io, "Tree $(i) / $(n_trees)")
        print_tree(io, forest.trees[i], args...; kwargs...)
    end
end

function print_tree(
        io::IO,
        leaf::DTLeaf{<:CLabel};
        indentation_str="",
        metrics_kwargs...,
    )
    metrics = get_metrics(leaf; metrics_kwargs...)

    metrics_str = ""

    metrics_str *= "conf = $(@sprintf "%.4f" metrics.confidence)"

    if haskey(metrics,:lift)
        metrics_str *= ", lift = $(@sprintf "%.2f" metrics.lift)"
    end

    if haskey(metrics,:support)
        metrics_str *= ", supp = $(@sprintf "%.4f" metrics.support)"
    end
    
    if haskey(metrics,:conviction)
        metrics_str *= ", conv = $(@sprintf "%.4f" metrics.conviction)"
    end
    
    if haskey(metrics,:sensitivity_share)
        metrics_str *= ", sensitivity_share = $(@sprintf "%.4f" metrics.sensitivity_share)"
    end
    
    println(io, "$(leaf.label) : $(metrics.n_correct)/$(metrics.n_inst) ($(metrics_str))")
end

function print_tree(
        io::IO,
        leaf::DTLeaf{<:RLabel};
        indentation_str="",
        metrics_kwargs...,
    )
    metrics = get_metrics(leaf; metrics_kwargs...)
    
    metrics_str = ""

    metrics_str *= "var = $(@sprintf "%.4f" metrics.var)"
    
    metrics_str *= ", mae = $(@sprintf "%.4f" metrics.mae)"
    
    metrics_str *= ", rmse = $(@sprintf "%.4f" metrics.rmse)"

    if haskey(metrics,:support)
        metrics_str *= ", supp = $(@sprintf "%.4f" metrics.support)"
    end

    println(io, "$(leaf.label) : $(metrics.n_inst) ($(metrics_str))")
end

function print_tree(
    io::IO,
    node::DTInternal;
    indentation_str="",
    # TODO print_rules = false,
    metrics_kwargs...,
)
    print(io, "$(display_decision(node))\t\t\t")
    print_tree(io, node.this; indentation_str = "", metrics_kwargs...)
    print(io, indentation_str * "✔ ") # "╭✔ "
    print_tree(io, node.left; indentation_str = indentation_str*"│", metrics_kwargs...)
    print(io, indentation_str * "✘ ") # "╰✘ "
    print_tree(io, node.right; indentation_str = indentation_str*" ", metrics_kwargs...)
end

function print_tree(
    io::IO,
    tree::DTree;
    metrics_kwargs...,
)
    print_tree(io, tree.root; metrics_kwargs...)
end
