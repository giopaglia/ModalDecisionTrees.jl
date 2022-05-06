export print_model, print_tree, print_forest

print_model(tree::DTree;     kwargs...) = print_tree(tree;     kwargs...)
print_model(forest::DForest; kwargs...) = print_forest(forest; kwargs...)

print_model(io::IO, tree::DTree;     kwargs...) = print_tree(io, tree;     kwargs...)
print_model(io::IO, forest::DForest; kwargs...) = print_forest(io, forest; kwargs...)

print_forest(forest::DForest, args...; kwargs...) = print_forest(stdout, forest, args...; kwargs...)
print_tree(tree::Union{DTree,DTNode}, args...; kwargs...) = print_tree(stdout, tree, args...; kwargs...)

function brief_prediction_str(leaf::DTLeaf)
    string(prediction(leaf))
end

function brief_prediction_str(leaf::NSDTLeaf)
    # "{$(leaf.predicting_function), size = $(Base.summarysize(leaf.predicting_function))}"
    "{$(leaf.predicting_function)}"
end

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


function get_metrics_str(metrics::NamedTuple)
    metrics_str_pieces = []
    if haskey(metrics,:confidence)
        push!(metrics_str_pieces, "conf = $(@sprintf "%.4f" metrics.confidence)")
    end
    if haskey(metrics,:lift)
        push!(metrics_str_pieces, "lift = $(@sprintf "%.2f" metrics.lift)")
    end
    if haskey(metrics,:support)
        push!(metrics_str_pieces, "supp = $(@sprintf "%.4f" metrics.support)")
    end
    if haskey(metrics,:conviction)
        push!(metrics_str_pieces, "conv = $(@sprintf "%.4f" metrics.conviction)")
    end
    if haskey(metrics,:sensitivity_share)
        push!(metrics_str_pieces, "sensitivity_share = $(@sprintf "%.4f" metrics.sensitivity_share)")
    end
    if haskey(metrics,:var)
        push!(metrics_str_pieces, "var = $(@sprintf "%.4f" metrics.var)")
    end
    if haskey(metrics,:mae)
        push!(metrics_str_pieces, "mae = $(@sprintf "%.4f" metrics.mae)")
    end
    if haskey(metrics,:rmse)
        push!(metrics_str_pieces, "rmse = $(@sprintf "%.4f" metrics.rmse)")
    end
    if haskey(metrics,:support)
        push!(metrics_str_pieces, "supp = $(@sprintf "%.4f" metrics.support)")
    end
    metrics_str = join(metrics_str_pieces, ", ")
    if haskey(metrics,:n_correct) # Classification
        "$(metrics.n_correct)/$(metrics.n_inst) ($(metrics_str))"
    else # Regression
        "$(metrics.n_inst) ($(metrics_str))"
    end
end

function print_tree(
        io::IO,
        leaf::DTLeaf;
        indentation_str="",
        metrics_kwargs...,
    )
    metrics = get_metrics(leaf; metrics_kwargs...)
    metrics_str = get_metrics_str(metrics)
    println(io, "$(brief_prediction_str(leaf)) : $(metrics_str)")
end

function print_tree(
        io::IO,
        leaf::NSDTLeaf;
        indentation_str="",
        metrics_kwargs...,
    )
    train_metrics_str = metrics_str(get_metrics(leaf; train_or_valid = true, metrics_kwargs...))
    valid_metrics_str = metrics_str(get_metrics(leaf; train_or_valid = false, metrics_kwargs...))
    println(io, "$(brief_prediction_str(leaf)) : {TRAIN: $(train_metrics_str); VALID: $(valid_metrics_str)}")
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
    # print_relative_confidence = false,
    # if print_relative_confidence && L<:CLabel
    #     print_tree(io, tree; rel_confidence_class_counts = countmap(Y))
    # else
    #     print_tree(io, tree)
    # end
    print_tree(io, tree.root; metrics_kwargs...)
end
