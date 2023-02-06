export print_model, print_tree, print_forest

# print model
function print_model(model::Union{DTNode,DTree,DForest}; kwargs...)
    print_model(stdout, model; kwargs...)
end
function print_model(io::IO, model::Union{DTNode,DTree}; kwargs...)
    print_tree(io, model; kwargs...)
end
function print_model(io::IO, model::DForest; kwargs...)
    print_forest(io, model; kwargs...)
end

# print tree and forest
function print_tree(tree::Union{DTNode,DTree}, args...; kwargs...)
    print_tree(stdout, tree, args...; kwargs...)
end
function print_forest(forest::DForest, args...; kwargs...)
    print_forest(stdout, forest, args...; kwargs...)
end

function print_tree(io::IO, tree::Union{DTNode,DTree}, args...; kwargs...)
    print(io, display_model(tree; args..., kwargs...))
end
function print_forest(io::IO, forest::DForest, args...; kwargs...)
    print(io, display_model(forest; args..., kwargs...))
end

function brief_prediction_str(leaf::DTLeaf)
    string(prediction(leaf))
end

function brief_prediction_str(leaf::NSDTLeaf)
    # "{$(leaf.predicting_function), size = $(Base.summarysize(leaf.predicting_function))}"
    "<$(leaf.predicting_function)>"
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

function display_model(
    forest::DForest,
    args...;
    kwargs...,
)
    outstr = ""
    n_trees = length(forest)
    for i in 1:n_trees
        outstr *= "Tree $(i) / $(n_trees)"
        outstr *= display_model(trees(forest)[i], args...; kwargs...)
    end
    return outstr
end


function display_model(
        leaf::DTLeaf;
        indentation_str="",
        attribute_names_map = nothing,
        max_depth = nothing,
        kwargs...,
    )
    metrics = get_metrics(leaf; kwargs...)
    metrics_str = get_metrics_str(metrics)
    return "$(brief_prediction_str(leaf)) : $(metrics_str)\n"
end

function display_model(
        leaf::NSDTLeaf;
        indentation_str="",
        attribute_names_map = nothing,
        max_depth = nothing,
        kwargs...,
    )
    train_metrics_str = get_metrics_str(get_metrics(leaf; train_or_valid = true, kwargs...))
    valid_metrics_str = get_metrics_str(get_metrics(leaf; train_or_valid = false, kwargs...))
    return "$(brief_prediction_str(leaf)) : {TRAIN: $(train_metrics_str); VALID: $(valid_metrics_str)}\n"
end

function display_model(
    node::DTInternal;
    indentation_str="",
    attribute_names_map = nothing,
    max_depth = nothing,
    # TODO print_rules = false,
    metrics_kwargs...,
)
    outstr = ""
    outstr *= "$(display_decision(node; attribute_names_map = attribute_names_map))\t\t\t"
    outstr *= display_model(this(node); indentation_str = "", metrics_kwargs...)
    if isnothing(max_depth) || length(indentation_str) < max_depth
        outstr *= indentation_str * "✔ " # "╭✔ 
        outstr *= display_model(left(node);
            indentation_str = indentation_str*"│",
            attribute_names_map = attribute_names_map,
            max_depth = max_depth,
            metrics_kwargs...,
        )
        outstr *= indentation_str * "✘ " # "╰✘ 
        outstr *= display_model(right(node);
            indentation_str = indentation_str*" ",
            attribute_names_map = attribute_names_map,
            max_depth = max_depth,
            metrics_kwargs...,
        )
    else
        outstr *= " [...]\n"
    end
    return outstr
end

function display_model(
    tree::DTree;
    metrics_kwargs...,
)
    # print_relative_confidence = false,
    # if print_relative_confidence && L<:CLabel
    #     outstr *= display_model(tree; rel_confidence_class_counts = countmap(Y))
    # else
    #     outstr *= display_model(tree)
    # end
    return display_model(root(tree); metrics_kwargs...)
end
