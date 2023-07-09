export printmodel, print_tree, print_forest

# print model
function printmodel(model::Union{DTNode,DTree,DForest,RootLevelNeuroSymbolicHybrid}; kwargs...)
    printmodel(stdout, model; kwargs...)
end
function printmodel(io::IO, model::Union{DTNode,DTree}; kwargs...)
    print_tree(io, model; kwargs...)
end
function printmodel(io::IO, model::DForest; kwargs...)
    print_forest(io, model; kwargs...)
end
function printmodel(io::IO, model::RootLevelNeuroSymbolicHybrid; kwargs...)
    print_rlnsdt(io, model; kwargs...)
end


function print_tree(tree::Union{DTNode,DTree}, args...; kwargs...)
    print_tree(stdout, tree, args...; kwargs...)
end
function print_forest(forest::DForest, args...; kwargs...)
    print_forest(stdout, forest, args...; kwargs...)
end
function print_rlnsdt(rlnsdt::RootLevelNeuroSymbolicHybrid, args...; kwargs...)
    print_rlnsdt(stdout, rlnsdt, args...; kwargs...)
end


function print_tree(io::IO, tree::Union{DTNode,DTree}, args...; kwargs...)
    print(io, displaymodel(tree; args..., kwargs...))
end
function print_forest(io::IO, forest::DForest, args...; kwargs...)
    print(io, displaymodel(forest; args..., kwargs...))
end
function print_rlnsdt(io::IO, rlnstd::RootLevelNeuroSymbolicHybrid, args...; kwargs...)
    print(io, displaymodel(rlnstd; args..., kwargs...))
end

############################################################################################

function displaybriefprediction(leaf::DTLeaf)
    string(prediction(leaf))
end

function displaybriefprediction(leaf::NSDTLeaf)
    # "{$(leaf.predicting_function), size = $(Base.summarysize(leaf.predicting_function))}"
    "<$(leaf.predicting_function)>"
end

function get_metrics_str(metrics::NamedTuple)
    metrics_str_pieces = []
    # if haskey(metrics,:n_inst)
    #     push!(metrics_str_pieces, "ninst = $(metrics.n_inst)")
    # end
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
    metrics_str = join(metrics_str_pieces, ", ")
    if haskey(metrics,:n_correct) # Classification
        "$(metrics.n_correct)/$(metrics.n_inst) ($(metrics_str))"
    else # Regression
        "$(metrics.n_inst) ($(metrics_str))"
    end
end

function displaymodel(
    tree::DTree;
    metrics_kwargs...,
)
    return displaymodel(root(tree); metrics_kwargs...)
end

function displaymodel(
    forest::DForest,
    args...;
    kwargs...,
)
    outstr = ""
    _ntrees = ntrees(forest)
    for i_tree in 1:_ntrees
        outstr *= "Tree $(i_tree) / $(_ntrees)\n"
        outstr *= displaymodel(trees(forest)[i_tree], args...; kwargs...)
    end
    return outstr
end

function displaymodel(
    nsdt::RootLevelNeuroSymbolicHybrid,
    args...;
    kwargs...,
)
    outstr = ""
    outstr *= "Feature function: $(nsdt.feature_function)"
    _ntrees = ntrees(forest)
    for (i_tree,tree) in enumerate(nsdt.trees)
        outstr *= "Tree $(i_tree) / $(_ntrees)\n"
        outstr *= displaymodel(tree, args...; kwargs...)
    end
    return outstr
end

function displaymodel(
    leaf::DTLeaf;
    indentation_str="",
    depth=0,
    variable_names_map = nothing,
    max_depth = nothing,
    kwargs...,
)
    metrics = get_metrics(leaf; kwargs...)
    metrics_str = get_metrics_str(metrics)
    return "$(displaybriefprediction(leaf)) : $(metrics_str)\n"
end

function displaymodel(
    leaf::NSDTLeaf;
    indentation_str="",
    depth=0,
    variable_names_map = nothing,
    max_depth = nothing,
    kwargs...,
)
    train_metrics_str = get_metrics_str(get_metrics(leaf; train_or_valid = true, kwargs...))
    valid_metrics_str = get_metrics_str(get_metrics(leaf; train_or_valid = false, kwargs...))
    return "$(displaybriefprediction(leaf)) : {TRAIN: $(train_metrics_str); VALID: $(valid_metrics_str)}\n"
end

function displaymodel(
    node::DTInternal;
    indentation_str="",
    depth=0,
    variable_names_map = nothing,
    max_depth = nothing,
    # TODO print_rules = false,
    metrics_kwargs...,
)
    outstr = ""
    if isnothing(max_depth) || depth < max_depth
        dec_str = displaydecision(node; variable_names_map = variable_names_map)
        outstr *= "$(rpad(dec_str, 59-(length(indentation_str) == 0 ? length(indentation_str)-1 : length(indentation_str)))) "
        # outstr *= "$(60-max(length(indentation_str)+1)) "
        outstr *= displaymodel(this(node); indentation_str = "", metrics_kwargs...)
        outstr *= indentation_str * "✔ " # "╭✔
        outstr *= displaymodel(left(node);
            indentation_str = indentation_str*"│",
            depth = depth+1,
            variable_names_map = variable_names_map,
            max_depth = max_depth,
            metrics_kwargs...,
        )
        outstr *= indentation_str * "✘ " # "╰✘
        outstr *= displaymodel(right(node);
            indentation_str = indentation_str*" ",
            depth = depth+1,
            variable_names_map = variable_names_map,
            max_depth = max_depth,
            metrics_kwargs...,
        )
    else
        depth != 0 && (outstr *= " ")
        outstr *= "[...]\n"
    end
    return outstr
end
