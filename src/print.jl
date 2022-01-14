export print_model, print_tree, print_forest

print_model(tree::DTree;    kwargs...) = print_tree(tree;     kwargs...)
print_model(forest::DForest; kwargs...) = print_forest(forest; kwargs...)

print_tree(a::Union{DTree,DTNode}, args...; kwargs...) = print_tree(stdout, a, args...; kwargs...)
print_forest(a::DForest, args...; kwargs...) = print_forest(stdout, a, args...; kwargs...)

function print_tree(
    io::IO,
    leaf::DTLeaf{String},
    depth=-1,
    indent=0,
    indent_guides=[];
    n_tot_inst = nothing,
    rel_confidence_class_counts = nothing,
)
    n_correct = sum(leaf.supp_labels .== leaf.label)
    n_inst = length(leaf.supp_labels)
    confidence = n_correct/n_inst

    metrics_str = "conf: $(@sprintf "%.4f" confidence)"
    
    if !isnothing(rel_confidence_class_counts)
        if !isnothing(n_tot_inst)
            @assert n_tot_inst == sum(values(rel_confidence_class_counts)) "n_tot_inst != sum(values(rel_confidence_class_counts)): $(n_tot_inst) $(sum(values(rel_confidence_class_counts))) sum($(values(rel_confidence_class_counts)))"
        else
            n_tot_inst = sum(values(rel_confidence_class_counts))
        end
    end


    if !isnothing(rel_confidence_class_counts)
        cur_class_counts = begin
            cur_class_counts = countmap(leaf.supp_labels)
            for class in keys(rel_confidence_class_counts)
                if !haskey(cur_class_counts, class)
                    cur_class_counts[class] = 0
                end
            end
            cur_class_counts
        end

        # println(io, cur_class_counts)
        # println(io, rel_confidence_class_counts)
        rel_tot_inst = sum([cur_class_counts[class]/rel_confidence_class_counts[class] for class in keys(rel_confidence_class_counts)])
        # "rel_conf: $(n_correct/rel_confidence_class_counts[leaf.label])"

        if !isnothing(n_tot_inst)
            class_support = get(rel_confidence_class_counts, leaf.label, 0)/n_tot_inst
            lift = confidence/class_support
            metrics_str *= ", lift: $(@sprintf "%.2f" lift)"
        end
        # TODO what was the rationale behind this?
        # rel_conf = (cur_class_counts[leaf.label]/get(rel_confidence_class_counts, leaf.label, 0))/rel_tot_inst
        # metrics_str *= ", rel_conf: $(@sprintf "%.4f" rel_conf)"
    end

    if !isnothing(n_tot_inst)
        support = n_inst/n_tot_inst
        metrics_str *= ", supp = $(@sprintf "%.4f" support)"
    end

    if !isnothing(rel_confidence_class_counts) && !isnothing(n_tot_inst)
        conv = (1-class_support)/(1-confidence)
        metrics_str *= ", conv: $(@sprintf "%.4f" conv)"
    end

    # - , quale è la sua porzione di responsabilità per la corretta classificazione di C, ovvero della sua sensitività: per farlo, considera i due numerini a/b, e calcola a/# di istanze con classe C nel test set), e TODOrapportala alla sensitività, così

    if !isnothing(rel_confidence_class_counts)
        sensitivity_share = n_correct/get(rel_confidence_class_counts, leaf.label, 0)
        metrics_str *= ", sensitivity_share: $(@sprintf "%.4f" sensitivity_share)"
    end

    println(io, "$(leaf.label) : $(n_correct)/$(n_inst) ($(metrics_str))")
end

function print_tree(
    io::IO,
    leaf::DTLeaf{<:Float64},
    depth=-1,
    indent=0,
    indent_guides=[];
    n_tot_inst = nothing,
    rel_confidence_class_counts = nothing
)
    
    n_inst = length(leaf.supp_labels)
    
    mae = sum(abs.(leaf.supp_labels .- leaf.label)) / n_inst
    rmse = StatsBase.rmsd(leaf.supp_labels, [leaf.label for i in 1:length(leaf.supp_labels)])
    var = StatsBase.var(leaf.supp_labels)
    
    metrics_str = ""
    # metrics_str *= "$(leaf.supp_labels) "
    metrics_str *= "var: $(@sprintf "%.4f" mae)"
    metrics_str *= ", mae: $(@sprintf "%.4f" mae)"
    metrics_str *= ", rmse: $(@sprintf "%.4f" rmse)"
    
    if !isnothing(n_tot_inst)
        support = n_inst/n_tot_inst
        metrics_str *= ", supp = $(@sprintf "%.4f" support)"
    end

    println(io, "$(leaf.label) : $(n_inst) ($(metrics_str))")
end

function print_tree(
    io::IO,
    tree::DTInternal,
    depth=-1,
    indent=0,
    indent_guides=[];
    n_tot_inst = nothing,
    rel_confidence_class_counts = nothing,
)
    if depth == indent
        println(io, "")
        return
    end

    if !isnothing(rel_confidence_class_counts)
        if !isnothing(n_tot_inst)
            @assert n_tot_inst == sum(values(rel_confidence_class_counts)) "n_tot_inst != sum(values(rel_confidence_class_counts)): $(n_tot_inst) $(sum(values(rel_confidence_class_counts))) sum($(values(rel_confidence_class_counts)))"
        else
            n_tot_inst = sum(values(rel_confidence_class_counts))
        end
    end
    
    # TODO show tree.this
    println(io, display_decision(tree))
    # indent_str = " " ^ indent
    indent_str = reduce(*, [i == 1 ? "│" : " " for i in indent_guides])
    # print(io, indent_str * "╭✔")
    print(io, indent_str * "✔ ")
    print_tree(io, tree.left, depth, indent + 1, [indent_guides..., 1]; n_tot_inst = n_tot_inst, rel_confidence_class_counts)
    # print(io, indent_str * "╰✘")
    print(io, indent_str * "✘ ")
    print_tree(io, tree.right, depth, indent + 1, [indent_guides..., 0]; n_tot_inst = n_tot_inst, rel_confidence_class_counts)
end

function print_tree(
    io::IO,
    tree::DTree;
    n_tot_inst = nothing,
    rel_confidence_class_counts = nothing,
)
    println(io, "worldTypes: $(tree.worldTypes)")
    println(io, "initConditions: $(tree.initConditions)")

    if !isnothing(rel_confidence_class_counts)
        if !isnothing(n_tot_inst)
            @assert n_tot_inst == sum(values(rel_confidence_class_counts)) "n_tot_inst != sum(values(rel_confidence_class_counts)): $(n_tot_inst) $(sum(values(rel_confidence_class_counts))) sum($(values(rel_confidence_class_counts)))"
        else
            n_tot_inst = sum(values(rel_confidence_class_counts))
        end
    end
    
    print_tree(io, tree.root, n_tot_inst = n_tot_inst, rel_confidence_class_counts = rel_confidence_class_counts)
end

function print_forest(
    io::IO,
    forest::DForest,
    args...;
    kwargs...
)
    n_trees = length(forest)
    for i in 1:n_trees
        println(io, "Tree $(i) / $(n_trees)")
        print_tree(io, forest.trees[i], args...; kwargs...)
    end
end
