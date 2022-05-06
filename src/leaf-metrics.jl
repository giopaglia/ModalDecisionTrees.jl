
function get_metrics(
        leaf::AbstractDecisionLeaf{<:CLabel};
        n_tot_inst = nothing,
        rel_confidence_class_counts = nothing,
        train_or_valid = true,
    )
    metrics = (;)

    supporting_labels      = supp_labels(leaf; train_or_valid = train_or_valid)
    supporting_predictions = predictions(leaf; train_or_valid = train_or_valid)

    ############################################################################
    # Confidence, # of supporting labels, # of correctly classified instances
    n_inst = length(supporting_labels)
    n_correct = sum(supporting_labels .== supporting_predictions)
    confidence = n_correct/n_inst
    
    metrics = merge(metrics, (
        n_inst            = n_inst,            
        n_correct         = n_correct,         
        confidence        = confidence,                
    ))

    ############################################################################
    # Total # of instances

    if !isnothing(rel_confidence_class_counts)
        if !isnothing(n_tot_inst)
            @assert n_tot_inst == sum(values(rel_confidence_class_counts)) "n_tot_inst != sum(values(rel_confidence_class_counts)): $(n_tot_inst) $(sum(values(rel_confidence_class_counts))) sum($(values(rel_confidence_class_counts)))"
        else
            n_tot_inst = sum(values(rel_confidence_class_counts))
        end
        metrics = merge(metrics, (
            n_tot_inst = n_tot_inst,
        ))
    end
    
    ############################################################################
    # Lift, class support and others

    if !isnothing(rel_confidence_class_counts)
        cur_class_counts = begin
            cur_class_counts = countmap(supporting_labels)
            for class in keys(rel_confidence_class_counts)
                if !haskey(cur_class_counts, class)
                    cur_class_counts[class] = 0
                end
            end
            cur_class_counts
        end

        rel_tot_inst = sum([cur_class_counts[class]/rel_confidence_class_counts[class] for class in keys(rel_confidence_class_counts)])

        # TODO can't remember the rationale behind this?
        # if isa(leaf, DTLeaf)
        # "rel_conf: $(n_correct/rel_confidence_class_counts[prediction(leaf)])"
        # rel_conf = (cur_class_counts[prediction(leaf)]/get(rel_confidence_class_counts, prediction(leaf), 0))/rel_tot_inst
        # end

        metrics = merge(metrics, (
            cur_class_counts = cur_class_counts,
            rel_tot_inst = rel_tot_inst,
            rel_conf = rel_conf,
        ))

        if !isnothing(n_tot_inst) && isa(leaf, DTLeaf)
            class_support = get(rel_confidence_class_counts, prediction(leaf), 0)/n_tot_inst
            lift = confidence/class_support
            metrics = merge(metrics, (
                class_support = class_support,
                lift = lift,
            ))
        end
    end
    
    ############################################################################
    # Support

    if !isnothing(n_tot_inst)
        support = n_inst/n_tot_inst
        metrics = merge(metrics, (
            support = support,
        ))
    end

    ############################################################################
    # Conviction

    if !isnothing(rel_confidence_class_counts) && !isnothing(n_tot_inst)
        conviction = (1-class_support)/(1-confidence)
        metrics = merge(metrics, (
            conviction = conviction,
        ))
    end

    ############################################################################
    # Sensitivity share: the portion of "responsibility" for the correct classification of class L

    if !isnothing(rel_confidence_class_counts) && isa(leaf, DTLeaf)
        sensitivity_share = n_correct/get(rel_confidence_class_counts, prediction(leaf), 0)
        metrics = merge(metrics, (
            sensitivity_share = sensitivity_share,
        ))
    end

    metrics
end

function get_metrics(
        leaf::AbstractDecisionLeaf{<:RLabel};
        n_tot_inst = nothing,
        rel_confidence_class_counts = nothing,
        train_or_valid = true,
    )
    @assert isnothing(rel_confidence_class_counts)

    metrics = (;)
    
    supporting_labels      = supp_labels(leaf; train_or_valid = train_or_valid)
    supporting_predictions = predictions(leaf; train_or_valid = train_or_valid)

    n_inst = length(supporting_labels)
    
    mae = sum(abs.(supporting_labels .- supporting_predictions)) / n_inst
    rmse = StatsBase.rmsd(supporting_labels, supporting_predictions)
    var = StatsBase.var(supporting_labels)
    
    metrics = merge(metrics, (
        n_inst = n_inst,
        mae = mae,
        rmse = rmse,
        var = var,
    ))

    if !isnothing(n_tot_inst)
        support = n_inst/n_tot_inst
        metrics = merge(metrics, (
            support = support,
        ))
    end

    metrics
end
