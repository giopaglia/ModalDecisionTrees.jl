function compute_featureimportance(model, var_grouping = nothing; normalize = true)
    feature_importance_by_count = MDT.variable_countmap(model)

    if !isnothing(var_grouping)
        feature_importance_by_count = Dict([
            # i_var => var_grouping[i_modality][i_var]
            var_grouping[i_modality][i_var] => count
            for ((i_modality, i_var), count) in feature_importance_by_count])
    end

    if normalize
        sumcount = sum(values(feature_importance_by_count))
        feature_importance_by_count = Dict([
            feature => (count/sumcount)
            for (feature, count) in feature_importance_by_count])
    end

    feature_importance_by_count
end
