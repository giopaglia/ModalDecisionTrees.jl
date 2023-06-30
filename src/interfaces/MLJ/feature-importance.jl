function compute_featureimportance(model, var_grouping)
    feature_importance_by_count = MDT.variable_countmap(model)

    if !isnothing(var_grouping)
        feature_importance_by_count = Dict([i_var => var_grouping[i_modality][i_var] for ((i_modality, i_var), count) in feature_importance_by_count])
    end
    feature_importance_by_count
end
