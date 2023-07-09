
function get_kwargs(m::SymbolicModel, X)
    return (;
        loss_function             = nothing,
        max_depth                 = m.max_depth,
        min_samples_leaf          = m.min_samples_leaf,
        min_purity_increase       = m.min_purity_increase,
        max_purity_at_leaf        = m.max_purity_at_leaf,
        max_modal_depth           = m.max_modal_depth,
        ####################################################################################
        n_subrelations            = identity,
        n_subfeatures             = m.n_subfeatures,
        initconditions            = readinitconditions(m, X),
        allow_global_splits       = ALLOW_GLOBAL_SPLITS,
        ####################################################################################
        use_minification          = false,
        perform_consistency_check = false,
        ####################################################################################
        rng                       = m.rng,
        print_progress            = m.print_progress,
    )
end

function MMI.clean!(model::SymbolicModel)
    warning = ""

    if !(isnothing(model.max_depth) || model.max_depth ≥ -1)
        warning *= "max_depth must be ≥ -1, but $(model.max_depth) " *
            "was provided. Defaulting to $(mlj_default_max_depth).\n"
        model.max_depth = mlj_default_max_depth
    end

    if !(isnothing(model.min_samples_leaf) || model.min_samples_leaf ≥ 1)
        warning *= "min_samples_leaf must be ≥ 1, but $(model.min_samples_leaf) " *
            "was provided. Defaulting to $(mlj_mdt_default_min_samples_leaf).\n"
        model.min_samples_leaf = mlj_mdt_default_min_samples_leaf
    end

    if !(isnothing(model.max_modal_depth) || model.max_modal_depth ≥ -1)
        warning *= "max_modal_depth must be ≥ -1, but $(model.max_modal_depth) " *
            "was provided. Defaulting to $(mlj_default_max_modal_depth).\n"
        model.max_modal_depth = mlj_default_max_depth
    end

    # Patch parameters: -1 -> nothing
    model.max_depth == -1 && (model.max_depth = nothing)
    model.max_modal_depth == -1 && (model.max_modal_depth = nothing)
    model.display_depth == -1 && (model.display_depth = nothing)

    # Patch parameters: nothing -> default value
    isnothing(model.max_depth)           && (model.max_depth           = mlj_default_max_depth)
    isnothing(model.min_samples_leaf)    && (model.min_samples_leaf    = mlj_mdt_default_min_samples_leaf)
    isnothing(model.min_purity_increase) && (model.min_purity_increase = mlj_mdt_default_min_purity_increase)
    isnothing(model.max_purity_at_leaf)  && (model.max_purity_at_leaf  = mlj_mdt_default_max_purity_at_leaf)
    isnothing(model.max_modal_depth)     && (model.max_modal_depth     = mlj_default_max_modal_depth)

    ########################################################################################
    ########################################################################################
    ########################################################################################

    if !(isnothing(model.relations) ||
        model.relations isa Symbol && model.relations in keys(AVAILABLE_RELATIONS) ||
        model.relations isa Vector{<:AbstractRelation} ||
        model.relations isa Function
    )
        warning *= "relations should be in $(collect(keys(AVAILABLE_RELATIONS))) " *
            "or a vector of SoleLogics.AbstractRelation's, " *
            "but $(model.relations) " *
            "was provided. Defaulting to $(mlj_default_relations_str).\n"
        model.relations = nothing
    end

    isnothing(model.relations)                      && (model.relations  = mlj_default_relations)
    model.relations isa Vector{<:AbstractRelation}  && (model.relations  = model.relations)

    if !(isnothing(model.conditions) ||
        model.conditions isa Vector{<:Union{SoleModels.VarFeature,Base.Callable}} ||
        model.conditions isa Vector{<:Tuple{Base.Callable,Integer}} ||
        model.conditions isa Vector{<:Tuple{TestOperator,<:Union{SoleModels.VarFeature,Base.Callable}}} ||
        model.conditions isa Vector{<:SoleModels.ScalarMetaCondition}
    )
        warning *= "conditions should be either:" *
            "a) a vector of features (i.e., callables to be associated to all variables, or SoleModels.VarFeature objects);\n" *
            "b) a vector of tuples (callable,var_id);\n" *
            "c) a vector of tuples (test_operator,features);\n" *
            "d) a vector of SoleModels.ScalarMetaCondition;\n" *
            "but $(model.conditions) " *
            "was provided. Defaulting to $(mlj_default_conditions_str).\n"
        model.conditions = nothing
    end

    isnothing(model.conditions) && (model.conditions  = mlj_default_conditions)

    if !(isnothing(model.initconditions) ||
        model.initconditions isa Symbol && model.initconditions in keys(AVAILABLE_INITCONDITIONS) ||
        model.initconditions isa InitialCondition
    )
        warning *= "initconditions should be in $(collect(keys(AVAILABLE_INITCONDITIONS))), " *
            "but $(model.initconditions) " *
            "was provided. Defaulting to $(mlj_default_initconditions_str).\n"
        model.initconditions = nothing
    end

    isnothing(model.initconditions) && (model.initconditions  = mlj_default_initconditions)

    ########################################################################################
    ########################################################################################
    ########################################################################################

    model.downsize = begin
        if model.downsize == true
            if model isa TreeModel
                tree_downsizing_function
            elseif model isa ForestModel
                forest_downsizing_function
            else
                error("Unexpected model type $(typeof(model)). Is it a tree or a forest?")
            end
        elseif model.downsize == false
            identity
        elseif model.downsize isa NTuple{N,Integer} where N
            make_downsizing_function(model.downsize)
        elseif model.downsize isa Function
            model.downsize
        else
            error("Unexpected value for `downsize` encountered: $(model.downsize)")
        end
    end

    if model.rng isa Integer
        model.rng = Random.MersenneTwister(model.rng)
    end

    ########################################################################################
    ########################################################################################
    ########################################################################################

    if !(isnothing(model.min_samples_split) || model.min_samples_split ≥ 2)
        warning *= "min_samples_split must be ≥ 2, but $(model.min_samples_split) " *
            "was provided. Defaulting to $(nothing).\n"
        model.min_samples_split = nothing
    end

    # Note:
    # (min_samples_leaf * 2 >  ninstances) || (min_samples_split >  ninstances)   <=>
    # (max(min_samples_leaf * 2, min_samples_split) >  ninstances)                <=>
    # (max(min_samples_leaf, div(min_samples_split, 2)) * 2 >  ninstances)

    if !isnothing(model.min_samples_split)
        model.min_samples_leaf = max(model.min_samples_leaf, div(model.min_samples_split, 2))
    end

    if model.n_subfeatures isa Integer && !(model.n_subfeatures > 0)
        warning *= "n_subfeatures must be > 0, but $(model.n_subfeatures) " *
            "was provided. Defaulting to $(nothing).\n"
        model.n_subfeatures = nothing
    end

    model.n_subfeatures == -1 && (model.n_subfeatures = nothing)

    model.n_subfeatures = begin
        if isnothing(model.n_subfeatures)
            identity
        elseif model.n_subfeatures isa Integer
            warning *= "An absolute n_subfeatures was provided $(model.n_subfeatures). " *
                "It is recommended to use relative values (between 0 and 1), interpreted " *
                " as the share of the random portion of feature space explored at each split."
            x -> convert(Int64, model.n_subfeatures)
        elseif model.n_subfeatures isa AbstractFloat && 0 ≤ model.n_subfeatures ≤ 1
            x -> ceil(Int64, x*model.n_subfeatures)
        elseif model.n_subfeatures isa Function
            # x -> ceil(Int64, model.n_subfeatures(x)) # Generates too much nesting
            model.n_subfeatures
        else
            error("Unexpected value for n_subfeatures: $(n_subfeatures) " *
                "(type: $(typeof(n_subfeatures)))")
        end
    end

    # Only true for classification:
    # if !(0 ≤ model.merge_purity_threshold ≤ 1)
    #     warning *= "merge_purity_threshold should be between 0 and 1, " *
    #         "but $(model.merge_purity_threshold) " *
    #         "was provided.\n"
    # end

    if model.feature_importance == :impurity
        error("feature_importance = :impurity is currently not supported." *
            "Defaulting to $(:split).\n")
        model.feature_importance == :split
    end

    if !(model.feature_importance in [:split])
        warning *= "feature_importance should be in [:split], " *
            "but $(model.feature_importance) " *
            "was provided.\n"
    end

    return warning
end
