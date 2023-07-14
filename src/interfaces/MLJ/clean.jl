
function get_kwargs(m::SymbolicModel, X)
    base_kwargs = (;
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

    additional_kwargs = begin
        if m isa TreeModel
            (;)
        elseif m isa ForestModel
            (;
                partial_sampling = m.sampling_fraction,
                ntrees = m.ntrees,
                suppress_parity_warning = true,
            )
        else
            error("Unexpected model type: $(typeof(m))")
        end
    end
    merge(base_kwargs, additional_kwargs)
end

function MMI.clean!(m::SymbolicModel)
    warning = ""

    if m isa TreeModel
        mlj_default_min_samples_leaf = mlj_mdt_default_min_samples_leaf
        mlj_default_min_purity_increase = mlj_mdt_default_min_purity_increase
        mlj_default_max_purity_at_leaf = mlj_mdt_default_max_purity_at_leaf
        mlj_default_n_subfeatures = mlj_mdt_default_n_subfeatures
    elseif m isa ForestModel
        mlj_default_min_samples_leaf = mlj_mrf_default_min_samples_leaf
        mlj_default_min_purity_increase = mlj_mrf_default_min_purity_increase
        mlj_default_max_purity_at_leaf = mlj_mrf_default_max_purity_at_leaf
        mlj_default_n_subfeatures = mlj_mrf_default_n_subfeatures
        mlj_default_ntrees = mlj_mrf_default_ntrees
        mlj_default_sampling_fraction = mlj_mrf_default_sampling_fraction
    else
        error("Unexpected model type: $(typeof(m))")
    end

    if !(isnothing(m.max_depth) || m.max_depth ≥ -1)
        warning *= "max_depth must be ≥ -1, but $(m.max_depth) " *
            "was provided. Defaulting to $(mlj_default_max_depth).\n"
        m.max_depth = mlj_default_max_depth
    end

    if !(isnothing(m.min_samples_leaf) || m.min_samples_leaf ≥ 1)
        warning *= "min_samples_leaf must be ≥ 1, but $(m.min_samples_leaf) " *
            "was provided. Defaulting to $(mlj_default_min_samples_leaf).\n"
        m.min_samples_leaf = mlj_default_min_samples_leaf
    end

    if !(isnothing(m.max_modal_depth) || m.max_modal_depth ≥ -1)
        warning *= "max_modal_depth must be ≥ -1, but $(m.max_modal_depth) " *
            "was provided. Defaulting to $(mlj_default_max_modal_depth).\n"
        m.max_modal_depth = mlj_default_max_depth
    end

    # Patch parameters: -1 -> nothing
    m.max_depth == -1 && (m.max_depth = nothing)
    m.max_modal_depth == -1 && (m.max_modal_depth = nothing)
    m.display_depth == -1 && (m.display_depth = nothing)

    # Patch parameters: nothing -> default value
    isnothing(m.max_depth)           && (m.max_depth           = mlj_default_max_depth)
    isnothing(m.min_samples_leaf)    && (m.min_samples_leaf    = mlj_default_min_samples_leaf)
    isnothing(m.min_purity_increase) && (m.min_purity_increase = mlj_default_min_purity_increase)
    isnothing(m.max_purity_at_leaf)  && (m.max_purity_at_leaf  = mlj_default_max_purity_at_leaf)
    isnothing(m.max_modal_depth)     && (m.max_modal_depth     = mlj_default_max_modal_depth)

    ########################################################################################
    ########################################################################################
    ########################################################################################

    if !(isnothing(m.relations) ||
        m.relations isa Symbol && m.relations in keys(AVAILABLE_RELATIONS) ||
        m.relations isa Vector{<:AbstractRelation} ||
        m.relations isa Function
    )
        warning *= "relations should be in $(collect(keys(AVAILABLE_RELATIONS))) " *
            "or a vector of SoleLogics.AbstractRelation's, " *
            "but $(m.relations) " *
            "was provided. Defaulting to $(mlj_default_relations_str).\n"
        m.relations = nothing
    end

    isnothing(m.relations)                      && (m.relations  = mlj_default_relations)
    m.relations isa Vector{<:AbstractRelation}  && (m.relations  = m.relations)

    if !(isnothing(m.conditions) ||
        m.conditions isa Vector{<:Union{SoleModels.VarFeature,Base.Callable}} ||
        m.conditions isa Vector{<:Tuple{Base.Callable,Integer}} ||
        m.conditions isa Vector{<:Tuple{TestOperator,<:Union{SoleModels.VarFeature,Base.Callable}}} ||
        m.conditions isa Vector{<:SoleModels.ScalarMetaCondition}
    )
        warning *= "conditions should be either:" *
            "a) a vector of features (i.e., callables to be associated to all variables, or SoleModels.VarFeature objects);\n" *
            "b) a vector of tuples (callable,var_id);\n" *
            "c) a vector of tuples (test_operator,features);\n" *
            "d) a vector of SoleModels.ScalarMetaCondition;\n" *
            "but $(m.conditions) " *
            "was provided. Defaulting to $(mlj_default_conditions_str).\n"
        m.conditions = nothing
    end

    isnothing(m.conditions) && (m.conditions  = mlj_default_conditions)

    if !(isnothing(m.initconditions) ||
        m.initconditions isa Symbol && m.initconditions in keys(AVAILABLE_INITCONDITIONS) ||
        m.initconditions isa InitialCondition
    )
        warning *= "initconditions should be in $(collect(keys(AVAILABLE_INITCONDITIONS))), " *
            "but $(m.initconditions) " *
            "was provided. Defaulting to $(mlj_default_initconditions_str).\n"
        m.initconditions = nothing
    end

    isnothing(m.initconditions) && (m.initconditions  = mlj_default_initconditions)

    ########################################################################################
    ########################################################################################
    ########################################################################################

    m.downsize = begin
        if m.downsize == true
            make_downsizing_function(m)
        elseif m.downsize == false
            identity
        elseif m.downsize isa NTuple{N,Integer} where N
            make_downsizing_function(m.downsize)
        elseif m.downsize isa Function
            m.downsize
        else
            error("Unexpected value for `downsize` encountered: $(m.downsize)")
        end
    end

    if m.rng isa Integer
        m.rng = Random.MersenneTwister(m.rng)
    end

    ########################################################################################
    ########################################################################################
    ########################################################################################

    if !(isnothing(m.min_samples_split) || m.min_samples_split ≥ 2)
        warning *= "min_samples_split must be ≥ 2, but $(m.min_samples_split) " *
            "was provided. Defaulting to $(nothing).\n"
        m.min_samples_split = nothing
    end

    # Note:
    # (min_samples_leaf * 2 >  ninstances) || (min_samples_split >  ninstances)   ⇔
    # (max(min_samples_leaf * 2, min_samples_split) >  ninstances)                ⇔
    # (max(min_samples_leaf, div(min_samples_split, 2)) * 2 >  ninstances)

    if !isnothing(m.min_samples_split)
        m.min_samples_leaf = max(m.min_samples_leaf, div(m.min_samples_split, 2))
    end

    if m.n_subfeatures isa Integer && !(m.n_subfeatures > 0)
        warning *= "n_subfeatures must be > 0, but $(m.n_subfeatures) " *
            "was provided. Defaulting to $(nothing).\n"
        m.n_subfeatures = nothing
    end

    # Legacy behaviour
    m.n_subfeatures == -1 && (m.n_subfeatures = sqrt_f)
    m.n_subfeatures == 0 && (m.n_subfeatures = identity)

    function make_n_subfeatures_function(n_subfeatures)
        if isnothing(n_subfeatures)
            mlj_default_n_subfeatures
        elseif n_subfeatures isa Integer
            warning *= "An absolute n_subfeatures was provided $(n_subfeatures). " *
                "It is recommended to use relative values (between 0 and 1), interpreted " *
                "as the share of the random portion of feature space explored at each split."
            x -> convert(Int64, n_subfeatures)
        elseif n_subfeatures isa AbstractFloat
            @assert 0 ≤ n_subfeatures ≤ 1 "Unexpected value for " *
                "n_subfeatures: $(n_subfeatures). It should be ∈ [0,1]"
            x -> ceil(Int64, x*n_subfeatures)
        elseif n_subfeatures isa Function
            # x -> ceil(Int64, n_subfeatures(x)) # Generates too much nesting
            n_subfeatures
        else
            error("Unexpected value for n_subfeatures: $(n_subfeatures) " *
                "(type: $(typeof(n_subfeatures)))")
        end
    end

    m.n_subfeatures = make_n_subfeatures_function(m.n_subfeatures)

    # Only true for classification:
    # if !(0 ≤ m.merge_purity_threshold ≤ 1)
    #     warning *= "merge_purity_threshold should be between 0 and 1, " *
    #         "but $(m.merge_purity_threshold) " *
    #         "was provided.\n"
    # end

    if m.feature_importance == :impurity
        warning *= "feature_importance = :impurity is currently not supported." *
            "Defaulting to $(:split).\n"
        m.feature_importance == :split
    end

    if !(m.feature_importance in [:split])
        warning *= "feature_importance should be in [:split], " *
            "but $(m.feature_importance) " *
            "was provided.\n"
    end

    if m isa ForestModel

        isnothing(m.sampling_fraction) && (m.sampling_fraction  = mlj_default_sampling_fraction)

        if !(0 ≤ m.sampling_fraction ≤ 1)
            warning *= "sampling_fraction should be ∈ [0,1], " *
                "but $(m.sampling_fraction) " *
                "was provided.\n"
        end

        isnothing(m.ntrees) && (m.ntrees  = mlj_default_ntrees)

        if !(m.ntrees > 0)
            warning *= "ntrees should be > 0, " *
                "but $(m.ntrees) " *
                "was provided.\n"
        end

    end

    return warning
end
