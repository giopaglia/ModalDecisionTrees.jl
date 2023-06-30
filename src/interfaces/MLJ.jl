module MLJInterface

using MLJBase
using MLJModelInterface
using MLJModelInterface.ScientificTypesBase
using CategoricalArrays
using DataFrames
using DataStructures
using Tables
using Random
using Random: GLOBAL_RNG
# using Distributions: Normal

using SoleLogics
using SoleLogics: AbstractRelation
using SoleData
using SoleModels
using SoleModels.MLJUtils
using SoleModels: TestOperator

using ModalDecisionTrees
using ModalDecisionTrees: InitialCondition


const MMI = MLJModelInterface
const MDT = ModalDecisionTrees

const repo_url = "https://github.com/giopaglia/$(MDT).jl"

export ModalDecisionTree

include("MLJ/default-parameters.jl")
include("MLJ/sanity-checks.jl")
include("MLJ/downsize.jl")
include("MLJ/printer.jl")
include("MLJ/wrapdataset.jl")
include("MLJ/feature-importance.jl")

############################################################################################
############################################################################################
############################################################################################

mutable struct ModalDecisionTree <: MMI.Deterministic

    ## Pruning conditions
    max_depth              :: Union{Nothing,Int}
    min_samples_leaf       :: Union{Nothing,Int}
    min_purity_increase    :: Union{Nothing,Float64}
    max_purity_at_leaf     :: Union{Nothing,Float64}

    ## Logic parameters

    # Relation set
    relations              :: Union{
        Nothing,                                            # defaults to a well-known relation set, depending on the data;
        Symbol,                                             # one of the relation sets specified in AVAILABLE_RELATIONS;
        Vector{<:AbstractRelation},                         # explicitly specify the relation set;
        # Vector{<:Union{Symbol,Vector{<:AbstractRelation}}}, # MULTIMODAL CASE: specify a relation set for each modality;
        Function                                            # A function worldtype -> relation set.
    }

    # Condition set
    conditions             :: Union{
        Nothing,                                                                     # defaults to scalar conditions (with ≥ and <) on well-known feature functions (e.g., minimum, maximum), applied to all variables;
        Vector{<:Union{SoleModels.VarFeature,Base.Callable}},                        # scalar conditions with ≥ and <, on an explicitly specified feature set (callables to be applied to each variable, or VarFeature objects);
        Vector{<:Tuple{Base.Callable,Integer}},                                      # scalar conditions with ≥ and <, on a set of features specified as a set of callables to be applied to a set of variables each;
        Vector{<:Tuple{TestOperator,<:Union{SoleModels.VarFeature,Base.Callable}}},  # explicitly specify the pairs (test operator, feature);
        Vector{<:SoleModels.ScalarMetaCondition},                                    # explicitly specify the scalar condition set.
    }

    # Initial conditions
    initconditions         :: Union{
        Nothing,                                                                     # defaults to standard conditions (e.g., start_without_world)
        Symbol,                                                                      # one of the initial conditions specified in AVAILABLE_INITIALCONDITIONS;
        InitialCondition,                                                            # explicitly specify an initial condition for the learning algorithm.
    }

    ## Miscellaneous
    downsize               :: Union{Bool,NTuple{N,Integer} where N,Function}
    check_conditions       :: Bool
    print_progress         :: Bool
    rng                    :: Random.AbstractRNG

    ## DecisionTree-specific
    display_depth          :: Union{Nothing,Int}
end

function MMI.clean!(model::ModalDecisionTree)
    warning = ""

    if !(isnothing(model.max_depth) || model.max_depth ≥ -1)
        warning *= "max_depth must be ≥ -1, but $(model.max_depth) " *
            "was provided. Defaulting to $(mlj_default_max_depth).\n"
        model.max_depth = mlj_default_max_depth
    end

    if !(isnothing(model.min_samples_leaf) || model.min_samples_leaf ≥ 1)
        warning *= "min_samples_leaf must be ≥ -1, but $(model.min_samples_leaf) " *
            "was provided. Defaulting to $(mlj_mdt_default_min_samples_leaf).\n"
        model.min_samples_leaf = mlj_mdt_default_min_samples_leaf
    end

    # Patch parameters: -1 -> nothing
    model.max_depth == -1 && (model.max_depth = nothing)
    model.display_depth == -1 && (model.display_depth = nothing)

    # Patch parameters: nothing -> default value
    isnothing(model.max_depth)           && (model.max_depth           = mlj_default_max_depth)
    isnothing(model.min_samples_leaf)    && (model.min_samples_leaf    = mlj_mdt_default_min_samples_leaf)
    isnothing(model.min_purity_increase) && (model.min_purity_increase = mlj_mdt_default_min_purity_increase)
    isnothing(model.max_purity_at_leaf)  && (model.max_purity_at_leaf  = mlj_mdt_default_max_purity_at_leaf)

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
    model.relations isa Symbol                      && (model.relations  = AVAILABLE_RELATIONS[model.relations])
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
    model.initconditions isa Symbol && (model.initconditions  = AVAILABLE_INITCONDITIONS[model.initconditions])

    ########################################################################################
    ########################################################################################
    ########################################################################################

    if model.check_conditions == true
        check_conditions(model.conditions)
    end

    if model.downsize == true
        model.downsize = tree_downsizing_function
    elseif model.downsize == false
        model.downsize = identity
    elseif model.downsize isa NTuple{N,Integer} where N
        model.downsize = X->moving_average(X, model.downsize)
    end

    return warning
end

# keyword constructor
function ModalDecisionTree(;
    max_depth = nothing,
    min_samples_leaf = nothing,
    min_purity_increase = nothing,
    max_purity_at_leaf = nothing,
    relations = nothing,
    conditions = nothing,
    initconditions = nothing,
    #
    downsize = true,
    check_conditions = true,
    print_progress = false,
    rng = Random.GLOBAL_RNG,
    #
    display_depth = nothing,
)
    model = ModalDecisionTree(
        max_depth,
        min_samples_leaf,
        min_purity_increase,
        max_purity_at_leaf,
        relations,
        conditions,
        initconditions,
        downsize,
        check_conditions,
        print_progress,
        rng,
        display_depth,
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end


############################################################################################
############################################################################################
############################################################################################

function MMI.fit(m::ModalDecisionTree, verbosity::Integer, X, y, w=nothing)
    X, var_grouping = wrapdataset(X, m)
    y, classes_seen = fix_y(y)

    ########################################################################################

    model = MDT.build_tree(
        X,
        y,
        w,
        ####################################################################################
        loss_function        = nothing,
        max_depth            = m.max_depth,
        min_samples_leaf     = m.min_samples_leaf,
        min_purity_increase  = m.min_purity_increase,
        max_purity_at_leaf   = m.max_purity_at_leaf,
        ####################################################################################
        n_subrelations       = identity,
        n_subfeatures        = identity,
        initconditions       = m.initconditions,
        allow_global_splits  = ALLOW_GLOBAL_SPLITS,
        ####################################################################################
        perform_consistency_check = false,
    )

    verbosity < 2 || MDT.printmodel(model; max_depth = m.display_depth, variable_names_map = var_grouping)

    solemodel = ModalDecisionTrees.translate(model)

    # Compute feature importance
    feature_importance_by_count = compute_featureimportance(model, var_grouping)

    fitresult = (
        model         = model,
        var_grouping  = var_grouping,
    )

    cache  = nothing
    report = (
        printmodel                  = ModelPrinter(model, solemodel, var_grouping),
        solemodel                   = solemodel,
        var_grouping                = var_grouping,
        feature_importance_by_count = feature_importance_by_count,
    )

    if !isnothing(classes_seen)
        report = merge(report, (;
            classes_seen    = classes_seen,
        ))
        fitresult = merge(fitresult, (;
            classes_seen    = classes_seen,
        ))
    end

    return fitresult, cache, report
end

function MMI.predict(m::ModalDecisionTree, fitresult, Xnew, ynew = nothing)
    Xnew, var_grouping = wrapdataset(Xnew, m, fitresult.var_grouping)
    ynew, classes_seen = fix_y(ynew)

    if isnothing(ynew)
        MDT.apply_model(fitresult.model, Xnew)
    else
        MDT.print_apply(fitresult.model, Xnew, ynew)
    end
end

############################################################################################
############################################################################################
############################################################################################

MMI.fitted_params(::ModalDecisionTree, fitresult) =
    (
        model           = fitresult.model,
        var_grouping  = fitresult.var_grouping,
    )

############################################################################################
############################################################################################
############################################################################################

end
using .MLJInterface
