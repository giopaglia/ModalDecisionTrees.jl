# Inspired from JuliaAI/MLJDecisionTreeInterface.jl

module MLJInterface

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
export depth

include("MLJ/default-parameters.jl")
include("MLJ/sanity-checks.jl")
include("MLJ/downsize.jl")
include("MLJ/printer.jl")
include("MLJ/wrapdataset.jl")
include("MLJ/feature-importance.jl")

############################################################################################
############################################################################################
############################################################################################

# DecisionTree.jl (https://github.com/JuliaAI/DecisionTree.jl) is the main package
#  for decision tree learning in Julia. These definitions allow for ModalDecisionTrees.jl
#  to act as a drop-in replacement for DecisionTree.jl. Well, more or less.

depth(t::MDT.DTree) = height(t)

############################################################################################
############################################################################################
############################################################################################

mutable struct ModalDecisionTree <: MMI.Probabilistic

    ## Pruning conditions
    max_depth              :: Union{Nothing,Int}
    min_samples_leaf       :: Union{Nothing,Int}
    min_purity_increase    :: Union{Nothing,Float64}
    max_purity_at_leaf     :: Union{Nothing,Float64}

    max_modal_depth        :: Union{Nothing,Int}

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
    # Type for the extracted feature values
    featvaltype            :: Type

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
    rng                    :: Union{Random.AbstractRNG,Integer}

    ## DecisionTree.jl parameters
    display_depth          :: Union{Nothing,Int}
    min_samples_split      :: Union{Nothing,Int}
    n_subfeatures          :: Union{Nothing,Int,Float64,Function}
    post_prune             :: Bool
    merge_purity_threshold :: Float64
    feature_importance     :: Symbol
end

# keyword constructor
function ModalDecisionTree(;
    max_depth = nothing,
    min_samples_leaf = nothing,
    min_purity_increase = nothing,
    max_purity_at_leaf = nothing,
    max_modal_depth = nothing,
    #
    relations = nothing,
    conditions = nothing,
    featvaltype = Float64,
    initconditions = nothing,
    #
    downsize = true,
    check_conditions = true,
    print_progress = false,
    rng = Random.GLOBAL_RNG,
    #
    display_depth = 5,
    min_samples_split = nothing,
    n_subfeatures = nothing,
    post_prune = false,
    merge_purity_threshold = 1.0,
    feature_importance = :split,
)
    model = ModalDecisionTree(
        max_depth,
        min_samples_leaf,
        min_purity_increase,
        max_purity_at_leaf,
        max_modal_depth,
        #
        relations,
        conditions,
        featvaltype,
        initconditions,
        #
        downsize,
        check_conditions,
        print_progress,
        rng,
        #
        display_depth,
        min_samples_split,
        n_subfeatures,
        post_prune,
        merge_purity_threshold,
        feature_importance,
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end


############################################################################################
############################################################################################
############################################################################################

function MMI.fit(m::ModalDecisionTree, verbosity::Integer, X, y, var_grouping, classes_seen=nothing, w=nothing)

    model = MDT.build_tree(
        X,
        y,
        w;
        ####################################################################################
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

    if m.post_prune
        model = MDT.prune(model; max_performance_at_split = m.merge_purity_threshold)
    end

    verbosity < 2 || MDT.printmodel(model; max_depth = m.display_depth, variable_names_map = var_grouping)

    solemodel = ModalDecisionTrees.translate(model, (;
        # syntaxstring_kwargs = (; hidemodality = (length(var_grouping) == 1), variable_names_map = var_grouping)
    ))

    fitresult = (
        model         = model,
        var_grouping  = var_grouping,
    )

    cache  = nothing
    report = (
        printmodel                  = ModelPrinter(m, model, solemodel, var_grouping),
        printapply                  = (Xnew, ynew)->begin
            (Xnew, ynew, var_grouping, classes_seen, w) = MMI.reformat(m, Xnew, ynew)
            print_apply(model, Xnew, ynew)
        end,
        solemodel                   = solemodel,
        var_grouping                = var_grouping,
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

MMI.fitted_params(::ModalDecisionTree, fitresult) = merge(fitresult, (; tree = fitresult.model))

############################################################################################
############################################################################################
############################################################################################

const SymbolicModel = Union{
    ModalDecisionTree
}

const TreeModel = Union{
    ModalDecisionTree
}

const ForestModel = Union{

}

include("MLJ/parameters.jl")

function MMI.predict(m::SymbolicModel, fitresult, Xnew, var_grouping = nothing)
    if !isnothing(var_grouping) && var_grouping != fitresult.var_grouping
        @warn "variable grouping differs from the one used in training! " *
            "training var_grouping: $(fitresult.var_grouping)" *
            "var_grouping = $(var_grouping)" *
            "\n"
    end
    MDT.apply_proba(fitresult.model, Xnew, get(fitresult, :classes_seen, nothing))
end

############################################################################################
# DATA FRONT END
############################################################################################

function MMI.reformat(m::SymbolicModel, X, y, w = nothing)
    X, var_grouping = wrapdataset(X, m)
    y, classes_seen = fix_y(y)
    (X, y, var_grouping, classes_seen, w)
end

MMI.selectrows(::SymbolicModel, I, X, y, var_grouping, classes_seen, w = nothing) =
    (MMI.selectrows(X, I), MMI.selectrows(y, I), var_grouping, classes_seen, MMI.selectrows(w, I),)

# For predict
function MMI.reformat(m::SymbolicModel, Xnew)
    Xnew, var_grouping = wrapdataset(Xnew, m)
    (Xnew, var_grouping)
end
MMI.selectrows(::SymbolicModel, I, Xnew, var_grouping) =
    (MMI.selectrows(Xnew, I), var_grouping,)

# MMI.fitted_params(::SymbolicModel, fitresult) = fitresult

############################################################################################
# FEATURE IMPORTANCES
############################################################################################

MMI.reports_feature_importances(::Type{<:SymbolicModel}) = true

function MMI.feature_importances(m::SymbolicModel, fitresult, report)
    # generate feature importances for report
    if !(m.feature_importance == :split)
        error("Unexpected feature_importance encountered: $(m.feature_importance).")
    end

    featimportance_dict = compute_featureimportance(fitresult.model, fitresult.var_grouping; normalize=true)
    featimportance_vec = collect(featimportance_dict)
    sort!(featimportance_vec, rev=true, by=x->last(x))

    return featimportance_vec
end

############################################################################################
# METADATA (MODEL TRAITS)
############################################################################################

MMI.metadata_pkg.(
    (
        ModalDecisionTree,
        # ModalRandomForest,
        # DecisionTreeRegressor,
        # RandomForestRegressor,
        # AdaBoostStumpClassifier,
    ),
    name = "$(MDT)",
    package_uuid = "e54bda2e-c571-11ec-9d64-0242ac120002",
    package_url = repo_url,
    is_pure_julia = true,
    is_wrapper=false,
    package_license = "MIT",
)

MMI.metadata_model(
    ModalDecisionTree,
    input_scitype = Union{
        Table(
            Continuous,     AbstractVector{<:Continuous},    AbstractMatrix{<:Continuous},
            Count,          AbstractVector{<:Count},         AbstractMatrix{<:Count},
            OrderedFactor,  AbstractVector{<:OrderedFactor}, AbstractMatrix{<:OrderedFactor},
        ),
        # AbstractArray{Continuous,2},     AbstractArray{Continuous,3},    AbstractArray{Continuous,4},
        # AbstractArray{Count,2},          AbstractArray{Count,3},         AbstractArray{Count,4},
        # AbstractArray{OrderedFactor,2},  AbstractArray{OrderedFactor,3}, AbstractArray{OrderedFactor,4},
    },
    target_scitype = Union{AbstractVector{<:Continuous},AbstractVector{<:Finite},AbstractVector{<:Textual}},
    human_name = "Modal Decision Tree",
    descr   = "A Modal Decision Tree is a probabilistic, symbolic model " *
        "for classification and regression tasks with dimensional data " *
        "(e.g., images and time-series)." *
        "The model is able to extract logical descriptions of the data "
        "in terms of logical formulas (see SoleLogics.jl) on propositions that are, "
        "for example, min[V2] ≥ 10, that is, \"the minimum of variable 2 is not less than 10\"."
        "As such, the model offers an interesting level of interpretability." *
        ""
        ,
    supports_weights = true,
    reports_feature_importances=true,
    load_path = "$MDT.ModalDecisionTree",
)

############################################################################################
############################################################################################
############################################################################################

end
using .MLJInterface
