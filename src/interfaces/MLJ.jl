# Inspired from JuliaAI/MLJDecisionTreeInterface.jl

module MLJInterface

export ModalDecisionTree, ModalRandomForest
export depth, wrapdataset

using MLJModelInterface
using MLJModelInterface.ScientificTypesBase
using CategoricalArrays
using DataFrames
using DataStructures
using Tables
using Random
using Random: GLOBAL_RNG

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

const _package_url = "https://github.com/giopaglia/$(MDT).jl"

include("MLJ/default-parameters.jl")
include("MLJ/sanity-checks.jl")
include("MLJ/printer.jl")
include("MLJ/wrapdataset.jl")
include("MLJ/feature-importance.jl")

include("MLJ/ModalDecisionTree.jl")
include("MLJ/ModalRandomForest.jl")

include("MLJ/docstrings.jl")

const SymbolicModel = Union{
    ModalDecisionTree,
    ModalRandomForest,
}

const TreeModel = Union{
    ModalDecisionTree,
}

const ForestModel = Union{
    ModalRandomForest,
}

include("MLJ/downsize.jl")
include("MLJ/clean.jl")

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

function MMI.fit(m::SymbolicModel, verbosity::Integer, X, y, var_grouping, classes_seen=nothing, w=nothing)
    # @show get_kwargs(m, X)
    model = begin
        if m isa ModalDecisionTree
            MDT.build_tree(X, y, w; get_kwargs(m, X)...)
        elseif m isa ModalRandomForest
            MDT.build_forest(X, y, w; get_kwargs(m, X)...)
        else
            error("Unexpected model type: $(typeof(m))")
        end
    end

    if m.post_prune
        merge_purity_threshold = m.merge_purity_threshold
        if isnothing(merge_purity_threshold)
            if !isnothing(classes_seen)
                merge_purity_threshold = 1.0
            else
                error("Please, provide a `merge_purity_threshold` parameter (maximum MAE at splits).")
            end
        end
        model = MDT.prune(model; simplify = true, max_performance_at_split = merge_purity_threshold)
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
        sprinkle                    = (Xnew, ynew)->begin
            (Xnew, ynew, var_grouping, classes_seen, w) = MMI.reformat(m, Xnew, ynew; passive_mode = true)
            ModalDecisionTrees.sprinkle(model, Xnew, ynew)
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

MMI.fitted_params(::TreeModel, fitresult) = merge(fitresult, (; tree = fitresult.model))
MMI.fitted_params(::ForestModel, fitresult) = merge(fitresult, (; forest = fitresult.model))

############################################################################################
############################################################################################
############################################################################################

function MMI.predict(m::SymbolicModel, fitresult, Xnew, var_grouping = nothing)
    if !isnothing(var_grouping) && var_grouping != fitresult.var_grouping
        @warn "variable grouping differs from the one used in training! " *
            "training var_grouping: $(fitresult.var_grouping)" *
            "var_grouping = $(var_grouping)" *
            "\n"
    end
    MDT.apply_proba(fitresult.model, Xnew, get(fitresult, :classes_seen, nothing); suppress_parity_warning = true)
end

############################################################################################
# DATA FRONT END
############################################################################################

function MMI.reformat(m::SymbolicModel, X, y, w = nothing; passive_mode = false)
    X, var_grouping = wrapdataset(X, m; passive_mode = passive_mode)
    y, classes_seen = fix_y(y)
    (X, y, var_grouping, classes_seen, w)
end

MMI.selectrows(::SymbolicModel, I, X, y, var_grouping, classes_seen, w = nothing) =
    (MMI.selectrows(X, I), MMI.selectrows(y, I), var_grouping, classes_seen, MMI.selectrows(w, I),)

# For predict
function MMI.reformat(m::SymbolicModel, Xnew)
    Xnew, var_grouping = wrapdataset(Xnew, m; passive_mode = true)
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
        ModalRandomForest,
        # DecisionTreeRegressor,
        # RandomForestRegressor,
        # AdaBoostStumpClassifier,
    ),
    name = "$(MDT)",
    package_uuid = "e54bda2e-c571-11ec-9d64-0242ac120002",
    package_url = _package_url,
    is_pure_julia = true,
    is_wrapper=false,
    package_license = "MIT",
)

for (model, human_name) in [
    (ModalDecisionTree, "Modal Decision Tree"),
    (ModalRandomForest, "Modal Random Forest"),
]
    MMI.metadata_model(
        model,
        input_scitype = Union{
            Table(
                Continuous,     AbstractArray{<:Continuous,0},    AbstractArray{<:Continuous,1},    AbstractArray{<:Continuous,2},
                Count,          AbstractArray{<:Count,0},         AbstractArray{<:Count,1},         AbstractArray{<:Count,2},
                OrderedFactor,  AbstractArray{<:OrderedFactor,0}, AbstractArray{<:OrderedFactor,1}, AbstractArray{<:OrderedFactor,2},
            ),
        },
        target_scitype = Union{
            AbstractVector{<:Continuous},
            AbstractVector{<:Count},
            AbstractVector{<:Finite},
            AbstractVector{<:Textual}
        },
        human_name = human_name,
        supports_weights = true,
        load_path = "$MDT.$(model)",
    )
end

end

using .MLJInterface
