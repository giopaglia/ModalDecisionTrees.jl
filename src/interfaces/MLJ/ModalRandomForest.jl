mutable struct ModalRandomForest <: MMI.Probabilistic

    sampling_fraction      :: Float64
    ntrees                 :: Int

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
    print_progress         :: Bool
    rng                    :: Union{Random.AbstractRNG,Integer}

    ## DecisionTree.jl parameters
    display_depth          :: Union{Nothing,Int}
    min_samples_split      :: Union{Nothing,Int}
    n_subfeatures          :: Union{Nothing,Int,Float64,Function}
    post_prune             :: Bool
    merge_purity_threshold :: Union{Nothing,Float64}
    feature_importance     :: Symbol
end

# keyword constructor
function ModalRandomForest(;
    sampling_fraction = 0.7,
    ntrees = 10,
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
    print_progress = (ntrees > 50),
    rng = Random.GLOBAL_RNG,
    #
    display_depth = nothing,
    min_samples_split = nothing,
    n_subfeatures = nothing,
    post_prune = false,
    merge_purity_threshold = nothing,
    feature_importance = :split,
)
    model = ModalRandomForest(
        sampling_fraction,
        ntrees,
        #
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
