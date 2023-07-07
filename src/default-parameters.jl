
DEFAULT_PERFORM_CONSISTENCY_CHECK = false

BOTTOM_MAX_DEPTH = typemax(Int64)
BOTTOM_MIN_SAMPLES_LEAF = 1
BOTTOM_MIN_PURITY_INCREASE = -Inf
BOTTOM_MAX_PURITY_AT_LEAF = Inf
BOTTOM_NTREES = typemax(Int64)

BOTTOM_MAX_PERFORMANCE_AT_SPLIT = Inf
BOTTOM_MIN_PERFORMANCE_AT_SPLIT = -Inf

BOTTOM_MAX_MODAL_DEPTH = typemax(Int64)

# function parametrization_is_going_to_prune(pruning_params)
#     (haskey(pruning_params, :max_depth)           && pruning_params.max_depth            < BOTTOM_MAX_DEPTH) ||
#     # (haskey(pruning_params, :min_samples_leaf)    && pruning_params.min_samples_leaf     > BOTTOM_MIN_SAMPLES_LEAF) ||
#     (haskey(pruning_params, :min_purity_increase) && pruning_params.min_purity_increase  > BOTTOM_MIN_PURITY_INCREASE) ||
#     (haskey(pruning_params, :max_purity_at_leaf)  && pruning_params.max_purity_at_leaf   < BOTTOM_MAX_PURITY_AT_LEAF) ||
#     (haskey(pruning_params, :ntrees)             && pruning_params.ntrees              < BOTTOM_NTREES)
# end
