
DEFAULT_PERFORM_CONSISTENCY_CHECK = false

DEFAULT_MAX_DEPTH = typemax(Int64)
DEFAULT_MIN_SAMPLES_LEAF = 1
DEFAULT_MIN_PURITY_INCREASE = -Inf
DEFAULT_MAX_PURITY_AT_LEAF = Inf
DEFAULT_NTREES = typemax(Int64)

# function parametrization_is_going_to_prune(pruning_params)
#     (haskey(pruning_params, :max_depth)           && pruning_params.max_depth            < DEFAULT_MAX_DEPTH) ||
#     # (haskey(pruning_params, :min_samples_leaf)    && pruning_params.min_samples_leaf     > DEFAULT_MIN_SAMPLES_LEAF) ||
#     (haskey(pruning_params, :min_purity_increase) && pruning_params.min_purity_increase  > DEFAULT_MIN_PURITY_INCREASE) ||
#     (haskey(pruning_params, :max_purity_at_leaf)  && pruning_params.max_purity_at_leaf   < DEFAULT_MAX_PURITY_AT_LEAF) ||
#     (haskey(pruning_params, :ntrees)             && pruning_params.ntrees              < DEFAULT_NTREES)
# end
