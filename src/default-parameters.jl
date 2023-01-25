
default_max_depth = typemax(Int64)
default_min_samples_leaf = 1
default_min_purity_increase = -Inf
default_max_purity_at_leaf = Inf
default_n_trees = typemax(Int64)

# function parametrization_is_going_to_prune(pruning_params)
#     (haskey(pruning_params, :max_depth)           && pruning_params.max_depth            < default_max_depth) ||
#     # (haskey(pruning_params, :min_samples_leaf)    && pruning_params.min_samples_leaf     > default_min_samples_leaf) ||
#     (haskey(pruning_params, :min_purity_increase) && pruning_params.min_purity_increase  > default_min_purity_increase) ||
#     (haskey(pruning_params, :max_purity_at_leaf)  && pruning_params.max_purity_at_leaf   < default_max_purity_at_leaf) ||
#     (haskey(pruning_params, :n_trees)             && pruning_params.n_trees              < default_n_trees)
# end
