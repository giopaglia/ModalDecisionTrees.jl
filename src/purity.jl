
function dishonor_min_purity_increase(::Type{L}, min_purity_increase, purity, best_purity_times_nt, nt) where {L<:CLabel}
    (best_purity_times_nt/nt - purity < min_purity_increase)
end
function dishonor_min_purity_increase(::Type{L}, min_purity_increase, purity, best_purity_times_nt, nt) where {L<:RLabel}
    # (best_purity_times_nt - tsum * label <= min_purity_increase * nt) # ORIGINAL
    (best_purity_times_nt/nt - purity < min_purity_increase * nt)
end

# TODO fix
# function _compute_purity( # faster_version assuming L<:Integer and labels going from 1:n_classes
#     labels           ::AbstractVector{L},
#     n_classes        ::Int,
#     weights          ::AbstractVector{U} = default_weights(length(labels));
#     loss_function    ::Union{Nothing,Function} = default_loss_function(L),
# ) where {L<:CLabel, L<:Integer, U}
#     nc = fill(zero(U), n_classes)
#     @simd for i in 1:max(length(labels),length(weights))
#         nc[labels[i]] += weights[i]
#     end
#     nt = sum(nc)
#     return loss_function(nc, nt)::Float64
# end
function compute_purity(
    labels           ::AbstractVector{L},
    weights          ::AbstractVector{U} = default_weights(length(labels));
    loss_function    ::Union{Nothing,Function} = default_loss_function(L),
) where {L<:CLabel, U}
    nc = Dict{L, U}()
    @simd for i in 1:max(length(labels),length(weights))
        nc[labels[i]] = get(nc, labels[i], 0) + weights[i]
    end
    nc = collect(values(nc))
    nt = sum(nc)
    return loss_function(nc, nt)::Float64
end
# function _compute_purity(
#     labels           ::AbstractVector{L},
#     weights          ::AbstractVector{U} = default_weights(length(labels));
#     loss_function    ::Union{Nothing,Function} = default_loss_function(L),
# ) where {L<:RLabel, U}
#     sums = labels .* weights
#     nt = sum(weights)
#     return -(loss_function(sums, nt))::Float64
# end
function compute_purity(
    labels           ::AbstractVector{L},
    weights          ::AbstractVector{U} = default_weights(length(labels));
    loss_function    ::Union{Nothing,Function} = default_loss_function(L),
) where {L<:RLabel, U}
    _compute_purity = _compute_purity(labels, weights = weights; loss_function = loss_function)
end
