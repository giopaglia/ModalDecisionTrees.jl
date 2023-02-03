############################################################################################
############################################################################################
############################################################################################

# Perform the modal step, that is, evaluate a modal formula
#  on a domain, and eventually compute the new world set.
function modal_step(
        X::ActiveModalDataset{T,W},
        i_sample::Integer,
        worlds::WorldSetType,
        decision::ExistentialDimensionalDecision{T},
        returns_survivors::Union{Val{true},Val{false}} = Val(false)
    ) where {T,W<:AbstractWorld,WorldSetType<:AbstractWorldSet{W}}
    @logmsg LogDetail "modal_step" worlds display_decision(decision)

    satisfied = false
    
    # TODO space for optimization here: with some relations (e.g. IA_A, IA_L) can be made smaller

    if returns_survivors isa Val{true}
        worlds_map = Dict{W,AbstractWorldSet{W}}()
    end
    if length(worlds) == 0
        # If there are no neighboring worlds, then the modal decision is not met
        @logmsg LogDetail "   No accessible world"
    else
        # Otherwise, check whether at least one of the accessible worlds witnesses truth of the decision.
        # TODO rewrite with new_worlds = map(...acc_worlds)
        # Initialize new worldset
        new_worlds = WorldSetType()

        # List all accessible worlds
        acc_worlds = 
            if returns_survivors isa Val{true}
                Threads.@threads for curr_w in worlds
                    l = ReentrantLock()
                    acc = accessibles(X, i_sample, curr_w, relation(decision)) |> collect
                    lock(l)
                    worlds_map[curr_w] = acc
                    unlock(l)
                end
                unique(cat([ worlds_map[k] for k in keys(worlds_map) ]...; dims = 1))
            else
                accessibles(X, i_sample, worlds, relation(decision))
            end

        for w in acc_worlds
            if test_decision(X, i_sample, w, feature(decision), test_operator(decision), threshold(decision))
                # @logmsg LogDetail " Found world " w ch_readWorld ... ch_readWorld(w, channel)
                satisfied = true
                push!(new_worlds, w)
            end
        end

        if satisfied == true
            worlds = new_worlds
        else
            # If none of the neighboring worlds satisfies the decision, then 
            #  the new set is left unchanged
        end
    end
    if satisfied
        @logmsg LogDetail "   YES" worlds
    else
        @logmsg LogDetail "   NO"
    end
    if returns_survivors isa Val{true}
        return (satisfied, worlds, worlds_map)
    else
        return (satisfied, worlds)
    end
end

test_decision(
        X::ModalDataset{T},
        i_sample::Integer,
        w::AbstractWorld,
        feature::AbstractFeature,
        test_operator::TestOperatorFun,
        threshold::T) where {T} = begin
    gamma = get_gamma(X, i_sample, w, feature)
    evaluate_thresh_decision(test_operator, gamma, threshold)
end

test_decision(
        X::ModalDataset{T},
        i_sample::Integer,
        w::AbstractWorld,
        decision::ExistentialDimensionalDecision{T}) where {T} = begin
    instance = get_instance(X, i_sample)

    aggregator = existential_aggregator(test_operator(decision))
    
    worlds = representatives(FullDimensionalFrame(instance_channel_size(instance)), w, relation(decision), feature(decision), aggregator)
    gamma = if length(worlds |> collect) == 0
        aggregator_bottom(aggregator, T)
    else
        aggregator((w)->get_gamma(X, i_sample, w, feature(decision)), worlds)
    end

    evaluate_thresh_decision(test_operator(decision), gamma, threshold(decision))
end



# function slice_dataset(x::Any, dataset_slice::AbstractVector{<:Integer}; allow_no_instances = false, kwargs...)
#     @assert (allow_no_instances || length(dataset_slice) > 0) "Can't apply empty slice to dataset."
#     slice_dataset(x, dataset_slice; kwargs...)
# end
