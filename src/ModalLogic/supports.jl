
struct GenericRelationalSupport{T,W} <: AbstractRelationalSupport{T,W}
    d :: AbstractArray{Dict{W,T}, 3}
end

goeswith(::Type{GenericRelationalSupport}, ::Type{<:AbstractWorld}) = true
# default_fwd_rs_type(::Type{<:AbstractWorld}) = GenericRelationalSupport # TODO implement similar pattern used for fwd

hasnans(emds::GenericRelationalSupport) = begin
    # @show any(map(d->(any(_isnan.(collect(values(d))))), emds.d))
    any(map(d->(any(_isnan.(collect(values(d))))), emds.d))
end

nsamples(emds::GenericRelationalSupport)     = size(emds, 1)
nfeatsnaggrs(emds::GenericRelationalSupport) = size(emds, 2)
nrelations(emds::GenericRelationalSupport)   = size(emds, 3)
capacity(emds::GenericRelationalSupport)     = Inf

Base.getindex(
    emds         :: GenericRelationalSupport{T,W},
    i_sample     :: Integer,
    w            :: W,
    i_featsnaggr :: Integer,
    i_relation   :: Integer) where {T,W<:AbstractWorld} = emds.d[i_sample, i_featsnaggr, i_relation][w]
Base.size(emds::GenericRelationalSupport, args...) = size(emds.d, args...)

fwd_rs_init(emd::ExplicitModalDataset{T,W}, nfeatsnaggrs::Integer, nrelations::Integer; perform_initialization = false) where {T,W} = begin
    if perform_initialization
        _fwd_rs = fill!(Array{Dict{W,Union{T,Nothing}}, 3}(undef, nsamples(emd), nfeatsnaggrs, nrelations), nothing)
        GenericRelationalSupport{Union{T,Nothing}, W}(_fwd_rs)
    else
        _fwd_rs = Array{Dict{W,T}, 3}(undef, nsamples(emd), nfeatsnaggrs, nrelations)
        GenericRelationalSupport{T,W}(_fwd_rs)
    end
end
fwd_rs_init_world_slice(emds::GenericRelationalSupport{T,W}, i_sample::Integer, i_featsnaggr::Integer, i_relation::Integer) where {T,W} =
    emds.d[i_sample, i_featsnaggr, i_relation] = Dict{W,T}()
fwd_rs_set(emds::GenericRelationalSupport{T,W}, i_sample::Integer, w::AbstractWorld, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T,W} =
    emds.d[i_sample, i_featsnaggr, i_relation][w] = threshold
function slice_dataset(emds::GenericRelationalSupport{T,W}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T,W}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    GenericRelationalSupport{T,W}(if return_view @view emds.d[inds,:,:] else emds.d[inds,:,:] end)
end

############################################################################################

# Note: the global support is world-agnostic
struct GenericGlobalSupport{T} <: AbstractGlobalSupport{T}
    d :: AbstractArray{T,2}
end

goeswith(::Type{AbstractGlobalSupport}, ::Type{<:AbstractWorld}) = true
# default_fwd_gs_type(::Type{<:AbstractWorld}) = GenericGlobalSupport # TODO implement similar pattern used for fwd

hasnans(emds::GenericGlobalSupport) = begin
    # @show any(_isnan.(emds.d))
    any(_isnan.(emds.d))
end

nsamples(emds::GenericGlobalSupport{T}) where {T}  = size(emds, 1)
nfeatsnaggrs(emds::GenericGlobalSupport{T}) where {T} = size(emds, 2)
Base.getindex(
    emds         :: GenericGlobalSupport{T},
    i_sample     :: Integer,
    i_featsnaggr  :: Integer) where {T} = emds.d[i_sample, i_featsnaggr]
Base.size(emds::GenericGlobalSupport{T}, args...) where {T} = size(emds.d, args...)

fwd_gs_init(emd::ExplicitModalDataset{T}, nfeatsnaggrs::Integer) where {T} =
    GenericGlobalSupport{T}(Array{T,2}(undef, nsamples(emd), nfeatsnaggrs))
fwd_gs_set(emds::GenericGlobalSupport{T}, i_sample::Integer, i_featsnaggr::Integer, threshold::T) where {T} =
    emds.d[i_sample, i_featsnaggr] = threshold
function slice_dataset(emds::GenericGlobalSupport{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
    GenericGlobalSupport{T}(if return_view @view emds.d[inds,:] else emds.d[inds,:] end)
end

# A function that computes fwd_rs and fwd_gs from an explicit modal dataset
Base.@propagate_inbounds function compute_fwd_supports(
        emd                 :: ExplicitModalDataset{T,W},
        grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}};
        compute_relation_glob = false,
        simply_init_rs = false,
    ) where {T,W<:AbstractWorld}

    # @logmsg LogOverview "ExplicitModalDataset -> ExplicitModalDatasetS "

    fwd = emd.fwd
    _features = features(emd)
    _relations = relations(emd)

    compute_fwd_gs = begin
        if RelationGlob in _relations
            throw_n_log("RelationGlob in relations: $(_relations)")
            _relations = filter!(l->l≠RelationGlob, _relations)
            true
        elseif compute_relation_glob
            true
        else
            false
        end
    end

    _n_samples = nsamples(emd)
    nrelations = length(_relations)
    nfeatsnaggrs = sum(length.(grouped_featsnaggrs))

    # println(_n_samples)
    # println(nrelations)
    # println(nfeatsnaggrs)
    # println(grouped_featsnaggrs)

    # Prepare fwd_rs
    # println("perform_initialization = $(perform_initialization)")
    fwd_rs = fwd_rs_init(emd, nfeatsnaggrs, nrelations; perform_initialization = simply_init_rs)

    # Prepare fwd_gs
    fwd_gs = begin
        if compute_fwd_gs
            fwd_gs_init(emd, nfeatsnaggrs)
        else
            nothing
        end
    end

    # p = Progress(_n_samples, 1, "Computing EMD supports...")
    Threads.@threads for i_sample in 1:_n_samples
        @logmsg LogDebug "Instance $(i_sample)/$(_n_samples)"

        # if i_sample == 1 || ((i_sample+1) % (floor(Int, ((_n_samples)/4))+1)) == 0
        #     @logmsg LogOverview "Instance $(i_sample)/$(_n_samples)"
        # end

        for (i_feature,aggregators) in enumerate(grouped_featsnaggrs)

            @logmsg LogDebug "Feature $(i_feature)"

            cur_fwd_slice = fwd_get_channel(fwd, i_sample, i_feature)

            @logmsg LogDebug cur_fwd_slice

            # Global relation (independent of the current world)
            if compute_fwd_gs
                @logmsg LogDebug "RelationGlob"

                # TODO optimize: all aggregators are likely reading the same raw values.
                for (i_featsnaggr,aggregator) in aggregators
                # Threads.@threads for (i_featsnaggr,aggregator) in aggregators
                    
                    # accessible_worlds = allworlds(emd, i_sample)
                    accessible_worlds = allworlds_aggr(emd, i_sample, _features[i_feature], aggregator)

                    threshold = compute_modal_gamma(cur_fwd_slice, accessible_worlds, aggregator)

                    @logmsg LogDebug "Aggregator[$(i_featsnaggr)]=$(aggregator)  -->  $(threshold)"

                    # @logmsg LogDebug "Aggregator" aggregator threshold

                    fwd_gs_set(fwd_gs, i_sample, i_featsnaggr, threshold)
                end
            end
            # readline()

            if !simply_init_rs
                # Other relations
                for (i_relation,relation) in enumerate(_relations)

                    @logmsg LogDebug "Relation $(i_relation)/$(nrelations)"

                    for (i_featsnaggr,aggregator) in aggregators
                        fwd_rs_init_world_slice(fwd_rs, i_sample, i_featsnaggr, i_relation)
                    end

                    for w in allworlds(emd, i_sample)

                        @logmsg LogDebug "World" w

                        # TODO optimize: all aggregators are likely reading the same raw values.
                        for (i_featsnaggr,aggregator) in aggregators
                            
                            # accessible_worlds = accessibles(emd, i_sample, w, relation)
                            accessible_worlds = representatives(emd, i_sample, w, relation, _features[i_feature], aggregator)
                        
                            threshold = compute_modal_gamma(cur_fwd_slice, accessible_worlds, aggregator)

                            # @logmsg LogDebug "Aggregator" aggregator threshold

                            fwd_rs_set(fwd_rs, i_sample, w, i_featsnaggr, i_relation, threshold)
                        end
                    end
                end
            end
        end
        # next!(p)
    end
    fwd_rs, fwd_gs
end
