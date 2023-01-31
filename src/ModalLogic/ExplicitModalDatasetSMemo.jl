struct ExplicitModalDatasetSMemo{T<:Number,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}} <: ExplicitModalDatasetWithSupport{T,W,FR}

    # Core dataset
    emd                 :: ExplicitModalDataset{T,W,FR}

    # Relational and global support
    fwd_rs              :: AbstractRelationalSupport{<:Union{T,Nothing}, W}
    fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing}

    # Features and Aggregators
    featsnaggrs         :: AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}}
    grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}}

    function ExplicitModalDatasetSMemo{T,W,FR}(
        emd                 :: ExplicitModalDataset{T,W},
        fwd_rs              :: AbstractRelationalSupport{<:Union{T,Nothing}, W},
        fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing},
        featsnaggrs         :: AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}},
        grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        @assert nsamples(emd) == nsamples(fwd_rs)                                    "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W), $(FR)} with unmatching nsamples for emd and fwd_rs support: $(nsamples(emd)) and $(nsamples(fwd_rs))"
        @assert nrelations(emd) == nrelations(fwd_rs)                                "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W), $(FR)} with unmatching nrelations for emd and fwd_rs support: $(nrelations(emd)) and $(nrelations(fwd_rs))"
        @assert sum(length.(grouped_featsnaggrs)) == length(featsnaggrs)             "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W), $(FR)} with unmatching nfeatsnaggrs (grouped vs flattened structure): $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == length(featsnaggrs)      "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W), $(FR)} with unmatching nfeatsnaggrs for emd and provided featsnaggrs: $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == nfeatsnaggrs(fwd_rs)     "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W), $(FR)} with unmatching nfeatsnaggrs for emd and fwd_rs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(nfeatsnaggrs(fwd_rs))"

        if fwd_gs != nothing
            @assert nsamples(emd) == nsamples(fwd_gs)                                "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W), $(FR)} with unmatching nsamples for emd and fwd_gs support: $(nsamples(emd)) and $(nsamples(fwd_gs))"
            # @assert somethinglike(emd) == nfeatsnaggrs(fwd_gs)                     "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W), $(FR)} with unmatching somethinglike for emd and fwd_gs support: $(somethinglike(emd)) and $(nfeatsnaggrs(fwd_gs))"
            @assert sum(length.(emd.grouped_featsaggrsnops)) == nfeatsnaggrs(fwd_gs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W), $(FR)} with unmatching nfeatsnaggrs for emd and fwd_gs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(nfeatsnaggrs(fwd_gs))"
        end

        new{T,W,FR}(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs)
    end

    function ExplicitModalDatasetSMemo{T,W,FR}(
        emd                 :: ExplicitModalDataset{T,W},
        args...,
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        ExplicitModalDatasetSMemo{T,W,FR}(emd, args...)
    end

    function ExplicitModalDatasetSMemo{T,W}(
        emd                 :: ExplicitModalDataset{T,W,FR},
        args...,
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        ExplicitModalDatasetSMemo{T,W,FR}(emd, args...)
    end

    function ExplicitModalDatasetSMemo(
        emd                 :: ExplicitModalDataset{T,W};
        compute_relation_glob :: Bool = true,
    ) where {T,W<:AbstractWorld}
        ExplicitModalDatasetSMemo{T,W}(emd, compute_relation_glob = compute_relation_glob)
    end

    function ExplicitModalDatasetSMemo{T,W}(
        emd                   :: ExplicitModalDataset{T,W};
        compute_relation_glob :: Bool = true,
    ) where {T,W<:AbstractWorld}
        featsnaggrs = Tuple{<:AbstractFeature,<:Aggregator}[]
        grouped_featsnaggrs = AbstractVector{Tuple{<:Integer,<:Aggregator}}[]

        i_featsnaggr = 1
        for (feat,aggrsnops) in zip(emd.features,emd.grouped_featsaggrsnops)
            aggrs = []
            for aggr in keys(aggrsnops)
                push!(featsnaggrs, (feat,aggr))
                push!(aggrs, (i_featsnaggr,aggr))
                i_featsnaggr += 1
            end
            push!(grouped_featsnaggrs, aggrs)
        end

        # Compute modal dataset propositions and 1-modal decisions
        fwd_rs, fwd_gs = compute_fwd_supports(emd, grouped_featsnaggrs, compute_relation_glob = compute_relation_glob, simply_init_modal = true);

        ExplicitModalDatasetSMemo{T,W}(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs)
    end

    function ExplicitModalDatasetSMemo(
        X                   :: InterpretedModalDataset{T,N,W};
        compute_relation_glob :: Bool = true,
    ) where {T,N,W<:AbstractWorld}
        ExplicitModalDatasetSMemo{T,W}(X, compute_relation_glob = compute_relation_glob)
    end

    function ExplicitModalDatasetSMemo{T,W}(
        X                   :: InterpretedModalDataset{T,N,W};
        compute_relation_glob :: Bool = true,
    ) where {T,N,W<:AbstractWorld}

        # Compute modal dataset propositions
        emd = ExplicitModalDataset(X);

        ExplicitModalDatasetSMemo{T,W}(emd, compute_relation_glob = compute_relation_glob)
    end
end
