
struct ExplicitModalDatasetS{T<:Number,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}} <: ExplicitModalDatasetWithSupport{T,W,FR}

    # Core dataset
    emd                 :: ExplicitModalDataset{T,W,FR}

    # Relational and global support
    fwd_rs              :: AbstractRelationalSupport{T,W}
    fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing}

    # Features and Aggregators
    featsnaggrs         :: AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}}
    grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}}

    function ExplicitModalDatasetS{T,W,FR}(
        emd                 :: ExplicitModalDataset{T,W,FR},
        fwd_rs              :: AbstractRelationalSupport{T,W},
        fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing},
        featsnaggrs         :: AbstractVector{Tuple{<:AbstractFeature,<:Aggregator}},
        grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        @assert nsamples(emd) == nsamples(fwd_rs)                               "Can't instantiate ExplicitModalDatasetS{$(T), $(W), $(FR)} with unmatching nsamples for emd and fwd_rs support: $(nsamples(emd)) and $(nsamples(fwd_rs))"
        @assert nrelations(emd) == nrelations(fwd_rs)                           "Can't instantiate ExplicitModalDatasetS{$(T), $(W), $(FR)} with unmatching nrelations for emd and fwd_rs support: $(nrelations(emd)) and $(nrelations(fwd_rs))"
        @assert sum(length.(grouped_featsnaggrs)) == length(featsnaggrs)          "Can't instantiate ExplicitModalDatasetS{$(T), $(W), $(FR)} with unmatching nfeatsnaggrs (grouped vs flattened structure): $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == length(featsnaggrs)   "Can't instantiate ExplicitModalDatasetS{$(T), $(W), $(FR)} with unmatching nfeatsnaggrs for emd and provided featsnaggrs: $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == nfeatsnaggrs(fwd_rs) "Can't instantiate ExplicitModalDatasetS{$(T), $(W), $(FR)} with unmatching nfeatsnaggrs for emd and fwd_rs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(nfeatsnaggrs(fwd_rs))"

        if fwd_gs != nothing
            @assert nsamples(emd) == nsamples(fwd_gs) "Can't instantiate ExplicitModalDatasetS{$(T), $(W), $(FR)} with unmatching nsamples for emd and fwd_gs support: $(nsamples(emd)) and $(nsamples(fwd_gs))"
            # @assert somethinglike(emd) == nfeatsnaggrs(fwd_gs) "Can't instantiate ExplicitModalDatasetS{$(T), $(W), $(FR)} with unmatching somethinglike for emd and fwd_gs support: $(somethinglike(emd)) and $(nfeatsnaggrs(fwd_gs))"
            @assert sum(length.(emd.grouped_featsaggrsnops)) == nfeatsnaggrs(fwd_gs) "Can't instantiate ExplicitModalDatasetS{$(T), $(W), $(FR)} with unmatching nfeatsnaggrs for emd and fwd_gs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(nfeatsnaggrs(fwd_gs))"
        end

        new{T,W,FR}(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs)
    end

    function ExplicitModalDatasetS{T,W}(
        emd                 :: ExplicitModalDataset{T,W,FR},
        args...
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        ExplicitModalDatasetS{T,W,FR}(emd, args...)
    end

    function ExplicitModalDatasetS(
        emd                 :: ExplicitModalDataset{T,W,FR};
        compute_relation_glob :: Bool = true,
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        ExplicitModalDatasetS{T,W}(emd, compute_relation_glob = compute_relation_glob)
    end

    function ExplicitModalDatasetS{T,W}(
        emd                   :: ExplicitModalDataset{T,W,FR};
        compute_relation_glob :: Bool = true,
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        
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
        fwd_rs, fwd_gs = compute_fwd_supports(emd, grouped_featsnaggrs, compute_relation_glob = compute_relation_glob);

        ExplicitModalDatasetS{T,W,FR}(emd, fwd_rs, fwd_gs, featsnaggrs, grouped_featsnaggrs)
    end

    function ExplicitModalDatasetS(
        X                   :: InterpretedModalDataset{T,N,W};
        compute_relation_glob :: Bool = true,
    ) where {T,N,W<:AbstractWorld}
        ExplicitModalDatasetS{T,W}(X, compute_relation_glob = compute_relation_glob)
    end

    function ExplicitModalDatasetS{T,W}(
        X                   :: InterpretedModalDataset{T,N,W};
        compute_relation_glob :: Bool = true,
    ) where {T,N,W<:AbstractWorld}

        # Compute modal dataset propositions
        emd = ExplicitModalDataset(X);

        ExplicitModalDatasetS{T,W}(emd, compute_relation_glob = compute_relation_glob)
    end
end

mutable struct ExplicitModalDatasetSMemo{T<:Number,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}} <: ExplicitModalDatasetWithSupport{T,W,FR}

    # Core dataset
    emd                 :: ExplicitModalDataset{T,W}

    # Relational and global support
    fwd_rs              :: AbstractRelationalSupport{<:Union{T,Nothing}, W}
    fwd_gs              :: Union{AbstractGlobalSupport{T},Nothing} # TODO maybe nothing is not needed here

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
        @assert nsamples(emd) == nsamples(fwd_rs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching nsamples for emd and fwd_rs support: $(nsamples(emd)) and $(nsamples(fwd_rs))"
        @assert nrelations(emd) == nrelations(fwd_rs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching nrelations for emd and fwd_rs support: $(nrelations(emd)) and $(nrelations(fwd_rs))"
        @assert sum(length.(grouped_featsnaggrs)) == length(featsnaggrs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching nfeatsnaggrs (grouped vs flattened structure): $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == length(featsnaggrs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching nfeatsnaggrs for emd and provided featsnaggrs: $(sum(length.(emd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(emd.grouped_featsaggrsnops)) == nfeatsnaggrs(fwd_rs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching nfeatsnaggrs for emd and fwd_rs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(nfeatsnaggrs(fwd_rs))"

        if fwd_gs != nothing
            @assert nsamples(emd) == nsamples(fwd_gs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching nsamples for emd and fwd_gs support: $(nsamples(emd)) and $(nsamples(fwd_gs))"
            # @assert somethinglike(emd) == nfeatsnaggrs(fwd_gs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching somethinglike for emd and fwd_gs support: $(somethinglike(emd)) and $(nfeatsnaggrs(fwd_gs))"
            @assert sum(length.(emd.grouped_featsaggrsnops)) == nfeatsnaggrs(fwd_gs) "Can't instantiate ExplicitModalDatasetSMemo{$(T), $(W)} with unmatching nfeatsnaggrs for emd and fwd_gs support: $(sum(length.(emd.grouped_featsaggrsnops))) and $(nfeatsnaggrs(fwd_gs))"
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

# getindex(X::ExplicitModalDatasetWithSupport{T,W}, args...) where {T,W} = getindex(X.emd, args...)
Base.size(X::ExplicitModalDatasetWithSupport)                                      =  (size(X.emd), size(X.fwd_rs), (isnothing(X.fwd_gs) ? nothing : size(X.fwd_gs)))
featsnaggrs(X::ExplicitModalDatasetWithSupport)                                    = X.featsnaggrs
features(X::ExplicitModalDatasetWithSupport)                                       = features(X.emd)
grouped_featsaggrsnops(X::ExplicitModalDatasetWithSupport)                         = grouped_featsaggrsnops(X.emd)
grouped_featsnaggrs(X::ExplicitModalDatasetWithSupport)                            = X.grouped_featsnaggrs
nfeatures(X::ExplicitModalDatasetWithSupport)                                      = nfeatures(X.emd)
nrelations(X::ExplicitModalDatasetWithSupport)                                     = nrelations(X.emd)
nsamples(X::ExplicitModalDatasetWithSupport)                                       = nsamples(X.emd)::Int64
relations(X::ExplicitModalDatasetWithSupport)                                      = relations(X.emd)
world_type(X::ExplicitModalDatasetWithSupport{T,W}) where {T,W}    = W

initialworldset(X::ExplicitModalDatasetWithSupport,  args...) = initialworldset(X.emd, args...)
accessibles(X::ExplicitModalDatasetWithSupport,     args...) = accessibles(X.emd, args...)
representatives(X::ExplicitModalDatasetWithSupport, args...) = representatives(X.emd, args...)
allworlds(X::ExplicitModalDatasetWithSupport,  args...) = allworlds(X.emd, args...)

function slice_dataset(X::ExplicitModalDatasetWithSupport, inds::AbstractVector{<:Integer}, args...; kwargs...)
    typeof(X)(
        slice_dataset(X.emd, inds, args...; kwargs...),
        slice_dataset(X.fwd_rs, inds, args...; kwargs...),
        (isnothing(X.fwd_gs) ? nothing : slice_dataset(X.fwd_gs, inds, args...; kwargs...)),
        X.featsnaggrs,
        X.grouped_featsnaggrs)
end

find_feature_id(X::ExplicitModalDatasetWithSupport, feature::AbstractFeature) =
    findall(x->x==feature, features(X))[1]
find_relation_id(X::ExplicitModalDatasetWithSupport, relation::AbstractRelation) =
    findall(x->x==relation, relations(X))[1]
find_featsnaggr_id(X::ExplicitModalDatasetWithSupport, feature::AbstractFeature, aggregator::Aggregator) =
    findall(x->x==(feature, aggregator), featsnaggrs(X))[1]

hasnans(X::ExplicitModalDatasetWithSupport) = begin
    # @show hasnans(X.emd)
    # @show hasnans(X.fwd_rs)
    # @show (!isnothing(X.fwd_gs) && hasnans(X.fwd_gs))
    hasnans(X.emd) || hasnans(X.fwd_rs) || (!isnothing(X.fwd_gs) && hasnans(X.fwd_gs))
end

Base.@propagate_inbounds @inline get_gamma(
        X::ExplicitModalDatasetWithSupport{T,W},
        i_sample::Integer,
        w::W,
        feature::AbstractFeature) where {T,W<:AbstractWorld} = get_gamma(X.emd, i_sample, w, feature)

isminifiable(::ExplicitModalDatasetWithSupport) = true

function minify(X::EMD) where {EMD<:ExplicitModalDatasetWithSupport}
    (new_emd, new_fwd_rs, new_fwd_gs), backmap =
        util.minify([
            X.emd,
            X.fwd_rs,
            X.fwd_gs,
        ])

    X = EMD(
        new_emd,
        new_fwd_rs,
        new_fwd_gs,
        featsnaggrs,
        grouped_featsnaggrs,
    )
    X, backmap
end

function display_structure(X::ExplicitModalDatasetS; indent_str = "")
    out = "$(typeof(X))\t$((Base.summarysize(X.emd) + Base.summarysize(X.fwd_rs) + Base.summarysize(X.fwd_gs)) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(X.emd))))\t$(relations(X.emd))\n"
    out *= indent_str * "├ emd\t$(Base.summarysize(X.emd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t(shape $(Base.size(X.emd)))\n"
    out *= indent_str * "├ fwd_rs\t$(Base.summarysize(X.fwd_rs) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t(shape $(Base.size(X.fwd_rs)))\n"
    out *= indent_str * "└ fwd_gs\t$(Base.summarysize(X.fwd_gs) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t"
    if !isnothing(X.fwd_gs)
        out *= "\t(shape $(Base.size(X.fwd_gs)))"
    else
        out *= "\t−"
    end
    out
end

function display_structure(X::ExplicitModalDatasetSMemo; indent_str = "")
    out = "$(typeof(X))\t$((Base.summarysize(X.emd) + Base.summarysize(X.fwd_rs) + Base.summarysize(X.fwd_gs)) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(X.emd))))\t$(relations(X.emd))\n"
    out *= indent_str * "├ emd\t$(Base.summarysize(X.emd) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
        out *= "\t(shape $(Base.size(X.emd.fwd)))\n"
        # out *= "\t(shape $(Base.size(X.emd.fwd)), $(n_nothing) nothings, $((1-(n_nothing    / size(X.emd)))*100)%)\n"
    n_nothing_m = count(isnothing, X.fwd_rs.d)
    out *= indent_str * "├ fwd_rs\t$(Base.summarysize(X.fwd_rs) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
        out *= "\t(shape $(Base.size(X.fwd_rs)), $(n_nothing_m) nothings)\n" # , $((1-(n_nothing_m  / n_v(X.fwd_rs)))*100)%)\n"
    out *= indent_str * "└ fwd_gs\t$(Base.summarysize(X.fwd_gs) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t"
    if !isnothing(X.fwd_gs)
        # n_nothing_g = count(isnothing, X.fwd_gs)
        # out *= "\t(shape $(Base.size(X.fwd_gs)), $(n_nothing_g) nothings, $((1-(n_nothing_g  / size(X.fwd_gs)))*100)%)"
        out *= "\t(shape $(Base.size(X.fwd_gs)))"
    else
        out *= "\t−"
    end
    out
end

############################################################################################

get_global_gamma(
        X::ExplicitModalDatasetWithSupport{T,W},
        i_sample::Integer,
        feature::AbstractFeature,
        test_operator::TestOperatorFun) where {T,W<:AbstractWorld} = begin
            # @assert !isnothing(X.fwd_gs) "Error. ExplicitModalDatasetWithSupport must be built with compute_relation_glob = true for it to be ready to test global decisions."
            i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
            X.fwd_gs[i_sample, i_featsnaggr]
end

get_modal_gamma(
        X::ExplicitModalDatasetS{T,W},
        i_sample::Integer,
        w::W,
        relation::AbstractRelation,
        feature::AbstractFeature,
        test_operator::TestOperatorFun) where {T,W<:AbstractWorld} = begin
            i_relation = find_relation_id(X, relation)
            i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
            X.fwd_rs[i_sample, w, i_featsnaggr, i_relation]
end
