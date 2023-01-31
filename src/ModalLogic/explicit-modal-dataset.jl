
############################################################################################
# Explicit modal dataset
# 
# An explicit modal dataset is the generic form of a modal dataset, and consists of
#  a wrapper around an fwd lookup table. The information it adds are the relation set,
#  a few functions for enumerating worlds (`accessibles`, `representatives`),
#  and a world set initialization function representing initial conditions (initializing world sets).
# 
############################################################################################

struct ExplicitModalDataset{T<:Number,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},FWD<:AbstractFWD{T,W,FR}} <: ActiveModalDataset{T,W,FR}
    
    # Core data (fwd lookup table)
    fwd                :: FWD

    ## Modal frame:
    # Accessibility relations
    relations          :: AbstractVector{<:AbstractRelation}
    
    # Features
    features           :: AbstractVector{<:AbstractFeature}

    # Test operators associated with each feature, grouped by their respective aggregator
    grouped_featsaggrsnops  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}

    function ExplicitModalDataset{T,W,FR,FWD}(
        fwd                     :: FWD,
        relations               :: AbstractVector{<:AbstractRelation},
        features                :: AbstractVector{<:AbstractFeature},
        grouped_featsaggrsnops  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}};
        allow_no_instances = false,
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},FWD<:AbstractFWD{T,W,FR}}
        @assert allow_no_instances || nsamples(fwd) > 0     "Can't instantiate ExplicitModalDataset{$(T), $(W)} with no instance. (fwd's type $(typeof(fwd)))"
        @assert length(grouped_featsaggrsnops) > 0 && sum(length.(grouped_featsaggrsnops)) > 0 && sum(vcat([[length(test_ops) for test_ops in aggrs] for aggrs in grouped_featsaggrsnops]...)) > 0 "Can't instantiate ExplicitModalDataset{$(T), $(W)} with no test operator: grouped_featsaggrsnops"
        @assert nfeatures(fwd) == length(features)          "Can't instantiate ExplicitModalDataset{$(T), $(W)} with different numbers of instances $(nsamples(fwd)) and of features $(length(features))."
        new{T,W,FR,FWD}(fwd, relations, features, grouped_featsaggrsnops)
    end

    function ExplicitModalDataset{T,W,FR}(
        fwd                     :: FWD,
        args...;
        kwargs...
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},FWD<:AbstractFWD{T,W,FR}}
        ExplicitModalDataset{T,W,FR,FWD}(fwd, args...; kwargs...)
    end

    function ExplicitModalDataset{T,W}(
        fwd                     :: AbstractFWD{T,W,FR},
        args...;
        kwargs...
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        ExplicitModalDataset{T,W,FR}(fwd, args...; kwargs...)
    end

    ExplicitModalDataset(
        fwd                    :: AbstractFWD{T,W},
        relations              :: AbstractVector{<:AbstractRelation},
        features               :: AbstractVector{<:AbstractFeature},
        grouped_featsaggrsnops_or_featsnops, # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
        args...;
        kwargs...,
    ) where {T,W} = begin ExplicitModalDataset{T,W}(fwd, relations, features, grouped_featsaggrsnops_or_featsnops, args...; kwargs...) end

    function ExplicitModalDataset(
        fwd                    :: AbstractFWD{T,W},
        relations              :: AbstractVector{<:AbstractRelation},
        features               :: AbstractVector{<:AbstractFeature},
        grouped_featsnops      :: AbstractVector{<:AbstractVector{<:TestOperatorFun}},
        args...;
        kwargs...,
    ) where {T,W<:AbstractWorld}

        grouped_featsaggrsnops = grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops)
 
        ExplicitModalDataset(fwd, relations, features, grouped_featsaggrsnops, args...; kwargs...)
    end

    # Quite importantly, an fwd can be computed from a dataset in implicit form (domain + ontology + features)
    Base.@propagate_inbounds function ExplicitModalDataset(
        imd                  :: InterpretedModalDataset{T,N,W},
        # FWD                  ::Type{<:AbstractFWD{T,W}} = default_fwd_type(W),
        FWD                  ::Type = default_fwd_type(W),
        args...;
        kwargs...,
    ) where {T,N,W<:AbstractWorld}

        fwd = begin

            # @logmsg LogOverview "InterpretedModalDataset -> ExplicitModalDataset"

            _features = features(imd)

            _n_samples = nsamples(imd)

            @assert goeswith(FWD, W)

            # Initialize the fwd structure
            fwd = fwd_init(FWD, imd)

            # Load any (possible) external features
            if any(isa.(_features, ExternalFWDFeature))
                i_external_features = first.(filter(((i_feature,is_external_fwd),)->(is_external_fwd), collect(enumerate(isa.(_features, ExternalFWDFeature)))))
                for i_feature in i_external_features
                    feature = _features[i_feature]
                    fwd_set_feature_slice(fwd, i_feature, feature.fwd)
                end
            end

            # Load any internal features
            i_features = first.(filter(((i_feature,is_external_fwd),)->!(is_external_fwd), collect(enumerate(isa.(_features, ExternalFWDFeature)))))
            enum_features = zip(i_features, _features[i_features])

            # Compute features
            # p = Progress(_n_samples, 1, "Computing EMD...")
            @inbounds Threads.@threads for i_sample in 1:_n_samples
                @logmsg LogDebug "Instance $(i_sample)/$(_n_samples)"

                # if i_sample == 1 || ((i_sample+1) % (floor(Int, ((_n_samples)/4))+1)) == 0
                #     @logmsg LogOverview "Instance $(i_sample)/$(_n_samples)"
                # end

                # instance = get_instance(imd, i_sample)
                # @logmsg LogDebug "instance" instance

                for w in allworlds(imd, i_sample)
                    
                    fwd_init_world_slice(fwd, i_sample, w)

                    @logmsg LogDebug "World" w

                    for (i_feature,feature) in enum_features

                        # threshold = computePropositionalThreshold(feature, w, instance)
                        threshold = get_gamma(imd, i_sample, w, feature)

                        @logmsg LogDebug "Feature $(i_feature)" threshold

                        fwd_set(fwd, w, i_sample, i_feature, threshold)

                    end
                end
                # next!(p)
            end
            fwd
        end

        ExplicitModalDataset(fwd, relations(imd), _features, grouped_featsaggrsnops(imd), args...; kwargs...)
    end

end

Base.getindex(X::ExplicitModalDataset{T,W}, args...) where {T,W} = getindex(X.fwd, args...)
Base.size(X::ExplicitModalDataset)              = size(X.fwd) # TODO fix not always defined?
features(X::ExplicitModalDataset)               = X.features
grouped_featsaggrsnops(X::ExplicitModalDataset) = X.grouped_featsaggrsnops
nfeatures(X::ExplicitModalDataset)              = length(X.features)
nrelations(X::ExplicitModalDataset)             = length(X.relations)
nsamples(X::ExplicitModalDataset)               = nsamples(X.fwd)::Int64
relations(X::ExplicitModalDataset)              = X.relations
world_type(X::ExplicitModalDataset{T,W}) where {T,W<:AbstractWorld} = W


initialworldset(X::ExplicitModalDataset, i_sample, args...) = initialworldset(X.fwd, i_sample, args...)
accessibles(X::ExplicitModalDataset, i_sample, args...) = accessibles(X.fwd, i_sample, args...)
representatives(X::ExplicitModalDataset, i_sample, args...) = representatives(X.fwd, i_sample, args...)
allworlds(X::ExplicitModalDataset, i_sample, args...) = allworlds(X.fwd, i_sample, args...)


slice_dataset(X::ExplicitModalDataset{T,W}, inds::AbstractVector{<:Integer}, args...; allow_no_instances = false, kwargs...) where {T,W} =
    ExplicitModalDataset{T,W}(
        slice_dataset(X.fwd, inds, args...; allow_no_instances = allow_no_instances, kwargs...),
        X.relations,
        X.features,
        X.grouped_featsaggrsnops;
        allow_no_instances = allow_no_instances
    )


function display_structure(emd::ExplicitModalDataset; indent_str = "")
    out = "$(typeof(emd))\t$(Base.summarysize(emd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(emd))))\t$(relations(emd))\n"
    out *= indent_str * "└ fwd: \t$(typeof(emd.fwd))\t$(Base.summarysize(emd.fwd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out
end


find_feature_id(X::ExplicitModalDataset{T,W}, feature::AbstractFeature) where {T,W} =
    findall(x->x==feature, features(X))[1]
find_relation_id(X::ExplicitModalDataset{T,W}, relation::AbstractRelation) where {T,W} =
    findall(x->x==relation, relations(X))[1]

hasnans(emd::ExplicitModalDataset) = begin
    # @show hasnans(emd.fwd)
    hasnans(emd.fwd)
end

Base.@propagate_inbounds @inline get_gamma(
        X::ExplicitModalDataset{T,W},
        i_sample::Integer,
        w::W,
        feature::AbstractFeature) where {T,W<:AbstractWorld} = begin
    i_feature = find_feature_id(X, feature)
    X[i_sample, w, i_feature]
end

isminifiable(::ExplicitModalDataset) = true

function minify(X::ExplicitModalDataset)
    new_fwd, backmap = minify(X.fwd)
    X = ExplicitModalDataset(
        new_fwd,
        X.relations,
        X.features,
        X.grouped_featsaggrsnops,
    )
    X, backmap
end
