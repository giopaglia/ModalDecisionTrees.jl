struct ExplicitModalDatasetS{
    T<:Number,
    W<:AbstractWorld,
    FR<:AbstractFrame{W,Bool},
    S<:SupportingModalDataset{T,W,FR},
} <: ActiveModalDataset{T,W,FR}

    # Core dataset
    emd                 :: ExplicitModalDataset{T,W,FR}

    # Support structure
    support             :: S
    
    ########################################################################################
    
    function ExplicitModalDatasetS{T,W,FR,S}(
        emd                 :: ExplicitModalDataset{T,W,FR},
        support             :: S,
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},S<:SupportingModalDataset{T,W,FR}}
        ty = "ExplicitModalDatasetS{$(T), $(W), $(FR), $(S)}"
        @assert checksupportconsistency(emd, support) "Can't instantiate $(ty) with an inconsistent support:\n\nemd:\n$(display_structure(emd))\n\nsupport:\n$(display_structure(support))"
        new{T,W,FR,S}(emd, support)
    end

    function ExplicitModalDatasetS{T,W,FR}(emd::ExplicitModalDataset{T,W}, support::S, args...; kwargs...) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool},S<:SupportingModalDataset{T,W,FR}}
        ExplicitModalDatasetS{T,W,FR,S}(emd, support, args...; kwargs...)
    end

    function ExplicitModalDatasetS{T,W}(emd::ExplicitModalDataset{T,W,FR}, args...; kwargs...) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        ExplicitModalDatasetS{T,W,FR}(emd, args...; kwargs...)
    end

    function ExplicitModalDatasetS{T}(emd::ExplicitModalDataset{T,W,FR}, args...; kwargs...) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        ExplicitModalDatasetS{T,W}(emd, args...; kwargs...)
    end

    function ExplicitModalDatasetS(emd::ExplicitModalDataset{T,W,FR}, args...; kwargs...) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        ExplicitModalDatasetS{T}(emd, args...; kwargs...)
    end
    
    ########################################################################################
    
    function ExplicitModalDatasetS(
        emd                   :: ExplicitModalDataset{T,W,FR};
        compute_relation_glob :: Bool = true,
        use_memoization       :: Bool = true,
    ) where {T,W<:AbstractWorld,FR<:AbstractFrame{W,Bool}}
        
        support = OneStepSupportingDataset(
            emd,
            compute_relation_glob = compute_relation_glob,
            use_memoization = use_memoization
        );

        ExplicitModalDatasetS(emd, support)
    end

    ########################################################################################
    
    function ExplicitModalDatasetS(
        X                   :: InterpretedModalDataset{T,N,W};
        kwargs...,
    ) where {T,N,W<:AbstractWorld}
        emd = ExplicitModalDataset(X);
        ExplicitModalDatasetS(emd; kwargs...)
    end

end

emd(X::ExplicitModalDatasetS)                                            = X.emd
support(X::ExplicitModalDatasetS)                                        = X.support

Base.size(X::ExplicitModalDatasetS)                                      = (size(emd(X)), size(support(X)))
features(X::ExplicitModalDatasetS)                                       = features(emd(X))
grouped_featsaggrsnops(X::ExplicitModalDatasetS)                         = grouped_featsaggrsnops(emd(X))
grouped_featsnaggrs(X::ExplicitModalDatasetS)                            = grouped_featsnaggrs(emd(X))
nfeatures(X::ExplicitModalDatasetS)                                      = nfeatures(emd(X))
nrelations(X::ExplicitModalDatasetS)                                     = nrelations(emd(X))
nsamples(X::ExplicitModalDatasetS)                                       = nsamples(emd(X))
relations(X::ExplicitModalDatasetS)                                      = relations(emd(X))
fwd(X::ExplicitModalDatasetS)                                            = fwd(emd(X))
world_type(X::ExplicitModalDatasetS{T,W}) where {T,W}    = W

usesmemo(X::ExplicitModalDatasetS) = usesmemo(support(X))

initialworldset(X::ExplicitModalDatasetS,  args...) = initialworldset(emd(X), args...)
accessibles(X::ExplicitModalDatasetS,     args...) = accessibles(emd(X), args...)
representatives(X::ExplicitModalDatasetS, args...) = representatives(emd(X), args...)
allworlds(X::ExplicitModalDatasetS,  args...) = allworlds(emd(X), args...)

function slice_dataset(X::ExplicitModalDatasetS, inds::AbstractVector{<:Integer}, args...; kwargs...)
    ExplicitModalDatasetS(
        slice_dataset(emd(X), inds, args...; kwargs...),
        slice_dataset(support(X), inds, args...; kwargs...),
    )
end

find_feature_id(X::ExplicitModalDatasetS, feature::AbstractFeature) = findall(x->x==feature, features(X))[1]
find_relation_id(X::ExplicitModalDatasetS, relation::AbstractRelation) = findall(x->x==relation, relations(X))[1]

hasnans(X::ExplicitModalDatasetS) = hasnans(emd(X)) || hasnans(support(X))

isminifiable(X::ExplicitModalDatasetS) = isminifiable(emd(X)) && isminifiable(emd(X))

function minify(X::EMD) where {EMD<:ExplicitModalDatasetS}
    (new_emd, new_support), backmap =
        minify([
            emd(X),
            support(X),
        ])

    X = EMD(
        new_emd,
        new_support,
    )
    X, backmap
end

function display_structure(X::ExplicitModalDatasetS; indent_str = "")
    out = "$(typeof(X))\t$((Base.summarysize(emd(X)) + Base.summarysize(support(X))) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ relations: \t$((length(relations(emd(X)))))\t$(relations(emd(X)))\n"
    out *= indent_str * "├ emd\t$(Base.summarysize(emd(X)) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
        out *= "\t(shape $(Base.size(emd(X))))\n"
    out *= indent_str * "└ support: $(display_structure(support(X); indent_str = "  "))"
    out
end

############################################################################################
############################################################################################
############################################################################################

Base.@propagate_inbounds @inline get_gamma(
    X::ExplicitModalDatasetS{T,W},
    i_sample::Integer,
    w::W,
    feature::AbstractFeature) where {T,W<:AbstractWorld} = get_gamma(emd(X), i_sample, w, feature)

function _get_global_gamma(
    X::ExplicitModalDatasetS{T,W},
    i_sample::Integer,
    feature::AbstractFeature,
    aggregator::Aggregator
) where {T,W<:AbstractWorld}
    compute_global_gamma(support(X), emd(X), i_sample, feature, aggregator)
end

function _get_modal_gamma(
    X::ExplicitModalDatasetS{T,W},
    i_sample::Integer,
    w::W,
    relation::AbstractRelation,
    feature::AbstractFeature,
    aggregator::Aggregator
) where {T,W<:AbstractWorld}
    i_relation = find_relation_id(X, relation)
    compute_modal_gamma(support(X), emd(X), i_sample, w, i_relation, relation, feature, aggregator)
end

function __get_modal_gamma(
    X::ExplicitModalDatasetS{T,W},
    i_sample::Integer,
    w::W,
    i_featsnaggr,
    i_relation,
    relation::AbstractRelation,
    feature::AbstractFeature,
    aggregator::Aggregator
)::T where {T,W<:AbstractWorld}
    _compute_modal_gamma(support(X), emd(X), i_sample, w, i_featsnaggr, i_relation, relation, feature, aggregator)
end

############################################################################################

function test_decision(
    X::ExplicitModalDatasetS{T,W},
    i_sample::Integer,
    w::W,
    decision::ExistentialDimensionalDecision
) where {T,W<:AbstractWorld}
    
    if is_propositional_decision(decision)
        test_decision(X, i_sample, w, feature(decision), test_operator(decision), threshold(decision))
    else
        gamma = begin
            if relation(decision) isa _RelationGlob
                get_global_gamma(X, i_sample, feature(decision), test_operator(decision))
            else
                get_modal_gamma(X, i_sample, w, relation(decision), feature(decision), test_operator(decision))
            end
        end
        evaluate_thresh_decision(test_operator(decision), gamma, threshold(decision))
    end
end

