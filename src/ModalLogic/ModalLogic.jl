module ModalLogic
using IterTools
import Base: argmax, argmin, size, show, convert, getindex, iterate, length
using Logging: @logmsg
using ..DecisionTree

using ComputedFieldTypes

using ResumableFunctions

using DataStructures

using BenchmarkTools # TODO use this to compare timings and use @btime

using DecisionTree.util

export AbstractWorld, AbstractRelation,
                Ontology,
                AbstractWorldSet, WorldSet,
                RelationGlob, RelationNone, RelationId,
                world_type, world_types # TODO maybe remove this function?
                # enumAccessibles, enumAccRepr

# Fix (not needed in Julia 1.7, see https://github.com/JuliaLang/julia/issues/34674 )
Base.keys(g::Base.Generator) = g.iter

# Abstract classes for world & relations
abstract type AbstractWorld end
abstract type AbstractRelation end

show(io::IO, r::AbstractRelation) = print(io, display_existential_relation(r))
display_existential_relation(r) = "⟨$(display_rel_short(r))⟩"
display_universal_relation(r) = "[$(display_rel_short(r))]"

# Concrete class for ontology models (world type + set of relations)
struct Ontology{WorldType<:AbstractWorld}
    relationSet :: AbstractVector{<:AbstractRelation}
    Ontology{WorldType}(relationSet) where {WorldType<:AbstractWorld} = begin
        relationSet = unique(relationSet)
        for relation in relationSet
            @assert goesWith(WorldType, relation) "Can't instantiate Ontology with WorldType $(WorldType) and relation $(relation)"
        end
        # if WorldType == OneWorld
        #   relationSet = []
        # end
        return new{WorldType}(relationSet)
    end
    # Ontology(worldType, relationSet) = new(worldType, relationSet)
end

world_type(::Ontology{WT}) where {WT<:AbstractWorld} = WT

# Actually, this will not work because relationSet does this collect(set(...)) thing... mh maybe better avoid that thing?
show(io::IO, o::Ontology{WorldType}) where {WorldType} = begin
    if o == OneWorldOntology
        print(io, "OneWorldOntology")
    else
        print(io, "Ontology{")
        show(io, WorldType)
        print(io, "}(")
        if issetequal(o.relationSet, IARelations)
            print(io, "IARelations")
        elseif issetequal(o.relationSet, IARelations_extended)
            print(io, "IARelations_extended")
        elseif issetequal(o.relationSet, IA2DRelations)
            print(io, "IA2DRelations")
        elseif issetequal(o.relationSet, IA2DRelations_U)
            print(io, "IA2DRelations_U")
        elseif issetequal(o.relationSet, IA2DRelations_extended)
            print(io, "IA2DRelations_extended")
        elseif issetequal(o.relationSet, RCC8Relations)
            print(io, "RCC8")
        elseif issetequal(o.relationSet, RCC5Relations)
            print(io, "RCC5")
        else
            show(io, o.relationSet)
        end
        print(io, ")")
    end
end

# strip_ontology(ontology::Ontology) = Ontology{OneWorld}(AbstractRelation[])


# This constant is used to create the default world for each WorldType
#  (e.g. Interval(emptyWorld) = Interval(-1,0))
struct _firstWorld end;    const firstWorld    = _firstWorld();
struct _emptyWorld end;    const emptyWorld    = _emptyWorld();
struct _centeredWorld end; const centeredWorld = _centeredWorld();

# World generators/enumerators and array/set-like structures
const AbstractWorldSet{W} = Union{AbstractVector{W},AbstractSet{W}} where {W<:AbstractWorld}
const WorldSet{W} = Vector{W} where {W<:AbstractWorld}
WorldSet{W}(S::WorldSet{W}) where {W<:AbstractWorld} = S

include("operators.jl")


abstract type CanonicalFeature end

export CanonicalFeature, CanonicalFeatureGeq, CanonicalFeatureLeq

# ⪴ and ⪳, that is, "*all* of the values on this world are at least, or at most ..."
struct _CanonicalFeatureGeq <: CanonicalFeature end; const CanonicalFeatureGeq  = _CanonicalFeatureGeq();
struct _CanonicalFeatureLeq <: CanonicalFeature end; const CanonicalFeatureLeq  = _CanonicalFeatureLeq();

export CanonicalFeatureGeq_95, CanonicalFeatureGeq_90, CanonicalFeatureGeq_85, CanonicalFeatureGeq_80, CanonicalFeatureGeq_75, CanonicalFeatureGeq_70, CanonicalFeatureGeq_60,
                CanonicalFeatureLeq_95, CanonicalFeatureLeq_90, CanonicalFeatureLeq_85, CanonicalFeatureLeq_80, CanonicalFeatureLeq_75, CanonicalFeatureLeq_70, CanonicalFeatureLeq_60

# ⪴_α and ⪳_α, that is, "*at least α⋅100 percent* of the values on this world are at least, or at most ..."

struct _CanonicalFeatureGeqSoft  <: CanonicalFeature
  alpha :: AbstractFloat
  _CanonicalFeatureGeqSoft(a::T) where {T<:Real} = (a > 0 && a < 1) ? new(a) : throw_n_log("Invalid instantiation for test operator: _CanonicalFeatureGeqSoft($(a))")
end;
struct _CanonicalFeatureLeqSoft  <: CanonicalFeature
  alpha :: AbstractFloat
  _CanonicalFeatureLeqSoft(a::T) where {T<:Real} = (a > 0 && a < 1) ? new(a) : throw_n_log("Invalid instantiation for test operator: _CanonicalFeatureLeqSoft($(a))")
end;

const CanonicalFeatureGeq_95  = _CanonicalFeatureGeqSoft((Rational(95,100)));
const CanonicalFeatureGeq_90  = _CanonicalFeatureGeqSoft((Rational(90,100)));
const CanonicalFeatureGeq_85  = _CanonicalFeatureGeqSoft((Rational(85,100)));
const CanonicalFeatureGeq_80  = _CanonicalFeatureGeqSoft((Rational(80,100)));
const CanonicalFeatureGeq_75  = _CanonicalFeatureGeqSoft((Rational(75,100)));
const CanonicalFeatureGeq_70  = _CanonicalFeatureGeqSoft((Rational(70,100)));
const CanonicalFeatureGeq_60  = _CanonicalFeatureGeqSoft((Rational(60,100)));

const CanonicalFeatureLeq_95  = _CanonicalFeatureLeqSoft((Rational(95,100)));
const CanonicalFeatureLeq_90  = _CanonicalFeatureLeqSoft((Rational(90,100)));
const CanonicalFeatureLeq_85  = _CanonicalFeatureLeqSoft((Rational(85,100)));
const CanonicalFeatureLeq_80  = _CanonicalFeatureLeqSoft((Rational(80,100)));
const CanonicalFeatureLeq_75  = _CanonicalFeatureLeqSoft((Rational(75,100)));
const CanonicalFeatureLeq_70  = _CanonicalFeatureLeqSoft((Rational(70,100)));
const CanonicalFeatureLeq_60  = _CanonicalFeatureLeqSoft((Rational(60,100)));

# TODO deprecated, remove
export TestOpGeq_95, TestOpGeq_90, TestOpGeq_85, TestOpGeq_80, TestOpGeq_75, TestOpGeq_70, TestOpGeq_60, TestOpLeq_95, TestOpLeq_90, TestOpLeq_85, TestOpLeq_80, TestOpLeq_75, TestOpLeq_70, TestOpLeq_60, TestOpGeq, TestOpLeq
TestOpGeq_95 = CanonicalFeatureGeq_95
TestOpGeq_90 = CanonicalFeatureGeq_90
TestOpGeq_85 = CanonicalFeatureGeq_85
TestOpGeq_80 = CanonicalFeatureGeq_80
TestOpGeq_75 = CanonicalFeatureGeq_75
TestOpGeq_70 = CanonicalFeatureGeq_70
TestOpGeq_60 = CanonicalFeatureGeq_60
TestOpLeq_95 = CanonicalFeatureLeq_95
TestOpLeq_90 = CanonicalFeatureLeq_90
TestOpLeq_85 = CanonicalFeatureLeq_85
TestOpLeq_80 = CanonicalFeatureLeq_80
TestOpLeq_75 = CanonicalFeatureLeq_75
TestOpLeq_70 = CanonicalFeatureLeq_70
TestOpLeq_60 = CanonicalFeatureLeq_60
TestOpGeq    = CanonicalFeatureGeq
TestOpLeq    = CanonicalFeatureLeq

################################################################################
# Decision
################################################################################

export Decision, is_modal_decision, display_decision, display_decision_inverse

# Split-Decision (e.g., ⟨L⟩ (minimum(A2) ≤ 10) )
struct Decision{T}
    
    # modal operator (e.g. RelationId for the propositional case)
    relation      :: AbstractRelation
    
    # scalar feature
    feature       :: ModalFeature
    
    # test_operator (e.g. ≤)
    test_operator :: TestOperatorFun

    # threshold value
    threshold     :: T
end

function show(io::IO, decision::Decision)
    println(io, display_decision(decision))
end

is_modal_decision(d::Decision) = !(d.relation isa ModalLogic._RelationId)


display_decision(i_frame::Integer, decision::Decision; threshold_display_method::Function = x -> x, is_intended_as_universal = false) = begin
    "{$i_frame} $(display_decision(decision; threshold_display_method = threshold_display_method, is_intended_as_universal = is_intended_as_universal))"
end

display_decision_inverse(i_frame::Integer, decision::Decision; threshold_display_method::Function = x -> x) = begin
    inv_decision = Decision{T}(decision.relation, decision.feature, test_operator_inverse(test_operator), decision.threshold)
    display_decision(i_frame, inv_decision; threshold_display_method = threshold_display_method, is_intended_as_universal = true)
end

display_decision(decision::Decision; threshold_display_method::Function = x -> x, is_intended_as_universal = false) = begin
    prop_decision_str = display_propositional_decision(decision.feature, decision.test_operator, decision.threshold; threshold_display_method = threshold_display_method)
    if is_modal_decision(decision)
        "$((is_intended_as_universal ? display_universal_relation : display_existential_relation)(decision.relation)) ($prop_decision_str)"
    else
        "$prop_decision_str"
    end
end

display_propositional_decision(feature::ModalFeature, test_operator::TestOperatorFun, threshold::Number; threshold_display_method::Function = x -> x) =
    "$(feature) $(test_operator) $(threshold_display_method(threshold))"

display_propositional_decision(feature::AttributeMinimumFeatureType, test_operator::typeof(≥), threshold::Number; threshold_display_method::Function = x -> x) =
    "A$(feature.i_attribute) ⪴ $(threshold_display_method(threshold))"
display_propositional_decision(feature::AttributeMaximumFeatureType, test_operator::typeof(≤), threshold::Number; threshold_display_method::Function = x -> x) =
    "A$(feature.i_attribute) ⪳ $(threshold_display_method(threshold))"
display_propositional_decision(feature::AttributeSoftMinimumFeatureType, test_operator::typeof(≥), threshold::Number; threshold_display_method::Function = x -> x) =
    "A$(feature.i_attribute) $("⪴" * util.subscriptnumber(rstrip(rstrip(string(alpha(feature)*100), '0'), '.'))) $(threshold_display_method(threshold))"
display_propositional_decision(feature::AttributeSoftMaximumFeatureType, test_operator::typeof(≤), threshold::Number; threshold_display_method::Function = x -> x) =
    "A$(feature.i_attribute) $("⪳" * util.subscriptnumber(rstrip(rstrip(string(alpha(feature)*100), '0'), '.'))) $(threshold_display_method(threshold))"

display_propositional_decision(feature::AttributeMinimumFeatureType, test_operator::typeof(<), threshold::Number; threshold_display_method::Function = x -> x) =
    "A$(feature.i_attribute) ⪶ $(threshold_display_method(threshold))"
display_propositional_decision(feature::AttributeMaximumFeatureType, test_operator::typeof(>), threshold::Number; threshold_display_method::Function = x -> x) =
    "A$(feature.i_attribute) ⪵ $(threshold_display_method(threshold))"
display_propositional_decision(feature::AttributeSoftMinimumFeatureType, test_operator::typeof(<), threshold::Number; threshold_display_method::Function = x -> x) =
    "A$(feature.i_attribute) $("⪶" * util.subscriptnumber(rstrip(rstrip(string(alpha(feature)*100), '0'), '.'))) $(threshold_display_method(threshold))"
display_propositional_decision(feature::AttributeSoftMaximumFeatureType, test_operator::typeof(>), threshold::Number; threshold_display_method::Function = x -> x) =
    "A$(feature.i_attribute) $("⪵" * util.subscriptnumber(rstrip(rstrip(string(alpha(feature)*100), '0'), '.'))) $(threshold_display_method(threshold))"

display_propositional_decision(feature::AttributeMinimumFeatureType, test_operator::typeof(≤), threshold::Number; threshold_display_method::Function = x -> x) =
    "A$(feature.i_attribute) ↘ $(threshold_display_method(threshold))"
display_propositional_decision(feature::AttributeMaximumFeatureType, test_operator::typeof(≥), threshold::Number; threshold_display_method::Function = x -> x) =
    "A$(feature.i_attribute) ↗ $(threshold_display_method(threshold))"
display_propositional_decision(feature::AttributeSoftMinimumFeatureType, test_operator::typeof(≤), threshold::Number; threshold_display_method::Function = x -> x) =
    "A$(feature.i_attribute) $("↘" * util.subscriptnumber(rstrip(rstrip(string(alpha(feature)*100), '0'), '.'))) $(threshold_display_method(threshold))"
display_propositional_decision(feature::AttributeSoftMaximumFeatureType, test_operator::typeof(≥), threshold::Number; threshold_display_method::Function = x -> x) =
    "A$(feature.i_attribute) $("↗" * util.subscriptnumber(rstrip(rstrip(string(alpha(feature)*100), '0'), '.'))) $(threshold_display_method(threshold))"

################################################################################
# Dataset structures
################################################################################


export n_samples, n_attributes, n_features, n_relations,
                channel_size, max_channel_size,
                n_frames, frames, get_frame, push_frame!,
                display_structure,
                get_gamma, test_decision,
                ##############################
                concat_datasets,
                dataset_has_nonevalues,
                ##############################
                relations,
                initws_function,
                acc_function,
                accrepr_functions,
                features,
                grouped_featsaggrsnops,
                featsnaggrs,
                grouped_featsnaggrs,
                ##############################
                SingleFrameGenericDataset,
                GenericDataset,
                MultiFrameModalDataset,
                AbstractModalDataset,
                OntologicalDataset, 
                AbstractFeaturedWorldDataset,
                FeatModalDataset,
                StumpFeatModalDataset,
                StumpFeatModalDatasetWithMemoization,
                MatricialInstance,
                MatricialDataset,
                # MatricialUniDataset,
                MatricialChannel,
                FeaturedWorldDataset

const initWorldSetFunction = Function
const accFunction = Function
const accReprFunction = Function

abstract type AbstractModalDataset{T<:Number,WorldType<:AbstractWorld} end

# A dataset, given by a set of N-dimensional (multi-attribute) matrices/instances,
#  and an Ontology to be interpreted on each of them.
# - The size of the domain array is {X×Y×...} × n_attributes × n_samples
# - N is the dimensionality of the domain itself (e.g. 1 for the temporal case, 2 for the spatial case)
#    and its dimensions are denoted as X,Y,Z,...
# - A uni-attribute dataset is of dimensionality S=N+1
# - A multi-attribute dataset is of dimensionality D=N+1+1
#  https://discourse.julialang.org/t/addition-to-parameter-of-parametric-type/20059/5

# TODO: It'd be nice to define these as a function of N, https://github.com/JuliaLang/julia/issues/8322
#   e.g. const MatricialUniDataset{T,N}       = AbstractArray{T,N+1}

const MatricialDataset{T<:Number,D}     = AbstractArray{T,D}
# const MatricialUniDataset{T<:Number,UD} = AbstractArray{T,UD}
const MatricialChannel{T<:Number,N}     = AbstractArray{T,N}
const MatricialInstance{T<:Number,MN}   = AbstractArray{T,MN}

# TODO use d[i,[(:) for i in 1:N]...] for accessing it, instead of writing blocks of functions

n_samples(d::MatricialDataset{T,D})    where {T,D} = size(d, D)::Int64
n_attributes(d::MatricialDataset{T,D}) where {T,D} = size(d, D-1)
channel_size(d::MatricialDataset{T,D}) where {T,D} = size(d)[1:end-2]
# length(d::MatricialDataset{T,N})        where {T,N} = n_samples(d)
# Base.iterate(d::MatricialDataset{T,D}, state=1) where {T, D} = state > length(d) ? nothing : (getInstance(d, state), state+1)
max_channel_size = channel_size
# TODO rename channel_size into max_channel_size and define channel_size for single instance
# channel_size(d::MatricialDataset{T,2}, idx_i::Integer) where T = size(d[      1, idx_i])
# channel_size(d::MatricialDataset{T,3}, idx_i::Integer) where T = size(d[:,    1, idx_i])
# channel_size(d::MatricialDataset{T,4}, idx_i::Integer) where T = size(d[:, :, 1, idx_i])

# channel_size(d::MatricialDataset{T,D}, idx_i::Integer) where {T,D} = size(d[idx_i])[1:end-2]
inst_channel_size(inst::MatricialInstance{T,MN}) where {T,MN} = size(inst)[1:end-1]

getInstance(d::MatricialDataset{T,2},     idx::Integer) where T = @views d[:, idx]         # N=0
getInstance(d::MatricialDataset{T,3},     idx::Integer) where T = @views d[:, :, idx]      # N=1
getInstance(d::MatricialDataset{T,4},     idx::Integer) where T = @views d[:, :, :, idx]   # N=2

slice_dataset(d::MatricialDataset{T,2}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[:, inds]       else d[:, inds]    end # N=0
slice_dataset(d::MatricialDataset{T,3}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[:, :, inds]    else d[:, :, inds] end # N=1
slice_dataset(d::MatricialDataset{T,4}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[:, :, :, inds] else d[:, :, :, inds] end # N=2

getChannel(d::MatricialDataset{T,2},      idx_i::Integer, idx_a::Integer) where T = @views d[      idx_a, idx_i]::T                     # N=0
getChannel(d::MatricialDataset{T,3},      idx_i::Integer, idx_a::Integer) where T = @views d[:,    idx_a, idx_i]::MatricialChannel{T,1} # N=1
getChannel(d::MatricialDataset{T,4},      idx_i::Integer, idx_a::Integer) where T = @views d[:, :, idx_a, idx_i]::MatricialChannel{T,2} # N=2
# getUniChannel(ud::MatricialUniDataset{T,1},  idx::Integer) where T = @views ud[idx]           # N=0
# getUniChannel(ud::MatricialUniDataset{T,2},  idx::Integer) where T = @views ud[:, idx]        # N=1
# getUniChannel(ud::MatricialUniDataset{T,3},  idx::Integer) where T = @views ud[:, :, idx]     # N=2
getInstanceAttribute(inst::MatricialInstance{T,1},      idx::Integer) where T = @views inst[      idx]::T                     # N=0
getInstanceAttribute(inst::MatricialInstance{T,2},      idx::Integer) where T = @views inst[:,    idx]::MatricialChannel{T,1} # N=1
getInstanceAttribute(inst::MatricialInstance{T,3},      idx::Integer) where T = @views inst[:, :, idx]::MatricialChannel{T,2} # N=2

dataset_has_nonevalues(d::MatricialDataset) = nothing in d || NaN in d || any(ismissing.(d))

concat_datasets(d1::MatricialDataset{T,2}, d2::MatricialDataset{T,2}) where {T} = cat(d1, d2; dims=2)
concat_datasets(d1::MatricialDataset{T,3}, d2::MatricialDataset{T,3}) where {T} = cat(d1, d2; dims=3)
concat_datasets(d1::MatricialDataset{T,4}, d2::MatricialDataset{T,4}) where {T} = cat(d1, d2; dims=4)
concat_datasets(d1::MatricialDataset{T,5}, d2::MatricialDataset{T,5}) where {T} = cat(d1, d2; dims=5)

# TODO maybe using views can improve performances
# @computed getChannel(X::OntologicalDataset{T,N}, idxs::AbstractVector{Integer}, attribute::Integer) where T = X[idxs, attribute, fill(:, N)...]::AbstractArray{T,N-1}
# attributeview(X::MatricialDataset{T,2}, idxs::AbstractVector{Integer}, attribute::Integer) = d[idxs, attribute]
# attributeview(X::MatricialDataset{T,3}, idxs::AbstractVector{Integer}, attribute::Integer) = view(d, idxs, attribute, :)
# attributeview(X::MatricialDataset{T,4}, idxs::AbstractVector{Integer}, attribute::Integer) = view(d, idxs, attribute, :, :)


# strip_domain(d::MatricialDataset{T,2}) where T = d  # N=0
# strip_domain(d::MatricialDataset{T,3}) where T = dropdims(d; dims=1)      # N=1
# strip_domain(d::MatricialDataset{T,4}) where T = dropdims(d; dims=(1,2))  # N=2

# Initialize MatricialUniDataset by slicing across the attribute dimension
# MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,2}) where T = Array{T, 1}(undef, n_samples(d))::MatricialUniDataset{T, 1}
# MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,3}) where T = Array{T, 2}(undef, size(d)[1:end-1])::MatricialUniDataset{T, 2}
# MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,4}) where T = Array{T, 3}(undef, size(d)[1:end-1])::MatricialUniDataset{T, 3}

get_gamma(X::MatricialDataset, i_instance::Integer, w::AbstractWorld, feature::ModalFeature) =
    yieldFunction(feature)(inst_readWorld(w, getInstance(X, i_instance)))


@computed struct OntologicalDataset{T, N, WorldType} <: AbstractModalDataset{T, WorldType}
    
    # Core data
    domain                  :: MatricialDataset{T,N+1+1}
    
    # Relations
    ontology                :: Ontology{WorldType} # Union{Nothing,}
    
    # Features
    features                :: AbstractVector{ModalFeature} # AbstractVector{<:ModalFeature} # Union{Nothing,}

    # Test operators associated with each feature, grouped by their respective aggregator
    grouped_featsaggrsnops  :: AbstractVector # TODO currently cannot use full type (probably due to @computed) # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}} # Union{Nothing,}

    # function OntologicalDataset(domain::MatricialDataset{T,D}, WorldType::Type{<:AbstractWorld}) where {T, D}
    #   N = D-1-1
    #   new{T,N,WorldType}(domain, nothing, nothing, nothing)
    #   # new{T,D-1-1,WorldType}(domain, nothing, nothing, nothing)
    # end

    # function OntologicalDataset(domain::MatricialDataset{T,D}, WorldType::Type{<:AbstractWorld}) where {T, D}
    #   OntologicalDataset{T,D-1-1,WorldType}(domain, nothing)
    # end

    # function OntologicalDataset{T, N, WorldType}(domain::MatricialDataset{T,D}, not::Nothing) where {T, N, D, WorldType<:AbstractWorld}
    #   OntologicalDataset{T, N, WorldType}(domain, not, not, not)
    # end

    # function OntologicalDataset{T, N, WorldType}(
    #   domain::MatricialDataset{T,D},
    #   ontology::Nothing,
    #   features::Nothing,
    #   grouped_featsaggrsnops::Nothing
    # ) where {T, N, D, WorldType<:AbstractWorld}
    #   @assert D == (N+1+1) "ERROR! Dimensionality mismatch: can't instantiate OntologicalDataset{$(T), $(N)} with MatricialDataset{$(T),$(D)}"
    #   OntologicalDataset{T, N, WorldType}(domain, ontology, features, grouped_featsaggrsnops)
    # end

    # function OntologicalDataset{T, N, WorldType}(
    #     domain::MatricialDataset{T,D},
    #     # ontology::Ontology{WorldType},
    #     # canonical_features,
    # ) where {T, N, D, WorldType<:AbstractWorld}
    #     features, featsnops = begin
    #         # features = ModalFeature[]
    #         # featsnops = Vector{<:TestOperatorFun}[]

    #         # # readymade
    #         # cnv_feat(cf::ModalFeature) = ([≥, ≤], cf)
    #         # cnv_feat(cf::Tuple{TestOperatorFun,ModalFeature}) = ([cf[1]], cf[2])
    #         # # attribute_specific
    #         # cnv_feat(cf::Any) = cf
    #         # cnv_feat(cf::Function) = ([≥, ≤], cf)
    #         # cnv_feat(cf::Tuple{TestOperatorFun,Function}) = ([cf[1]], cf[2])

    #         # canonical_features = cnv_feat.(canonical_features)

    #         # readymade_cfs          = filter(x->isa(x, Tuple{<:AbstractVector{<:TestOperatorFun},ModalFeature}), canonical_features)
    #         # attribute_specific_cfs = filter(x->isa(x, CanonicalFeature) || isa(x, Tuple{<:AbstractVector{<:TestOperatorFun},Function}), canonical_features)

    #         # @assert length(readymade_cfs) + length(attribute_specific_cfs) == length(canonical_features) "Unexpected canonical_features: $(filter(x->(! (x in readymade_cfs) && ! (x in attribute_specific_cfs)), canonical_features))"

    #         # for (test_ops,cf) in readymade_cfs
    #         #     push!(features, cf)
    #         #     push!(featsnops, test_ops)
    #         # end

    #         # single_attr_feats_n_featsnops(i_attr,cf::ModalLogic._CanonicalFeatureGeq) = ([≥],ModalLogic.AttributeMinimumFeatureType(i_attr))
    #         # single_attr_feats_n_featsnops(i_attr,cf::ModalLogic._CanonicalFeatureLeq) = ([≤],ModalLogic.AttributeMaximumFeatureType(i_attr))
    #         # single_attr_feats_n_featsnops(i_attr,cf::ModalLogic._CanonicalFeatureGeqSoft) = ([≥],ModalLogic.AttributeSoftMinimumFeatureType(i_attr, cf.alpha))
    #         # single_attr_feats_n_featsnops(i_attr,cf::ModalLogic._CanonicalFeatureLeqSoft) = ([≤],ModalLogic.AttributeSoftMaximumFeatureType(i_attr, cf.alpha))
    #         # single_attr_feats_n_featsnops(i_attr,(test_ops,cf)::Tuple{<:AbstractVector{<:TestOperatorFun},Function}) = (test_ops,AttributeFunctionFeatureType(i_attr, cf))
    #         # single_attr_feats_n_featsnops(i_attr,::Any) = throw_n_log("Unknown canonical_feature type: $(cf), $(typeof(cf))")

    #         # for i_attr in 1:n_attributes(domain)
    #         #     for (test_ops,cf) in map((cf)->single_attr_feats_n_featsnops(i_attr,cf),attribute_specific_cfs)
    #         #         push!(featsnops, test_ops)
    #         #         push!(features, cf)
    #         #     end
    #         # end
    #         # features, featsnops
    #         ModalFeature[], []
    #     end
    #     ontology = getIntervalOntologyOfDim(Val(D-1-1))
    #     OntologicalDataset{T, N, world_type(ontology)}(domain, ontology, features, featsnops)
    # end

    function OntologicalDataset(
        domain::MatricialDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops_or_featsnops, # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
    ) where {T, D, WorldType<:AbstractWorld}
        OntologicalDataset{T}(domain, ontology, features, grouped_featsaggrsnops_or_featsnops)
    end

    function OntologicalDataset{T}(
        domain::MatricialDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops_or_featsnops, # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
    ) where {T, D, WorldType<:AbstractWorld}
        OntologicalDataset{T, D-1-1}(domain, ontology, features, grouped_featsaggrsnops_or_featsnops)
    end

    function OntologicalDataset{T, N}(
        domain::MatricialDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops_or_featsnops, # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
    ) where {T, N, D, WorldType<:AbstractWorld}
        OntologicalDataset{T, N, WorldType}(domain, ontology, features, grouped_featsaggrsnops_or_featsnops)
    end
    
    function OntologicalDataset{T, N, WorldType}(
        domain::MatricialDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsnops  :: AbstractVector{<:AbstractVector{<:TestOperatorFun}},
    ) where {T, N, D, WorldType<:AbstractWorld}

        grouped_featsaggrsnops = grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops)
        
        OntologicalDataset{T, N, WorldType}(domain, ontology, features, grouped_featsaggrsnops)
    end
    function OntologicalDataset{T, N, WorldType}(
        domain::MatricialDataset{T,D},
        ontology::Ontology{WorldType},
        features::AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
    ) where {T, N, D, WorldType<:AbstractWorld}

        @assert n_samples(domain) > 0 "Can't instantiate OntologicalDataset{$(T), $(N), $(WorldType)} with no instance. (domain's type $(typeof(domain)))"
        @assert N == worldTypeDimensionality(WorldType) "ERROR! Dimensionality mismatch: can't interpret WorldType $(WorldType) (dimensionality = $(worldTypeDimensionality(WorldType)) on MatricialDataset of dimensionality = $(N))"
        @assert D == (N+1+1) "ERROR! Dimensionality mismatch: can't instantiate OntologicalDataset{$(T), $(N)} with MatricialDataset{$(T),$(D)}"
        @assert length(features) == length(grouped_featsaggrsnops) "Can't instantiate OntologicalDataset{$(T), $(N), $(WorldType)} with mismatching length(features) == length(grouped_featsaggrsnops): $(length(features)) != $(length(grouped_featsaggrsnops))"
        # @assert length(grouped_featsaggrsnops) > 0 && sum(length.(grouped_featsaggrsnops)) > 0 && sum(vcat([[length(test_ops) for test_ops in aggrs] for aggrs in grouped_featsaggrsnops]...)) > 0 "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with no test operator: $(grouped_featsaggrsnops)"

        # if prod(channel_size(domain)) == 1
        #   TODO throw warning
        # end
        
        new{T, N, WorldType}(domain, ontology, features, grouped_featsaggrsnops)
    end
end

relations(X::OntologicalDataset)        = X.ontology.relationSet
size(X::OntologicalDataset)             = size(X.domain)
n_samples(X::OntologicalDataset)        = n_samples(X.domain)::Int64
n_attributes(X::OntologicalDataset)     = n_attributes(X.domain)
n_relations(X::OntologicalDataset)      = length(relations(X))
world_type(X::OntologicalDataset{T,N,WT})    where {T,N,WT<:AbstractWorld} = WT

features(X::OntologicalDataset)               = X.features
grouped_featsaggrsnops(X::OntologicalDataset) = X.grouped_featsaggrsnops
n_features(X::OntologicalDataset)             = length(X.features)
# getindex(X::OntologicalDataset, args::Vararg) = getindex(X.domain[...], args...)

initws_function(X::OntologicalDataset{T, N, WorldType},  i_instance) where {T, N, WorldType} =
    (iC)->initWorldSet(iC, WorldType, inst_channel_size(getInstance(X, i_instance)))
acc_function(X::OntologicalDataset, i_instance) = (w,R)->enumAccessibles(w,R, inst_channel_size(getInstance(X, i_instance))...)
accAll_function(X::OntologicalDataset{T, N, WorldType}, i_instance) where {T, N, WorldType} = enumAll(WorldType, acc_function(X, i_instance))
accrepr_function(X::OntologicalDataset, i_instance)  = (f,a,w,R)->enumAccReprAggr(f,a,w,R,inst_channel_size(getInstance(X, i_instance))...)

length(X::OntologicalDataset)                = n_samples(X)
Base.iterate(X::OntologicalDataset, state=1) = state > length(X) ? nothing : (getInstance(X, state), state+1) # Base.iterate(X.domain, state=state)
channel_size(X::OntologicalDataset)          = channel_size(X.domain)

getInstance(X::OntologicalDataset, args::Vararg)     = getInstance(X.domain, args...)
getChannel(X::OntologicalDataset,   args::Vararg)    = getChannel(X.domain, args...)

slice_dataset(X::OntologicalDataset, inds::AbstractVector{<:Integer}; args...)    =
    OntologicalDataset(slice_dataset(X.domain, inds; args...), X.ontology, X.features, X.grouped_featsaggrsnops)


display_structure(X::OntologicalDataset; indent_str = "") = begin
    out = "$(typeof(X))\t$(Base.summarysize(X) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ domain shape\t$(Base.size(X.domain))\n"
    out *= indent_str * "├ $(length(relations(X))) relations\t$(relations(X))\n"
    out *= indent_str * "└ max_channel_size\t$(max_channel_size(X))"
    out
end

get_gamma(X::OntologicalDataset, args...) = get_gamma(X.domain, args...)

abstract type AbstractFeaturedWorldDataset{T, WorldType} end

struct FeatModalDataset{T, WorldType} <: AbstractModalDataset{T, WorldType}
    
    # Core data
    fwd                :: AbstractFeaturedWorldDataset{T,WorldType}
    
    ## Modal frame:
    # Accessibility relations
    relations          :: AbstractVector{<:AbstractRelation}
    
    # Worldset initialization functions (one per instance)
    #  with signature (initCondition) -> vs::AbstractWorldSet{WorldType}
    initws_functions   :: AbstractVector{<:initWorldSetFunction}
    # Accessibility functions (one per instance)
    #  with signature (w::WorldType/AbstractWorldSet{WorldType}, r::AbstractRelation) -> vs::AbstractVector{WorldType}
    acc_functions      :: AbstractVector{<:accFunction}
    # Representative accessibility functions (one per instance)
    #  with signature (feature::ModalFeature, aggregator::Aggregator, w::WorldType/AbstractWorldSet{WorldType}, r::AbstractRelation) -> vs::AbstractVector{WorldType}
    accrepr_functions  :: AbstractVector{<:accReprFunction}
    
    # Feature
    features           :: AbstractVector{<:ModalFeature}

    # Test operators associated with each feature, grouped by their respective aggregator
    grouped_featsaggrsnops  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}

    FeatModalDataset(
        fwd                :: AbstractFeaturedWorldDataset{T,WorldType},
        relations          :: AbstractVector{<:AbstractRelation},
        initws_functions   :: AbstractVector{<:initWorldSetFunction},
        acc_functions      :: AbstractVector{<:accFunction},
        accrepr_functions  :: AbstractVector{<:accReprFunction},
        features           :: AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops_or_featsnops, # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
    ) where {T,WorldType} = begin FeatModalDataset{T, WorldType}(fwd, relations, initws_functions, acc_functions, accrepr_functions, features, grouped_featsaggrsnops_or_featsnops) end

    function FeatModalDataset{T, WorldType}(
        fwd                :: AbstractFeaturedWorldDataset{T,WorldType},
        relations          :: AbstractVector{<:AbstractRelation},
        initws_functions   :: AbstractVector{<:initWorldSetFunction},
        acc_functions      :: AbstractVector{<:accFunction},
        accrepr_functions  :: AbstractVector{<:accReprFunction},
        features           :: AbstractVector{<:ModalFeature},
        grouped_featsaggrsnops  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}},
    ) where {T,WorldType<:AbstractWorld}

        @assert n_samples(fwd) > 0 "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with no instance. (fwd's type $(typeof(fwd)))"
        @assert length(grouped_featsaggrsnops) > 0 && sum(length.(grouped_featsaggrsnops)) > 0 && sum(vcat([[length(test_ops) for test_ops in aggrs] for aggrs in grouped_featsaggrsnops]...)) > 0 "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with no test operator: grouped_featsaggrsnops"
        @assert n_samples(fwd) == length(initws_functions) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of initws_functions $(length(initws_functions))."
        @assert n_samples(fwd) == length(acc_functions) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of acc_functions $(length(acc_functions))."
        @assert n_samples(fwd) == length(accrepr_functions) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of accrepr_functions $(length(accrepr_functions))."
        @assert n_features(fwd) == length(features) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of features $(length(features))."
        new{T, WorldType}(fwd, relations, initws_functions, acc_functions, accrepr_functions, features, grouped_featsaggrsnops)
    end

    function FeatModalDataset(
        fwd                :: AbstractFeaturedWorldDataset{T,WorldType},
        relations          :: AbstractVector{<:AbstractRelation},
        initws_functions   :: AbstractVector{<:initWorldSetFunction},
        acc_functions      :: AbstractVector{<:accFunction},
        accrepr_functions  :: AbstractVector{<:accReprFunction},
        features           :: AbstractVector{<:ModalFeature},
        grouped_featsnops  :: AbstractVector{<:AbstractVector{<:TestOperatorFun}},
    ) where {T,WorldType<:AbstractWorld}

        grouped_featsaggrsnops = grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops)
 
        FeatModalDataset(fwd, relations, initws_functions, acc_functions, accrepr_functions, features, grouped_featsaggrsnops)
    end

    FeatModalDataset(
        X                  :: OntologicalDataset{T, N, WorldType},
    ) where {T, N, WorldType<:AbstractWorld} = begin
        fwd = FeaturedWorldDataset(X);

        # TODO optimize this! When the underlying MatricialDataset is an AbstractArray, this is going to be an array of a single function.
        # How to achievi this? Think about it.
        initws_functions  = [initws_function(X,  i_instance) for i_instance in 1:n_samples(X)]
        acc_functions     = [acc_function(X,     i_instance) for i_instance in 1:n_samples(X)]
        accrepr_functions = [accrepr_function(X, i_instance) for i_instance in 1:n_samples(X)]

        FeatModalDataset(fwd, relations(X), initws_functions, acc_functions, accrepr_functions, features(X), grouped_featsaggrsnops(X))
    end

end

relations(X::FeatModalDataset)         = X.relations
initws_function(X::FeatModalDataset,  i_instance)  = X.initws_functions[i_instance]
acc_function(X::FeatModalDataset,     i_instance)  = X.acc_functions[i_instance]
accAll_function(X::FeatModalDataset{T, WorldType}, i_instance) where {T, WorldType} = enumAll(WorldType, acc_function(X, i_instance))
accrepr_function(X::FeatModalDataset, i_instance)  = X.accrepr_functions[i_instance]
features(X::FeatModalDataset)          = X.features
grouped_featsaggrsnops(X::FeatModalDataset) = X.grouped_featsaggrsnops

size(X::FeatModalDataset)             where {T,N} =  size(X.fwd)
n_samples(X::FeatModalDataset{T, WorldType}) where {T, WorldType}   = n_samples(X.fwd)::Int64
n_features(X::FeatModalDataset{T, WorldType}) where {T, WorldType}  = length(X.features)
n_relations(X::FeatModalDataset{T, WorldType}) where {T, WorldType} = length(X.relations)
# length(X::FeatModalDataset{T,WorldType})        where {T,WorldType} = n_samples(X)
# Base.iterate(X::FeatModalDataset{T,WorldType}, state=1) where {T, WorldType} = state > length(X) ? nothing : (getInstance(X, state), state+1)
getindex(X::FeatModalDataset{T,WorldType}, args::Vararg) where {T,WorldType} = getindex(X.fwd, args...)
world_type(X::FeatModalDataset{T,WorldType}) where {T,WorldType<:AbstractWorld} = WorldType


slice_dataset(X::FeatModalDataset{T,WorldType}, inds::AbstractVector{<:Integer}; args...) where {T,WorldType} =
    FeatModalDataset{T,WorldType}(
        slice_dataset(X.fwd, inds; args...),
        X.relations,
        X.initws_functions[inds],
        X.acc_functions[inds],
        X.accrepr_functions[inds],
        X.features,
        X.grouped_featsaggrsnops,
    )


function grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops::AbstractVector{<:AbstractVector{<:TestOperatorFun}})::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
    grouped_featsaggrsnops = Dict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}[]
    for (i_feature, test_operators) in enumerate(grouped_featsnops)
        aggrsnops = Dict{Aggregator,AbstractVector{<:TestOperatorFun}}()
        for test_operator in test_operators
            aggregator = ModalLogic.existential_aggregator(test_operator)
            if (!haskey(aggrsnops, aggregator))
                aggrsnops[aggregator] = TestOperatorFun[]
            end
            push!(aggrsnops[aggregator], test_operator)
        end
        push!(grouped_featsaggrsnops, aggrsnops)
    end
    grouped_featsaggrsnops
end


find_feature_id(X::FeatModalDataset{T,WorldType}, feature::ModalFeature) where {T,WorldType} =
    findall(x->x==feature, features(X))[1]
find_relation_id(X::FeatModalDataset{T,WorldType}, relation::AbstractRelation) where {T,WorldType} =
    findall(x->x==relation, relations(X))[1]

get_gamma(
        X::FeatModalDataset{T,WorldType},
        i_instance::Integer,
        w::WorldType,
        feature::ModalFeature) where {WorldType<:AbstractWorld, T} = begin
    i_feature = find_feature_id(X, feature)
    X[i_instance, w, i_feature]
end


Base.@propagate_inbounds @resumable function generate_propositional_feasible_decisions(
        X::FeatModalDataset{T,WorldType},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:AbstractWorld}
    relation = RelationId
    n_instances = length(instances_inds)

    # For each feature
    @inbounds for i_feature in features_inds
        feature = features(X)[i_feature]
        @logmsg DTDebug "Feature $(i_feature): $(feature)"

        # operators for each aggregator
        aggrsnops = grouped_featsaggrsnops(X)[i_feature]
        # Vector of aggregators
        aggregators = keys(aggrsnops) # Note: order-variant, but that's ok here
        
        # dict->vector
        # aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

        # Initialize thresholds with the bottoms
        thresholds = Array{T,2}(undef, length(aggregators), n_instances)
        for (i_aggr,aggr) in enumerate(aggregators)
            thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
        end

        # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
        for (i_instance,instance_id) in enumerate(instances_inds)
            @logmsg DTDetail " Instance $(i_instance)/$(n_instances)"
            worlds = Sf[i_instance]

            # TODO also try this instead
            # values = [X.fmd[instance_id, w, i_feature] for w in worlds]
            # thresholds[:,i_instance] = map(aggr->aggr(values), aggregators)
            
            for w in worlds
                gamma = X.fwd[instance_id, w, i_feature]
                for (i_aggr,aggr) in enumerate(aggregators)
                    thresholds[i_aggr,i_instance] = ModalLogic.aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,i_instance])
                end
            end
        end
        
        # tested_test_operator = TestOperatorFun[]

        # @logmsg DTDebug "thresholds: " thresholds
        # For each aggregator
        for (i_aggr,aggr) in enumerate(aggregators)
            aggr_thresholds = thresholds[i_aggr,:]
            aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))
            for (i_test_operator,test_operator) in enumerate(aggrsnops[aggr])
                # TODO figure out a solution to this issue: ≥ and ≤ in a propositional condition can find more or less the same optimum, so no need to check both; but which one of them should be the one on the left child, the one that makes the modal step?
                # if dual_test_operator(test_operator) in tested_test_operator
                #   throw_n_log("Double-check this part of the code: there's a foundational issue here to settle!")
                #   println("Found $(test_operator)'s dual $(dual_test_operator(test_operator)) in tested_test_operator = $(tested_test_operator)")
                #   continue
                # end
                @logmsg DTDetail " Test operator $(test_operator)"
                # Look for the best threshold 'a', as in propositions like "feature >= a"
                for threshold in aggr_domain
                    decision = Decision(relation, feature, test_operator, threshold)
                    @logmsg DTDebug " Testing decision: $(display_decision(decision))"
                    @yield decision, aggr_thresholds
                end # for threshold
                # push!(tested_test_operator, test_operator)
            end # for test_operator
        end # for aggregator
    end # for feature
end

abstract type AbstractFMDStumpSupport{T, WorldType} end
abstract type AbstractFMDStumpGlobalSupport{T} end

#  Stump support provides a structure for thresholds.
#   A threshold is the unique value γ for which w ⊨ <R> f ⋈ γ and:
#   if polarity(⋈) == true:      ∀ a > γ:    w ⊭ <R> f ⋈ a
#   if polarity(⋈) == false:     ∀ a < γ:    w ⊭ <R> f ⋈ a
#   for a given feature f, world w, relation R and feature f and test operator ⋈,

struct StumpFeatModalDataset{T, WorldType} <: AbstractModalDataset{T, WorldType}
    
    # Core data
    fmd                :: FeatModalDataset{T, WorldType}

    # Stump support
    fmd_m              :: AbstractFMDStumpSupport{T, WorldType}
    fmd_g              :: Union{AbstractFMDStumpGlobalSupport{T},Nothing}

    # Features and Aggregators
    featsnaggrs         :: AbstractVector{Tuple{<:ModalFeature,<:Aggregator}}
    grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}}

    function StumpFeatModalDataset{T, WorldType}(
        fmd                :: FeatModalDataset{T, WorldType},
        fmd_m              :: AbstractFMDStumpSupport{T, WorldType},
        fmd_g              :: Union{AbstractFMDStumpGlobalSupport{T},Nothing},
        featsnaggrs         :: AbstractVector{Tuple{<:ModalFeature,<:Aggregator}},
        grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
    ) where {T,WorldType<:AbstractWorld}
        @assert n_samples(fmd) == n_samples(fmd_m) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_samples for fmd and fmd_m support: $(n_samples(fmd)) and $(n_samples(fmd_m))"
        @assert n_relations(fmd) == n_relations(fmd_m) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_relations for fmd and fmd_m support: $(n_relations(fmd)) and $(n_relations(fmd_m))"
        @assert world_type(fmd) == world_type(fmd_m) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching world_type for fmd and fmd_m support: $(world_type(fmd)) and $(world_type(fmd_m))"
        @assert sum(length.(grouped_featsnaggrs)) == length(featsnaggrs) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_featsnaggrs (grouped vs flattened structure): $(sum(length.(fmd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(fmd.grouped_featsaggrsnops)) == length(featsnaggrs) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and provided featsnaggrs: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(fmd.grouped_featsaggrsnops)) == n_featsnaggrs(fmd_m) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and fmd_m support: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(n_featsnaggrs(fmd_m))"

        if fmd_g != nothing
            @assert n_samples(fmd) == n_samples(fmd_g) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_samples for fmd and fmd_g support: $(n_samples(fmd)) and $(n_samples(fmd_g))"
            # @assert somethinglike(fmd) == n_featsnaggrs(fmd_g) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching somethinglike for fmd and fmd_g support: $(somethinglike(fmd)) and $(n_featsnaggrs(fmd_g))"
            # @assert world_type(fmd) == world_type(fmd_g) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching world_type for fmd and fmd_g support: $(world_type(fmd)) and $(world_type(fmd_g))"
            @assert sum(length.(fmd.grouped_featsaggrsnops)) == n_featsnaggrs(fmd_g) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and fmd_g support: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(n_featsnaggrs(fmd_g))"
        end

        new{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
    end

    function StumpFeatModalDataset(
        fmd                 :: FeatModalDataset{T, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T,WorldType<:AbstractWorld}
        StumpFeatModalDataset{T, WorldType}(fmd, computeRelationGlob = computeRelationGlob)
    end

    function StumpFeatModalDataset{T, WorldType}(
        fmd                 :: FeatModalDataset{T, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T,WorldType<:AbstractWorld}
        
        featsnaggrs = Tuple{<:ModalFeature,<:Aggregator}[]
        grouped_featsnaggrs = AbstractVector{Tuple{<:Integer,<:Aggregator}}[]

        i_featsnaggr = 1
        for (feat,aggrsnops) in zip(fmd.features,fmd.grouped_featsaggrsnops)
            aggrs = []
            for aggr in keys(aggrsnops)
                push!(featsnaggrs, (feat,aggr))
                push!(aggrs, (i_featsnaggr,aggr))
                i_featsnaggr += 1
            end
            push!(grouped_featsnaggrs, aggrs)
        end

        # Compute modal dataset propositions and 1-modal decisions
        fmd_m, fmd_g = computeModalDatasetStumpSupport(fmd, grouped_featsnaggrs, computeRelationGlob = computeRelationGlob);

        StumpFeatModalDataset{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
    end

    function StumpFeatModalDataset(
        X                   :: OntologicalDataset{T, N, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T, N, WorldType<:AbstractWorld}
        StumpFeatModalDataset{T, WorldType}(X, computeRelationGlob = computeRelationGlob)
    end

    function StumpFeatModalDataset{T, WorldType}(
        X                   :: OntologicalDataset{T, N, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T, N, WorldType<:AbstractWorld}

        # Compute modal dataset propositions
        fmd = FeatModalDataset(X);

        StumpFeatModalDataset{T, WorldType}(fmd, computeRelationGlob = computeRelationGlob)

        # TODO bring back ModalDatasetStumpSupport computation from X. 

        # fmd_m, fmd_g = computeModalDatasetStumpSupport(X, relations, fmd.grouped_featsaggrsnops??, fmd, features, computeRelationGlob = computeRelationGlob);

        # new{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
    end
end

struct StumpFeatModalDatasetWithMemoization{T, WorldType} <: AbstractModalDataset{T, WorldType}
    
    # Core data
    fmd                :: FeatModalDataset{T, WorldType}

    # Stump support
    fmd_m              :: AbstractFMDStumpSupport{<:Union{T,Nothing}, WorldType}
    fmd_g              :: Union{AbstractFMDStumpGlobalSupport{T},Nothing}

    # Features and Aggregators
    featsnaggrs         :: AbstractVector{Tuple{<:ModalFeature,<:Aggregator}}
    grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}}

    function StumpFeatModalDatasetWithMemoization{T, WorldType}(
        fmd                :: FeatModalDataset{T, WorldType},
        fmd_m              :: AbstractFMDStumpSupport{<:Union{T,Nothing}, WorldType},
        fmd_g              :: Union{AbstractFMDStumpGlobalSupport{T},Nothing},
        featsnaggrs         :: AbstractVector{Tuple{<:ModalFeature,<:Aggregator}},
        grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
    ) where {T,WorldType<:AbstractWorld}
        @assert n_samples(fmd) == n_samples(fmd_m) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching n_samples for fmd and fmd_m support: $(n_samples(fmd)) and $(n_samples(fmd_m))"
        @assert n_relations(fmd) == n_relations(fmd_m) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching n_relations for fmd and fmd_m support: $(n_relations(fmd)) and $(n_relations(fmd_m))"
        @assert world_type(fmd) == world_type(fmd_m) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching world_type for fmd and fmd_m support: $(world_type(fmd)) and $(world_type(fmd_m))"
        @assert sum(length.(grouped_featsnaggrs)) == length(featsnaggrs) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching n_featsnaggrs (grouped vs flattened structure): $(sum(length.(fmd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(fmd.grouped_featsaggrsnops)) == length(featsnaggrs) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and provided featsnaggrs: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
        @assert sum(length.(fmd.grouped_featsaggrsnops)) == n_featsnaggrs(fmd_m) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and fmd_m support: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(n_featsnaggrs(fmd_m))"

        if fmd_g != nothing
            @assert n_samples(fmd) == n_samples(fmd_g) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching n_samples for fmd and fmd_g support: $(n_samples(fmd)) and $(n_samples(fmd_g))"
            # @assert somethinglike(fmd) == n_featsnaggrs(fmd_g) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching somethinglike for fmd and fmd_g support: $(somethinglike(fmd)) and $(n_featsnaggrs(fmd_g))"
            # @assert world_type(fmd) == world_type(fmd_g) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching world_type for fmd and fmd_g support: $(world_type(fmd)) and $(world_type(fmd_g))"
            @assert sum(length.(fmd.grouped_featsaggrsnops)) == n_featsnaggrs(fmd_g) "Can't instantiate StumpFeatModalDatasetWithMemoization{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and fmd_g support: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(n_featsnaggrs(fmd_g))"
        end

        new{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
    end

    function StumpFeatModalDatasetWithMemoization(
        fmd                 :: FeatModalDataset{T, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T,WorldType<:AbstractWorld}
        StumpFeatModalDatasetWithMemoization{T, WorldType}(fmd, computeRelationGlob = computeRelationGlob)
    end

    function StumpFeatModalDatasetWithMemoization{T, WorldType}(
        fmd                 :: FeatModalDataset{T, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T,WorldType<:AbstractWorld}
        
        featsnaggrs = Tuple{<:ModalFeature,<:Aggregator}[]
        grouped_featsnaggrs = AbstractVector{Tuple{<:Integer,<:Aggregator}}[]

        i_featsnaggr = 1
        for (feat,aggrsnops) in zip(fmd.features,fmd.grouped_featsaggrsnops)
            aggrs = []
            for aggr in keys(aggrsnops)
                push!(featsnaggrs, (feat,aggr))
                push!(aggrs, (i_featsnaggr,aggr))
                i_featsnaggr += 1
            end
            push!(grouped_featsnaggrs, aggrs)
        end

        # Compute modal dataset propositions and 1-modal decisions
        fmd_m, fmd_g = computeModalDatasetStumpSupport(fmd, grouped_featsnaggrs, computeRelationGlob = computeRelationGlob, simply_init_modal = true);

        StumpFeatModalDatasetWithMemoization{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
    end

    function StumpFeatModalDatasetWithMemoization(
        X                   :: OntologicalDataset{T, N, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T, N, WorldType<:AbstractWorld}
        StumpFeatModalDatasetWithMemoization{T, WorldType}(X, computeRelationGlob = computeRelationGlob)
    end

    function StumpFeatModalDatasetWithMemoization{T, WorldType}(
        X                   :: OntologicalDataset{T, N, WorldType};
        computeRelationGlob :: Bool = true,
    ) where {T, N, WorldType<:AbstractWorld}

        # Compute modal dataset propositions
        fmd = FeatModalDataset(X);

        StumpFeatModalDatasetWithMemoization{T, WorldType}(fmd, computeRelationGlob = computeRelationGlob)

        # TODO bring back ModalDatasetStumpSupport computation from X. 

        # fmd_m, fmd_g = computeModalDatasetStumpSupport(X, relations, fmd.grouped_featsaggrsnops??, fmd, features, computeRelationGlob = computeRelationGlob);

        # new{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
    end
end

StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType} = Union{StumpFeatModalDataset{T,WorldType},StumpFeatModalDatasetWithMemoization{T,WorldType}}


featsnaggrs(X::StumpFeatModalDatasetWithOrWithoutMemoization)         = X.featsnaggrs
grouped_featsnaggrs(X::StumpFeatModalDatasetWithOrWithoutMemoization) = X.grouped_featsnaggrs
relations(X::StumpFeatModalDatasetWithOrWithoutMemoization)           = relations(X.fmd)
initws_function(X::StumpFeatModalDatasetWithOrWithoutMemoization,  args...) = initws_function(X.fmd, args...)
acc_function(X::StumpFeatModalDatasetWithOrWithoutMemoization,     args...) = acc_function(X.fmd, args...)
accAll_function(X::StumpFeatModalDatasetWithOrWithoutMemoization,  args...) = accAll_function(X.fmd, args...)
accrepr_function(X::StumpFeatModalDatasetWithOrWithoutMemoization, args...) = accrepr_function(X.fmd, args...)
features(X::StumpFeatModalDatasetWithOrWithoutMemoization)           = features(X.fmd)
grouped_featsaggrsnops(X::StumpFeatModalDatasetWithOrWithoutMemoization)  = grouped_featsaggrsnops(X.fmd)


size(X::StumpFeatModalDatasetWithOrWithoutMemoization)             where {T,N} =  (size(X.fmd), size(X.fmd_m), (isnothing(X.fmd_g) ? nothing : size(X.fmd_g)))
n_samples(X::StumpFeatModalDatasetWithOrWithoutMemoization{T, WorldType}) where {T, WorldType}   = n_samples(X.fmd)::Int64
n_features(X::StumpFeatModalDatasetWithOrWithoutMemoization{T, WorldType}) where {T, WorldType}  = n_features(X.fmd)
n_relations(X::StumpFeatModalDatasetWithOrWithoutMemoization{T, WorldType}) where {T, WorldType} = n_relations(X.fmd)
# getindex(X::StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType}, args::Vararg) where {T,WorldType} = getindex(X.fmd, args...)
world_type(X::StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType}) where {T,WorldType<:AbstractWorld} = WorldType

slice_dataset(X::StumpFeatModalDatasetWithOrWithoutMemoization, inds::AbstractVector{<:Integer}; args...) =
    typeof(X)(
        slice_dataset(X.fmd, inds; args...),
        slice_dataset(X.fmd_m, inds; args...),
        (isnothing(X.fmd_g) ? nothing : slice_dataset(X.fmd_g, inds; args...)),
        X.featsnaggrs,
        X.grouped_featsnaggrs)

display_structure(X::StumpFeatModalDataset; indent_str = "") = begin
    out = "$(typeof(X))\t$((Base.summarysize(X.fmd) + Base.summarysize(X.fmd_m) + Base.summarysize(X.fmd_g)) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ fmd\t$(Base.summarysize(X.fmd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t(shape $(Base.size(X.fmd)))\n"
    out *= indent_str * "├ fmd_m\t$(Base.summarysize(X.fmd_m) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t(shape $(Base.size(X.fmd_m)))\n"
    out *= indent_str * "└ fmd_g\t$(Base.summarysize(X.fmd_g) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t"
    if !isnothing(X.fmd_g)
        out *= "\t(shape $(Base.size(X.fmd_g)))"
    else
        out *= "\t−"
    end
    out
end

find_feature_id(X::StumpFeatModalDatasetWithOrWithoutMemoization, feature::ModalFeature) =
    findall(x->x==feature, features(X))[1]
find_relation_id(X::StumpFeatModalDatasetWithOrWithoutMemoization, relation::AbstractRelation) =
    findall(x->x==relation, relations(X))[1]
find_featsnaggr_id(X::StumpFeatModalDatasetWithOrWithoutMemoization, feature::ModalFeature, aggregator::Aggregator) =
    findall(x->x==(feature, aggregator), featsnaggrs(X))[1]

get_gamma(
        X::StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType},
        i_instance::Integer,
        w::WorldType,
        feature::ModalFeature) where {WorldType<:AbstractWorld, T} = get_gamma(X.fmd, i_instance, w, feature)

get_global_gamma(
        X::StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType},
        i_instance::Integer,
        feature::ModalFeature,
        test_operator::TestOperatorFun) where {WorldType<:AbstractWorld, T} = begin
            # @assert !isnothing(X.fmd_g) "Error. StumpFeatModalDatasetWithOrWithoutMemoization must be built with computeRelationGlob = true for it to be ready to test global decisions."
            i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
            X.fmd_g[i_instance, i_featsnaggr]
end

get_modal_gamma(
        X::StumpFeatModalDataset{T,WorldType},
        i_instance::Integer,
        w::WorldType,
        relation::AbstractRelation,
        feature::ModalFeature,
        test_operator::TestOperatorFun) where {WorldType<:AbstractWorld, T} = begin
            i_relation = find_relation_id(X, relation)
            i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
            X.fmd_m[i_instance, w, i_featsnaggr, i_relation]
end

test_decision(
        X::StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType},
        i_instance::Integer,
        w::WorldType,
        decision::Decision) where {T, WorldType<:AbstractWorld} = begin
    if !is_modal_decision(decision)
        test_decision(X, i_instance, w, decision.feature, decision.test_operator, decision.threshold)
    else
        gamma = begin
            if relation isa ModalLogic._RelationGlob
                get_global_gamma(X, i_instance, decision.feature, decision.test_operator)
            else
                get_modal_gamma(X, i_instance, w, decision.relation, decision.feature, decision.test_operator)
            end
        end
        evaluate_thresh_decision(decision.test_operator, gamma, decision.threshold)
    end
end

Base.@propagate_inbounds @resumable function generate_propositional_feasible_decisions(
        X::StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType},
        args...
        ) where {T, WorldType<:AbstractWorld}
        for decision in generate_propositional_feasible_decisions(X.fmd, args...)
            @yield decision
        end
end

Base.@propagate_inbounds @resumable function generate_global_feasible_decisions(
        X::StumpFeatModalDatasetWithOrWithoutMemoization{T,WorldType},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:AbstractWorld}
    relation = RelationGlob
    n_instances = length(instances_inds)
    
    @assert !isnothing(X.fmd_g) "Error. StumpFeatModalDatasetWithOrWithoutMemoization must be built with computeRelationGlob = true for it to be ready to generate global decisions."

    # For each feature
    @inbounds for i_feature in features_inds
        feature = features(X)[i_feature]
        @logmsg DTDebug "Feature $(i_feature): $(feature)"

        # operators for each aggregator
        aggrsnops = grouped_featsaggrsnops(X)[i_feature]
        # println(aggrsnops)
        # Vector of aggregators
        aggregators_with_ids = grouped_featsnaggrs(X)[i_feature]
        # println(aggregators_with_ids)

        # dict->vector
        # aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

        # # TODO use this optimized version:
        #   thresholds can in fact be directly given by slicing fmd_g and permuting the two dimensions
        # aggregators_ids = fst.(aggregators_with_ids)
        # thresholds = transpose(X.fmd_g[instances_inds, aggregators_ids])

        # Initialize thresholds with the bottoms
        thresholds = Array{T,2}(undef, length(aggregators_with_ids), n_instances)
        for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)
            thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
        end
        
        # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
        for (instance_id,i_instance) in enumerate(instances_inds)
            @logmsg DTDetail " Instance $(instance_id)/$(n_instances)"
            for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
                gamma = X.fmd_g[i_instance, i_featsnaggr]
                thresholds[i_aggr,instance_id] = aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,instance_id])
                # println(gamma)
                # println(thresholds[i_aggr,instance_id])
            end
        end

        # println(thresholds)
        @logmsg DTDebug "thresholds: " thresholds

        # For each aggregator
        for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)

            # println(aggr)

            aggr_thresholds = thresholds[i_aggr,:]
            aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))

            for (i_test_operator,test_operator) in enumerate(aggrsnops[aggr])
                @logmsg DTDetail " Test operator $(test_operator)"
                
                # Look for the best threshold 'a', as in propositions like "feature >= a"
                for threshold in aggr_domain
                    decision = Decision(relation, feature, test_operator, threshold)
                    @logmsg DTDebug " Testing decision: $(display_decision(decision))"
                    @yield decision, aggr_thresholds
                end # for threshold
            end # for test_operator
        end # for aggregator
    end # for feature
end


Base.@propagate_inbounds @resumable function generate_modal_feasible_decisions(
        X::StumpFeatModalDataset{T,WorldType},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        modal_relations_inds::AbstractVector{<:Integer},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:AbstractWorld}
    n_instances = length(instances_inds)

    # For each relational operator
    @inbounds for i_relation in modal_relations_inds
        relation = relations(X)[i_relation]
        @logmsg DTDebug "Relation $(relation)..."

        # For each feature
        for i_feature in features_inds
            feature = features(X)[i_feature]
            @logmsg DTDebug "Feature $(i_feature): $(feature)"

            # operators for each aggregator
            aggrsnops = grouped_featsaggrsnops(X)[i_feature]
            # Vector of aggregators
            aggregators_with_ids = grouped_featsnaggrs(X)[i_feature]

            # dict->vector
            # aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

            # Initialize thresholds with the bottoms
            thresholds = Array{T,2}(undef, length(aggregators_with_ids), n_instances)
            for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)
                thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
            end

            # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
                for (i_instance,instance_id) in enumerate(instances_inds)
                @logmsg DTDetail " Instance $(i_instance)/$(n_instances)"
                worlds = Sf[i_instance] # TODO could also use accrepr_functions here?

                # TODO also try this instead (TODO fix first)
                # values = [X.fmd_m[instance_id, w, i_feature] for w in worlds]
                # thresholds[:,i_instance] = map((_,aggr)->aggr(values), aggregators_with_ids)
                    
                for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
                    for w in worlds
                        gamma = X.fmd_m[instance_id, w, i_featsnaggr, i_relation]
                        thresholds[i_aggr,i_instance] = ModalLogic.aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,i_instance])
                    end
                end
            end

            @logmsg DTDebug "thresholds: " thresholds

            # For each aggregator
            for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)

                aggr_thresholds = thresholds[i_aggr,:]
                aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))

                for (i_test_operator,test_operator) in enumerate(aggrsnops[aggr])
                    @logmsg DTDetail " Test operator $(test_operator)"
                    
                    # Look for the best threshold 'a', as in propositions like "feature >= a"
                    for threshold in aggr_domain
                        decision = Decision(relation, feature, test_operator, threshold)
                        @logmsg DTDebug " Testing decision: $(display_decision(decision))"
                        @yield decision, aggr_thresholds
                    end # for threshold
                end # for test_operator
            end # for aggregator
        end # for feature
    end # for relation
end


display_structure(X::StumpFeatModalDatasetWithMemoization; indent_str = "") = begin
    out = "$(typeof(X))\t$((Base.summarysize(X.fmd) + Base.summarysize(X.fmd_m) + Base.summarysize(X.fmd_g)) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
    out *= indent_str * "├ fmd\t$(Base.summarysize(X.fmd) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
        out *= "\t(shape $(Base.size(X.fmd.fwd)))\n"
        # out *= "\t(shape $(Base.size(X.fmd.fwd)), $(n_nothing) nothings, $((1-(n_nothing    / size(X.fmd)))*100)%)\n"
    # TODO n_nothing_m = count_nothings(X.fmd_m)
    n_nothing_m = count(isnothing, X.fmd_m.d)
    out *= indent_str * "├ fmd_m\t$(Base.summarysize(X.fmd_m) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
        out *= "\t(shape $(Base.size(X.fmd_m)), $(n_nothing_m) nothings)\n" # , $((1-(n_nothing_m  / n_v(X.fmd_m)))*100)%)\n"
    out *= indent_str * "└ fmd_g\t$(Base.summarysize(X.fmd_g) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t"
    if !isnothing(X.fmd_g)
        # n_nothing_g = count(isnothing, X.fmd_g)
        # out *= "\t(shape $(Base.size(X.fmd_g)), $(n_nothing_g) nothings, $((1-(n_nothing_g  / size(X.fmd_g)))*100)%)"
        out *= "\t(shape $(Base.size(X.fmd_g)))"
    else
        out *= "\t−"
    end
    out
end

# get_global_gamma(
#       X::StumpFeatModalDatasetWithMemoization{T,WorldType},
#       i_instance::Integer,
#       feature::ModalFeature,
#       test_operator::TestOperatorFun) where {WorldType<:AbstractWorld, T} = begin
#   @assert !isnothing(X.fmd_g) "Error. StumpFeatModalDatasetWithMemoization must be built with computeRelationGlob = true for it to be ready to test global decisions."
#   i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
#   # if !isnothing(X.fmd_g[i_instance, i_featsnaggr])
#   X.fmd_g[i_instance, i_featsnaggr]
#   # else
#   #   i_feature = find_feature_id(X, feature)
#   #   aggregator = existential_aggregator(test_operator)
#   #   cur_fwd_slice = modalDatasetChannelSlice(X.fmd.fwd, i_instance, i_feature)
#   #   accessible_worlds = enumReprAll(WorldType, accrepr_function(X.fmd, i_instance), feature, aggregator)
#   #   gamma = computeModalThreshold(cur_fwd_slice, accessible_worlds, aggregator)
#   #   FMDStumpGlobalSupportSet(X.fmd_g, i_instance, i_featsnaggr, gamma)
#   # end
# end

# TODO scan this value for an example problem and different number of threads

using Random
coin_flip_memoiz_rng = Random.default_rng()

cfnls_max = 0.8
# cfnls_k = 5.9
cfnls_k = 30
coin_flip_no_look_StumpFeatModalDatasetWithMemoization_value = cfnls_max*cfnls_k/((Threads.nthreads())-1+cfnls_k)
coin_flip_no_look_StumpFeatModalDatasetWithMemoization() = (rand(coin_flip_memoiz_rng) >= coin_flip_no_look_StumpFeatModalDatasetWithMemoization_value)
# coin_flip_no_look_StumpFeatModalDatasetWithMemoization() = false

get_modal_gamma(
        X::StumpFeatModalDatasetWithMemoization{T,WorldType},
        i_instance::Integer,
        w::WorldType,
        relation::AbstractRelation,
        feature::ModalFeature,
        test_operator::TestOperatorFun) where {WorldType<:AbstractWorld, T} = begin
    i_relation = find_relation_id(X, relation)
    aggregator = existential_aggregator(test_operator)
    i_featsnaggr = find_featsnaggr_id(X, feature, aggregator)
    # if coin_flip_no_look_StumpFeatModalDatasetWithMemoization() || 
    if false || 
            isnothing(X.fmd_m[i_instance, w, i_featsnaggr, i_relation])
        i_feature = find_feature_id(X, feature)
        cur_fwd_slice = modalDatasetChannelSlice(X.fmd.fwd, i_instance, i_feature)
        accessible_worlds = accrepr_function(X.fmd, i_instance)(feature, aggregator, w, relation)
        gamma = computeModalThreshold(cur_fwd_slice, accessible_worlds, aggregator)
        FMDStumpSupportSet(X.fmd_m, w, i_instance, i_featsnaggr, i_relation, gamma)
    else
        X.fmd_m[i_instance, w, i_featsnaggr, i_relation]
    end
end


Base.@propagate_inbounds @resumable function generate_modal_feasible_decisions(
        X::StumpFeatModalDatasetWithMemoization{T,WorldType},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        modal_relations_inds::AbstractVector{<:Integer},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:AbstractWorld}
    n_instances = length(instances_inds)

    # For each relational operator
    @inbounds for i_relation in modal_relations_inds
        relation = relations(X)[i_relation]
        @logmsg DTDebug "Relation $(relation)..."

        # For each feature
        for i_feature in features_inds
            feature = features(X)[i_feature]
            @logmsg DTDebug "Feature $(i_feature): $(feature)"

            # operators for each aggregator
            aggrsnops = grouped_featsaggrsnops(X)[i_feature]
            # Vector of aggregators
            aggregators_with_ids = grouped_featsnaggrs(X)[i_feature]

            # dict->vector
            # aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

            # Initialize thresholds with the bottoms
            thresholds = Array{T,2}(undef, length(aggregators_with_ids), n_instances)
            for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)
                thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
            end

            # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
            for (instance_id,i_instance) in enumerate(instances_inds)
                @logmsg DTDetail " Instance $(instance_id)/$(n_instances)"
                worlds = Sf[instance_id] # TODO could also use accrepr_functions here?

                # TODO also try this instead (TODO fix first)
                # values = [X.fmd_m[i_instance, w, i_feature] for w in worlds]
                # thresholds[:,instance_id] = map((_,aggr)->aggr(values), aggregators_with_ids)
                    
                for (i_aggr,(i_featsnaggr,aggregator)) in enumerate(aggregators_with_ids)
                    for w in worlds
                        gamma = 
                            # if coin_flip_no_look_StumpFeatModalDatasetWithMemoization() || 
                            if false || 
                                isnothing(X.fmd_m[i_instance, w, i_featsnaggr, i_relation])
                                cur_fwd_slice = modalDatasetChannelSlice(X.fmd.fwd, i_instance, i_feature)
                                accessible_worlds = accrepr_function(X.fmd, i_instance)(feature, aggregator, w, relation)
                                gamma = computeModalThreshold(cur_fwd_slice, accessible_worlds, aggregator)
                                FMDStumpSupportSet(X.fmd_m, w, i_instance, i_featsnaggr, i_relation, gamma)
                            else
                                X.fmd_m[i_instance, w, i_featsnaggr, i_relation]
                            end
                        thresholds[i_aggr,instance_id] = aggregator_to_binary(aggregator)(gamma, thresholds[i_aggr,instance_id])
                    end
                end
            end

            @logmsg DTDebug "thresholds: " thresholds

            # For each aggregator
            for (i_aggr,(_,aggregator)) in enumerate(aggregators_with_ids)

                aggr_thresholds = thresholds[i_aggr,:]
                aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))

                for (i_test_operator,test_operator) in enumerate(aggrsnops[aggregator])
                    @logmsg DTDetail " Test operator $(test_operator)"
                    
                    # Look for the best threshold 'a', as in propositions like "feature >= a"
                    for threshold in aggr_domain
                        decision = Decision(relation, feature, test_operator, threshold)
                        @logmsg DTDebug " Testing decision: $(display_decision(decision))"
                        @yield decision, aggr_thresholds
                    end # for threshold
                end # for test_operator
            end # for aggregator
        end # for feature
    end # for relation
end


const SingleFrameGenericDataset{T} = Union{MatricialDataset{T},OntologicalDataset{T},AbstractModalDataset{T}}

struct MultiFrameModalDataset
    frames  :: AbstractVector{<:SingleFrameGenericDataset}
    # function MultiFrameModalDataset(Xs::AbstractVector{<:SingleFrameGenericDataset{<:T, <:AbstractWorld}}) where {T}
    # function MultiFrameModalDataset(Xs::AbstractVector{<:SingleFrameGenericDataset{T, <:AbstractWorld}}) where {T}
    function MultiFrameModalDataset(Xs::AbstractVector{<:SingleFrameGenericDataset})
        @assert length(Xs) > 0 && length(unique(n_samples.(Xs))) == 1 "Can't create an empty MultiFrameModalDataset or with mismatching number of samples (n_frames: $(length(Xs)), frame_sizes: $(n_samples.(Xs)))."
        new(Xs)
    end
    function MultiFrameModalDataset(X::SingleFrameGenericDataset)
        new([X])
    end
    function MultiFrameModalDataset()
        new(SingleFrameGenericDataset[])
    end
end

# TODO: test all these methods
size(X::MultiFrameModalDataset) = map(size, X.frames)
get_frame(X::MultiFrameModalDataset, i) = X.frames[i]
push_frame!(X::MultiFrameModalDataset, f::AbstractModalDataset) = push!(X.frames, f)
n_frames(X::MultiFrameModalDataset)             = length(X.frames)
n_samples(X::MultiFrameModalDataset)            = n_samples(X.frames[1])::Int64 # n_frames(X) > 0 ? n_samples(X.frames[1]) : 0
length(X::MultiFrameModalDataset)               = n_samples(X)
frames(X::MultiFrameModalDataset) = X.frames
Base.iterate(X::MultiFrameModalDataset, state=1) = state > length(X) ? nothing : (getInstance(X, state), state+1)

channel_size(X::MultiFrameModalDataset) = map(channel_size, X.frames)
# get total number of features (TODO: figure if this is useless or not)
n_features(X::MultiFrameModalDataset) = map(n_features, X.frames)
# get number of features in a single frame
n_features(X::MultiFrameModalDataset, i_frame::Integer) = n_features(X.frames[i_frame])
# TODO: Note: channel_size doesn't make sense at this point. Only the acc_functions[i] functions.
n_relations(X::MultiFrameModalDataset) = map(n_relations, X.frames)
n_relations(X::MultiFrameModalDataset, i_frame::Integer) = n_relations(X.frames[i_frame])
world_type(X::MultiFrameModalDataset, i_frame::Integer) = world_type(X.frames[i_frame])
world_types(X::MultiFrameModalDataset) = Vector{Type{<:AbstractWorld}}(world_type.(X.frames))

getInstance(X::MultiFrameModalDataset,  i_frame::Integer, idx_i::Integer, args::Vararg)  = getInstance(X.frames[i_frame], idx_i, args...)
# slice_dataset(X::MultiFrameModalDataset, i_frame::Integer, inds::AbstractVector{<:Integer}, args::Vararg)  = slice_dataset(X.frames[i_frame], inds; args...)
getChannel(X::MultiFrameModalDataset,   i_frame::Integer, idx_i::Integer, idx_f::Integer, args::Vararg)  = getChannel(X.frames[i_frame], idx_i, idx_f, args...)

# getInstance(X::MultiFrameModalDataset, idx_i::Integer, args::Vararg)  = getInstance(X.frames[i], idx_i, args...) # TODO should slice across the frames!
slice_dataset(X::MultiFrameModalDataset, inds::AbstractVector{<:Integer}; args...) =
    MultiFrameModalDataset(map(frame->slice_dataset(frame, inds; args...), X.frames))


const GenericDataset = Union{SingleFrameGenericDataset,MultiFrameModalDataset}

display_structure(Xs::MultiFrameModalDataset; indent_str = "") = begin
    out = "$(typeof(Xs))" # * "\t\t\t$(Base.summarysize(Xs) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
    for (i_frame, X) in enumerate(frames(Xs))
        if i_frame == n_frames(Xs)
            out *= "\n$(indent_str)└ "
        else
            out *= "\n$(indent_str)├ "
        end
        out *= "[$(i_frame)] "
        # \t\t\t$(Base.summarysize(X) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t(world_type: $(world_type(X)))"
        out *= display_structure(X; indent_str = indent_str * (i_frame == n_frames(Xs) ? "   " : "│  ")) * "\n"
    end
    out
end


################################################################################
# Enumerate accessible worlds
################################################################################

# Fallback: enumAccessibles works with domains AND their dimensions
enumAccessibles(S::AbstractWorldSet{WorldType}, r::AbstractRelation, channel::MatricialChannel{T,N}) where {T,N,WorldType<:AbstractWorld} = enumAccessibles(S, r, size(channel)...)
# enumAccRepr(S::Any, r::AbstractRelation, channel::MatricialChannel{T,N}) where {T,N} = enumAccRepr(S, r, size(channel)...)
# Fallback: enumAccessibles for world sets maps to enumAcc-ing their elements
#  (note: one may overload this function to provide improved implementations for special cases (e.g. ⟨L⟩ of a world set in interval algebra))
enumAccessibles(S::AbstractWorldSet{WorldType}, r::AbstractRelation, XYZ::Vararg{Integer,N}) where {T,N,WorldType<:AbstractWorld} = begin
    IterTools.imap(WorldType,
        IterTools.distinct(Iterators.flatten((enumAccBare(w, r, XYZ...) for w in S)))
    )
end
enumAccessibles(w::WorldType, r::AbstractRelation, XYZ::Vararg{Integer,N}) where {T,N,WorldType<:AbstractWorld} = begin
    IterTools.imap(WorldType, enumAccBare(w, r, XYZ...))
end

################################################################################
################################################################################

## Basic, ontology-agnostic relations:

# None relation      (RelationNone)  =  Used as the "nothing" constant
struct _RelationNone  <: AbstractRelation end; const RelationNone = _RelationNone();

# Identity relation  (RelationId)    =  S -> S
struct _RelationId    <: AbstractRelation end; const RelationId   = _RelationId();

enumAccessibles(w::WorldType,           ::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = [w] # IterTools.imap(identity, [w])
enumAccessibles(S::AbstractWorldSet{W}, ::_RelationId, XYZ::Vararg{Integer,N}) where {W<:AbstractWorld,N} = S # TODO try IterTools.imap(identity, S) ?

enumAccReprAggr(::ModalFeature, ::Aggregator, w::WorldType, r::AbstractRelation, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = enumAccessibles(w, r, XYZ...)
enumAccReprAggr(::ModalFeature, ::Aggregator, w::WorldType, r::_RelationId,      XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = enumAccessibles(w, r, XYZ...)


# computeModalThresholdDual(test_operator::TestOperatorFun, w::WorldType, relation::_RelationId, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
#   computePropositionalThresholdDual(test_operator, w, channel)
# computeModalThreshold(test_operator::TestOperatorFun, w::WorldType, relation::_RelationId, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
#   computePropositionalThreshold(test_operator, w, channel)
# computeModalThresholdMany(test_ops::Vector{<:TestOperatorFun}, w::WorldType, relation::_RelationId, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
#   computeModalThresholdMany(test_ops, w, channel)

display_rel_short(::_RelationId)  = "Id"


################################################################################
################################################################################
#
# Global relation    (RelationGlob)   =  S -> all-worlds
struct _RelationGlob   <: AbstractRelation end; const RelationGlob  = _RelationGlob();

display_rel_short(::_RelationGlob) = "G"

# Shortcut for enumerating all worlds
enumAll(::Type{WorldType}, args::Vararg) where {WorldType<:AbstractWorld} = enumAccessibles(WorldType[], RelationGlob, args...)
enumAll(::Type{WorldType}, enumAccFun::Function) where {WorldType<:AbstractWorld} = enumAccFun(WorldType[], RelationGlob)
enumReprAll(::Type{WorldType}, enumReprFun::Function, f::ModalFeature, a::Aggregator) where {WorldType<:AbstractWorld} = enumReprFun(f, a, WorldType[], RelationGlob)


################################################################################
################################################################################

export OneWorldOntology,
        # genericIntervalOntology,
        IntervalOntology,
        Interval2DOntology,
        getIntervalOntologyOfDim,
        # genericIntervalRCC8Ontology,
        IntervalRCC8Ontology,
        Interval2DRCC8Ontology,
        getIntervalRCC8OntologyOfDim,
        getIntervalRCC5OntologyOfDim,
        IntervalRCC5Ontology,
        Interval2DRCC5Ontology

minExtrema(extr::Union{NTuple{N,NTuple{2,T}},AbstractVector{NTuple{2,T}}}) where {T<:Number,N} = reduce(((fst,snd),(f,s))->(min(fst,f),max(snd,s)), extr; init=(typemax(T),typemin(T)))
maxExtrema(extr::Union{NTuple{N,NTuple{2,T}},AbstractVector{NTuple{2,T}}}) where {T<:Number,N} = reduce(((fst,snd),(f,s))->(max(fst,f),min(snd,s)), extr; init=(typemin(T),typemax(T)))
minExtrema(extr::Vararg{NTuple{2,T}}) where {T<:Number} = minExtrema(extr)
maxExtrema(extr::Vararg{NTuple{2,T}}) where {T<:Number} = maxExtrema(extr)

include("OneWorld.jl")
# include("Point.jl")

include("Interval.jl")
include("IARelations.jl")
include("TopoRelations.jl")

include("Interval2D.jl")
include("IA2DRelations.jl")
include("Topo2DRelations.jl")

# abstract type OntologyType end

const OneWorldOntology   = Ontology{OneWorld}(AbstractRelation[])

# struct _genericIntervalOntology  <: OntologyType end; const genericIntervalOntology  = _genericIntervalOntology();
const IntervalOntology   = Ontology{Interval}(IARelations)
const Interval2DOntology = Ontology{Interval2D}(IA2DRelations)

# struct _genericIntervalRCC8Ontology  <: OntologyType end; const genericIntervalRCC8Ontology  = _genericIntervalRCC8Ontology();
const IntervalRCC8Ontology   = Ontology{Interval}(RCC8Relations)
const Interval2DRCC8Ontology = Ontology{Interval2D}(RCC8Relations)
const IntervalRCC5Ontology   = Ontology{Interval}(RCC5Relations)
const Interval2DRCC5Ontology = Ontology{Interval2D}(RCC5Relations)

getIntervalOntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalOntologyOfDim(Val(D-2))
getIntervalOntologyOfDim(::Val{1}) = IntervalOntology
getIntervalOntologyOfDim(::Val{2}) = Interval2DOntology

getIntervalRCC8OntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalRCC8OntologyOfDim(Val(D-2))
getIntervalRCC8OntologyOfDim(::Val{1}) = IntervalRCC8Ontology
getIntervalRCC8OntologyOfDim(::Val{2}) = Interval2DRCC8Ontology

getIntervalRCC5OntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalRCC5OntologyOfDim(Val(D-2))
getIntervalRCC5OntologyOfDim(::Val{1}) = IntervalRCC5Ontology
getIntervalRCC5OntologyOfDim(::Val{2}) = Interval2DRCC5Ontology


################################################################################
################################################################################

include("BuildStumpSupport.jl")

computePropositionalThreshold(feature::ModalFeature, w::AbstractWorld, instance::MatricialInstance) = yieldFunction(feature)(inst_readWorld(w, instance))


# TODO add AbstractWorldSet type
computeModalThreshold(fwd_propositional_slice::FeaturedWorldDatasetSlice{T}, worlds::Any, aggregator::Agg) where {T, Agg<:Aggregator} = begin
    
    # TODO try reduce(aggregator, worlds; init=ModalLogic.bottom(aggregator, T))
    # TODO remove this aggregator_to_binary...
    
    if length(worlds |> collect) == 0
        ModalLogic.aggregator_bottom(aggregator, T)
    else
        aggregator((w)->modalDatasetChannelSliceGet(fwd_propositional_slice, w), worlds)
    end

    # opt = aggregator_to_binary(aggregator)
    # threshold = ModalLogic.bottom(aggregator, T)
    # for w in worlds
    #   e = modalDatasetChannelSliceGet(fwd_propositional_slice, w)
    #   threshold = opt(threshold,e)
    # end
    # threshold
end

################################################################################
################################################################################

# TODO A relation can be defined as a union of other relations.
# In this case, thresholds can be computed by maximization/minimization of the
#  thresholds referred to the relations involved.
# abstract type AbstractRelation end
# struct _UnionOfRelations{T<:NTuple{N,<:AbstractRelation} where N} <: AbstractRelation end;

# computeModalThresholdDual(test_operator::TestOperatorFun, w::WorldType, relation::R where R<:_UnionOfRelations{relsTuple}, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
#   computePropositionalThresholdDual(test_operator, w, channel)
#   fieldtypes(relsTuple)
# computeModalThreshold(test_operator::TestOperatorFun, w::WorldType, relation::R where R<:_UnionOfRelations{relsTuple}, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
#   computePropositionalThreshold(test_operator, w, channel)
#   fieldtypes(relsTuple)

################################################################################
################################################################################

# Perform the modal step, that is, evaluate a modal formula
#  on a domain, and eventually compute the new world set.
function modal_step(
        X::Union{AbstractModalDataset{T,WorldType},OntologicalDataset{T,N,WorldType}},
        i_instance::Integer,
        worlds::WorldSetType,
        decision::Decision{T},
        returns_survivors::Union{Val{true},Val{false}} = Val(false)
    ) where {T, N, WorldType<:AbstractWorld, WorldSetType<:AbstractWorldSet{WorldType}}
    @logmsg DTDetail "modal_step" worlds display_decision(decision)

    satisfied = false
    
    # TODO space for optimization here: with some relations (e.g. IA_A, IA_L) can be made smaller

    if returns_survivors isa Val{true}
        worlds_map = Dict{AbstractWorld,AbstractWorldSet{WorldType}}()
    end
    if length(worlds) == 0
        # If there are no neighboring worlds, then the modal decision is not met
        @logmsg DTDetail "   No accessible world"
    else
        # Otherwise, check whether at least one of the accessible worlds witnesses truth of the decision.
        # TODO rewrite with new_worlds = map(...acc_worlds)
        # Initialize new worldset
        new_worlds = WorldSetType()

        # List all accessible worlds
        acc_worlds = 
            if returns_survivors isa Val{true}
                Threads.@threads for curr_w in worlds
                    worlds_map[curr_w] = acc_function(X, i_instance)(curr_w, decision.relation)
                end
                unique(cat([ worlds_map[k] for k in keys(worlds_map) ]...; dims = 1))
            else
                acc_function(X, i_instance)(worlds, decision.relation)
            end

        for w in acc_worlds
            if test_decision(X, i_instance, w, decision.feature, decision.test_operator, decision.threshold)
                # @logmsg DTDetail " Found world " w ch_readWorld ... ch_readWorld(w, channel)
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
        @logmsg DTDetail "   YES" worlds
    else
        @logmsg DTDetail "   NO"
    end
    if returns_survivors isa Val{true}
        return (satisfied, worlds, worlds_map)
    else
        return (satisfied, worlds)
    end
end

test_decision(
        X::SingleFrameGenericDataset{T},
        i_instance::Integer,
        w::AbstractWorld,
        feature::ModalFeature,
        test_operator::TestOperatorFun,
        threshold::T) where {T} = begin
    gamma = get_gamma(X, i_instance, w, feature)
    evaluate_thresh_decision(test_operator, gamma, threshold)
end

test_decision(
        X::SingleFrameGenericDataset{T},
        i_instance::Integer,
        w::AbstractWorld,
        decision::Decision{T}) where {T} = begin
    instance = getInstance(X, i_instance)

    aggregator = existential_aggregator(decision.test_operator)
    
    worlds = enumAccReprAggr(decision.feature, aggregator, w, decision.relation, inst_channel_size(instance)...)
    gamma = if length(worlds |> collect) == 0
        ModalLogic.aggregator_bottom(aggregator, T)
    else
        aggregator((w)->get_gamma(X, i_instance, w, decision.feature), worlds)
    end

    evaluate_thresh_decision(decision.test_operator, gamma, decision.threshold)
end


export generate_feasible_decisions
                # ,
                # generate_propositional_feasible_decisions,
                # generate_global_feasible_decisions,
                # generate_modal_feasible_decisions

Base.@propagate_inbounds @resumable function generate_feasible_decisions(
        X::AbstractModalDataset{T,WorldType},
        instances_inds::AbstractVector{<:Integer},
        Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
        allow_propositional_decisions::Bool,
        allow_modal_decisions::Bool,
        allow_global_decisions::Bool,
        modal_relations_inds::AbstractVector{<:Integer},
        features_inds::AbstractVector{<:Integer},
        ) where {T, WorldType<:AbstractWorld}
    # Propositional splits
    if allow_propositional_decisions
        for decision in generate_propositional_feasible_decisions(X, instances_inds, Sf, features_inds)
            @yield decision
        end
    end
    # Global splits
    if allow_global_decisions
        for decision in generate_global_feasible_decisions(X, instances_inds, Sf, features_inds)
            @yield decision
        end
    end
    # Modal splits
    if allow_modal_decisions
        for decision in generate_modal_feasible_decisions(X, instances_inds, Sf, modal_relations_inds, features_inds)
            @yield decision
        end
    end
end

end # module
