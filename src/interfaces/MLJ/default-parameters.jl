using SoleModels.DimensionalDatasets
using SoleModels.DimensionalDatasets: UniformFullDimensionalLogiset
using SoleModels: ScalarOneStepMemoset, AbstractFullMemoset
using SoleModels: naturalconditions

const ALLOW_GLOBAL_SPLITS = true

const mlj_default_max_depth = typemax(Int64)

const mlj_mdt_default_min_samples_leaf = 4
const mlj_mdt_default_min_purity_increase = 0.002
const mlj_mdt_default_max_purity_at_leaf = Inf

const mlj_mrf_default_min_samples_leaf = 1
const mlj_mrf_default_min_purity_increase = -Inf
const mlj_mrf_default_max_purity_at_leaf = Inf
const mlj_mrf_default_ntrees = 50

AVAILABLE_RELATIONS = OrderedDict{Symbol,Vector{<:AbstractRelation}}([
    :none       => AbstractRelation[],
    :IA         => [globalrel, SoleLogics.IARelations...],
    :IA3        => [globalrel, SoleLogics.IA3Relations...],
    :IA7        => [globalrel, SoleLogics.IA7Relations...],
    :RCC5       => [globalrel, SoleLogics.RCC8Relations...],
    :RCC8       => [globalrel, SoleLogics.RCC5Relations...],
])

mlj_default_relations = nothing

mlj_default_relations_str = "either no relation (adimensional data), " *
    "IA7 interval relations (1-dimensional data), or RCC5 relations " *
    "(2-dimensional data)."

function defaultrelations(dataset)
    if dataset isa Union{
        SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset}} where {W,U,FT,FR,L,N},
        SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset,<:AbstractFullMemoset}} where {W,U,FT,FR,L,N},
    }
        MDT.relations(dataset)
    elseif dimensionality(dataset) == 0
        AVAILABLE_RELATIONS[:none]
    elseif dimensionality(dataset) == 1
        AVAILABLE_RELATIONS[:IA7]
    elseif dimensionality(dataset) == 2
        AVAILABLE_RELATIONS[:RCC5]
    else
        error("Cannot infer relation set for dimensionality $(dimensionality(dataset)). " *
            "Dimensionality should be 0, 1 or 2.")
    end
end

# Infer relation set from model.relations parameter and the (unimodal) dataset.
function readrelations(model, dataset)
    if model.relations == mlj_default_relations
        defaultrelations(dataset)
    else
        if dataset isa Union{
            SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset}} where {W,U,FT,FR,L,N},
            SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset,<:AbstractFullMemoset}} where {W,U,FT,FR,L,N},
        }
            rels = model.relations(dataset)
            @assert issubset(rels, MDT.relations(dataset)) "Could not find " *
                "specified relations $(displaysyntaxvector(rels)) in " *
                "logiset relations $(displaysyntaxvector(MDT.relations(dataset)))."
            rels
        else
            model.relations(dataset)
        end
    end
end


mlj_default_conditions = nothing

mlj_default_conditions_str = "scalar conditions (test operators ≥ and <) " *
    "on either minimum and maximum feature functions (if dimensional data is provided), " *
    "or the features of the logiset, if one is provided."

function defaultconditions(dataset)
    if dataset isa Union{
        SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset}} where {W,U,FT,FR,L,N},
        SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset,<:AbstractFullMemoset}} where {W,U,FT,FR,L,N},
    }
        MDT.metaconditions(dataset)
    elseif dataset isa UniformFullDimensionalLogiset
        vcat([
            [
                ScalarMetaCondition(feature, ≥),
                (all(i_instance->SoleModels.nworlds(frame(dataset, i_instance)) == 1, 1:ninstances(dataset)) ?
                    [] :
                    [ScalarMetaCondition(feature, <)]
                )...
            ]
        for feature in features(dataset)]...)
    else
        if all(i_instance->SoleModels.nworlds(frame(dataset, i_instance)) == 1, 1:ninstances(dataset))
            [identity]
        else
            [minimum, maximum]
        end
    end
end

function readconditions(model, dataset)
    conditions = begin
        if model.conditions == mlj_default_conditions
            defaultconditions(dataset)
        else
            model.conditions
        end
    end

    if dataset isa Union{
        SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset}} where {W,U,FT,FR,L,N},
        SupportedLogiset{W,U,FT,FR,L,N,<:Tuple{<:ScalarOneStepMemoset,<:AbstractFullMemoset}} where {W,U,FT,FR,L,N},
    }
        @assert issubset(conditions, MDT.metaconditions(dataset)) "Could not find " *
            "specified conditions $(displaysyntaxvector(conditions)) in " *
            "logiset metaconditions $(displaysyntaxvector(MDT.metaconditions(dataset)))."
        conditions
    else
        naturalconditions(dataset, conditions)
    end
end

mlj_default_initconditions = MDT.start_without_world

mlj_default_initconditions_str = ":start_with_global (i.e., starting with a global decision, such as ⟨G⟩ min(V1) > 2)."

AVAILABLE_INITCONDITIONS = OrderedDict{Symbol,InitialCondition}([
    :start_with_global => MDT.start_without_world,
    :start_at_center   => MDT.start_at_center,
])
