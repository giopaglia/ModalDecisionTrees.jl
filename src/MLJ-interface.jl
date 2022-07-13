# Inspired from JuliaAI/MLJDecisionTreeInterface.jl

# Reference: https://alan-turing-institute.github.io/MLJ.jl/dev/quick_start_guide_to_adding_models/#Quick-Start-Guide-to-Adding-Models
# Reference: https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/

# TODO remove redundance
export ModalDecisionTree, ModalRandomForest

module MLJInterface

export ModalDecisionTree, ModalRandomForest


using MLJBase
import MLJModelInterface
using MLJModelInterface.ScientificTypesBase
using CategoricalArrays
using ..ModalDecisionTrees
import Tables

import Base: show

using Random
using DataFrames
import Random.GLOBAL_RNG

const MMI = MLJModelInterface
const MDT = ModalDecisionTrees
const PKG = "ModalDecisionTrees"

############################################################################################
############################################################################################
############################################################################################

struct ModelPrinter{T}
    model::T
    frame_grouping::Union{Nothing,AbstractVector{<:AbstractVector},AbstractVector{<:AbstractDict}}
end
(c::ModelPrinter)(; args...) = c(c.model; args...)
(c::ModelPrinter)(model; max_depth = 5) = MDT.print_model(model; attribute_names_map = c.frame_grouping, max_depth = max_depth)

Base.show(io::IO, c::ModelPrinter) =
    print(io, "ModelPrinter object (call with display depth)")

############################################################################################
############################################################################################
############################################################################################

_size = ((x)->(hasmethod(size, (typeof(x),)) ? size(x) : missing))

function separate_attributes_into_frames(X)

    columns = names(X)
    channel_sizes = [unique(_size.(X)[:,col]) for col in columns]

    # Must have common channel size across instances
    _uniform_columns = (length.(channel_sizes) .== 1)
    _nonmissing_columns = (((cs)->all((!).(ismissing.(cs)))).(channel_sizes))

    __uniform_cols = columns[(!).(_uniform_columns)]
    if length(__uniform_cols) > 0
        println("Dropping columns due to non-uniform channel size across instances: $(__uniform_cols)...")
    end
    __uniform_non_missing_cols = columns[_uniform_columns .&& (!).(_nonmissing_columns)]
    if length(__uniform_non_missing_cols) > 0
        println("Dropping columns due to missings: $(__uniform_non_missing_cols)...")
    end
    _good_columns = _uniform_columns .&& _nonmissing_columns

    _good_columns = _uniform_columns .&& _nonmissing_columns
    channel_sizes = channel_sizes[_good_columns]
    columns = columns[_good_columns]
    channel_sizes = getindex.(channel_sizes, 1)

    unique_channel_sizes = sort(unique(channel_sizes))

    frame_ids = [findfirst((ucs)->(ucs==cs), unique_channel_sizes) for cs in channel_sizes]

    frames = Dict([frame_id => [] for frame_id in unique(frame_ids)])
    for (frame_id, col) in zip(frame_ids, columns)
        push!(frames[frame_id], col)
    end
    frames = [frames[frame_id] for frame_id in unique(frame_ids)]

    # println("Frames:");
    # println()
    # display(collect(((x)->Pair(x...)).((enumerate(frames)))));

    frames
end


function __moving_window_without_overflow_fixed_num(
    npoints::Integer;
    nwindows::Integer,
    relative_overlap::AbstractFloat,
)::AbstractVector{UnitRange{Int}}
    # Code by Giovanni Pagliarini (@giopaglia) & Federico Manzella (@ferdiu)
    # 
    # start = 1+half_context
    # stop = npoints-half_context
    # step = (stop-start+1)/nwindows
    # half_context = step*relative_overlap/2

    # half_context = relative_overlap * (npoints-1) / (2* nwindows+2*relative_overlap)
    half_context = relative_overlap * npoints / (2* nwindows+2*relative_overlap)
    start = 1+half_context
    stop = npoints-half_context
    step = (stop-start+1)/nwindows
    
    # _w = floor(Int, step+2*half_context)
    # _w = floor(Int, ((stop-start+1)/nwindows)+2*half_context)
    # _w = floor(Int, ((npoints-half_context)-(1+half_context)+1)/nwindows+2*half_context)
    # _w = floor(Int, (npoints-2*half_context)/nwindows+2*half_context)
    _w = floor(Int, (npoints-2*half_context)/nwindows + 2*half_context)

    # println("step: ($(stop)-$(start)+1)/$(nwindows) = ($(stop-start+1)/$(nwindows) = $(step)")
    # println("half_context: $(half_context)")
    # first_points = range(start=start, stop=stop, length=nwindows+1)[1:end-1]
    first_points = range(start=start, stop=stop, length=nwindows+1)[1:end-1] # TODO needs Julia 1.7: warn user
    first_points = map((x)->x-half_context, first_points)
    @assert isapprox(first_points[1], 1.0)
    # println("first_points: $(collect(first_points))")
    # println("window: $(step)+$(2*half_context) = $(step+2*half_context)")
    # println("windowi: $(_w)")
    first_points = map((x)->round(Int, x), first_points)
    # first_points .|> (x)->(x+step/2) .|> (x)->(x-size/2,x+size/2)
    # first_points .|> (x)->(max(1.0,x-half_context),min(x+step+half_context,npoints))
    # first_points .|> (x)->(x-half_context,x+step+half_context)
    first_points .|> (xi)->(xi:xi+_w-1)
end

function moving_average(
    X::AbstractArray{T,3},
    nwindows::Integer,
    relative_overlap::AbstractFloat = .5,
) where {T}
    npoints, n_attributes, n_instances = size(X)
    new_X = similar(X, (nwindows, n_attributes, n_instances))
    for i_instance in 1:n_instances
        for i_attribute in 1:n_attributes
            new_X[:, i_attribute, i_instance] .= [mean(X[idxs, i_attribute, i_instance]) for idxs in __moving_window_without_overflow_fixed_num(npoints; nwindows = nwindows, relative_overlap = relative_overlap)]
        end
    end
    return new_X
end

function moving_average(
    X::AbstractArray{T,4},
    new_channel_size::Tuple{Integer,Integer},
    relative_overlap::AbstractFloat = .5,
) where {T}
    n_X, n_Y, n_attributes, n_instances = size(X)
    windows_1 = __moving_window_without_overflow_fixed_num(n_X; nwindows = new_channel_size[1], relative_overlap = relative_overlap)
    windows_2 = __moving_window_without_overflow_fixed_num(n_Y; nwindows = new_channel_size[2], relative_overlap = relative_overlap)
    new_X = similar(X, (new_channel_size..., n_attributes, n_instances))
    for i_instance in 1:n_instances
        for i_attribute in 1:n_attributes
            new_X[:, :, i_attribute, i_instance] .= [mean(X[idxs1, idxs2, i_attribute, i_instance]) for idxs1 in windows_1, idxs2 in windows_2]
        end
    end
    return new_X
end

function DataFrame2MultiFrameModalDataset(
        X,
        frame_grouping,
        relations,
        mixed_features,
        init_conditions,
        allow_global_splits,
        mode;
        downsizing_function = (channel_size, n_samples)->identity,
    )

    @assert mode in [:explicit, :implicit]

    @assert all((<:).(eltype.(eachcol(X)), Union{Real,AbstractVector{<:Real},AbstractMatrix{<:Real}})) "ModalDecisionTrees.jl only allows attributes that are `Real`, `AbstractVector{<:Real}` or `AbstractMatrix{<:Real}`"
    @assert ! any(map((x)->(any(MDT.ModalLogic.hasnans.(x))), eachcol(X))) "ModalDecisionTrees.jl doesn't allow NaN values"

    Xs_ic = [begin
        X_frame = X[:,frame]
        
        channel_size = unique([unique(_size.(X_frame[:, col])) for col in names(X_frame)])
        @assert length(channel_size) == 1
        @assert length(channel_size[1]) == 1
        channel_size = channel_size[1][1]
        
        channel_dim = length(channel_size)
        
        # println("$(i_frame)\tchannel size: $(channel_size)\t => $(frame_grouping)")

        _X = begin
            n_attributes = DataFrames.ncol(X_frame)
            n_samples = DataFrames.nrow(X_frame)

            # dataframe2cube(X_frame)
            common_type = Union{eltype.(eltype.(eachcol(X_frame)))...}
            common_type = common_type == Any ? Real : common_type
            _X = Array{common_type}(undef, channel_size..., n_attributes, n_samples)
            for (i_col, col) in enumerate(eachcol(X_frame))
                for (i_row, row) in enumerate(col)
                    _X[[(:) for i in 1:length(size(row))]...,i_col,i_row] = row
                end
            end

            _X = downsizing_function(channel_size, n_samples)(_X)

            _X
        end

        _relations = if isnothing(relations)
            if channel_dim == 1
                :IA
            else
                # :RCC5 TODO
                :RCC8
            end
        else
            relations
        end
        _init_conditions = if isnothing(init_conditions)
            if channel_dim in [0, 1]
                init_conditions_d[:start_with_global]
            else
                # :RCC5 TODO
                init_conditions_d[:start_at_center]
            end
        else
            init_conditions_d[init_conditions]
        end

        ontology = MDT.get_interval_ontology(channel_dim, _relations)
        # println(eltype(_X))
        __X = MDT.InterpretedModalDataset(_X, ontology, mixed_features)
        # println(MDT.display_structure(__X))

        (if mode == :implicit
            __X
        else
            WorldType = MDT.world_type(ontology)
            
            compute_relation_glob =
                WorldType != MDT.OneWorld && (
                    (allow_global_splits || _init_conditions == MDT.start_without_world)
                )
            MDT.ExplicitModalDatasetSMemo(__X, compute_relation_glob = compute_relation_glob)
        end, _init_conditions)
    end for (i_frame, frame) in enumerate(frame_grouping)]
    Xs, init_conditions = zip(Xs_ic...)
    Xs, init_conditions = collect(Xs), collect(init_conditions)
    Xs = MDT.MultiFrameModalDataset{MDT.ModalLogic.ActiveModalDataset}(Xs)
    # println(MDT.display_structure(Xs))
    Xs, init_conditions
end

tree_downsizing_function(channel_size, n_samples) = function (_X)
    channel_dim = length(channel_size)
    if channel_dim == 1
        n_points = channel_size[1]
        if n_samples > 300 && n_points > 100
            println("Warning: downsizing series of size $(n_points) to $(100) points ($(n_samples) samples). If this process gets killed, please downsize your dataset beforehand.")
            _X = moving_average(_X, 100)
        elseif n_points > 150
            println("Warning: downsizing series of size $(n_points) to $(150) points ($(n_samples) samples). If this process gets killed, please downsize your dataset beforehand.")
            _X = moving_average(_X, 150)
        end
    elseif channel_dim == 2
        if n_samples > 300 && prod(channel_size) > prod((7,7),)
            new_channel_size = min.(channel_size, (7,7))
            println("Warning: downsizing image of size $(channel_size) to $(new_channel_size) pixels ($(n_samples) samples). If this process gets killed, please downsize your dataset beforehand.")
            _X = moving_average(_X, new_channel_size)
        elseif prod(channel_size) > prod((10,10),)
            new_channel_size = min.(channel_size, (10,10))
            println("Warning: downsizing image of size $(channel_size) to $(new_channel_size) pixels ($(n_samples) samples). If this process gets killed, please downsize your dataset beforehand.")
            _X = moving_average(_X, new_channel_size)
        end
    end
    _X
end

forest_downsizing_function(channel_size, n_samples; n_trees) = function (_X)
    channel_dim = length(channel_size)
    if channel_dim == 1
        n_points = channel_size[1]
        if n_samples > 300 && n_points > 100
            println("Warning: downsizing series of size $(n_points) to $(100) points ($(n_samples) samples). If this process gets killed, please downsize your dataset beforehand.")
            _X = moving_average(_X, 100)
        elseif n_points > 150
            println("Warning: downsizing series of size $(n_points) to $(150) points ($(n_samples) samples). If this process gets killed, please downsize your dataset beforehand.")
            _X = moving_average(_X, 150)
        end
    elseif channel_dim == 2
        if n_samples > 300 && prod(channel_size) > prod((4,4),)
            new_channel_size = min.(channel_size, (4,4))
            println("Warning: downsizing image of size $(channel_size) to $(new_channel_size) pixels ($(n_samples) samples). If this process gets killed, please downsize your dataset beforehand.")
            _X = moving_average(_X, new_channel_size)
        elseif prod(channel_size) > prod((7,7),)
            new_channel_size = min.(channel_size, (7,7))
            println("Warning: downsizing image of size $(channel_size) to $(new_channel_size) pixels ($(n_samples) samples). If this process gets killed, please downsize your dataset beforehand.")
            _X = moving_average(_X, new_channel_size)
        end
    end
    _X
end

init_conditions_d = Dict([
    :start_with_global => MDT.start_without_world,
    :start_at_center   => MDT.start_at_center,
])

function _check_features(features)
    good = [
        [
            map(
            (f)->(f isa CanonicalFeature || (ret = f(ch); isa(ret, Real) && typeof(ret) == eltype(ch))),
            features) for ch in [collect(1:10), collect(1.:10.)]
        ]
    ] |> Iterators.flatten |> all
    @assert good "`features` should be a vector of scalar functions accepting on object of type `AbstractVector{T}` and returning an object of type `T`."
    # println(typeof(good))
    good
end

using ModalDecisionTrees: default_min_samples_leaf
using ModalDecisionTrees: default_min_purity_increase
using ModalDecisionTrees: default_max_purity_at_leaf
using ModalDecisionTrees: Relation
using ModalDecisionTrees: CanonicalFeature
using ModalDecisionTrees: CanonicalFeatureGeq, canonical_geq
using ModalDecisionTrees: CanonicalFeatureLeq, canonical_leq
using ModalDecisionTrees: start_without_world
using ModalDecisionTrees: start_at_center

############################################################################################
############################################################################################
############################################################################################

MMI.@mlj_model mutable struct ModalDecisionTree <: MMI.Deterministic
    # Pruning hyper-parameters
    max_depth              :: Union{Nothing,Int}           = nothing::(isnothing(_) || _ ≥ -1)
    min_samples_leaf       :: Int                          = default_min_samples_leaf::(_ ≥ 1)
    min_purity_increase    :: Float64                      = default_min_purity_increase
    max_purity_at_leaf     :: Float64                      = default_max_purity_at_leaf
    # Modal hyper-parameters
    relations              :: Union{Nothing,Symbol,Vector{<:Relation}} = (nothing)::(isnothing(nothing) || _ in [:IA, :IA3, :IA7, :RCC5, :RCC8] || _ isa AbstractVector{<:Relation})
    # TODO expand to ModalFeature
    # features               :: Vector{<:Function}           = [minimum, maximum]::(all(Iterators.flatten([(f)->(ret = f(ch); isa(ret, Real) && typeof(ret) == eltype(ch)), _) for ch in [collect(1:10), collect(1.:10.)]]))
    # features               :: Vector{<:Union{CanonicalFeature,Function}}       TODO = Vector{<:Union{CanonicalFeature,Function}}([canonical_geq, canonical_leq]) # TODO: ::(_check_features(_))
    # features               :: AbstractVector{<:CanonicalFeature}       = CanonicalFeature[canonical_geq, canonical_leq] # TODO: ::(_check_features(_))
    # features               :: Vector                       = [canonical_geq, canonical_leq] # TODO: ::(_check_features(_))
    init_conditions        :: Union{Nothing,Symbol}                       = nothing::(isnothing(_) || _ in [:start_with_global, :start_at_center])
    allow_global_splits    :: Bool                         = true
    automatic_downsizing   :: Bool                         = true
    
    # ModalDecisionTree-specific
    display_depth          :: Union{Nothing,Int}           = 5::(isnothing(_) || _ ≥ 0)
end


function MMI.fit(m::ModalDecisionTree, verbosity::Int, X, y, w=nothing)
    X = DataFrame(X)
    classes_seen  = y isa CategoricalArray ? filter(in(unique(y)), MMI.classes(y)) : unique(y)
    y  = Vector{String}(y)

    ########################################################################################
    
    max_depth            = m.max_depth
    max_depth            = isnothing(max_depth) ? typemax(Int64) : max_depth
    min_samples_leaf     = m.min_samples_leaf
    min_purity_increase  = m.min_purity_increase
    max_purity_at_leaf   = m.max_purity_at_leaf
    relations            = m.relations
    features             = [canonical_geq, canonical_leq] # TODO: m.features
    init_conditions      = m.init_conditions
    allow_global_splits  = m.allow_global_splits
    automatic_downsizing = m.automatic_downsizing
    
    display_depth        = m.display_depth

    ########################################################################################
    
    frame_grouping = separate_attributes_into_frames(X)
    Xs, init_conditions = DataFrame2MultiFrameModalDataset(
        X,
        frame_grouping,
        relations,
        features,
        init_conditions,
        allow_global_splits,
        :explicit;
        downsizing_function = (automatic_downsizing ? tree_downsizing_function : identity),
    )

    model = MDT.build_tree(
        Xs,
        y,
        w,
        ##############################################################################
        loss_function        = nothing,
        max_depth            = max_depth,
        min_samples_leaf     = min_samples_leaf,
        min_purity_increase  = min_purity_increase,
        max_purity_at_leaf   = max_purity_at_leaf,
        ##############################################################################
        n_subrelations       = identity,
        n_subfeatures        = identity,
        init_conditions      = init_conditions,
        allow_global_splits  = allow_global_splits,
        ##############################################################################
        perform_consistency_check = false,
    )

    verbosity < 2 || MDT.print_model(model; max_depth = display_depth, attribute_names_map = frame_grouping)

    feature_importance_by_count = Dict([i_attr => frame_grouping[i_frame][i_attr] for ((i_frame, i_attr), count) in MDT.attribute_countmap(model)])

    fitresult = (
        model           = model,
        frame_grouping  = frame_grouping,
        classes_seen    = classes_seen,
    )

    cache  = nothing
    report = (
        print_tree                   = ModelPrinter(model, frame_grouping),
        print_model                  = ModelPrinter(model, frame_grouping),
        frame_grouping              = frame_grouping,
        feature_importance_by_count = feature_importance_by_count,
        classes_seen = classes_seen,
    )

    return fitresult, cache, report
end

MMI.fitted_params(::ModalDecisionTree, fitresult) =
    (
        model           = fitresult.model,
        frame_grouping  = fitresult.frame_grouping,
    )

function MMI.predict(m::ModalDecisionTree, fitresult, Xnew) #, ynew = nothing)
    Xnew = DataFrame(Xnew)
    # ynew = Vector{String}(ynew)
    ynew = nothing
    @assert isnothing(ynew)

    relations = m.relations
    features = [canonical_geq, canonical_leq] # TODO: m.features
    init_conditions = m.init_conditions
    allow_global_splits = m.allow_global_splits
    automatic_downsizing = m.automatic_downsizing
    
    missing_columns = setdiff(Iterators.flatten(fitresult.frame_grouping), names(Xnew))
    @assert length(missing_columns) == 0 "Can't perform prediction due to missing DataFrame columns: $(missing_columns)"
    
    Xs, init_conditions = DataFrame2MultiFrameModalDataset(
        Xnew,
        fitresult.frame_grouping,
        relations,
        features,
        init_conditions,
        allow_global_splits,
        :implicit,
        downsizing_function = (automatic_downsizing ? tree_downsizing_function : identity),
    )

    if isnothing(ynew)
        MDT.apply_model(fitresult.model, Xs)
    else
        MDT.print_apply(fitresult.model, Xs, ynew)
    end
end

############################################################################################
############################################################################################
############################################################################################

MMI.@mlj_model mutable struct ModalRandomForest <: MMI.Probabilistic
    # Pruning hyper-parameters
    max_depth              :: Union{Nothing,Int}           = nothing::(isnothing(_) || _ ≥ -1)
    min_samples_leaf       :: Int                          = default_min_samples_leaf::(_ ≥ 1)
    min_purity_increase    :: Float64                      = default_min_purity_increase
    max_purity_at_leaf     :: Float64                      = default_max_purity_at_leaf
    # Modal hyper-parameters
    relations              :: Union{Nothing,Symbol,Vector{<:Relation}} = (nothing)::(isnothing(nothing) || _ in [:IA, :IA3, :IA7, :RCC5, :RCC8] || _ isa AbstractVector{<:Relation})
    # TODO expand to ModalFeature
    # features               :: Vector{<:Function}           = [minimum, maximum]::(all(Iterators.flatten([(f)->(ret = f(ch); isa(ret, Real) && typeof(ret) == eltype(ch)), _) for ch in [collect(1:10), collect(1.:10.)]]))
    # features               :: Vector{<:Union{CanonicalFeature,Function}}       TODO = Vector{<:Union{CanonicalFeature,Function}}([canonical_geq, canonical_leq]) # TODO: ::(_check_features(_))
    # features               :: AbstractVector{<:CanonicalFeature}       = CanonicalFeature[canonical_geq, canonical_leq] # TODO: ::(_check_features(_))
    # features               :: Vector                       = [canonical_geq, canonical_leq] # TODO: ::(_check_features(_))
    init_conditions        :: Union{Nothing,Symbol}                       = nothing::(isnothing(_) || _ in [:start_with_global, :start_at_center])
    allow_global_splits    :: Bool                         = true
    automatic_downsizing   :: Bool                         = true

    # ModalDecisionTree-specific
    
    n_subrelations         ::Union{Nothing,Int,Function}   = nothing::(isnothing(_) || _ isa Function || _ ≥ -1)
    n_subfeatures          ::Union{Nothing,Int,Function}   = nothing::(isnothing(_) || _ isa Function || _ ≥ -1)
    n_trees                ::Int                 = 10::(_ ≥ 2)
    sampling_fraction      ::Float64   = 0.7::(0 < _ ≤ 1)
    rng                    ::Union{AbstractRNG,Integer} = GLOBAL_RNG
end


function MMI.fit(m::ModalRandomForest, verbosity::Int, X, y, w=nothing)
    X = DataFrame(X)
    classes_seen  = y isa CategoricalArray ? filter(in(unique(y)), MMI.classes(y)) : unique(y)
    y  = Vector{String}(y)

    ########################################################################################
    
    max_depth            = m.max_depth
    max_depth            = isnothing(max_depth) || max_depth == -1 ? typemax(Int64) : max_depth
    min_samples_leaf     = m.min_samples_leaf
    min_purity_increase  = m.min_purity_increase
    max_purity_at_leaf   = m.max_purity_at_leaf
    relations            = m.relations
    features             = [canonical_geq, canonical_leq] # TODO: m.features
    init_conditions      = m.init_conditions
    allow_global_splits  = m.allow_global_splits
    automatic_downsizing = m.automatic_downsizing

    _n_subrelations      = m.n_subrelations
    n_subrelations       = isnothing(_n_subrelations) ? identity : (isa(_n_subrelations, Integer) ? (x)->(_n_subrelations) : _n_subrelations)
    _n_subfeatures       = m.n_subfeatures
    n_subfeatures        = isnothing(_n_subfeatures) ? identity : (isa(_n_subfeatures, Integer) ? (x)->(_n_subfeatures) : _n_subfeatures)
    n_trees              = m.n_trees
    sampling_fraction    = m.sampling_fraction
    rng                  = m.rng

    ########################################################################################
    
    frame_grouping = separate_attributes_into_frames(X)
    Xs, init_conditions = DataFrame2MultiFrameModalDataset(
        X,
        frame_grouping,
        relations,
        features,
        init_conditions,
        allow_global_splits,
        :explicit;
        downsizing_function = (automatic_downsizing ? (args...)->forest_downsizing_function(args...; n_trees = m.n_trees) : identity),
    )
    
    model = MDT.build_forest(
        Xs,
        y,
        w,
        ##############################################################################
        n_trees              = n_trees,
        partial_sampling     = sampling_fraction,
        ##############################################################################
        loss_function        = nothing,
        max_depth            = max_depth,
        min_samples_leaf     = min_samples_leaf,
        min_purity_increase  = min_purity_increase,
        max_purity_at_leaf   = max_purity_at_leaf,
        ##############################################################################
        n_subrelations       = n_subrelations,
        n_subfeatures        = n_subfeatures,
        init_conditions      = init_conditions,
        allow_global_splits  = allow_global_splits,
        ##############################################################################
        perform_consistency_check = false,
        rng = rng,
    )
    # println(MDT.attribute_countmap(model))
    feature_importance_by_count = Dict([i_attr => frame_grouping[i_frame][i_attr] for ((i_frame, i_attr), count) in MDT.attribute_countmap(model)])

    fitresult = (
        model           = model,
        frame_grouping  = frame_grouping,
        classes_seen    = classes_seen,
    )

    cache  = nothing
    report = (
        print_forest                 = ModelPrinter(model, frame_grouping),
        print_model                  = ModelPrinter(model, frame_grouping),
        frame_grouping              = frame_grouping,
        feature_importance_by_count = feature_importance_by_count,
        classes_seen = classes_seen,
    )

    return fitresult, cache, report
end


MMI.fitted_params(::ModalRandomForest, fitresult) =
    (
        model           = fitresult.model,
        frame_grouping  = fitresult.frame_grouping,
    )

function MMI.predict(m::ModalRandomForest, fitresult, Xnew) #, ynew = nothing)
    Xnew = DataFrame(Xnew)
    # ynew = Vector{String}(ynew)
    ynew = nothing
    @assert isnothing(ynew)

    relations = m.relations
    features = [canonical_geq, canonical_leq] # TODO: m.features
    init_conditions = m.init_conditions
    allow_global_splits = m.allow_global_splits
    automatic_downsizing = m.automatic_downsizing
    
    missing_columns = setdiff(Iterators.flatten(fitresult.frame_grouping), names(Xnew))
    @assert length(missing_columns) == 0 "Can't perform prediction due to missing DataFrame columns: $(missing_columns)"
    
    Xs, init_conditions = DataFrame2MultiFrameModalDataset(
        Xnew,
        fitresult.frame_grouping,
        relations,
        features,
        init_conditions,
        allow_global_splits,
        :implicit,
        downsizing_function = (automatic_downsizing ? (args...)->forest_downsizing_function(args...; n_trees = m.n_trees) : identity),
    )

    scores = MDT.apply_model_proba(fitresult.model, Xs, fitresult.classes_seen)
    return MMI.UnivariateFinite(fitresult.classes_seen, scores)
end

############################################################################################
############################################################################################
############################################################################################

# # ADA BOOST STUMP CLASSIFIER

# TODO
# MMI.@mlj_model mutable struct AdaBoostStumpClassifier <: MMI.Probabilistic
#     n_iter::Int            = 10::(_ ≥ 1)
# end

# function MMI.fit(m::AdaBoostStumpClassifier, verbosity::Int, X, y)
#     Xmatrix = MMI.matrix(X)
#     yplain  = MMI.int(y)

#     classes_seen  = filter(in(unique(y)), MMI.classes(y[1]))
#     integers_seen = MMI.int(classes_seen)

#     stumps, coefs = MDT.build_adaboost_stumps(yplain, Xmatrix,
#                                              m.n_iter)
#     cache  = nothing
#     report = NamedTuple()
#     return (stumps, coefs, classes_seen, integers_seen), cache, report
# end

# MMI.fitted_params(::AdaBoostStumpClassifier, (stumps,coefs,_)) =
#     (stumps=stumps,coefs=coefs)

# function MMI.predict(m::AdaBoostStumpClassifier, fitresult, Xnew)
#     Xmatrix = MMI.matrix(Xnew)
#     stumps, coefs, classes_seen, integers_seen = fitresult
#     scores = MDT.apply_adaboost_stumps_proba(stumps, coefs,
#                                             Xmatrix, integers_seen)
#     return MMI.UnivariateFinite(classes_seen, scores)
# end


# # # DECISION TREE REGRESSOR

# MMI.@mlj_model mutable struct DecisionTreeRegressor <: MMI.Deterministic
#     max_depth::Int                               = (-)(1)::(_ ≥ -1)
#     min_samples_leaf::Int                = 5::(_ ≥ 0)
#     min_samples_split::Int               = 2::(_ ≥ 2)
#     min_purity_increase::Float64 = 0.0::(_ ≥ 0)
#     n_subfeatures::Int                   = 0::(_ ≥ -1)
#     post_prune::Bool                     = false
#     merge_purity_threshold::Float64 = 1.0::(0 ≤ _ ≤ 1)
#     rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
# end

# function MMI.fit(m::DecisionTreeRegressor, verbosity::Int, X, y)
#     Xmatrix = MMI.matrix(X)
#     tree    = MDT.build_tree(float(y), Xmatrix,
#                             m.n_subfeatures,
#                             m.max_depth,
#                             m.min_samples_leaf,
#                             m.min_samples_split,
#                             m.min_purity_increase;
#                             rng=m.rng)

#     if m.post_prune
#         tree = MDT.prune_tree(tree, m.merge_purity_threshold)
#     end
#     cache  = nothing
#     report = NamedTuple()
#     return tree, cache, report
# end

# MMI.fitted_params(::DecisionTreeRegressor, tree) = (tree=tree,)

# function MMI.predict(::DecisionTreeRegressor, tree, Xnew)
#     Xmatrix = MMI.matrix(Xnew)
#     return MDT.apply_tree(tree, Xmatrix)
# end


# # # RANDOM FOREST REGRESSOR

# MMI.@mlj_model mutable struct RandomForestRegressor <: MMI.Deterministic
#     max_depth::Int               = (-)(1)::(_ ≥ -1)
#     min_samples_leaf::Int        = 1::(_ ≥ 0)
#     min_samples_split::Int       = 2::(_ ≥ 2)
#     min_purity_increase::Float64 = 0.0::(_ ≥ 0)
#     n_subfeatures::Int           = (-)(1)::(_ ≥ -1)
#     n_trees::Int                 = 10::(_ ≥ 2)
#     sampling_fraction::Float64   = 0.7::(0 < _ ≤ 1)
#     rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
# end

# function MMI.fit(m::RandomForestRegressor, verbosity::Int, X, y)
#     Xmatrix = MMI.matrix(X)
#     forest  = MDT.build_forest(float(y), Xmatrix,
#                               m.n_subfeatures,
#                               m.n_trees,
#                               m.sampling_fraction,
#                               m.max_depth,
#                               m.min_samples_leaf,
#                               m.min_samples_split,
#                               m.min_purity_increase,
#                               rng=m.rng)
#     cache  = nothing
#     report = NamedTuple()
#     return forest, cache, report
# end

# MMI.fitted_params(::RandomForestRegressor, forest) = (forest=forest,)

# function MMI.predict(::RandomForestRegressor, forest, Xnew)
#     Xmatrix = MMI.matrix(Xnew)
#     return MDT.apply_forest(forest, Xmatrix)
# end


############################################################################################
############################################################################################
############################################################################################


# # METADATA (MODEL TRAITS)

MMI.metadata_pkg.(
    (
        ModalDecisionTree,
        ModalRandomForest,
        # DecisionTreeRegressor,
        # RandomForestRegressor,
        # AdaBoostStumpClassifier,
    ),
    name = "ModalDecisionTrees",
    package_uuid = "e54bda2e-c571-11ec-9d64-0242ac120002",
    package_url = "https://github.com/giopaglia/ModalDecisionTrees.jl",
    is_pure_julia = true,
    package_license = "MIT",
    is_wrapper = false,
)

MMI.metadata_model(
    ModalDecisionTree,
    input_scitype = Table(
        Continuous,     AbstractVector{Continuous},    AbstractMatrix{Continuous},
        Count,          AbstractVector{Count},         AbstractMatrix{Count},
        OrderedFactor,  AbstractVector{OrderedFactor}, AbstractMatrix{OrderedFactor},
    ),
    target_scitype = AbstractVector{<:Finite},
    human_name = "Modal Decision Tree (MDT)",
    descr   = "A Modal Decision Tree (MDT) offers a high level of interpretability for classification tasks with images and time-series.",
    supports_weights = true,
    load_path = "$PKG.ModalDecisionTree",
)

# MMI.metadata_model(
#     ModalRandomForest,
#     input_scitype = Table(Continuous, Count, OrderedFactor),
#     target_scitype = AbstractVector{<:Finite},
#     human_name = "CART random forest classifier",
#     load_path = "$PKG.ModalRandomForest"
# )

# MMI.metadata_model(
#     AdaBoostStumpClassifier,
#     input_scitype = Table(Continuous, Count, OrderedFactor),
#     target_scitype = AbstractVector{<:Finite},
#     human_name = "Ada-boosted stump classifier",
#     load_path = "$PKG.AdaBoostStumpClassifier"
# )

# MMI.metadata_model(
#     DecisionTreeRegressor,
#     input_scitype = Table(Continuous, Count, OrderedFactor),
#     target_scitype = AbstractVector{Continuous},
#     human_name = "CART decision tree regressor",
#     load_path = "$PKG.DecisionTreeRegressor"
# )

# MMI.metadata_model(
#     RandomForestRegressor,
#     input_scitype = Table(Continuous, Count, OrderedFactor),
#     target_scitype = AbstractVector{Continuous},
#     human_name = "CART random forest regressor",
#     load_path = "$PKG.RandomForestRegressor")


# # DOCUMENT STRINGS

const DOC_CART = "[CART algorithm](https://en.wikipedia.org/wiki/Decision_tree_learning)"*
", originally published in Breiman, Leo; Friedman, J. H.; Olshen, R. A.; "*
"Stone, C. J. (1984): \"Classification and regression trees\". *Monterey, "*
"CA: Wadsworth & Brooks/Cole Advanced Books & Software.*"

const DOC_RANDOM_FOREST = "[Random Forest algorithm]"*
    "(https://en.wikipedia.org/wiki/Random_forest), originally published in "*
    "Breiman, L. (2001): \"Random Forests.\", *Machine Learning*, vol. 45, pp. 5–32"

"""
$(MMI.doc_header(ModalDecisionTree))
`ModalDecisionTree` implements a variation of the $DOC_CART that adopts modal logics of time and space
to perform temporal/spatial reasoning on non-scalar data such as time-series and image.
# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)
where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`
Train the machine with `fit!(mach)`.
# Hyper-parameters
- `max_depth=-1`:          Maximum depth of the decision tree (-1=any)
- `min_samples_leaf=$(default_min_samples_leaf)`:    Minimum number of samples required at each leaf
- `min_purity_increase=$(default_min_purity_increase)`: Minimum purity needed for a split
- `max_purity_at_leaf=$(default_max_purity_at_leaf)`: Minimum purity needed for a split
- `relations=nothing`       Relations that the model uses to "move" around the image; it can be a symbol in [:IA, :IA3, :IA7, :RCC5, :RCC8],
                            where :IA stands [Allen's Interval Algebra](https://en.m.wikipedia.org/wiki/Allen%27s_interval_algebra) (13 relations in 1D, 169 relations in 2D),
                            :IA3 and :IA7 are [coarser fragments with 3 and 7 relations, respectively](https://www.sciencedirect.com/science/article/pii/S0004370218305964),
                            :RCC5 and :RCC8 are [Region Connection Calculus algebras](https://en.m.wikipedia.org/wiki/Region_connection_calculus) with 5 and 8 topological operators, respectively.
                            Relations from :IA, :IA3, :IA7, capture directional aspects of the relative arrangement of two intervals in time (or rectangles in a 2D space),
                             while relations from :RCC5 and :RCC8 only capture topological aspects and are therefore rotation-invariant.
                            This hyper-parameter defaults to :IA for temporal variables (1D), and to :RCC8 for spatial variables (2D).
- `init_conditions=nothing` initial conditions for evaluating modal decisions at the root; it can be a symbol in [:start_with_global, :start_at_center].
                            :start_with_global forces the first decision to be a *global* decision (e.g., `⟨G⟩ (minimum(A2) ≥ 10)`, which translates to "there exists a region where the minimum of attribute 2 is higher than 10").
                            :start_at_center forces the first decision to be evaluated on the smallest central world, that is, the central value of a time-series, or the central pixel of an image.
                            This hyper-parameter defaults to :start_with_global for temporal variables (1D), and to :start_at_center for spatial variables (2D).
- `allow_global_splits=true`  Whether to allow global splits (e.g. `⟨G⟩ (minimum(A2) ≥ 10)`) at any point of the tree.
- `automatic_downsizing=true` Whether to perform automatic downsizing. In fact, this algorithm has high complexity (both time and space), and can only handle small time-series (< 100 points) & small images (< 10 x 10 pixels).

- `display_depth=5`:       max depth to show when displaying the tree
# Operations
- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above.

# Fitted parameters
The fields of `fitted_params(mach)` are:
- `model`: the tree object returned by the core algorithm
- `frame_grouping`: the adopted grouping of the features encountered in training, in an order consistent with the output of `print_model` (see below).
# Report
The fields of `report(mach)` are:
- `print_model`: method to print a pretty representation of the fitted
  tree, with single argument the tree depth; interpretation requires
  internal integer-class encoding (see "Fitted parameters" above).
- `frame_grouping`: the adopted grouping of the features encountered in training, in an order consistent with the output of `print_model` (see below).
- `feature_importance_by_count`: a simple count of each of the occurrences of the features across the model, in an order consistent with `frame_grouping`.
- `classes_seen`: list of target classes actually observed in training.
# Examples
```
using MLJ
MDT = @load ModalDecisionTree pkg=ModalDecisionTrees
tree = MDT(max_depth=4, min_samples_split=3)
X, y = @load_iris
mach = machine(tree, X, y) |> fit!
Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)
yhat = predict(mach, Xnew) # probabilistic predictions
predict_mode(mach, Xnew)   # point predictions
pdf.(yhat, "virginica")    # probabilities for the "verginica" class
fitted_params(mach).tree # raw tree or stump object from DecisionTrees.jl
julia> report(mach).print_model(3)
Feature 4, Threshold 0.8
L-> 1 : 50/50
R-> Feature 4, Threshold 1.75
    L-> Feature 3, Threshold 4.95
        L->
        R->
    R-> Feature 3, Threshold 4.85
        L->
        R-> 3 : 43/43
```
To interpret the internal class labelling:
```
julia> fitted_params(mach).encoding
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, UInt32} with 3 entries:
  "virginica"  => 0x00000003
  "setosa"     => 0x00000001
  "versicolor" => 0x00000002
```
See also
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
the unwrapped model type [`MLJDecisionTreeInterface.DecisionTree.ModalDecisionTree`](@ref).
"""
ModalDecisionTree

# """
# $(MMI.doc_header(ModalRandomForest))
# `ModalRandomForest` implements the standard $DOC_RANDOM_FOREST.
# # Training data
# In MLJ or MLJBase, bind an instance `model` to data with
#     mach = machine(model, X, y)
# where
# - `X`: any table of input features (eg, a `DataFrame`) whose columns
#   each have one of the following element scitypes: `Continuous`,
#   `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
# - `y`: the target, which can be any `AbstractVector` whose element
#   scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
#   with `scitype(y)`
# Train the machine with `fit!(mach, rows=...)`.
# # Hyper-parameters
# - `max_depth=-1`:          max depth of the decision tree (-1=any)
# - `min_samples_leaf=1`:    min number of samples each leaf needs to have
# - `min_samples_split=2`:   min number of samples needed for a split
# - `min_purity_increase=0`: min purity needed for a split
# - `n_subfeatures=-1`: number of features to select at random (0 for all,
#   -1 for square root of number of features)
# - `n_trees=10`:            number of trees to train
# - `sampling_fraction=0.7`  fraction of samples to train each tree on
# - `rng=Random.GLOBAL_RNG`: random number generator or seed
# # Operations
# - `predict(mach, Xnew)`: return predictions of the target given
#   features `Xnew` having the same scitype as `X` above. Predictions
#   are probabilistic, but uncalibrated.
# - `predict_mode(mach, Xnew)`: instead return the mode of each
#   prediction above.
# # Fitted parameters
# The fields of `fitted_params(mach)` are:
# - `forest`: the `Ensemble` object returned by the core DecisionTree.jl algorithm
# # Examples
# ```
# using MLJ
# MRF = @load ModalRandomForest pkg=ModalDecisionTrees
# forest = MRF(min_samples_split=6, n_subfeatures=3)
# X, y = @load_iris
# mach = machine(forest, X, y) |> fit!
# Xnew = (sepal_length = [6.4, 7.2, 7.4],
#         sepal_width = [2.8, 3.0, 2.8],
#         petal_length = [5.6, 5.8, 6.1],
#         petal_width = [2.1, 1.6, 1.9],)
# yhat = predict(mach, Xnew) # probabilistic predictions
# predict_mode(mach, Xnew)   # point predictions
# pdf.(yhat, "virginica")    # probabilities for the "verginica" class
# fitted_params(mach).forest # raw `Ensemble` object from DecisionTrees.jl
# ```
# See also
# [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
# the unwrapped model type
# [`MLJDecisionTreeInterface.DecisionTree.ModalRandomForest`](@ref).
# """
ModalRandomForest

# """
# $(MMI.doc_header(AdaBoostStumpClassifier))
# # Training data
# In MLJ or MLJBase, bind an instance `model` to data with
#     mach = machine(model, X, y)
# where:
# - `X`: any table of input features (eg, a `DataFrame`) whose columns
#   each have one of the following element scitypes: `Continuous`,
#   `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
# - `y`: the target, which can be any `AbstractVector` whose element
#   scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
#   with `scitype(y)`
# Train the machine with `fit!(mach, rows=...)`.
# # Hyper-parameters
# - `n_iter=10`:   number of iterations of AdaBoost
# # Operations
# - `predict(mach, Xnew)`: return predictions of the target given
#   features `Xnew` having the same scitype as `X` above. Predictions
#   are probabilistic, but uncalibrated.
# - `predict_mode(mach, Xnew)`: instead return the mode of each
#   prediction above.
# # Fitted Parameters
# The fields of `fitted_params(mach)` are:
# - `stumps`: the `Ensemble` object returned by the core DecisionTree.jl
#   algorithm.
# - `coefficients`: the stump coefficients (one per stump)
# ```
# using MLJ
# Booster = @load AdaBoostStumpClassifier pkg=ModalDecisionTrees
# booster = Booster(n_iter=15)
# X, y = @load_iris
# mach = machine(booster, X, y) |> fit!
# Xnew = (sepal_length = [6.4, 7.2, 7.4],
#         sepal_width = [2.8, 3.0, 2.8],
#         petal_length = [5.6, 5.8, 6.1],
#         petal_width = [2.1, 1.6, 1.9],)
# yhat = predict(mach, Xnew) # probabilistic predictions
# predict_mode(mach, Xnew)   # point predictions
# pdf.(yhat, "virginica")    # probabilities for the "verginica" class
# fitted_params(mach).stumps # raw `Ensemble` object from DecisionTree.jl
# fitted_params(mach).coefs  # coefficient associated with each stump
# ```
# See also
# [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
# the unwrapped model type
# [`MLJDecisionTreeInterface.DecisionTree.AdaBoostStumpClassifier`](@ref).
# """
# # AdaBoostStumpClassifier

# """
# $(MMI.doc_header(DecisionTreeRegressor))
# `DecisionTreeRegressor` implements the $DOC_CART.
# # Training data
# In MLJ or MLJBase, bind an instance `model` to data with
#     mach = machine(model, X, y)
# where
# - `X`: any table of input features (eg, a `DataFrame`) whose columns
#   each have one of the following element scitypes: `Continuous`,
#   `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
# - `y`: the target, which can be any `AbstractVector` whose element
#   scitype is `Continuous`; check the scitype with `scitype(y)`
# Train the machine with `fit!(mach, rows=...)`.
# # Hyper-parameters
# - `max_depth=-1`:          max depth of the decision tree (-1=any)
# - `min_samples_leaf=1`:    max number of samples each leaf needs to have
# - `min_samples_split=2`:   min number of samples needed for a split
# - `min_purity_increase=0`: min purity needed for a split
# - `n_subfeatures=0`: number of features to select at random (0 for all,
#   -1 for square root of number of features)
# - `post_prune=false`:      set to `true` for post-fit pruning
# - `merge_purity_threshold=1.0`: (post-pruning) merge leaves having
#                            combined purity `>= merge_purity_threshold`
# - `rng=Random.GLOBAL_RNG`: random number generator or seed
# # Operations
# - `predict(mach, Xnew)`: return predictions of the target given new
#   features `Xnew` having the same scitype as `X` above.
# # Fitted parameters
# The fields of `fitted_params(mach)` are:
# - `tree`: the tree or stump object returned by the core
#   DecisionTree.jl algorithm
# # Examples
# ```
# using MLJ
# MDT = @load DecisionTreeRegressor pkg=ModalDecisionTrees
# tree = MDT(max_depth=4, min_samples_split=3)
# X, y = make_regression(100, 2) # synthetic data
# mach = machine(tree, X, y) |> fit!
# Xnew, _ = make_regression(3, 2)
# yhat = predict(mach, Xnew) # new predictions
# fitted_params(mach).tree # raw tree or stump object from DecisionTree.jl
# ```
# See also
# [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
# the unwrapped model type
# [`MLJDecisionTreeInterface.DecisionTree.DecisionTreeRegressor`](@ref).
# """
# DecisionTreeRegressor

# """
# $(MMI.doc_header(RandomForestRegressor))
# `DecisionTreeRegressor` implements the standard $DOC_RANDOM_FOREST
# # Training data
# In MLJ or MLJBase, bind an instance `model` to data with
#     mach = machine(model, X, y)
# where
# - `X`: any table of input features (eg, a `DataFrame`) whose columns
#   each have one of the following element scitypes: `Continuous`,
#   `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
# - `y`: the target, which can be any `AbstractVector` whose element
#   scitype is `Continuous`; check the scitype with `scitype(y)`
# Train the machine with `fit!(mach, rows=...)`.
# # Hyper-parameters
# - `max_depth=-1`:          max depth of the decision tree (-1=any)
# - `min_samples_leaf=1`:    min number of samples each leaf needs to have
# - `min_samples_split=2`:   min number of samples needed for a split
# - `min_purity_increase=0`: min purity needed for a split
# - `n_subfeatures=-1`: number of features to select at random (0 for all,
#   -1 for square root of number of features)
# - `n_trees=10`:            number of trees to train
# - `sampling_fraction=0.7`  fraction of samples to train each tree on
# - `rng=Random.GLOBAL_RNG`: random number generator or seed
# # Operations
# - `predict(mach, Xnew)`: return predictions of the target given new
#   features `Xnew` having the same scitype as `X` above.
# # Fitted parameters
# The fields of `fitted_params(mach)` are:
# - `forest`: the `Ensemble` object returned by the core DecisionTree.jl algorithm
# # Examples
# ```
# using MLJ
# Forest = @load RandomForestRegressor pkg=ModalDecisionTrees
# forest = Forest(max_depth=4, min_samples_split=3)
# X, y = make_regression(100, 2) # synthetic data
# mach = machine(forest, X, y) |> fit!
# Xnew, _ = make_regression(3, 2)
# yhat = predict(mach, Xnew) # new predictions
# fitted_params(mach).forest # raw `Ensemble` object from DecisionTree.jl
# ```
# See also
# [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
# the unwrapped model type
# [`MLJDecisionTreeInterface.DecisionTree.RandomForestRegressor`](@ref).
# """
# # RandomForestRegressor

end

using .MLJInterface
