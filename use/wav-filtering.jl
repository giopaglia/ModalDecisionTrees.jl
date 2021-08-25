
include("wav2stft_time_series.jl")
include("local.jl")

# using DSP: filt
# using Weave
import Pkg
Pkg.activate("..")

using DecisionTree
using DecisionTree.ModalLogic

using Plots
using MFCC: hz2mel, mel2hz

MFCC.mel2hz(f::AbstractFloat, htk=false)  = mel2hz([f], htk)[1]

function FIRfreqz(b::Array; w = range(0, stop=π, length=1024))::Array{ComplexF32}
    n = length(w)
    h = Array{ComplexF32}(undef, n)
    sw = 0
    for i = 1:n
        for j = 1:length(b)
        sw += b[j]*exp(-im*w[i])^-j
        end
        h[i] = sw
        sw = 0
    end
    h
end

function plotfilter(
        filter;
        samplerate = 100,
        xlims::Union{Nothing,Tuple{Real,Real}} = (0, 0),
        ylims::Union{Nothing,Tuple{Real,Real}} = (-24, 0),
        plotfunc = plot
    )
    
    w = range(0, stop=π, length=1024)
    h = FIRfreqz(filter; w = w)
    ws = w / π * (samplerate / 2)
    if xlims[2] == 0
        xlims = (xlims[1], maximum(ws))
    end
    plotfunc(ws, amp2db.(abs.(h)), xlabel="Frequency (Hz)", ylabel="Magnitude (db)", xlims = xlims, ylims = ylims, leg = false)
end
plotfilter(filter::Filters.Filter; kwargs...) = plotfilter(digitalfilter(filter, firwindow); kwargs...)

plotfilter!(filter::Filters.Filter; args...) = plotfilter(filter; args..., plotfunc = plot!)
plotfilter!(filter; args...)                 = plotfilter(filter; args..., plotfunc = plot!)

function multibandpass_digitalfilter(
        vec::Vector{Tuple{T, T}},
        fs::Real,
        window_f::Function;
        nbands::Integer = 60,
        nwin = nbands,
        weights::Vector{F} where F <:AbstractFloat = fill(1., length(vec))
    )::AbstractVector where T

    @assert length(weights) == length(vec) "length(weights) != length(vec): $(length(weights)) != $(length(vec))"

    result_filter = zeros(T, nbands)
    i = 1
    @simd for t in vec
        result_filter += digitalfilter(Filters.Bandpass(t..., fs = fs), FIRWindow(window_f(nwin))) * weights[i]
        i=i+1
    end
    result_filter
end

function multibandpass_digitalfilter(
        selected_bands::Vector{Int},
        fs::Real,
        window_f::Function;
        nbands::Integer = 60,
        lower_freq = 0,
        higher_freq = fs / 2,
        nwin = nbands,
        weights::Vector{F} where F <:AbstractFloat = fill(1., length(selected_bands))
    )::AbstractVector

    @assert length(weights) == length(selected_bands) "length(weights) != length(selected_bands): $(length(weights)) != $(length(selected_bands))"

    band_width = (higher_freq - lower_freq) / nbands

    result_filter = zeros(Float64, nbands)
    i = 1
    @simd for b in selected_bands
        result_filter += digitalfilter(Filters.Bandpass(b * band_width, ((b+1) * band_width) - 1, fs = fs), FIRWindow(window_f(nwin))) * weights
        i=i+1
    end
    result_filter
end

function multibandpass_digitalfilter_mel(
        selected_bands::Vector{Int},
        fs::Real,
        window_f::Function;
        nbands::Integer = 60,
        lower_freq = 0,
        higher_freq = hz2mel(fs / 2),
        nwin = nbands,
        weights::Vector{F} where F <:AbstractFloat = fill(1., length(selected_bands))
    )::AbstractVector

    @assert length(weights) == length(selected_bands) "length(weights) != length(selected_bands): $(length(weights)) != $(length(selected_bands))"

    correct_selected_bands = selected_bands .- 1
    band_width = (higher_freq - lower_freq) / nbands

    result_filter = zeros(Float64, nbands)
    i = 1
    @simd for b in correct_selected_bands
        result_filter += digitalfilter(Filters.Bandpass(
            mel2hz(max(b * band_width, eps(Float64))),
            mel2hz(((b+1) * band_width)),
            fs = fs
        ), FIRWindow(window_f((nwin)))) * weights[i]
        i=i+1
    end
    result_filter
end

function draw_mel_filters_graph(fs::Real, window_f::Function; nbands::Integer = 60, lower_freq = 0, higher_freq = hz2mel(fs / 2))
    filters = [ multibandpass_digitalfilter_mel([i], fs, window_f; nbands = nbands, lower_freq = lower_freq, higher_freq = higher_freq) for i in 1:nbands ]
    plotfilter(filters[1]; samplerate = fs)
    for i in 2:(length(filters)-1)
        plotfilter!(filters[i]; samplerate = fs)
    end
    # last call to plot has to "return" from function otherwise the graph will not be displayed
    plotfilter!(filters[end]; samplerate = fs)
end

timerange2points(range::Tuple{T, T} where T <:Number, fs::Real)::UnitRange{Int64} = max(1, round(Int64, range[1] * fs)):round(Int64, range[2] * fs)

function draw_audio_anim(
        # TODO: figure out a way to generalize this Float64 and Float32 without getting error...
        audio_files    :: Vector{Tuple{Vector{Float64},Float32}};
        labels         :: Vector{String} = fill("", length(audio_files)),
        colors         :: Union{Vector{Symbol},Vector{RGB{Float64}}} = fill(:auto, length(audio_files)),
        outfile        :: String = homedir() * "/gif.gif",
        size           :: Tuple{Int64,Int64} = (1000, 150 * length(audio_files)),
        fps            :: Int64 = 60,
        # selected_range:
        # - 1:1000 means from point 1 to point 1000
        # - (1.1, 2.3) means from time 1.1 to time 2.3 (in seconds)
        # - :whole means "do not slice"
        selected_range :: Union{UnitRange{Int64},Tuple{Number,Number},Symbol} = :whole,
        single_graph   :: Bool = false
    )
    function draw_wav(points::Vector{Float64}, fs::Number; color = :auto, title = "", func = plot)
        func(
            collect(0:(length(points) - 1)),
            points,
            title = title,
            xlims = (0, length(points)),
            ylims = (-1, 1),
            framestyle = :zerolines,       # show axis at zeroes
            fill = 0,                      # show area under curve
            leg = false,                   # hide legend
            yshowaxis = false,             # hide y axis
            grid = false,                  # hide y grid
            ticks = false,                 # hide y ticks
            tick_direction = :none,
            linecolor = color,
            fillcolor = color,
            size = size
        )
    end
    draw_wav!(points::Vector{Float64}, fs::Number; title = "", color = :auto) = draw_wav(points, fs, func = plot!; title = title, color = color)

    @assert length(audio_files) > 0 "No audio file provided"
    @assert length(audio_files) == length(labels) "audio_files and labels mismatch in length: $(length(audio_files)) != $(length(labels))"

    wavs = []
    fss = []
    for f in audio_files
        push!(wavs, merge_channels(f[1]))
        push!(fss, f[2])
    end

    @assert length(unique(fss)) == 1 "Inconsistent bitrate across multiple files"
    @assert length(unique([x -> length(x) for wav in wavs])) == 1 "Inconsistent length across multiple files"

    if selected_range isa Tuple
        # convert seconds to points
        println("Selected time range from $(selected_range[1])s to $(selected_range[2])s")
        selected_range = timerange2points(selected_range, fss[1])
        #round(Int64, selected_range[1] * fss[1]):round(Int64, selected_range[2] * fss[1])
    end
    # slice all wavs
    if selected_range != :whole
        println("Slicing from point $(collect(selected_range)[1]) to $(collect(selected_range)[end])")
        for i in 1:length(wavs)
            wavs[i] = wavs[i][selected_range]
        end
    end
    wavlength = length(wavs[1])
    freq = fss[1]
    wavlength_seconds = wavlength / freq

    total_frames = ceil(Int64, wavlength_seconds * fps)
    step = wavlength / total_frames

    anim = nothing
    plts = []
    for (i, w) in enumerate(wavs)
        if i == 1
            push!(plts, draw_wav(w, freq; title = labels[i], color = colors[i]))
        else
            if single_graph
                draw_wav!(w, freq; title = labels[i], color = colors[i])
            else
                push!(plts, draw_wav(w, freq; title = labels[i], color = colors[i]))
            end
        end
    end

    anim = @animate for f in 1:total_frames
        println("Processing frame $(f) / $(total_frames)")
        if f != 1
            Threads.@threads for p in 1:length(plts)
                # Make previous vline invisible
                plts[p].series_list[end][:linealpha] = 0.0
            end
        end
        #Threads.@threads 
        for p in 1:length(plts)
            vline!(plts[p], [ (f-1) * step ], line = (:black, 1))
        end
        plot(plts..., layout = (length(wavs), 1))
    end

    gif(anim, outfile, fps = fps)
end

function draw_audio_anim(audio_files::Vector{String}; kwargs...)
    @assert length(audio_files) > 0 "No audio file provided"

    converted_input = []
    for f in audio_files
        wav, fs = wavread(f)
        push!(converted_input, (merge_channels(wav), fs))
    end

    draw_audio_anim(converted_input; kwargs...)
end

struct DecisionPathNode
    taken         :: Bool
    feature       :: ModalLogic.FeatureTypeFun
    test_operator :: TestOperatorFun
    threshold     :: T where T
    # TODO: add here info about the world(s)
end

const DecisionPath = Vector{DecisionPathNode}

mutable struct InstancePathInTree{S}
    file_name    :: String
    label        :: S
    tree         :: Union{Nothing,DTree{T}} where T
    predicted    :: Union{Nothing,S}
    path         :: DecisionPath
    dataset_info :: Any
    InstancePathInTree{S}(file_name::String, label::S) where S = new(file_name, label, nothing, nothing, [], ())
    InstancePathInTree{S}(file_name::String, label::S, dataset_info) where S = new(file_name, label, nothing, nothing, [], dataset_info)
    InstancePathInTree{S}(file_name::String, label::S, tree::DTree) where S = new(file_name, label, tree, nothing, [], ())
    InstancePathInTree{S}(file_name::String, label::S, tree::DTree, dataset_info) where S = new(file_name, label, tree, nothing, [], dataset_info)
end

is_correctly_classified(inst::InstancePathInTree)::Bool = inst.label === inst.predicted

get_path_in_tree(leaf::DTLeaf, X::Any, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, path::DecisionPath = DecisionPath())::DecisionPath = return path
function get_path_in_tree(tree::DTInternal, X::MultiFrameModalDataset, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, path::DecisionPath = DecisionPath())::DecisionPath
    satisfied = true
	(satisfied,new_worlds) =
		ModalLogic.modal_step(
						get_frame(X, tree.i_frame),
						i_instance,
						worlds[tree.i_frame],
						tree.relation,
						tree.feature,
						tree.test_operator,
						tree.threshold)

    # TODO: add here info about the worlds that generated the decision
    push!(path, DecisionPathNode(satisfied, tree.feature, tree.test_operator, tree.threshold))

	worlds[tree.i_frame] = new_worlds
	get_path_in_tree((satisfied ? tree.left : tree.right), X, i_instance, worlds)
end
function get_path_in_tree(tree::DTree{S}, X::GenericDataset)::Vector{DecisionPath} where {S}
	n_instances = n_samples(X)
    println("instances: ", n_instances)
	paths = fill(DecisionPath(), n_instances)
	for i_instance in 1:n_instances
		worlds = DecisionTree.inst_init_world_sets(X, tree, i_instance)
		get_path_in_tree(tree.root, X, i_instance, worlds, paths[i_instance])
	end
	paths
end

function apply_tree_to_datasets_wavs(
        tree_hash::String,
        tree::DTree{S},
        dataset::GenericDataset,
        wav_paths::Vector{String},
        labels::Vector{S};
        filter_kwargs = (),
        window_f::Function = hamming,
        destination_dir::String = "filtering-results/filtered"
    ) where {S}

    n_instances = n_samples(dataset)

    @assert n_instances == length(wav_paths) "dataset and wav_paths length mismatch! $(n_instances) != $(length(wav_paths))"
    @assert n_instances == length(labels) "dataset and labels length mismatch! $(n_instances) != $(length(labels))"

    if dataset isa MultiFrameModalDataset
        @assert n_frames(dataset) == 1 "MultiFrameModalDataset with more than one frame is still not supported! n_frames(dataset): $(n_frames(dataset))"
    end

    results = Vector{InstancePathInTree}(undef, n_instances)
    predictions = apply_tree(tree, dataset)
    paths = get_path_in_tree(tree, dataset)
    Threads.@threads for i in 1:n_instances
        results[i] = InstancePathInTree{S}(wav_paths[i], labels[i], tree)
        
        results[i].predicted = predictions[i]
        results[i].path = paths[i]
    end

    originals = Vector{Vector{Float64}}(undef, n_instances)
    samplerates = Vector{Number}(undef, n_instances)
    Threads.@threads for i in 1:n_instances
        curr_orig, samplerates[i] = wavread(wav_paths[i])
        originals[i] = merge_channels(curr_orig)
    end
    
    filtered = Vector{Vector{Float64}}(undef, n_instances)
    Threads.@threads for i in 1:n_instances
        # TODO: use path + worlds to generate dynamic filters
        if !is_correctly_classified(results[i])
            println("Skipping file $(wav_paths[i]) because it was not correctly classified...")
            continue
        end
        n_features = length(results[i].path)
        bands = Vector{Int64}(undef, n_features)
        weights = Vector{AbstractFloat}(undef, n_features)
        for j in 1:n_features
            # TODO: here goes the logic interpretation of the tree
            bands[j] = results[i].path[j].feature.i_attribute
            weights[j] = 1.
        end
        println("Applying filter to file $(wav_paths[i]) with bands $(string(bands))...")
        filter = multibandpass_digitalfilter_mel(bands, samplerates[i], window_f; weights = weights, filter_kwargs...)
        filtered[i] = filt(filter, originals[i])
    end

    real_destination = destination_dir * "/" * tree_hash
    mkpath(real_destination)
    Threads.@threads for i in 1:n_instances
        if !is_correctly_classified(results[i])
            println("Skipping file $(wav_paths[i]) because it was not correctly classified...")
            continue
        end
        save_path = wav_paths[i]
        while startswith(save_path, "../")
            save_path = replace(save_path, "../" => "")
        end
        mkpath(dirname(real_destination * "/" * save_path))
        println("Saving filtered file $(real_destination * "/" * save_path)...")
        wavwrite(filtered[i], real_destination * "/" * save_path; Fs = samplerates[i])
    end

    results
end

# TODO: implement this function for real (it is just a draft for now...)
# function apply_dynfilter!(
#         dynamic_filter::Matrix{Integer},
#         sample::AbstractVector{T};
#         samplerate = 16000,
#         wintime = 0.025,
#         steptime = 0.01,
#         window_f::Function
#     )::Vector{T} where {T <: Real}

#     nbands = size(dynamic_filter, 2)
#     ntimechunks = ntimechunks
#     nwin = round(Integer, wintime * sr)
#     nstep = round(Integer, steptime * sr)

#     window = window_f(nwin)

#     winsize = (samplerate / 2) / nwin

#     # init filters
#     filters = [ digitalfilter(Filters.Bandpass((i-1) * winsize, (i * winsize) - 1, fs = samplerate), window) for i in 1:nbands ]

#     # combine filters to generate EQs
#     slice_filters = Vector{Vector{Float64}}(undef, ntimechunks)
#     for i in 1:ntimechunks
#         step_filters = (@view filters[findall(isequal(1), dynamic_filter[i])])
#         slice_filters[i] = maximum.(collect(zip(step_filters...)))
#     end

#     # write filtered time chunks to new_track
#     new_track = Vector{T}(undef, length(sample) - 1)
#     time_chunk_length = length(sample) / ntimechunks
#     for i in 1:ntimechunks
#         new_track[i:(i*time_chunk_length) - 1] = filt(slice_filters[i], sample[i:(i*time_chunk_length) - 1])
#     end

#     new_track
# end
