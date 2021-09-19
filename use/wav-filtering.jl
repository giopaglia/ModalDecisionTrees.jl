
include("scanner.jl")
# include("datasets.jl")
# include("lib.jl")
# include("caching.jl")
include("wav2stft_time_series.jl")
include("local.jl")

# using DSP: filt
# using Weave
# import Pkg
# Pkg.activate("..")

using DecisionTree
using DecisionTree.ModalLogic

using Plots
using Images
using MFCC: hz2mel, mel2hz
using Plots.Measures
using JSON

MFCC.mel2hz(f::AbstractFloat, htk=false)  = mel2hz([f], htk)[1]

function printprogress(io::Base.TTY, string::String)
    print(io, "\033[2K\033[1000D" * string)
end
function printprogress(io::IO, string::String)
    println(io, string)
end
function printprogress(string::String)
    printprogress(stdout, string)
end

default_plot_size = (1920, 1080)
default_plot_margins = (
    left_margin = 15mm,
    right_margin = 15mm,
    top_margin = 15mm,
    bottom_margin = 15mm
)

"""
Calculate frequency response
"""
# https://weavejl.mpastell.com/stable/examples/FIR_design.pdf
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

"""
Plot the frequency and impulse response
"""
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
    plotfunc(ws, amp2db.(abs.(h)), xlabel="Frequency (Hz)", ylabel="Magnitude (db)", xlims = xlims, ylims = ylims, leg = false, size = default_plot_size, default_plot_margins...)
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
        minfreq = 0.0,
        maxfreq = fs / 2,
        nwin = nbands,
        weights::Vector{F} where F <:AbstractFloat = fill(1., length(selected_bands))
    )::AbstractVector

    @assert length(weights) == length(selected_bands) "length(weights) != length(selected_bands): $(length(weights)) != $(length(selected_bands))"

    band_width = (maxfreq - minfreq) / nbands

    result_filter = zeros(Float64, nwin)
    i = 1
    @simd for b in selected_bands
        l = b * band_width
        r = ((b+1) * band_width) - 1
        result_filter += digitalfilter(Filters.Bandpass(l <= 0 ? eps(typof(l)) : l, r >= maxfreq ? r - 0.000001 : r, fs = fs), FIRWindow(window_f(nwin))) * weights
        i=i+1
    end
    result_filter
end

struct MelBand
    left  :: Real
    right :: Real
    peak  :: Real
    MelBand(left::Real, right::Real, peak::Real) = new(max(eps(Float64), left), right, peak)
end

struct MelScale
    nbands :: Int
    bands  :: Vector{MelBand}
    MelScale(nbands::Int) = new(nbands, Vector{MelBand}(undef, nbands))
    MelScale(nbands::Int, bands::Vector{MelBand}) = begin
        @assert length(bands) == nbands "nbands != length(bands): $(nbands) != $(length(bands))"
        new(nbands, bands)
    end
    MelScale(bands::Vector{MelBand}) = new(length(bands), bands)
end

import Base: getindex, setindex!, length

length(scale::MelScale)::Int = length(scale.bands)
getindex(scale::MelScale, idx::Int)::MelBand = scale.bands[idx]
setindex!(scale::MelScale, band::MelBand, idx::Int)::MelBand = scale.bands[idx] = band

function melbands(nbands::Int, minfreq::Real = 0.0, maxfreq::Real = 8_000.0; htkmel = false)::Vector{Float64}
    minmel = hz2mel(minfreq, htkmel)
    maxmel = hz2mel(maxfreq, htkmel)
    mel2hz(minmel .+ collect(0:(nbands+1)) / (nbands+1) * (maxmel-minmel), htkmel)
end
function get_mel_bands(nbands::Int, minfreq::Real = 0.0, maxfreq::Real = 8_000.0; htkmel = false)::MelScale
    bands = melbands(nbands, minfreq, maxfreq; htkmel = htkmel)
    MelScale(nbands, [ MelBand(bands[i], bands[i+2] >= maxfreq ? bands[i+2] - 0.0000001 : bands[i+2], bands[i+1]) for i in 1:(length(bands)-2) ])
end

# TODO: create dispatch of this function on presence of 'window_f` argument
function digitalfilter_mel(band::MelBand, fs::Real, window_f::Function = triang; nwin = 60, filter_type = Filters.Bandpass)
    digitalfilter(filter_type(band.left, band.right, fs = fs), FIRWindow(; transitionwidth = 0.01, attenuation = 160))
end

function multibandpass_digitalfilter_mel(
        selected_bands :: Vector{Int},
        fs             :: Real,
        window_f       :: Function;
        nbands         :: Integer                           = 60,
        minfreq        :: Real                              = 0.0,
        maxfreq        :: Real                              = fs / 2,
        nwin           :: Int                               = nbands,
        weights        :: Vector{F} where F <:AbstractFloat = fill(1., length(selected_bands))
    )::AbstractVector

    @assert length(weights) == length(selected_bands) "length(weights) != length(selected_bands): $(length(weights)) != $(length(selected_bands))"

    result_filter = nothing#zeros(Float64, nwin)
    scale = get_mel_bands(nbands, minfreq, maxfreq)
    i = 1
    @simd for b in selected_bands
        if isnothing(result_filter)
            result_filter = digitalfilter_mel(scale[b], fs, window_f, nwin = nwin) * weights[i]
        else
            result_filter += digitalfilter_mel(scale[b], fs, window_f, nwin = nwin) * weights[i]
        end
        i=i+1
    end
    result_filter
end

function draw_mel_filters_graph(fs::Real, window_f::Function; nbands::Integer = 60, minfreq::Real = 0.0, maxfreq::Real = fs / 2)
    scale = get_mel_bands(nbands, minfreq, maxfreq)
    filters = [ digitalfilter_mel(scale[i], fs, window_f; nwin = nbands) for i in 1:nbands ]
    plotfilter(filters[1]; samplerate = fs)
    for i in 2:(length(filters)-1)
        plotfilter!(filters[i]; samplerate = fs)
    end
    # last call to plot has to "return" from function otherwise the graph will not be displayed
    plotfilter!(filters[end]; samplerate = fs)
end

function plot_band(band::MelBand; minfreq::Real = 0.0, maxfreq::Real = 8_000.0, ylims::Tuple{Number,Number} = (0.0, 1.0), show_freq::Bool = true, plot_func::Function = plot)
    common_args = (ylims = ylims, xlims = (minfreq, maxfreq), xguide = "Frequency (Hz)", yguide = "Amplitude", leg = false)
    texts = ["", show_freq ? text(string(round(Int64, band.peak)), font(pointsize = 8)) : "", ""]
    plot_func([band.left, band.peak, band.right], [ylims[1], ylims[2], ylims[1]]; annotationfontsize = 8, texts = texts, size = default_plot_size, common_args..., default_plot_margins...)
end
plot_band!(band::MelBand; kwargs...) = plot_band(band; plot_func = plot!, kwargs...)

function draw_synthetic_mel_filters_graph(; nbands::Integer = 60, minfreq::Real = 0.0, maxfreq::Real = 8_000.0)
    scale = get_mel_bands(nbands, minfreq, maxfreq)
    plot_band(scale[1]; minfreq = minfreq, maxfreq = maxfreq)
    for i in 2:(length(scale)-1)
        plot_band!(scale[i]; minfreq = minfreq, maxfreq = maxfreq)
    end
    # last call to plot has to "return" from function otherwise the graph will not be displayed
    plot_band!(scale[length(scale)]; minfreq = minfreq, maxfreq = maxfreq)
end

timerange2points(range::Tuple{T, T} where T <:Number, fs::Real)::UnitRange{Int64} = max(1, round(Int64, range[1] * fs)):round(Int64, range[2] * fs)
points2timerange(range::UnitRange{Int64}, fs::Real)::Tuple{T, T} where T <:Real = ((range.start - 1) / fs, (range.stop) / fs)
frame2points(index::Int64, framesize::Int64, stepsize::Int64)::UnitRange{Int64} = begin
    start = round(Int64, (index - 1) * stepsize) + 1
    start:(start+framesize-1)
end
frame2points(index::Int64, framesize::AbstractFloat, stepsize::AbstractFloat, fs::AbstractFloat)::UnitRange{Int64} = frame2points(index, round(Int64, framesize * fs), round(Int64, stepsize * fs))
frame2timerange(index::Int64, framesize::AbstractFloat, stepsize::AbstractFloat, fs::AbstractFloat)::Tuple{T, T} where T <:Number = points2timerange(frame2points(index, framesize, stepsize, fs), fs)

timerange2points(ranges::Vector{Tuple{T, T}} where T <:Number, fs::Real)::Vector{UnitRange{Int64}} = [ timerange2points(r, fs) for r in ranges ]
points2timerange(ranges::Vector{UnitRange{Int64}}, fs::Real)::Vector{Tuple{T, T}} where T <:Real = [ points2timerange(r, fs) for r in ranges ]
frame2points(indices::Vector{Int64}, framesize::Int64, stepsize::Int64)::Vector{UnitRange{Int64}} = [ frame2points(i, framesize, stepsize) for i in indices ]
frame2points(indices::Vector{Int64}, framesize::AbstractFloat, stepsize::AbstractFloat, fs::AbstractFloat)::Vector{UnitRange{Int64}} = [ frame2points(i, framesize, stepsize, fs) for i in indices ]
frame2timerange(indices::Vector{Int64}, framesize::AbstractFloat, stepsize::AbstractFloat, fs::AbstractFloat)::Vector{Tuple{T, T}} where T <:Number = [ frame2timerange(i, framesize, stepsize, fs) for i in indices ]

function approx_wav(
        samples   :: Vector{T},
        fs        :: Real;
        mode      :: Function    = maximum, # rms
        width     :: Real        = 1000.0,
        scale_res :: Real        = 1.0
    ):: Tuple{Vector{T}, Real} where T<:Real

    num_frames = ceil(Int64, (width * scale_res) / 2) + 1

    step_size = floor(Int64, length(samples) / num_frames)
    frame_size = step_size

    # TODO: optimize
    frames = []
    for i in 1:num_frames
        interval = frame2points(i, frame_size, step_size)
        # println(interval)
        interval = UnitRange(max(interval.start, 1), min(interval.stop, length(samples)))
        push!(frames, samples[interval])
    end

    frames_contracted = mode.(frames)

    # TODO: optimize
    n = []
    for point in frames_contracted
        push!(n, point)
        push!(n, -point)
    end

    n, (fs * (length(n) / length(samples)))
end
function approx_wav(filepath::String; kwargs...):: Tuple{Vector{T}, Real} where T<:Real
    samples, fs = wavread(filepath)
    samples = merge_channels(samples)

    approx_wav(samples, fs; kwargs...)
end

function draw_audio_anim(
        audio_files               :: Vector{Tuple{Vector{T1},T2}} where {T1<:AbstractFloat, T2<:AbstractFloat};
        labels                    :: Vector{String}                                      = fill("", length(audio_files)),
        colors                    :: Union{Vector{Symbol},Vector{RGB{Float64}}}          = fill(:auto, length(audio_files)),
        outfile                   :: String                                              = homedir() * "/gif.gif",
        size                      :: Tuple{Int64,Int64}                                  = (1000, 150 * length(audio_files)),
        fps                       :: Int64                                               = 30,
        resample_at_rate          :: Real                                                = 8000.0, # 0 for "no resample"
        reset_canvas_every_frames :: Integer                                             = 50,
        # selected_range:
        # - 1:1000 means from point 1 to point 1000
        # - (1.1, 2.3) means from time 1.1 to time 2.3 (in seconds)
        # - :whole means "do not slice"
        selected_range            :: Union{UnitRange{Int64},Tuple{Number,Number},Symbol} = :whole,
        single_graph              :: Bool                                                = false,
        use_wav_apporximation     :: Bool                                                = true,
        wav_apporximation_scale   :: Real                                                = 1.0
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

    if resample_at_rate > 0
        Threads.@threads for i in 1:length(wavs)
            if fss[i] == resample_at_rate continue end
            wavs[i] = resample(wavs[i], resample_at_rate / fss[i])
        end
        fss .= resample_at_rate
    end

    @assert length(unique(fss)) == 1 "Inconsistent bitrate across multiple files (try using resample keyword argument)"
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

    real_wavs = []
    real_fss = []
    if use_wav_apporximation
        # TODO: optimize
        for i in 1:length(wavs)
            curr_samps, curr_fs = approx_wav(wavs[i], fss[i]; scale_res = wav_apporximation_scale, width = size[1])
            push!(real_wavs, curr_samps)
            push!(real_fss, curr_fs)
        end
    else
        # TODO: optimize
        real_wavs = wavs
        real_fss = fss
    end

    @assert length(real_wavs) == length(wavs) "Transformed wavs length != original wavs length: $(length(real_wavs)) != $(length(wavs))"
    @assert length(real_fss) == length(fss) "Transformed fss length != original fss length: $(length(real_fss)) != $(length(fss))"

    if real_fss[1] < fps
        fps = floor(Int64, real_fss[1])
        @warn "Reducing FPS to $(fps) due to sampling reate too low"
    end

    wavlength = length(real_wavs[1])
    freq = real_fss[1]
    wavlength_seconds = wavlength / freq

    total_frames = ceil(Int64, wavlength_seconds * fps)
    step = wavlength / total_frames

    anim = nothing
    plts = []
    for (i, w) in enumerate(real_wavs)
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

    plts_orig = deepcopy(plts)
    anim = @animate for f in 1:total_frames
        printprogress("Processing frame $(f) / $(total_frames)")
        if f != 1
            if f % reset_canvas_every_frames == 0
                plts = deepcopy(plts_orig)
            else
                Threads.@threads for p in 1:length(plts)
                    # Make previous vline invisible
                    plts[p].series_list[end][:linealpha] = 0.0
                end
            end
        end
        for p in 1:length(plts)
            vline!(plts[p], [ (f-1) * step ], line = (:black, 1))
        end
        plot(plts..., layout = (length(real_wavs), 1))
    end
    printprogress("Completed all frames ($(total_frames))\n")

    gif(anim, outfile, fps = fps)
end

function draw_audio_anim(audio_files::Vector{String}; kwargs...)
    # TODO: test this dispatch
    @assert length(audio_files) > 0 "No audio file provided"

    converted_input::Vector{Tuple{Vector{AbstractFloat},AbstractFloat}} = []
    for f in audio_files
        wav, fs = wavread(f)
        push!(converted_input, (merge_channels(wav), fs))
    end

    draw_audio_anim(converted_input; kwargs...)
end

function draw_spectrogram(
        samples                  :: Vector{T},
        fs                       :: Real;
        gran                     :: Int                           = 50,
        title                    :: String                        = "",
        clims                    :: Tuple{Number,Number}          = (-100, 0),
        spectrogram_plot_options :: NamedTuple                    = NamedTuple(),
        melbands                 :: NamedTuple                    = (draw = false, nbands = 60, minfreq = 0.0, maxfreq = fs / 2, htkmel = false),
        only_bands               :: Union{Symbol,Vector{Int64}}   = :all
    ) where T <: AbstractFloat
    nw_orig::Int = round(Int64, length(samples) / gran)

    default_melbands = (draw = false, nbands = 60, minfreq = 0.0, maxfreq = fs / 2, htkmel = false)
    melbands = merge(default_melbands, melbands)

    if only_bands isa Symbol
        only_bands = collect(1:melbands[:nbands])
    elseif melbands[:draw] == false
        @warn "Selected bands to display but melbands[:draw] is false => no band will be displayed in the spectrogram"
    end

    default_heatmap_kwargs = (xguide = "Time (s)", yguide = "Frequency (Hz)", ylims = (0, fs / 2),  background_color_inside = :black, size = default_plot_size, leg = true, )
    total_heatmap_kwargs = merge(default_heatmap_kwargs, default_plot_margins)
    total_heatmap_kwargs = merge(total_heatmap_kwargs, spectrogram_plot_options)

    spec = spectrogram(samples, nw_orig, round(Int64, nw_orig/2); fs = fs)
    hm = heatmap(spec.time, spec.freq, pow2db.(spec.power); title = title, clims = clims, total_heatmap_kwargs...)

    # Draw horizontal line on spectrograms corresponding for selected bands
    if melbands[:draw]
        bands = get_mel_bands(melbands[:nbands], melbands[:minfreq], melbands[:maxfreq]; htkmel = melbands[:htkmel])
        for i in 1:melbands[:nbands]
            if i in only_bands
                hline!(hm, [ bands[i].left, bands[i].right ], line = (1, :white), leg = false)
            end
        end
        yticks!(hm, [ melbands[:minfreq], [ bands[i].peak for i in only_bands ]..., melbands[:maxfreq] ], [ string(round(Int64, melbands[:minfreq])), [ string("A", i) for i in only_bands ]..., string(round(Int64, melbands[:maxfreq])) ])
    end

    hm
end
function draw_spectrogram(filepath::String; kwargs...)
    samp, sr = wavread(filepath)
    samp = merge_channels(samp)

    draw_spectrogram(samp, sr; kwargs...)
end

struct DecisionPathNode
    taken         :: Bool
    feature       :: ModalLogic.FeatureTypeFun
    test_operator :: TestOperatorFun
    threshold     :: T where T
    worlds        :: AbstractWorldSet
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

get_path_in_tree(leaf::DTLeaf, X::Any, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, paths::Vector{DecisionPath} = Vector(DecisionPath()))::Vector{DecisionPath} = return paths
function get_path_in_tree(tree::DTInternal, X::MultiFrameModalDataset, i_instance::Integer, worlds::AbstractVector{<:AbstractWorldSet}, paths::Vector{DecisionPath} = Vector(DecisionPath()))::Vector{DecisionPath}
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

    push!(paths[i_instance], DecisionPathNode(satisfied, tree.feature, tree.test_operator, tree.threshold, deepcopy(new_worlds)))

	worlds[tree.i_frame] = new_worlds
	get_path_in_tree((satisfied ? tree.left : tree.right), X, i_instance, worlds, paths)
end
function get_path_in_tree(tree::DTree{S}, X::GenericDataset)::Vector{DecisionPath} where {S}
	n_instances = n_samples(X)
	paths::Vector{DecisionPath} = fill([], n_instances)
	for i_instance in 1:n_instances
		worlds = DecisionTree.inst_init_world_sets(X, tree, i_instance)
		get_path_in_tree(tree.root, X, i_instance, worlds, paths)
	end
	paths
end

function get_internalnode_dirname(node::DTInternal)::String
    replace(DecisionTree.display_decision(node), " " => "_")
end

mk_tree_path(leaf::DTLeaf; path::String = "") = touch(path * "/" * string(leaf.majority) * ".txt")
function mk_tree_path(node::DTInternal; path::String = "")
    dir_name = get_internalnode_dirname(node)
    mkpath(path * "/Y_" * dir_name)
    mkpath(path * "/N_" * dir_name)
    mk_tree_path(node.left; path = path * "/Y_" * dir_name)
    mk_tree_path(node.right; path = path * "/N_" * dir_name)
end
function mk_tree_path(tree_hash::String, tree::DTree; path::String = "filtering-results/filtered")
    mkpath(path * "/" * tree_hash)
    mk_tree_path(tree.root; path = path * "/" * tree_hash)
end

function get_tree_path_as_dirpath(tree_hash::String, tree::DTree, decpath::DecisionPath; path::String = "filtering-results/filtered")::String
    current = tree.root
    result = path * "/" * tree_hash
    for node in decpath
        if current isa DTLeaf break end
        result *= "/" * (node.taken ? "Y" : "N") * "_" * get_internalnode_dirname(current)
        current = node.taken ? current.left : current.right
    end
    result
end

function apply_tree_to_datasets_wavs(
        tree_hash                             :: String,
        tree                                  :: DTree{S},
        dataset                               :: GenericDataset,
        wav_paths                             :: Vector{String},
        labels                                :: Vector{S};
        # MODE OPTIONS
        filter_type                           :: Symbol             = :static,   # :dynamic
        mode                                  :: Symbol             = :bandpass, # :emphasize
        # OUTPUT OPTIONS
        only_files                            :: Vector{String}     = Vector{String}(),
        files_already_generated               :: Bool               = false,
        use_original_dataset_filesystem_tree  :: Bool               = false,
        destination_dir                       :: String             = "filtering-results/filtered",
        remove_from_path                      :: String             = "",
        save_single_band_tracks               :: Bool               = true,
        generate_json                         :: Bool               = true,
        json_file_name                        :: String             = "files.json",
        verbose                               :: Bool               = false,
        # WAV POST-PROCESS
        postprocess_wavs                      :: Vector{Function}   = [ trim_wav!,      normalize! ],
        postprocess_wavs_kwargs               :: AbstractVector     = [ (level = 0.0,), (level = 1.0,) ],
        # FILTER OPTIONS
        filter_kwargs                         :: NamedTuple         = (nbands = 40,),
        window_f                              :: Function           = triang,
        # SPECTROGRAM OPTIONS
        generate_spectrogram                  :: Bool               = true,
        spectrograms_kwargs                   :: NamedTuple         = NamedTuple(),
        spec_show_only_found_bands            :: Bool               = true,
        # ANIM AND VIDEO OPTIONS (note: can't create video without animation)
        draw_anim_for_instances               :: Vector{Int64}      = Vector{Int64}(),
        anim_kwargs                           :: NamedTuple         = NamedTuple(),
        normalize_before_draw_anim            :: Bool               = false,
        generate_video_from_gif               :: Bool               = true,
        video_kwargs                          :: NamedTuple         = NamedTuple()
    ) where {S}

    @assert mode in [ :bandpass, :emphasize ] "Got unsupported mode Symbol $(mode). It can be $([ :bandpass, :emphasize ])."
    @assert filter_type in [ :dynamic, :static ] "Got unsupported filter_type Symbol $(filter_type). It can be $([ :dynamic, :static ])."

    # TODO:
    if mode == :emphasize
        @warn "The value :emphasize for 'mode` is still experimental!"
    end
    if filter_type == :dynamic
        @warn "The value :emphasize for 'filter_type` is still experimental!"
    end

    n_instances = n_samples(dataset)

    println()
    println("Applying tree $(tree_hash):")
    print_tree(tree)
    println()
    print_apply_tree(tree, dataset, labels)
    println()

    if length(postprocess_wavs) == 0
        empty!(postprocess_wavs_kwargs)
    end

    @assert n_instances == length(wav_paths) "dataset and wav_paths length mismatch! $(n_instances) != $(length(wav_paths))"
    @assert n_instances == length(labels) "dataset and labels length mismatch! $(n_instances) != $(length(labels))"

    @assert length(postprocess_wavs) == length(postprocess_wavs_kwargs) "length(postprocess_wavs) != length(postprocess_wavs_kwargs): $(length(postprocess_wavs)) != $(length(postprocess_wavs_kwargs))"

    if dataset isa MultiFrameModalDataset
        @assert n_frames(dataset) == 1 "MultiFrameModalDataset with more than one frame is still not supported! n_frames(dataset): $(n_frames(dataset))"
    end

    json_archive = Dict{String, Vector{String}}()

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

    if :maxfreq in keys(filter_kwargs)
        min_sr = minimum(samplerates)
        @assert (min_sr / 2) >= filter_kwargs[:maxfreq] "maxfreq ($(filter_kwargs[:maxfreq])) is too high: lower samplerate in dataset is $min_sr (Nyquist freq: $(min_sr / 2))"
    end

    #############################################################
    ########## FILTER USING TREE FEATURES #######################
    #############################################################

    filtered = Vector{Vector{Float64}}(undef, n_instances)
    Threads.@threads for i in 1:n_instances
        if length(only_files) > 0 && !(wav_paths[i] in only_files)
            if verbose println("Skipping file $(wav_paths[i]) because it is not in the list...") end
            continue
        end
        if !is_correctly_classified(results[i])
            # TODO: handle not correctly classified instances
            if verbose println("Skipping file $(wav_paths[i]) because it was not correctly classified...") end
            continue
        end
        n_features = length(results[i].path)
        bands = Vector{Int64}(undef, n_features)
        weights = Vector{AbstractFloat}(undef, n_features)
        for j in 1:n_features
            if mode == :bandpass
                weights[j] = 1.0
            elseif mode == :emphasize
                # TODO: set a proper way to choose the threshold
                # TODO: real behaviour to emphasizes
                if (in(results[i].path[j].test_operator, [ >, >= ]) && results[i].path[j].taken) ||
                   (in(results[i].path[j].test_operator, [ <, <= ]) && !results[i].path[j].taken)
                    if results[i].path[j].threshold <= 1 # > than low
                        0.5
                    else # > than high
                        1.0
                    end
                else
                    if results[i].path[j].threshold <= 1 # < than low
                        0.25
                    else # < than high
                        0.5
                    end
                end
            end
            bands[j] = results[i].path[j].feature.i_attribute
        end
        println("Applying filter to file $(wav_paths[i]) with bands $(string(collect(zip(bands, weights))))...")
        # TODO: add here support for dynamic filter
        # if filter_type == :dynamic
        filter = multibandpass_digitalfilter_mel(bands, samplerates[i], window_f; weights = weights, filter_kwargs...)
        filtered[i] = filt(filter, originals[i])
        # elseif filter_type == :static
        #     filter = multibandpass_digitalfilter_mel(bands, samplerates[i], window_f; weights = weights, filter_kwargs...)
        #     filtered[i] = filt(filter, originals[i])
        # else
            # assert should be performed prior this point
        # end
    end

    #############################################################
    ######## APPLY POST-PROCESS AND SAVE FILTERED ###############
    #############################################################

    mk_tree_path(tree_hash, tree; path = destination_dir)

    real_destination = destination_dir * "/" * tree_hash
    mkpath(real_destination)
    heatmap_png_path = Vector{String}(undef, n_instances)
    json_lock = Threads.Condition()
    Threads.@threads for i in 1:n_instances
        if length(only_files) > 0 && !(wav_paths[i] in only_files)
            if verbose println("Skipping file $(wav_paths[i]) because it is not in the list...") end
            continue
        end
        if !is_correctly_classified(results[i])
            if verbose println("Skipping file $(wav_paths[i]) because it was not correctly classified...") end
            continue
        end
        save_path = replace(wav_paths[i], remove_from_path => "")
        if use_original_dataset_filesystem_tree
            while startswith(save_path, "../")
                save_path = replace(save_path, "../" => "")
            end
            save_dir = real_destination
        else
            save_dir = get_tree_path_as_dirpath(tree_hash, tree, results[i].path; path = destination_dir)
        end
        filtered_file_path = save_dir * "/" * replace(save_path, ".wav" => ".filt.wav")
        original_file_path = save_dir * "/" * replace(save_path, ".wav" => ".orig.wav")
        heatmap_png_path[i] = save_dir * "/" * replace(save_path, ".wav" => ".spectrogram.png")
        mkpath(dirname(filtered_file_path))
        for (i_pp, pp) in enumerate(postprocess_wavs)
            pp(filtered[i]; (postprocess_wavs_kwargs[i_pp])...)
            pp(originals[i]; (postprocess_wavs_kwargs[i_pp])...)
        end
        if files_already_generated
            continue
        end
        println("Saving filtered file $(filtered_file_path)...")
        wavwrite(filtered[i], filtered_file_path; Fs = samplerates[i])
        wavwrite(originals[i], original_file_path; Fs = samplerates[i])
        if generate_json
            if !haskey(json_archive, string(results[i].label))
                json_archive[string(results[i].label)] = []
            end
            Threads.lock(json_lock)
            push!(json_archive[string(results[i].label)], filtered_file_path)
            Threads.unlock(json_lock)
        end
    end

    #############################################################
    ######## GENERATE JSON WITH FILTERED WAV PATHS ##############
    #############################################################

    if generate_json && !files_already_generated
        try
            # TODO: handle "only_files" situation
            f = open(real_destination * "/" * json_file_name, "w")
            write(f, JSON.json(json_archive))
            close(f)
        catch
            println(stderr, "Unable to write file $(real_destination * "/" * json_file_name)")
        end
    end

    #############################################################
    ################ GENERATE SPECTROGRAMS ######################
    #############################################################

    spectrograms = Vector{NamedTuple{(:original, :filtered), Tuple{Plots.Plot, Plots.Plot}}}(undef, n_instances)
    if generate_spectrogram
        println("Generating spectrograms...")
        for i in 1:n_instances
            if length(only_files) > 0 && !(wav_paths[i] in only_files) continue end
            if !is_correctly_classified(results[i]) continue end
            println("Generating spectrogram $(heatmap_png_path[i])...")
            freqs =
                if spec_show_only_found_bands
                    [ path_node.feature.i_attribute for path_node in results[i].path ]
                else
                    :all
                end
            spectrograms[i] = (
                original = draw_spectrogram(originals[i], samplerates[i]; title = "Original", only_bands = freqs, spectrograms_kwargs...),
                filtered = draw_spectrogram(filtered[i], samplerates[i]; title = "Filtered", only_bands = freqs, spectrograms_kwargs...)
            )
            plot(spectrograms[i].original, spectrograms[i].filtered, layout = (1, 2))
            savefig(heatmap_png_path[i])
        end
    end

    #############################################################
    ########### SIGNLE BANDS, ANIMATION AND VIDEO ###############
    #############################################################

    single_band_tracks = Vector{Vector{Tuple{Int64,Vector{Float64}}}}(undef, n_instances)
    if save_single_band_tracks
        println("Generating single band tracks...")
        for i in 1:n_instances
            if length(only_files) > 0 && !(wav_paths[i] in only_files) continue end
            if !is_correctly_classified(results[i]) continue end
            single_band_tracks[i] = Vector{Tuple{Int64,Vector{Float64}}}(undef, length(results[i].path))
            n_features = length(results[i].path)
            for j in 1:n_features
                feat = results[i].path[j].feature.i_attribute
                one_band_sample = filt(multibandpass_digitalfilter_mel([feat], samplerates[i], window_f; filter_kwargs...), originals[i])
                single_band_tracks[i][j] = (feat, one_band_sample)
            end
        end
        nbands = filter_kwargs[:nbands]
        features_colors = [ RGB(1 - (1 * (i/(nbands - 1))), 0, 1 * (i/(nbands - 1))) for i in 0:(nbands-1) ]
        for i in 1:n_instances
            if length(only_files) > 0 && !(wav_paths[i] in only_files) continue end
            if !is_correctly_classified(results[i]) continue end
            for (feat, samples) in single_band_tracks[i]
                save_path = replace(heatmap_png_path[i], ".spectrogram.png" => ".A$(feat).wav")
                println("Saving single band track to file $(save_path)...")
                wavwrite(samples, save_path; Fs = samplerates[i])
            end
            if i in draw_anim_for_instances
                gifout = replace(heatmap_png_path[i], ".spectrogram.png" => ".anim.gif")
                wws = 
                    if normalize_before_draw_anim
                        orig_norm = normalize!(deepcopy(originals[i]))
                        norm_rate = maximum(abs, orig_norm) / maximum(abs, originals[i])
                        collect(zip(
                            [ orig_norm, filtered[i] * norm_rate, [single_band_tracks[i][j][2] * norm_rate for j in 1:length(single_band_tracks[i])]... ],
                            fill(samplerates[i], 2 + length(single_band_tracks[i]))
                        ))
                    else
                        [ (originals[i], samplerates[i]), (filtered[i], samplerates[i]), [ (single_band_tracks[i][j][2], samplerates[i]) for j in 1:length(single_band_tracks[i]) ]... ]
                    end
                draw_audio_anim(
                    wws;
                    labels = [ "Original", "Filtered", [ string("A", single_band_tracks[i][j][1]) for j in 1:length(single_band_tracks[i]) ]... ],
                    colors = [ RGB(.3, .3, 1), RGB(1, .3, .3), features_colors[ [ b[1] for b in single_band_tracks[i] ] ]...],
                    outfile = gifout,
                    anim_kwargs...
                )
                if generate_video_from_gif
                    orig = replace(gifout, ".anim.gif" => ".orig.wav")
                    filt = replace(gifout, ".anim.gif" => ".filt.wav")
                    generate_video(gifout, [ orig, filt ]; video_kwargs...)
                end
            end
        end
    end

    return results, spectrograms
end

# const DynamicFilter = Matrix{Integer}

# n_bands(dynamic_filter::DynamicFilter) = size(dynamic_filter, 2)
# n_chunks(dynamic_filter::DynamicFilter) = size(dynamic_filter, 1)

# # TODO: implement this function for real (it is just a draft for now...)
# function dynfilt(
#         dynamic_filter      :: Matrix{Integer},
#         sample              :: AbstractVector{T};
#         samplerate          :: Real = 8_000.0,
#         wintime             :: Real = 0.025,
#         steptime            :: Real = 0.01,
#         minfreq             :: Real = 0.0,
#         maxfreq             :: Real = samplerate / 2,
#         window_f            :: Function = triang
#     )::Vector{T} where {T <: Real}

#     nbands = n_bands(dynamic_filter)
#     ntimechunks = n_chunks(dynamic_filter)
#     winlength = round(Integer, wintime * sr)
#     winstep = round(Integer, steptime * sr)

#     # init filters
#     filters = [
#             multibandpass_digitalfilter_mel(
#                 [i],
#                 samplerate,
#                 window_f,
#                 nbands = nbands,
#                 minfreq = minfreq,
#                 maxfreq = maxfreq,
#                 nwin = nbands,
#             ) for i in 1:nbands ]

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

function generate_video(
        gif                      :: String,
        wavs                     :: Vector{String};
        outpath                  :: Union{String,Nothing}        = nothing,
        outputfile_name          :: Union{String,Nothing}        = nothing,
        ffmpeg_output_file       :: Union{String,IO,Nothing}     = nothing,
        ffmpeg_error_output_file :: Union{String,IO,Nothing}     = nothing,
        video_codec              :: String                       = "libx265",
        audio_codec              :: String                       = "copy",
        output_ext               :: String                       = "mkv",
        additional_ffmpeg_args_v :: Union{Vector{String},String} = Vector{String}(),
        additional_ffmpeg_args_a :: Union{Vector{String},String} = Vector{String}()
    )

    @assert isfile(gif) "File $(gif) does not exist."
    for w in wavs
        @assert isfile(w) "File $(w) does not exist."
    end

    if !isnothing(outputfile_name) && endswith(outputfile_name, "." * output_ext)
        outputfile_name = replace(outputfile_name, "." * output_ext => "")
    end

    if !isnothing(outpath)
        try
            mkpath(outpath)
        catch
            error("Unable to create directory $(outpat)")
        end
    end

    if additional_ffmpeg_args_v isa String
        additional_ffmpeg_args_v = convert.(String, split(additional_ffmpeg_args_v, ' '; keepempty = false))
    end

    if additional_ffmpeg_args_a isa String
        additional_ffmpeg_args_a = convert.(String, split(additional_ffmpeg_args_a, ' '; keepempty = false))
    end

    try
        for w in wavs
            total_output_path = isnothing(outpath) ?
                (isnothing(outputfile_name) ?
                    replace(w, ".wav" => "." * output_ext) :
                    outputfile_name * "." * output_ext
                ) :
                outpath * "/" * (isnothing(outputfile_name) ?
                    replace(basename(w), ".wav" => "." * output_ext) :
                    outputfile_name * "." * output_ext
                )

            ffmpeg_output_file_manually_open = false
            ffmpeg_error_output_file_manually_open = false

            tmp_ffmpeg_output_file::Union{String,IO,Nothing} = ffmpeg_output_file
            tmp_ffmpeg_error_output_file::Union{String,IO,Nothing} = ffmpeg_error_output_file
            if isnothing(tmp_ffmpeg_output_file)
                tmp_ffmpeg_output_file = relpath(total_output_path) * "-ffmpeg.out"
            end
            if tmp_ffmpeg_output_file isa String
                tmp_ffmpeg_output_file = open(tmp_ffmpeg_output_file, "w+")
                ffmpeg_output_file_manually_open = true
            end

            if isnothing(tmp_ffmpeg_error_output_file)
                tmp_ffmpeg_error_output_file = tmp_ffmpeg_output_file
            end
            if tmp_ffmpeg_error_output_file isa String
                tmp_ffmpeg_error_output_file = open(tmp_ffmpeg_error_output_file, "w+")
                ffmpeg_error_output_file_manually_open = true
            end

            print("Generating video in $(total_output_path)...")
            run(pipeline(`ffmpeg -i $gif -i $w -y -c:a $audio_codec $additional_ffmpeg_args_a -c:v $video_codec $additional_ffmpeg_args_v $total_output_path`, stdout = tmp_ffmpeg_output_file, stderr = tmp_ffmpeg_error_output_file))
            println(" done")
            
            if ffmpeg_output_file_manually_open close(tmp_ffmpeg_output_file) end
            if ffmpeg_error_output_file_manually_open close(tmp_ffmpeg_error_output_file) end
        end
    catch
        println(" fail")
        if ffmpeg_error_output_file != stderr
            println("Look at file $(ffmpeg_error_output_file) to understand what went wrong!")
        end
        error("unable to generate video: is ffmpeg installed?")
    end
end

mkpath("tree-anim")
function draw_tree_anim(
        wav_descriptor    :: Vector{Bool},
        blank_image       :: AbstractMatrix,
        highlighted_image :: AbstractMatrix,
        fs                :: Real;
        outfile           :: String               = "tree-anim/tree-anim.gif",
        size              :: Tuple{Number,Number} = Tuple((max(size(blank_image,2), size(highlighted_image,2)), max(size(blank_image,1), size(highlighted_image,1)))),
        fps               :: Int64                = 30
    )
    function plot_image(image::AbstractMatrix; size = size)
        plot(
            image,
            framestyle     = :zerolines, # show axis at zeroes
            leg            = false,      # hide legend
            showaxis       = false,      # hide y axis
            grid           = false,      # hide y grid
            ticks          = false,      # hide y ticks
            tick_direction = :none,
            margin         = 0mm,
            size           = (size[1] + 25, size[2] + 25)
        )
    end
    wavlength = length(wav_descriptor)
    wavlength_seconds = wavlength / fs

    total_frames = ceil(Int64, wavlength_seconds * fps)
    step = wavlength / total_frames

    anim = @animate for f in 1:total_frames
        printprogress("Processing frame $(f) / $(total_frames)")
        point = floor(Int64, clamp((f-1) * step, 1, wavlength))
        plot_image(wav_descriptor[point] ? highlighted_image : blank_image; size = size)
    end
    printprogress("Completed all frames ($(total_frames))\n")

    gif(anim, outfile, fps = fps)
end
draw_tree_anim(wav_descriptor::Vector{Bool}, blank_image::String, highlighted_image::String, fs::Real; kwargs...) = draw_tree_anim(wav_descriptor, load(blank_image), load(highlighted_image), fs; kwargs...)

"""
    winsize and stepsize should have to ma_size and ma_step
"""
function get_points_and_seconds_from_worlds(worlds::DecisionTree.ModalLogic.AbstractWorldSet, winsize::Int64, stepsize::Int64, samplerate::Real)
    # keep largest worlds
    dict = Dict{Int64,DecisionTree.ModalLogic.AbstractWorld}()
    for w in worlds
        if !haskey(dict, w.x) || dict[w.x].y < w.y
            dict[w.x] = w
        end
    end

    # TODO: optimize this algorithm

    highest_key = maximum(keys(dict))
    # join overlapping worlds
    for i in 1:highest_key
        if haskey(dict, i)
            for j in 1:highest_key
                if i == j continue end
                if haskey(dict, j)
                    # do they overlap and is j.y > than i.y?
                    if dict[i].y >= dict[j].x && dict[j].y >= dict[i].y
                        dict[i] = DecisionTree.ModalLogic.Interval(dict[i].x, dict[j].y)
                        delete!(dict, j)
                    end
                end
            end
        end
    end

    timeranges = []
    for i in 1:highest_key
        if haskey(dict, i)
            w = dict[i]
            # println("Considering interval $(w.x:(w.y-1))")
            frames = frame2points(collect(w.x:(w.y-1)), winsize, stepsize)
            push!(timeranges, (frames[1][1], frames[end][2]))
        end
    end

    points = []
    seconds = []
    wav_descriptor = fill(false, length(samples))
    for tr in timeranges
        # println("Points: ", (tr[1], tr[2]),"; Time (s): ", points2timerange(tr[1]:tr[2], samplerate))
        push!(points, (tr[1], tr[2]))
        push!(seconds, points2timerange(tr[1]:tr[2], samplerate))
        wav_descriptor[max(1, tr[1]):min(tr[2], length(samples))] .= true
    end

    wav_descriptor, points, seconds
end

function join_tree_video_and_waveform_video(
            tree_v_path      :: String,
            tree_v_size      :: Tuple{Number,Number},
            wave_form_v_path :: String,
            wave_form_v_size :: Tuple{Number,Number},
            output_file      :: String                = "prova.mkv"
        )

    final_video_resolution = (max(tree_v_size[1], wave_form_v_size[1]), 1 + max(tree_v_size[2], wave_form_v_size[2]))
    final_tree_position = (round(Int64, (final_video_resolution[1] - tree_v_size[1]) / 2), 0)
    final_wave_form_position = (round(Int64, (final_video_resolution[1] - wave_form_v_size[1]) / 2), tree_v_size[2] + 1)

    final_video_resolution = (
        ((final_video_resolution[1] % 2 == 1) ? final_video_resolution[1] + 1 : final_video_resolution[1]),
        ((final_video_resolution[2] % 2 == 1) ? final_video_resolution[2] + 1 : final_video_resolution[2])
    )
    final_tree_position = (
        ((final_tree_position[1] % 2 == 1) ? final_tree_position[1] + 1 : final_tree_position[1]),
        ((final_tree_position[2] % 2 == 1) ? final_tree_position[2] + 1 : final_tree_position[2])
    )
    final_wave_form_position = (
        ((final_wave_form_position[1] % 2 == 1) ? final_wave_form_position[1] + 1 : final_wave_form_position[1]),
        ((final_wave_form_position[2] % 2 == 1) ? final_wave_form_position[2] + 1 : final_wave_form_position[2])
    )

    color_input = "color=white:$(final_video_resolution[1])x$(final_video_resolution[2]),format=rgb24"
    # color_input = "color=white:$(final_video_resolution[1])x$(final_video_resolution[2]):d=3,format=rgb24"

    total_complex_filter =  "[0:v]pad=$(final_video_resolution[1]):$(final_video_resolution[2]):0:[a]; "
    total_complex_filter *= "[a][1:v]overlay=$(final_tree_position[1]):$(final_tree_position[2]):0:[b]; "
    total_complex_filter *= "[b][2:v]overlay=$(final_wave_form_position[1]):$(final_wave_form_position[2]):0:[c]"

    run(`ffmpeg -f lavfi -i $color_input -i $tree_v_path -i $wave_form_v_path -y -filter_complex "$total_complex_filter" -map '[b]' -map '[c]' -map 2:a -c:a copy -c:v libx264 -crf 23 -preset veryfast $output_file`)

    # I have two videos and I need to place them side-by-side in a new video.
    # The dimensions first video (blob.mp4) was 1280×720. The second video (aoc.mp4) was 406×720.

    # ffmpeg -i blob.mp4 -i aoc.mp4 -filter_complex "[0:v]pad=1686:720:0:[a]; [a][1:v]overlay=1280:0[b]" -map "[b]" -pix_fmt yuv420p aoc-meets-garbage-disposal.mp4

    # In the first part of the filter, I specified that the first input video (blob.mp4)
    # will have a dimension of 1686×720, where 1686 is the total width of the both videos side-by-side.
    # A map identifier “[a]” is used for this screen canvas. Next, with a semicolon delimiter, the second
    # video is specified as appearing at 1280 pixels from the left edge at 0. (Yes, it begins at zero.
    # I got a stupid “Overlay area … not within the main area … or zero-sized… Failed to configure input
    # pad on Parsed_overlay_1” error when I thought I had to use 1281 as the starting point of the second video.)
    # The overlaid video is then map identified as “[b]”. This map is then encoded to the output video.
end