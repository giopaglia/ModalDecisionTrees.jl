
# TODO: all "outfile" arguments should be empty and return the graph instead of saving it on the HDD

using DSP, WAV, Plots, Images, Plots.Measures

default_plot_size = (1920, 1080)
default_plot_margins = (
    left_margin = 4mm,
    right_margin = 4mm,
    top_margin = 4mm,
    bottom_margin = 4mm
)

"""
    plotfilter(filter; samplerate = 100.0, xlims = (0,0), ylims = (-24, 0), plotfunc = plot)
    plotfilter!(filter; samplerate = 100.0, xlims = (0,0), ylims = (-24, 0))

Plot the frequency and impulse response of a filter.
"""
function plotfilter(
        filter;
        samplerate :: Real                            = 100.0,
        xlims      :: Union{Nothing,Tuple{Real,Real}} = (0, 0),
        ylims      :: Union{Nothing,Tuple{Real,Real}} = (-24, 0),
        plotfunc   :: Function                        = plot
    )

    w = range(0, stop=π, length=1024)
    h = FIRfreqz(filter; w = w)
    ws = w / π * (samplerate / 2)
    if xlims[2] == 0
        xlims = (xlims[1], maximum(ws))
    end
    plotfunc(ws, amp2db.(abs.(h)), xlabel="Frequency (Hz)", ylabel="Magnitude (db)", xlims = xlims, ylims = ylims, leg = false, size = default_plot_size, default_plot_margins...)
end
# TODO: test all dispatch
plotfilter(filter::Filters.Filter; kwargs...) = plotfilter(digitalfilter(filter, firwindow); kwargs...)
plotfilter!(filter::Filters.Filter; args...)  = plotfilter(filter; args..., plotfunc = plot!)
plotfilter!(filter; args...)                  = plotfilter(filter; args..., plotfunc = plot!)

"""
    draw_mel_filters_graph(samplerate, window_f; nbands = 40, minfreq = 0.0, maxfreq = 8000.0)

Draw a mel filters graph between `minfreq` and `maxfreq` using
`nbands` bands building the actual filters.

NB: if you just want to display the mel bands use the
function [`draw_synthetic_mel_filters_graph`](@ref) instead.
"""
function draw_mel_filters_graph(
            samplerate :: Real,
            window_f   :: Function;
            nbands     :: Integer   = 40,
            minfreq    :: Real      = 0.0,
            maxfreq    :: Real      = samplerate / 2
        )

    scale = get_mel_bands(nbands, minfreq, maxfreq)

    filters = [ digitalfilter_mel(scale[i], samplerate, window_f; nwin = nbands) for i in 1:nbands ]

    plotfilter(filters[1]; samplerate = samplerate)
    for i in 2:(length(filters)-1)
        plotfilter!(filters[i]; samplerate = samplerate)
    end
    # last call to plot has to "return" from function otherwise the graph will not be displayed
    plotfilter!(filters[end]; samplerate = samplerate)
end

"""
    plot_band(band; minfreq = 0.0, maxfreq = 8000.0, ylims = (0.0, 1.0), show_freq = true, plot_func = plot)
    plot_band!(band; minfreq = 0.0, maxfreq = 8000.0, ylims = (0.0, 1.0), show_freq = true)

Plot a `MelBand` in a graph using plot function `plot_func`.

NB: `plot_func` kwarg is provided to be able to create
function `plot_band!` and sould not be changed manually.

Defaults:
* `minfreq` = 0.0
* `maxfreq` = 8000.0
"""
function plot_band(
            band      :: MelBand;
            minfreq   :: Real                 = 0.0,
            maxfreq   :: Real                 = 8_000.0,
            ylims     :: Tuple{Number,Number} = (0.0, 1.0),
            show_freq :: Bool                 = true,
            plot_func :: Function             = plot
        )

    common_args = (ylims = ylims, xlims = (minfreq, maxfreq), xguide = "Frequency (Hz)", yguide = "Amplitude", leg = false)

    texts = [ "", show_freq ? text(string(round(Int64, band.peak)), font(pointsize = 8)) : "", ""]

    plot_func([band.left, band.peak, band.right], [ylims[1], ylims[2], ylims[1]]; annotationfontsize = 8, texts = texts, size = default_plot_size, common_args..., default_plot_margins...)
end
plot_band!(band::MelBand; kwargs...) = plot_band(band; plot_func = plot!, kwargs...)

"""
    draw_synthetic_mel_filters_graph(; nbands = 40, minfreq = 0.0, maxfreq = 8_000.0)

Draw a mel filters graph between `minfreq` and `maxfreq` using `nbands` bands.

NB: if you want to display the real mel bands use the function [`draw_mel_filters_graph`](@ref) instead.
"""
function draw_synthetic_mel_filters_graph(; nbands::Integer = 40, minfreq::Real = 0.0, maxfreq::Real = 8_000.0)
    scale = get_mel_bands(nbands, minfreq, maxfreq)

    plot_band(scale[1]; minfreq = minfreq, maxfreq = maxfreq)
    for i in 2:(length(scale)-1)
        plot_band!(scale[i]; minfreq = minfreq, maxfreq = maxfreq)
    end

    # last call to plot has to "return" from function otherwise the graph will not be displayed
    plot_band!(scale[length(scale)]; minfreq = minfreq, maxfreq = maxfreq)
end

"""
    draw_wav(samples[, samplerate]; color = :auto, title = "", size = (1000, 150), plotfunc = plot)
    draw_wav!(samples[, samplerate]; color = :auto, title = "", size = (1000, 150))

Plot a wave using its `samples`.

NB: `samplerate` is not used inside this function but inserted to
maintain interface consistency.
"""
function draw_wav(
            samples        :: Vector{T},
            samplerate     :: Number;
            color                                  = :auto, # TODO: find proper Union Type for this argument
            title          :: String               = "",
            size           :: Tuple{Number,Number} = (1000, 150),
            plotfunc       :: Function             = plot
        )::Plots.Plot where T <: AbstractFloat
    plot = plotfunc(
        collect(0:(length(samples) - 1)),
        samples,
        title = title,
        xlims = (0, length(samples)),
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
function draw_wav(samples::Vector{T}; kwargs...)::Plots.Plot where T <: AbstractFloat
    draw_wav(samples::Vector{T}, 44100; kwargs...)
end
function draw_wav!(samples::Vector{T}, samplerate::Number; kwargs...)::Plots.Plot where T <: AbstractFloat
    draw_wav(samples::Vector{T}, samplerate::Number; plotfunc = plot!, kwargs...)
end
function draw_wav!(samples::Vector{T}; kwargs...)::Plots.Plot where T <: AbstractFloat
    draw_wav(samples::Vector{T}, 44100; plotfunc = plot!, kwargs...)
end

"""
"""
function draw_audio_anim(
            audio_files               :: Vector{Tuple{Vector{T1},T2}};
            labels                    :: Vector{String}                             = fill("", length(audio_files)),
            colors                    :: Union{Vector{Symbol},Vector{RGB{Float64}}} = fill(:auto, length(audio_files)),
            single_graph              :: Bool                                       = false,
            outfile                   :: String                                     = "", # TODO: handle empty string
            size                      :: Tuple{Int64,Int64}                         = single_graph ? (1000, 150) : (1000, 150 * length(audio_files)),
            fps                       :: Int64                                      = 30,
            resample_at_rate          :: Real                                       = 0.0,
            reset_canvas_every_frames :: Integer                                    = 50,
            # selected_range:
            # - 1:1000 means from point 1 to point 1000
            # - (1.1, 2.3) means from time 1.1 to time 2.3 (in seconds)
            # - :whole means "do not slice"
            selected_range            :: Union{UnitRange{Int64},Tuple{Number,Number},Symbol} = :whole,
            use_wav_apporximation     :: Bool                                                = true,
            wav_approximation_scale   :: Real                                                = 1.0
        )::Plots.AnimatedGif where {T1<:AbstractFloat, T2<:AbstractFloat}

    @assert length(audio_files) > 0 "No audio file provided"
    if !single_graph
        @assert length(audio_files) == length(labels) "audio_files and labels mismatch in length: $(length(audio_files)) != $(length(labels))"
    end
    @assert length(audio_files) == length(colors) "audio_files and colors mismatch in length: $(length(audio_files)) != $(length(colors))"

    wavs = []
    samplerates = []
    for f in audio_files
        push!(wavs, merge_channels(f[1]))
        push!(samplerates, f[2])
    end

    if resample_at_rate > 0
        Threads.@threads for i in 1:length(wavs)
            if samplerates[i] == resample_at_rate continue end
            wavs[i] = resample(wavs[i], resample_at_rate / samplerates[i])
        end
        samplerates .= resample_at_rate
    end

    @assert length(unique(samplerates)) == 1 "Inconsistent bitrate across multiple files (try using resample keyword argument)"
    @assert length(unique([x -> length(x) for wav in wavs])) == 1 "Inconsistent length across multiple files"

    if selected_range isa Tuple
        # convert seconds to points
        println("Selected time range from $(selected_range[1])s to $(selected_range[2])s")
        selected_range = timerange2points(selected_range, samplerates[1])
    end
    # slice all wavs
    if selected_range != :whole
        println("Slicing from point $(collect(selected_range)[1]) to $(collect(selected_range)[end])")
        for i in 1:length(wavs)
            wavs[i] = wavs[i][selected_range]
        end
    end

    real_wavs = []
    real_samplerates = []
    if use_wav_apporximation
        # TODO: optimize
        for i in 1:length(wavs)
            curr_samps, curr_samplerate = approx_wav(wavs[i], samplerates[i]; scale_res = wav_approximation_scale, width = size[1])
            push!(real_wavs, curr_samps)
            push!(real_samplerates, curr_samplerate)
        end
    else
        # TODO: optimize
        real_wavs = wavs
        real_samplerates = samplerates
    end

    @assert length(real_wavs) == length(wavs) "Transformed wavs length != original wavs length: $(length(real_wavs)) != $(length(wavs))"
    @assert length(real_samplerates) == length(samplerates) "Transformed samplerates length != original samplerates length: $(length(real_samplerates)) != $(length(samplerates))"

    if real_samplerates[1] < fps
        fps = floor(Int64, real_samplerates[1])
        # TODO: maybe auto-select an higher resolution automatically
        @warn "Reducing FPS to $(fps) due to sampling reate too low"
    end

    wavlength = length(real_wavs[1])
    freq = real_samplerates[1]
    wavlength_seconds = wavlength / freq

    total_frames = ceil(Int64, wavlength_seconds * fps)
    step = wavlength / total_frames

    anim = nothing
    plts = []
    for (i, w) in enumerate(real_wavs)
        if i == 1
            push!(plts, draw_wav(w, freq; title = labels[i], color = colors[i], size = size))
        else
            if single_graph
                draw_wav!(w, freq; color = colors[i], size = size)
            else
                push!(plts, draw_wav(w, freq; title = labels[i], color = colors[i], size = size))
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
                    # Make previous vlines invisible
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
        wav, samplerate = wavread(f)
        push!(converted_input, (merge_channels(wav), samplerate))
    end

    draw_audio_anim(converted_input; kwargs...)
end

"""
    draw_spectrogram(samples, samplerate; gran = 50, title = "", clims = (-100, 0), spectrogram_plot_options = (),
    melbands = (draw = false, nbands = 40, minfreq = 0.0, maxfreq = samplerate / 2, htkmel = false), only_bands = :all)

    draw_spectrogram(filepath; gran = 50, title = "", clims = (-100, 0), spectrogram_plot_options = (),
    melbands = (draw = false, nbands = 40, minfreq = 0.0, maxfreq = samplerate / 2, htkmel = false), only_bands = :all)

Draw spectrogram of a wav.
"""
function draw_spectrogram(
            samples                  :: Vector{T},
            samplerate               :: Real;
            gran                     :: Int                         = 50,
            title                    :: String                      = "",
            clims                    :: Tuple{Number,Number}        = (-100, 0),
            spectrogram_plot_options :: NamedTuple                  = NamedTuple(),
            melbands                 :: NamedTuple                  = (draw = false, nbands = 40, minfreq = 0.0, maxfreq = samplerate / 2, htkmel = false),
            only_bands               :: Union{Symbol,Vector{Int64}} = :all
        ) where T <: AbstractFloat

    nw_orig::Int = round(Int64, length(samples) / gran)

    default_melbands = (draw = false, nbands = 40, minfreq = 0.0, maxfreq = samplerate / 2, htkmel = false)
    melbands = merge(default_melbands, melbands)

    if only_bands isa Symbol
    only_bands = collect(1:melbands[:nbands])
    elseif melbands[:draw] == false
    @warn "Selected bands to display but melbands[:draw] is false => no band will be displayed in the spectrogram"
    end

    default_heatmap_kwargs = (xguide = "Time (s)", yguide = "Frequency (Hz)", ylims = (0, samplerate / 2),  background_color_inside = :black, size = default_plot_size, leg = true, )
    total_heatmap_kwargs = merge(default_heatmap_kwargs, default_plot_margins)
    total_heatmap_kwargs = merge(total_heatmap_kwargs, spectrogram_plot_options)

    spec = spectrogram(samples, nw_orig, round(Int64, nw_orig / 2); fs = samplerate)
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

"""
    generate_video(gif, wavs; outpath = nothing, outputfile_name = nothing, ffmpeg_output_file = nothing, ffmpeg_error_output_file = nothing, video_codec = "libx265", audio_codec = "copy", output_ext = "mkv", additional_ffmpeg_args_v = [], additional_ffmpeg_args_a = [])

Generate `length(wavs)` videos with same video (the `gif`) but different audios (the `wavs`).

NB: it need ffmpeg to be installed on the system.
"""
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
            run(pipeline(`ffmpeg -hide_banner -i $gif -i $w -y -c:a $audio_codec $additional_ffmpeg_args_a -c:v $video_codec $additional_ffmpeg_args_v $total_output_path`, stdout = tmp_ffmpeg_output_file, stderr = tmp_ffmpeg_error_output_file))
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

"""
    draw_tree_anim(wav_descriptor, blank_image, highlighted_image, samplerate; outfile = "", size = `calculated from images`, fps = 30)

Draw a simple animation highlighting a tree branch when `wav_descriptor` is `true`.
TODO: make this more clear
"""
function draw_tree_anim(
            wav_descriptor    :: Vector{Bool},
            blank_image       :: AbstractMatrix,
            highlighted_image :: AbstractMatrix,
            samplerate        :: Real;
            outfile           :: String               = "", # TODO: handle empty string
            size              :: Tuple{Number,Number} = Tuple((max(size(blank_image,2), size(highlighted_image,2)), max(size(blank_image,1), size(highlighted_image,1)))),
            fps               :: Int64                = 30,
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
            size           = size
        )
    end

    wavlength = length(wav_descriptor)
    wavlength_seconds = wavlength / samplerate

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
draw_tree_anim(wav_descriptor::Vector{Bool}, blank_image::String, highlighted_image::String, samplerate::Real; kwargs...) = draw_tree_anim(wav_descriptor, load(blank_image), load(highlighted_image), samplerate; kwargs...)

"""
    draw_worlds(descriptor, samplerate; title = "", outfile = "", color = RGB(0.3, 0.3, 1), size = (1000, 150))

Draw a simple graph representing the worlds.

If `outfile` is an empty string the graph will not be saved on drive.
"""
function draw_worlds(
            descriptor :: Vector{Bool},
            samplerate :: Real;
            title      :: String               = "",
            outfile    :: String               = "",
            color      :: RGB                  = RGB(0.3, 0.3, 1),
            size       :: Tuple{Number,Number} = (1000, 150)
        )

    tot_length = length(descriptor)
    curr_pos = 0
    ticks_positions = Vector{Float64}()
    while tot_length > curr_pos
        push!(ticks_positions, curr_pos)
        curr_pos = curr_pos + floor(Int64, samplerate / 2)
    end
    ticks_string = Vector{String}(undef, length(ticks_positions))
    for i in 0:(length(ticks_positions)-1)
        ticks_string[i+1] = string(i*0.5)
    end

    p = plot(
        collect(0:(length(descriptor) - 1)),
        descriptor,
        title = title,
        xlims = (0, length(descriptor)),
        ylims = (0, 1),
        framestyle = :zerolines,       # show axis at zeroes
        fill = 0,                      # show area under curve
        leg = false,                   # hide legend
        yshowaxis = false,             # hide y axis
        ygrid = false,                 # hide y grid
        yticks = false,                # hide y ticks
        ytick_direction = :none,
        linecolor = color,
        fillcolor = color,
        xlabel = "Time (s)",
        size = size
    )
    xticks!(p, ticks_positions, ticks_string)

    if outfile != ""
        savefig(outfile)
    end

    return p
end

"""
    join_tree_video_and_waveform_video(tree_video_path, tree_video_size, wave_form_video_path, wave_form_video_size; output_file = "output.mkv")

Join tree animation video with waveform video using FFmpeg.

Assumption: the waveform video contains the audio.

NB: it need ffmpeg to be installed on the system.
"""
function join_tree_video_and_waveform_video(
            tree_v_path      :: String,
            tree_v_size      :: Tuple{Number,Number},
            wave_form_v_path :: String,
            wave_form_v_size :: Tuple{Number,Number};
            output_file      :: String                = "output.mkv"
        )

    final_video_resolution = (max(tree_v_size[1], wave_form_v_size[1]), 1 + tree_v_size[2] + wave_form_v_size[2])
    final_tree_position = (round(Int64, (final_video_resolution[1] - tree_v_size[1]) / 2), 0)
    final_wave_form_position = (round(Int64, (final_video_resolution[1] - wave_form_v_size[1]) / 2), tree_v_size[2] + 1)

    # for safety resize everything in even number to be compatible with all video encoders (ex: libx264 does not support odd numbers)
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

    total_complex_filter =  "[0:v]pad=$(final_video_resolution[1]):$(final_video_resolution[2]):0:[a]; "
    total_complex_filter *= "[a][1:v]overlay=$(final_tree_position[1]):$(final_tree_position[2]):0:[b]; "
    total_complex_filter *= "[b][2:v]overlay=$(final_wave_form_position[1]):$(final_wave_form_position[2]):0:[c]"

    # assumption: audio is in the wave form file
    run(`ffmpeg -hide_banner -f lavfi -i $color_input -i $tree_v_path -i $wave_form_v_path -y -filter_complex "$total_complex_filter" -shortest -map '[c]' -map 2:a:0 -c:a copy -c:v libx264 -crf 0 -preset veryfast $output_file`)
end

