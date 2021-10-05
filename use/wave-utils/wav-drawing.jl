
using DSP, WAV, Plots, Images, Plots.Measures

default_plot_size = (1920, 1080)
default_plot_margins = (
	left_margin = 4mm,
	right_margin = 4mm,
	top_margin = 4mm,
	bottom_margin = 4mm
)
_make_it_even(v::Integer) = v % 2 != 0 ? v+1 : v

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
	)::Plots.Plot

	w = range(0, stop=π, length=1024)
	h = FIRfreqz(filter; w = w)
	ws = w / π * (samplerate / 2)
	if xlims[2] == 0
		xlims = (xlims[1], maximum(ws))
	end
	plotfunc(ws, amp2db.(abs.(h)), xlabel="Frequency (Hz)", ylabel="Magnitude (db)", xlims = xlims, ylims = ylims, leg = false, size = default_plot_size, default_plot_margins...)
end
plotfilter!(filter; args...)::Plots.Plot = plotfilter(filter; args..., plotfunc = plot!)

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
	draw_wav(filepath; kwargs...)
	draw_wav!(samples[, samplerate]; color = :auto, title = "", size = (1000, 150))
	draw_wav!(filepath; kwargs...)

Plot a wave using its `samples`.

NB: `samplerate` is not used inside this function but inserted to
maintain interface consistency.
"""
function draw_wav(
			samples        :: Vector{T},
			samplerate     :: Number               = 44100.0;
			color                                  = :auto,
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
		size = _make_it_even.(size)
	)
end
function draw_wav(samples::Matrix{T}, samplerate::Real = 44100.0; kwargs...)::Plots.Plot where T <: AbstractFloat
	draw_wav(merge_channels(samples), samplerate; kwargs...)
end
function draw_wav(filepath::String; kwargs...)::Plots.Plot
	samps, sr = wavread(filepath)
	draw_wav(samps, sr; kwargs...)
end
function draw_wav(samples_n_samplerate::Tuple{Vector{T},Real}; kwargs...)::Plots.Plot where T<:AbstractFloat
	draw_wav(samples_n_samplerate...; kwargs...)
end

"""
	draw_wav!(samples, samplerate; kwargs...)
	draw_wav!(samples_n_samplerate; kwargs...)
	draw_wav!(filepath; kwargs...)

Same as [`draw_wav`](@ref) but this will draw on
the existing canvas.
"""
function draw_wav!(samples::Vector{T}, samplerate::Real = 44100.0; kwargs...)::Plots.Plot where T <: AbstractFloat
	draw_wav(samples, samplerate; plotfunc = plot!, kwargs...)
end
function draw_wav!(samples::Matrix{T}, samplerate::Real = 44100.0; kwargs...)::Plots.Plot where T <: AbstractFloat
	draw_wav(samples, samplerate; plotfunc = plot!, kwargs...)
end
function draw_wav!(filepath::String; kwargs...)::Plots.Plot
	samps, sr = wavread(filepath)
	draw_wav(samps, sr; plotfunc = plot!, kwargs...)
end
function draw_wav!(samples_n_samplerate::Tuple{Vector{T},Real}; kwargs...)::Plots.Plot where T<:AbstractFloat
	draw_wav(samples_n_samplerate...; plotfunc = plot!, kwargs...)
end

"""
	draw_wave_anim(samples_vec, samplerates; labels = [], colors = [], single_graph = false, outfile = "", size = (1000, 150), fps = 30, resample_at_rate = 0.0, reset_canvas_every_frames = 50, selected_range = :whole, use_wav_apporximation = true, wav_approximation_scale = 1.0)
	draw_wave_anim(filepaths; kargs...)
	draw_wave_anim(filepath; kargs...)
	draw_wave_anim(samples_and_samplerates_vec; kargs...)
	draw_wave_anim(samples_and_samplerates; kargs...)

Draw an animation of the waves represented by  form with a moving
vertical line as it was played by an audio player.

The resulting animation can containe a stack of wave forms or
a single graph containing all waveforms drawn one above each other,
depending on the value of `single_graph`.

If `single_graph` if `false` a `label` can be set for each
wave graph.

Before drawing a resampling of the waves can be done if there are
waves that mismatch in samplerate by setting the `resample_at_rate`
at the desired samplingrate.

`use_wav_apporximation` will create an approximation before drawing.
The default value of this parameter is set to `true` and in most cases
it is a good idea to leav it this way for performance reasons.
When `true` the wave is approximated in `2n` points where `n` is the
width setted in `size`.
If an accurate representation of the wave is needed this feature
should be turned off.

For a more fine tuning of the approximation of the wave it is available
the `wav_approximation_scale` parameter which will cause the wave to
be rendered at a scaled resoultion.

To avoid drawing a lot of transparent lines the canvas can be reset to
the original canvas (containing just the draw of the waveforms). This
will happen every `reset_canvas_every_frames` frames.
"""
function draw_wave_anim(
			wavs                      :: Vector{Vector{T1}},
			samplerates               :: Vector{T2};
			labels                    :: Vector{String}                             = fill("", length(wavs)),
			colors                    :: Union{Vector{Symbol},Vector{RGB{Float64}}} = fill(:auto, length(wavs)),
			single_graph              :: Bool                                       = false,
			outfile                   :: String                                     = "",
			size                      :: Tuple{Int64,Int64}                         = single_graph ? (1000, 150) : (1000, 150 * length(wavs)),
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
		)::Plots.Animation where {T1<:AbstractFloat, T2<:AbstractFloat}

	@assert length(wavs) > 0 "No audio file provided"

	@assert length(wavs) == length(samplerates) "wavs and samplerates mismatch in length: $(length(wavs)) != $(length(samplerates))"
	if !single_graph
		@assert length(wavs) == length(labels) "wavs and labels mismatch in length: $(length(wavs)) != $(length(labels))"
	end
	@assert length(wavs) == length(colors) "wavs and colors mismatch in length: $(length(wavs)) != $(length(colors))"

	if resample_at_rate > 0
		Threads.@threads for i in 1:length(wavs)
			if samplerates[i] == resample_at_rate continue end
			wavs[i] = resample(wavs[i], resample_at_rate / samplerates[i])
		end
		samplerates .= resample_at_rate
	end

	@assert length(unique(samplerates)) == 1 "Inconsistent bitrate across multiple files (try using resample_at_rate keyword argument)"
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

	real_wavs = Vector{Vector{T1}}(undef, length(wavs))
	real_samplerates = Vector{Real}(undef, length(samplerates))
	if use_wav_apporximation
		Threads.@threads for i in 1:length(wavs)
			real_wavs[i], real_samplerates[i] = approx_wav(wavs[i], samplerates[i]; scale_res = wav_approximation_scale, width = size[1])
		end
	else
		real_wavs = @view wavs[:]
		real_samplerates = @view samplerates[:]
	end

	@assert length(real_wavs) == length(wavs) "Transformed wavs length != original wavs length: $(length(real_wavs)) != $(length(wavs))"
	@assert length(real_samplerates) == length(samplerates) "Transformed samplerates length != original samplerates length: $(length(real_samplerates)) != $(length(samplerates))"

	if real_samplerates[1] < fps
		fps = floor(Int64, real_samplerates[1])
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

	if length(outfile) > 0
		gif(anim, outfile, fps = fps)
	end

	anim
end
function draw_wave_anim(audio_files::Vector{String}; kwargs...)::Plots.Animation
	@assert length(audio_files) > 0 "No audio file provided"

	converted_samps::Vector{Vector{AbstractFloat}} = Vector{Vector{AbstractFloat}}(undef, length(audio_files))
	samplerates::Vector{Real} = Vector{Real}(undef, length(audio_files))
	Threads.@threads for (i, f) in collect(enumerate(audio_files))
		converted_samps[i], samplerates[i] = wavread(f)
	end

	draw_wave_anim(converted_samps, samplerates; kwargs...)
end
draw_wave_anim(audio_file::String; kwargs...)::Plots.Animation = draw_wave_anim([audio_file]; kwargs...)
function draw_wave_anim(samples::Vector{Matrix{T}}, samplerates::Vector{Real}; kwargs...)::Plots.Animation where T<:AbstractFloat
	converted::Vector{Vector{AbstractFloat}} = Vector{Vector{AbstractFloat}}(undef, length(samples))
	Threads.@threads for (i, samps) in collect(enumerate(samples))
		converted[i] = merge_channels(samps)
	end

	draw_wave_anim(converted, samplerates; kwargs...)
end
function draw_wave_anim(samples_n_samplerate::Vector{Tuple{Vector{T1},T2}}; kwargs...)::Plots.Animation where {T1<:AbstractFloat, T2<:Real}
	draw_wave_anim([ samples_n_samplerate[i][1] for i in 1:length(samples_n_samplerate) ], [ samples_n_samplerate[i][2] for i in 1:length(samples_n_samplerate) ]; kwargs...)
end
function draw_wave_anim(samples_n_samplerate::Tuple{Vector{T},Real}; kwargs...)::Plots.Animation where T<:AbstractFloat
	draw_wave_anim(samples_n_samplerate...; kwargs...)
end

"""
	draw_spectrogram(samples, samplerate; gran = 50, title = "", clims = (-100, 0), spectrogram_plot_options = (),
	melbands = (draw = false, nbands = 40, minfreq = 0.0, maxfreq = samplerate / 2, htkmel = false), only_bands = :all)

	draw_spectrogram(filepath; gran = 50, title = "", clims = (-100, 0), spectrogram_plot_options = (),
	melbands = (draw = false, nbands = 40, minfreq = 0.0, maxfreq = samplerate / 2, htkmel = false), only_bands = :all)

Draw spectrogram of a wave.

`gran` controls how large is the window used to apply
the STFT determining this way the time axis resolution
of the spectrogram.

Mel bands can be drawn on the resulting image setting
`melbands.draw` to `true`.

`melbands` is a NamedTuple whose keys and default values are:

	melbands = (
		draw = false,
		nbands = 40,
		minfreq = 0.0,
		maxfreq = samplerate / 2,
		htkmel = false
	)

`only_bands` can be used to show only some of the melbands; it can
be a vector of Integers (the index of the bands you want to draw)
or a Symbol `:all` to show them all.
"""
function draw_spectrogram(
			samples                  :: Vector{T},
			samplerate               :: Real;
			gran                     :: Int                         = 50,
			title                    :: String                      = "",
			outfile                  :: String                      = "",
			clims                    :: Tuple{Number,Number}        = (-100, 0),
			spectrogram_plot_options :: NamedTuple                  = NamedTuple(),
			melbands                 :: NamedTuple                  = (draw = false, nbands = 40, minfreq = 0.0, maxfreq = samplerate / 2, htkmel = false),
			only_bands               :: Union{Symbol,Vector{Int64}} = :all
		)::Plots.Plot where T <: AbstractFloat

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

	if haskey(total_heatmap_kwargs, :size)
		total_heatmap_kwargs = merge(total_heatmap_kwargs, (size = _make_it_even.(total_heatmap_kwargs.size),))
	end

	spec = spectrogram(samples, nw_orig, round(Int64, nw_orig / 2); fs = samplerate)
	hm = heatmap(spec.time, spec.freq, pow2db.(spec.power); title = title, clims = clims, total_heatmap_kwargs...)

	# Draw horizontal line on spectrograms corresponding to selected bands
	if melbands[:draw]
	bands = get_mel_bands(melbands[:nbands], melbands[:minfreq], melbands[:maxfreq]; htkmel = melbands[:htkmel])
	for i in 1:melbands[:nbands]
		if i in only_bands
			hline!(hm, [ bands[i].left, bands[i].right ], line = (1, :white), leg = false)
		end
	end
	yticks!(hm, [ melbands[:minfreq], [ bands[i].peak for i in only_bands ]..., melbands[:maxfreq] ], [ string(round(Int64, melbands[:minfreq])), [ string("A", i) for i in only_bands ]..., string(round(Int64, melbands[:maxfreq])) ])
	end

	if length(outfile) > 0
		savefig(hm, outfile)
	end

	hm
end
function draw_spectrogram(samples::Matrix{T}, samplerate::Real; kwargs...)::Plots.Plot where T<:AbstractFloat
	samps = merge_channels(samples)
	draw_spectrogram(samps, samplerate; kwargs...)
end
function draw_spectrogram(filepath::String; kwargs...)::Plots.Plot
	samp, sr = wavread(filepath)
	draw_spectrogram(samp, sr; kwargs...)
end
function draw_spectrogram(samples_n_samplerate::Tuple{Vector{T},Real}; kwargs...)::Plots.Plot where T<:AbstractFloat
	draw_spectrogram(samples_n_samplerate...; kwargs...)
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

	if !isnothing(outpath) || (isa(outpath, String) && length(outpath) == 0)
		try
			mkpath(outpath)
		catch
			throw_n_log("Unable to create directory $(outpat)")
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
		throw_n_log("unable to generate video: is ffmpeg installed?")
	end
end

"""
	draw_tree_anim(wav_descriptor, blank_image, highlighted_image, samplerate; outfile = "", size = `calculated from images`, fps = 30)

Draw a simple animation which will highlight a tree
branch in time ranges accordingly to `wav_descriptor`.
"""
function draw_tree_anim(
			wav_descriptor    :: Vector{Bool},
			blank_image       :: AbstractMatrix,
			highlighted_image :: AbstractMatrix,
			samplerate        :: Real;
			outfile           :: String               = "",
			size              :: Tuple{Number,Number} = Tuple((max(size(blank_image,2), size(highlighted_image,2)), max(size(blank_image,1), size(highlighted_image,1)))),
			fps               :: Int64                = 30,
		)::Plots.Animation

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
			size           = _make_it_even.(size)
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

	if length(outfile) > 0
		gif(anim, outfile, fps = fps)
	end

	anim
end
draw_tree_anim(wav_descriptor::Vector{Bool}, blank_image::String, highlighted_image::String, samplerate::Real; kwargs...)::Plots.Animation = draw_tree_anim(wav_descriptor, load(blank_image), load(highlighted_image), samplerate; kwargs...)

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
		size = _make_it_even.(size)
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

	final_video_resolution = _make_it_even.((max(tree_v_size[1], wave_form_v_size[1]), 1 + tree_v_size[2] + wave_form_v_size[2]))
	final_tree_position = _make_it_even.((round(Int64, (final_video_resolution[1] - tree_v_size[1]) / 2), 0))
	final_wave_form_position = _make_it_even.((round(Int64, (final_video_resolution[1] - wave_form_v_size[1]) / 2), tree_v_size[2] + 1))

	color_input = "color=white:$(final_video_resolution[1])x$(final_video_resolution[2]),format=rgb24"

	total_complex_filter =  "[0:v]pad=$(final_video_resolution[1]):$(final_video_resolution[2]):0:[a]; "
	total_complex_filter *= "[a][1:v]overlay=$(final_tree_position[1]):$(final_tree_position[2]):0:[b]; "
	total_complex_filter *= "[b][2:v]overlay=$(final_wave_form_position[1]):$(final_wave_form_position[2]):0:[c]"

	# assumption: audio is in the wave-form file
	run(`ffmpeg -hide_banner -f lavfi -i $color_input -i $tree_v_path -i $wave_form_v_path -y -filter_complex "$total_complex_filter" -shortest -map '[c]' -map 2:a:0 -c:a copy -c:v libx264 -crf 0 -preset veryfast $output_file`)
end

