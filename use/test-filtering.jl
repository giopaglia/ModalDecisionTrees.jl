
include("wav2stft_time_series.jl")

# using DSP: filt
# using Weave
using Plots

gr()

function multibandpass_digitalfilter(vec::Vector{Tuple{T, T}}, fs::Real, window_f::Function; nbands::Integer = 60)::AbstractVector where T
    result_filter = zeros(T, nbands)
    @simd for t in vec
        result_filter += digitalfilter(Filters.Bandpass(t..., fs = fs), FIRWindow(window_f(nbands)))
    end
    result_filter
end

function multibandpass_digitalfilter(selected_bands::Vector{Int}, fs::Real, window_f::Function; nbands::Integer = 60, lower_freq = 0)::AbstractVector
    higher_freq = fs * 0.5
    band_width = (higher_freq - lower_freq) / nbands

    result_filter = zeros(Float64, nbands)
    @simd for b in selected_bands
        result_filter += digitalfilter(Filters.Bandpass(b * band_width, ((b+1) * band_width) - 1, fs = fs), FIRWindow(window_f(nbands)))
    end
    result_filter
end

function draw_audio_anim(
        audio_files    :: Vector{String};
        outfile        :: String = homedir() * "/gif.gif",
        size           :: Tuple{Int64,Int64} = (1000, 400),
        fps            :: Int64 = 60,
        selected_range :: Union{UnitRange{Int64},Tuple{Number,Number},Symbol} = :whole
    )
    function draw_wav(points::Vector{Float64}, fs::Number; func = plot)
        func(
            collect(0:(length(points) - 1)),
            points,
            xlims = (0, length(points)),
            ylims = (-1, 1),
            framestyle = :zerolines,       # show axis at zeroes
            fill = 0,                      # show area under curve
            leg = false,                   # hide legend
            yshowaxis = false,             # hide y axis
            grid = false,                  # hide y grid
            ticks = false,                 # hide y ticks
            tick_direction = :none,
            size = size
        )
    end
    draw_wav!(points::Vector{Float64}, fs::Number) = draw_wav(points, fs, func = plot!)

    @assert length(audio_files) > 0 "No audio file provided"

    wavs = []
    fss = []
    for file_name in audio_files
        wav, fs = wavread(file_name)
        push!(wavs, merge_channels(wav))
        push!(fss, fs)
    end

    @assert length(unique(fss)) == 1 "Inconsistent bitrate across multiple files"
    @assert length(unique([x -> length(x) for wav in wavs])) == 1 "Inconsistent length across multiple files"

    if selected_range isa Tuple
        # convert seconds to points
        println("Selected time range from $(selected_range[1])s to $(selected_range[2])s")
        selected_range = round(Int64, selected_range[1] * fss[1]):round(Int64, selected_range[2] * fss[1])
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

    plt = nothing
    for (i, w) in enumerate(wavs)
        if i == 1
            plt = draw_wav(w, freq)
        else
            draw_wav!(w, freq)
        end
    end

    anim = @animate for p in 1:total_frames
        println("Processing frame $(p) / $(total_frames)")
        vline!(deepcopy(plt), [ (p-1) * step ], line = (:black, 1))
    end

    gif(anim, outfile, fps = fps)
end

# SETTINGS
outpath = homedir()

inputfile_healthy = "../datasets/KDD/healthyandroidwithcough/cough/cough_9me0RMtVww_1586943699308.wav_aug_noise1.wav"
outfile_healthy = "filtered_healthy.wav"
totaloutpath_healthy = outpath * "/" * outfile_healthy

inputfile_covid = "../datasets/KDD/covidwebwithcough/2020-04-17-19_23_25_139694/audio_file_cough.wav"
outfile_covid = "filtered_covid.wav"
totaloutpath_covid = outpath * "/" * outfile_covid

video_codec = "libx265"
gifout = outpath * "/" * "gif.gif"
mkv_original = outpath * "/" * "original.mkv"
mkv_filtered = outpath * "/" * "filtered.mkv"

original_copy_healty = outpath * "/" * "original_healthy.wav"
original_copy_covid = outpath * "/" * "original_covid.wav"

# # READ INPUT
# samps_healthy, sr_healthy, nbits_healthy = wavread(inputfile_healthy)
# samps_healthy = merge_channels(samps_healthy)

# samps_covid, sr_covid, nbits_covid = wavread(inputfile_covid)
# samps_covid = merge_channels(samps_covid)

# # COMPUTE
# filter_healthy = multibandpass_digitalfilter([ 48, 19, 39, 27 ], sr_healthy, hamming, nbands = 60)
# filtered_healthy = filt(filter_healthy, samps_healthy)

# filter_covid = multibandpass_digitalfilter([ 48, 19, 39, 27 ], sr_covid, hamming, nbands = 60)
# filtered_covid = filt(filter_covid, samps_covid)

# # OUTPUT
# cp(inputfile_healthy, original_copy_healty; force = true)
# wavwrite(filtered_healthy, totaloutpath_healthy; Fs = sr_healthy)#, nbits = nbits_healthy)

# cp(inputfile_covid, original_copy_covid; force = true)
# wavwrite(filtered_covid, totaloutpath_covid; Fs = sr_covid)#, nbits = nbits_covid)

draw_audio_anim(
    [ original_copy_covid, totaloutpath_covid ],
    outfile = gifout,
    fps = 1,
#    selected_range = (1.9, 3.15)
)

try
    run(`ffmpeg -i $gifout -i $original_copy_covid -y -c:a copy -c:v $video_codec $mkv_original`)
    run(`ffmpeg -i $gifout -i $totaloutpath_covid -y -c:a copy -c:v $video_codec $mkv_filtered`)
catch
    error("unable to generate video automatially: is ffmpeg installed?")
end

# plotfilter(filter, samplerate = sr, firwindow = FIRWindow(hamming(60)))

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
# 	nstep = round(Integer, steptime * sr)

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

#################### FOLLOWING SECTION IS USED TO DRAW FILTERS #########################

# function FIRfreqz(b::Array; w = range(0, stop=π, length=1024))::Array{ComplexF32}
#     n = length(w)
#     h = Array{ComplexF32}(undef, n)
#     sw = 0
#     for i = 1:n
#         for j = 1:length(b)
#         sw += b[j]*exp(-im*w[i])^-j
#         end
#         h[i] = sw
#         sw = 0
#     end
#     h
# end

# function plotfilter(filter::Filters.Filter; samplerate = 100, firwindow = FIRWindow(hamming(100)))
#     f = digitalfilter(filter, firwindow)
#     w = range(0, stop=π, length=1024)
#     h = FIRfreqz(f; w = w)
#     ws = w / π * (samplerate / 2)
#     plot(ws, amp2db.(abs.(h)), xlabel="Frequency(Hz)", ylabel="Magnitude(db)")
# end

# function plotfilter(filter; samplerate = 100)
#     w = range(0, stop=π, length=1024)
#     h = FIRfreqz(filter; w = w)
#     ws = w / π * (samplerate / 2)
#     plot(ws, amp2db.(abs.(h)), xlabel="Frequency(Hz)", ylabel="Magnitude(db)")
# end

# function plotfilter!(filter::Filters.Filter; samplerate = 100, firwindow = FIRWindow(hamming(100)))
#     f = digitalfilter(filter, firwindow)
#     w = range(0, stop=π, length=1024)
#     h = FIRfreqz(f; w = w)
#     ws = w / π * (samplerate / 2)
#     plot!(ws, amp2db.(abs.(h)), xlabel="Frequency(Hz)", ylabel="Magnitude(db)")
# end

# function plotfilter!(filter; samplerate = 100)
#     w = range(0, stop=π, length=1024)
#     h = FIRfreqz(filter; w = w)
#     ws = w / π * (samplerate / 2)
#     plot!(ws, amp2db.(abs.(h)), xlabel="Frequency(Hz)", ylabel="Magnitude(db)")
# end

# myf1 = digitalfilter(Filters.Bandpass(100, 199; fs = 8000), FIRWindow(hamming(100)))
# myf2 = digitalfilter(Filters.Bandpass(2000, 2599; fs = 8000), FIRWindow(hamming(100)))

# println("myf1", myf1[1:10])
# println("myf2", myf2[1:10])
# myf = myf1 + myf2 #maximum.(collect(zip(myf1, myf2)))
# println("myf", myf[1:10])
# println(eltype(myf))

# function drawnow()
#     plotfilter(myf; samplerate = 8000)
#     plotfilter!(myf1; samplerate = 8000)
#     plotfilter!(myf2; samplerate = 8000)
# end

# NOTES:
# It may be a better solution applying a bandstop filter for each "unwanted" frequency rather than
# traying to combine multiple bandpass filters
# They do not need to be combined but just to be applyed in sequence to each portion of the WAV file