
using DSP, WAV, MFCC, Printf
using MFCC: hz2mel, mel2hz

MFCC.mel2hz(f::AbstractFloat, htk=false)  = mel2hz([f], htk)[1]

##############################################################
######################### STRUCTS ############################
##############################################################

"""
A `MelBand` is an asymmetric interval representing
a triangular filter based in `left` and `right`
with its peak in `peak`.
"""
struct MelBand
    left  :: Real
    right :: Real
    peak  :: Real
    MelBand(left::Real, right::Real, peak::Real) = new(max(eps(Float64), left), right, peak)
end

"""
A Scale of `MelBand`s.
"""
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

import Base: getindex, setindex!, length, show

Base.length(scale::MelScale)::Int = length(scale.bands)
Base.getindex(scale::MelScale, idx::Int)::MelBand = scale.bands[idx]
Base.setindex!(scale::MelScale, band::MelBand, idx::Int)::MelBand = scale.bands[idx] = band

"""
    melbands(nbands, minfreq, maxfreq; htkmel = false)

Get `nbands` vector containing all the relevant frequencies
to build a Mel scale in the range `minfreq`:`maxfreq`.

A vector of length `nbands + 2` will return where the first
and last frequencies are the `minfreq` and `maxfreq` and
the values in the middle are the peak of each band.
"""
function melbands(nbands::Int, minfreq::Real = 0.0, maxfreq::Real = 8_000.0; htkmel = false)::Vector{Float64}
    minmel = hz2mel(minfreq, htkmel)
    maxmel = hz2mel(maxfreq, htkmel)
    mel2hz(minmel .+ collect(0:(nbands+1)) / (nbands+1) * (maxmel-minmel), htkmel)
end

"""
    melbands(nbands, minfreq, maxfreq; htkmel = false)

Get a `MelScale` composed by `nbands` `MelBand`s in the range `minfreq`:`maxfreq`.
"""
function get_mel_bands(nbands::Int, minfreq::Real = 0.0, maxfreq::Real = 8_000.0; htkmel = false)::MelScale
    bands = melbands(nbands, minfreq, maxfreq; htkmel = htkmel)
    MelScale(nbands, [ MelBand(bands[i], bands[i+2] >= maxfreq ? bands[i+2] - 0.0000001 : bands[i+2], bands[i+1]) for i in 1:(length(bands)-2) ])
end

"""
A DynamicFilterDescriptor is a `Matrix{Integer}` representing
the distribution of the Worlds on a per-band basis.
"""
const DynamicFilterDescriptor = Matrix{Integer}

"""
    n_chunks(dynamic_filter_descriptor)

Returns the number of `time chunks` in the `DynamicFilterDescriptor`.

This, depending on cases, may correpsond to the frames
of a framed wave.
"""
n_chunks(dynamic_filter_descriptor::DynamicFilterDescriptor) = size(dynamic_filter_descriptor, 1)

"""
    n_bands(dynamic_filter_descriptor)

Returns the number of `bands` in the `DynamicFilterDescriptor`.
"""
n_bands(dynamic_filter_descriptor::DynamicFilterDescriptor) = size(dynamic_filter_descriptor, 2)
Base.show(io::IO, dynamic_filter_descriptor::DynamicFilterDescriptor) = begin
    for i in 1:n_bands(dynamic_filter_descriptor)
        @printf(io, "A%2d ", i)
        for j in 1:n_chunks(dynamic_filter_descriptor)
            print(dynamic_filter_descriptor[j,i] == 1 ? "X" : "-")
        end
        println()
    end
end

"""
A `DynamicFilter` is a Filter that can be used
to filter a WAV depending on its Worlds distribution.
"""
struct DynamicFilter{T} <: DSP.Filters.FilterType
    descriptor :: DynamicFilterDescriptor
    samplerate :: Real
    winsize    :: Int64
    stepsize   :: Int64
    filters    :: Vector{Vector{T}}
end
Base.show(io::IO, dynamic_filter::DynamicFilter) = begin
    println(io,
        "DynamicFilter{$(eltype(eltype(dynamic_filter.filters)))}", "\n",
        "Samplerate: ", dynamic_filter.samplerate, "\n",
        "Window size: ", dynamic_filter.winsize, "\n",
        "Step size: ", dynamic_filter.stepsize
    )
    for i in 1:n_bands(dynamic_filter.descriptor)
        @printf(io, "A%2d ", i)
        for j in 1:n_chunks(dynamic_filter.descriptor)
            print(dynamic_filter.descriptor[j,i] == 1 ? "X" : "-")
        end
        println()
    end
end

"""
    n_chunks(dynamic_filter)

Get the number of `chunks` (AKA `worlds` or `frames`)
the `DynamicFilter` is built with.
"""
n_chunks(dynamic_filter::DynamicFilter)::Int64 = size(dynamic_filter.descriptor, 1)
"""
    n_chunks(dynamic_filter)

Get the number of `bands` the `DynamicFilter` is built with.
"""
n_bands(dynamic_filter::DynamicFilter)::Int64 = size(dynamic_filter.descriptor, 2)
"""
    winsize(dynamic_filter)

Get the size of the window of `dynamic_filter` in "number of samples".
"""
winsize(dynamic_filter::DynamicFilter)::Int64 = dynamic_filter.winsize
"""
    stepsize(dynamic_filter)

Get the step of the window of `dynamic_filter` in "number of samples".
"""
stepsize(dynamic_filter::DynamicFilter)::Int64 = dynamic_filter.stepsize
"""
    filter_n_windows(dynamic_filter)

Get the number of windows of the filters in the `dynamic_filter`.
"""
filter_n_windows(dynamic_filter::DynamicFilter)::Int64 = length(dynamic_filter.filters[1])
"""
    get_filter(dynamic_filter, band_index)
    get_filter(dynamic_filter, band_indices)

Get the `band_index`-th filter in the `dynamic_filter`.
"""
function get_filter(dynamic_filter::DynamicFilter{T}, band_index::Int64)::AbstractVector{T} where T
    dynamic_filter.filters[band_index]
end
function get_filter(dynamic_filter::DynamicFilter{T}, band_index::AbstractVector{Int64})::Vector{T} where T
    # combine filters to generate EQs
    result::Vector{T} = fill(0.0, filter_n_windows(dynamic_filter))
    for i in band_index
        result += get_filter(dynamic_filter, i)
    end
    result
end
"""
    get_chunk(dynamic_filter, chunk_index)
    get_chunk(dynamic_filter, chunk_indices)

Get the index of the samples corresponding to `chunk_index`.
"""
get_chunk(dynamic_filter::DynamicFilter, chunk_index::Int64)::Vector{Int64} = collect(frame2points(chunk_index, dynamic_filter.winsize, dynamic_filter.stepsize))
get_chunk(dynamic_filter::DynamicFilter, chunk_index::Vector{Int64})::Vector{Int64} = length(chunk_index) > 0 ? cat(collect.(frame2points(chunk_index, dynamic_filter.winsize, dynamic_filter.stepsize))...; dims = 1) : Vector{Int64}()
"""
    get_filter_for_chunk(dynamic_filter, chunk_index)

Get the `filter` that need to be applied for a certain `chunk`.

This will sum all the filters that need to be applied to the inputed `chunk`.

NB: this function can return `UndefInitializer` in the case of no filter found for the `chunk`.
"""
function get_filter_for_chunk(dynamic_filter::DynamicFilter{T}, chunk_index::Int64)::Union{Vector{T},UndefInitializer} where T
    filter_idxs = findall(isequal(1), dynamic_filter.descriptor[chunk_index,:])

    if length(filter_idxs) == 0
        return undef
    else
        return get_filter(dynamic_filter, filter_idxs)
    end
end

##############################################################
##############################################################
##############################################################


##############################################################
####################### CONVERSIONS ##########################
##############################################################

"""
    timerange2points(timerange, samplerate)

* `point`s are in the form of `UnitRange{Int64}`
* `timerange`s are in the form of `Tuple{Real,Real}`
* `frame`s are in the form of `Int64`
"""
timerange2points(range::Tuple{T, T} where T <:Number, samplerate::Real)::UnitRange{Int64} = max(1, round(Int64, range[1] * samplerate)):round(Int64, range[2] * samplerate)
"""
    points2timerange(points, samplerate)

* `point`s are in the form of `UnitRange{Int64}`
* `timerange`s are in the form of `Tuple{Real,Real}`
* `frame`s are in the form of `Int64`
"""
points2timerange(range::UnitRange{Int64}, samplerate::Real)::Tuple{T, T} where T <:Real = ((range.start - 1) / samplerate, (range.stop) / samplerate)
"""
    frame2points(index, framesize, stepsize)
    frame2points(index, frametime, steptime, samplerate)

* `point`s are in the form of `UnitRange{Int64}`
* `timerange`s are in the form of `Tuple{Real,Real}`
* `frame`s are in the form of `Int64`

* `framesize` = `frametime` * `samplerate`
* `stepsize` = `steptime` * `samplerate`
"""
frame2points(index::Int64, framesize::Int64, stepsize::Int64)::UnitRange{Int64} = begin
    start = round(Int64, (index - 1) * stepsize) + 1
    start:(start+framesize-1)
end
frame2points(index::Int64, frametime::AbstractFloat, steptime::AbstractFloat, samplerate::AbstractFloat)::UnitRange{Int64} = frame2points(index, round(Int64, frametime * samplerate), round(Int64, steptime * samplerate))
"""
    frame2timerange(index, frametime, steptime, samplerate)

* `point`s are in the form of `UnitRange{Int64}`
* `timerange`s are in the form of `Tuple{Real,Real}`
* `frame`s are in the form of `Int64`

* `framesize` = `frametime` * `samplerate`
* `stepsize` = `steptime` * `samplerate`
"""
frame2timerange(index::Int64, framesize::AbstractFloat, stepsize::AbstractFloat, samplerate::AbstractFloat)::Tuple{T, T} where T <:Number = points2timerange(frame2points(index, framesize, stepsize, samplerate), samplerate)

timerange2points(ranges::Vector{Tuple{T, T}} where T <:Number, samplerate::Real)::Vector{UnitRange{Int64}} = [ timerange2points(r, samplerate) for r in ranges ]
points2timerange(ranges::Vector{UnitRange{Int64}}, samplerate::Real)::Vector{Tuple{T, T}} where T <:Real = [ points2timerange(r, samplerate) for r in ranges ]
frame2points(indices::Vector{Int64}, framesize::Int64, stepsize::Int64)::Vector{UnitRange{Int64}} = [ frame2points(i, framesize, stepsize) for i in indices ]
frame2points(indices::Vector{Int64}, framesize::AbstractFloat, stepsize::AbstractFloat, samplerate::AbstractFloat)::Vector{UnitRange{Int64}} = [ frame2points(i, framesize, stepsize, samplerate) for i in indices ]
frame2timerange(indices::Vector{Int64}, framesize::AbstractFloat, stepsize::AbstractFloat, samplerate::AbstractFloat)::Vector{Tuple{T, T}} where T <:Number = [ frame2timerange(i, framesize, stepsize, samplerate) for i in indices ]

##############################################################
##############################################################
##############################################################


##############################################################
######################## WAVE UTILS ##########################
##############################################################

"""
    FIRfreqz(b; w = range(0, stop=π, length=1024))

Calculate frequency finite impulse response of the
`FIRFilter` represented by the `b` Array.

ref: https://weavejl.mpastell.com/stable/examples/FIR_design.pdf
"""
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
    framesample(samples; winsize = 200, stepsize = 80)
    framesample(samples, samplerate; wintime = 0.025, steptime = 0.01, moving_averate_size = 1, moving_average_step = 1)

Get a Vector of frames representing `samples`.

Note: difference between `{win,step}size` and `{win,step}time`:

* `winsize` = `samplerate` * `wintime` * `moving_average_size`
* `stepsize` = `samplerate` * `steptime` * `moving_average_step`
"""
function framesample(
            samples             :: Vector{T};
            winsize             :: Integer    = 200, # = 0.025 * 8_000 Hz
            stepsize            :: Integer    = 80   # = 0.01 * 8_000 Hz
        )::Vector{Vector{T}} where T<:AbstractFloat

    nwin::Int64 = ceil(Int64, length(samples) / stepsize)

    result = Vector{Vector{T}}(undef, nwin)
    Threads.@threads for (i, range) in collect(enumerate(frame2points(collect(1:nwin), winsize, stepsize)))
        left = max(1, range.start)
        right = min(length(samples), range.stop)
        result[i] = deepcopy(samples[left:right])
    end

    result
end
function framesample(
            samples             :: Vector{T},
            samplerate          :: Real;
            wintime             :: AbstractFloat = 0.025,
            steptime            :: AbstractFloat = 0.01,
            moving_average_size :: Integer       = 1,
            moving_average_step :: Integer       = 1
        )::Vector{Vector{T}} where T<:AbstractFloat

    winsize::Int64 = round(Int64, moving_average_size * (wintime * samplerate))
    stepsize::Int64 = round(Int64, moving_average_step * (steptime * samplerate))

    framesample(samples; winsize = winsize, stepsize = stepsize)
end

"""
    approx_wav(samples, samplerate; mode = maximum, width = 1000.0, sacle_res = 1.0)
    approx_wav(filepath; mode = maximum, width = 1000.0, sacle_res = 1.0)

Get an approximation of the wav using `mode` as "frame descriptor":
`mode` default value is `maximum` meaning that each frame the wav will
be divided into will be approximated by the `maximum` of the frame.

Another useful value for `mode` could be [`rms`](@ref).
"""
function approx_wav(
            samples    :: Vector{T},
            samplerate :: Real;
            mode       :: Function    = maximum,
            width      :: Real        = 1000.0,
            scale_res  :: Real        = 1.0
        ):: Tuple{Vector{T}, Real} where T<:Real

    num_frames = ceil(Int64, (width * scale_res) / 2) + 1

    step_size = floor(Int64, length(samples) / num_frames)
    frame_size = step_size

    frames = []
    for i in 1:num_frames
        interval = frame2points(i, frame_size, step_size)
        interval = UnitRange(max(interval.start, 1), min(interval.stop, length(samples)))
        push!(frames, samples[interval])
    end

    frames_contracted = mode.(frames)

    n = Vector{T}(undef, 2 * length(frames_contracted))
    Threads.@threads for i in collect(1:2:((length(frames_contracted)*2)-1))
        i_r = floor(Int64, i / 2) + 1
        n[i:i+1] = [ frames_contracted[i_r], -frames_contracted[i_r] ]
    end

    n, (samplerate * (length(n) / length(samples)))
end
function approx_wav(samples::Matrix, samplerate::Real; kwargs...)::Tuple{Vector{T}, Real} where T<:Real
    approx_wav(merge_channels(samples), samplerate; kwargs...)
end
function approx_wav(filepath::String; kwargs...)::Tuple{Vector{T}, Real} where T<:Real
    samples, samplerate = wavread(filepath)
    approx_wav(samples, samplerate; kwargs...)
end

"""
    get_points_and_seconds_from_worlds(worlds, winsize, stepsize, n_samps, samplerate)

The `winsize` and `stepsize` parameres should have `moving_average_size`
and `moving_average_step` already considered in them.

In general:

* `winsize` = `wintime` * `samplerate` * `moving_average_size`
* `stepsize` = `steptime` * `samplerate` * `moving_average_step`
"""
function get_points_and_seconds_from_worlds(
            worlds     :: DecisionTree.ModalLogic.AbstractWorldSet,
            winsize    :: Int64,
            stepsize   :: Int64,
            n_samps    :: Int64,
            samplerate :: Real
        )
    # keep largest worlds
    dict = Dict{Int64,DecisionTree.ModalLogic.AbstractWorld}()
    for w in worlds
        if !haskey(dict, w.x) || dict[w.x].y < w.y
            dict[w.x] = w
        end
    end

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

    # convert worlds into samples intervals
    timeranges = []
    for i in 1:highest_key
        if haskey(dict, i)
            w = dict[i]
            frames = frame2points(collect(w.x:(w.y-1)), winsize, stepsize)
            push!(timeranges, (frames[1][1], frames[end][2]))
        end
    end

    points = []
    seconds = []
    wav_descriptor = fill(false, n_samps)
    for tr in timeranges
        push!(points, (tr[1], tr[2]))
        push!(seconds, points2timerange(tr[1]:tr[2], samplerate))
        wav_descriptor[max(1, tr[1]):min(tr[2], n_samps)] .= true
    end

    wav_descriptor, points, seconds
end

##############################################################
##############################################################
##############################################################


##############################################################
################## DIGITAL FILTER CREATION ###################
##############################################################

"""
    multibandpass_digitalfilter(vec, samplerate, window_f; nbands = 40, nwin = nbands, weigths = fill(1, length(vec)))
    multibandpass_digitalfilter(selected_bands, samplerate, window_f; nbands = 40, minfreq = 0.0, maxfreq = samplerate / 2, nwin = nbands, weigths = fill(1, length(vec)))

Create a multi band pass digital filter using bands in
`vec` in the form of a tuple `(w1, w2)`.

`wheigts` is an array of values in the range 0.0 to 1.0
of the same length of `vec` which controls the
attenuation of the band (1.0 = let the band pass,
0.0 = is the same as not including the band in `vec`,
0.5 = let the band pass but with "half of the energy").

see: [`DSP.Filters.Bandpass`](@ref).
"""
function multibandpass_digitalfilter(
            vec        :: Vector{Tuple{T, T}},
            samplerate :: Real,
            window_f   :: Function;
            nbands     :: Integer              = 40,
            nwin       :: Integer              = nbands,
            weights    :: Vector{F}            = fill(1., length(vec))
        )::AbstractVector where {T, F<:AbstractFloat}

    @assert length(weights) == length(vec) "length(weights) != length(vec): $(length(weights)) != $(length(vec))"

    result_filter = zeros(T, nwin)
    i = 1
    @simd for t in vec
        result_filter += digitalfilter(Filters.Bandpass(t..., fs = samplerate), FIRWindow(window_f(nwin))) * weights[i]
        i=i+1
    end
    result_filter
end
function multibandpass_digitalfilter(
            selected_bands :: Vector{Int},
            samplerate     :: Real,
            window_f       :: Function;
            nbands         :: Integer      = 40,
            minfreq        :: Real         = 0.0,
            maxfreq        :: Real         = samplerate / 2,
            nwin           :: Int64        = nbands,
            weights        :: Vector{F}    = fill(1., length(selected_bands))
        )::AbstractVector where F <:AbstractFloat

    @assert length(weights) == length(selected_bands) "length(weights) != length(selected_bands): $(length(weights)) != $(length(selected_bands))"

    band_width = (maxfreq - minfreq) / nbands

    result_filter = zeros(Float64, nwin)
    i = 1
    @simd for b in selected_bands
        l = b * band_width
        r = ((b+1) * band_width) - 1
        result_filter += digitalfilter(Filters.Bandpass(l <= 0 ? eps(typeof(l)) : l, r >= maxfreq ? r - 0.000001 : r, fs = samplerate), FIRWindow(window_f(nwin))) * weights
        i=i+1
    end
    result_filter
end

# TODO: create dispatch of this function on presence of 'window_f` argument
"""
    function digitalfilter_mel(band, samplerate, window_f = triang; nwin = 40, filter_type = Filters.Bandpass)

Create a digital `Bandpass` filter for `band`.
"""
function digitalfilter_mel(band::MelBand, samplerate::Real, window_f::Function = triang; nwin::Int64 = 40, filter_type = Filters.Bandpass)
    digitalfilter(filter_type(band.left, band.right, fs = samplerate), FIRWindow(; transitionwidth = 0.01, attenuation = 160))
end

"""
    function multibandpass_digitalfilter_mel(selected_bands, samplerate, window_f = triang; nbands = 40, minfreq = 0.0, maxfreq = samplerate / 2, nwin = nbands, weights = fill(1, length(selected_bands)))

Create a digital multi-`Bandpass` filter for `selected_bands`.

`wheigts` is an array of values in the range 0.0 to 1.0
of the same length of `selected_bands` which controls the
attenuation of the band (1.0 = let the band pass,
0.0 = is the same as not including the band in `selected_bands`,
0.5 = let the band pass but with "half of the energy").
"""
function multibandpass_digitalfilter_mel(
            selected_bands :: Vector{Int},
            samplerate     :: Real,
            window_f       :: Function;
            nbands         :: Integer      = 40,
            minfreq        :: Real         = 0.0,
            maxfreq        :: Real         = samplerate / 2,
            nwin           :: Int          = nbands,
            weights        :: Vector{F}    = fill(1., length(selected_bands))
        )::AbstractVector where F <:AbstractFloat

    @assert length(weights) == length(selected_bands) "length(weights) != length(selected_bands): $(length(weights)) != $(length(selected_bands))"

    result_filter = undef
    scale = get_mel_bands(nbands, minfreq, maxfreq)
    i = 1
    @simd for b in selected_bands
        if result_filter isa UndefInitializer
            result_filter = digitalfilter_mel(scale[b], samplerate, window_f; nwin = nwin) * weights[i]
        else
            result_filter += digitalfilter_mel(scale[b], samplerate, window_f; nwin = nwin) * weights[i]
        end
        i=i+1
    end

    result_filter
end



# TODO: dynamic_multibandpass_digitalfilter_mel and multibandpass_digitalfilter_mel function should have almost the same kwargs (except for nbands)
"""
    dynamic_multibandpass_digitalfilter_mel(decision_path, samplerate, winsize, stepsize, nbands; nchunks = -1, window_f = triang, minfreq = 0.0, maxfreq = samplerate / 2, nwin = nbands, weights = fill(1, length(nbands)))

Create a `DynamicFilter` using the info in `decision_path`.

The `DynamicFilterDescriptor` is a "more general" representation
of a `DecisionPath` which allow the user to modify more freely
the dynamic filter.

The parameters are the same as any other multibandpass digital filter
except for `nchunks` which is used to control how many frames will
be used inside the `DynamicFilter`.

Note: `nchunks` doesn't necessarily mean that the final wave must
have exactly `nchunks` frames (but at least `nchunks`).
If the wave has more than `nchunks` frames the last will be
considered to be 0 and then they will be muted. 
"""
function dynamic_multibandpass_digitalfilter_mel(
            decision_path      :: DecisionPath,
            samplerate         :: Real,
            winsize            :: Int64,
            stepsize           :: Int64,
            nbands             :: Int64;
            nchunks            :: Int64     = -1,
            window_f           :: Function  = triang,
            minfreq            :: Real      = 0.0,
            maxfreq            :: Real      = samplerate / 2,
            nwin               :: Int64     = nbands,
            weights            :: Vector{F} = fill(1.0, nbands) # TODO: suppor weights in dynamic_multibandpass_digitalfilter_mel
        )::DynamicFilter{<:AbstractFloat} where F <:AbstractFloat

    if nchunks < 0
        nchunks = max([ (w.y-1) for node in decision_path for w in node.worlds ]...)
    end

    descriptor::DynamicFilterDescriptor = fill(0, max(1, nchunks), nbands)

    for node in decision_path
        curr_chunks = Vector{Int64}()
        for world_interval in [ filter(x -> x > 0, collect(w.x:(w.y-1))) for w in node.worlds ]
            append!(curr_chunks, world_interval)
        end
        if length(curr_chunks) > 0
            descriptor[unique(curr_chunks), node.feature.i_attribute] .= 1
        end
    end

    filters = [
            multibandpass_digitalfilter_mel(
                [i],
                samplerate,
                window_f,
                nbands = nbands,
                minfreq = minfreq,
                maxfreq = maxfreq,
                nwin = nwin,
            ) for i in 1:nbands ]

    DynamicFilter(descriptor, samplerate, winsize, stepsize, filters)
end

##############################################################
##############################################################
##############################################################


##############################################################
##################### APPLY FILTER ###########################
##############################################################

"""
    apply_filter(dynamic_filter, samples)
    apply_filter(dynamic_filter, samples, band_index)

Apply filter `dynamic_filter` to `samples`.

If the parameter `band_index` is specified will be
applied only the filter corresponding to the band at
index `band_index`.
This will result in a wave containing frequencies only
in that band and all the samples corresponding to zero-chunks
zeroed-out (no signal).

Note: if a chunk in the `DynamicFilter` is 0 in all of its
bands the resulting chunk after the filter is applied
will be muted.
"""
function apply_filter(dynamic_filter::DynamicFilter{T}, samples::Vector{Ts}, band_index::Int64)::Vector{Ts} where {T, Ts}
    new_track::Vector{Ts} = fill(zero(Ts), length(samples))

    f = get_filter(dynamic_filter, band_index)
    chunks = filter(x -> x <= length(samples), get_chunk(dynamic_filter, findall(isequal(1), dynamic_filter.descriptor[:,band_index])))

    if length(chunks) > 0
        new_track[chunks] = filt(f, samples[chunks])
    end

    new_track
end
function apply_filter(dynamic_filter::DynamicFilter{T}, samples::Vector{Ts})::Vector{Ts} where {T, Ts}
    new_track::Vector{T} = fill(zero(Ts), length(samples))

    for i in 1:n_chunks(dynamic_filter)
        f = get_filter_for_chunk(dynamic_filter, i)
        if !(f isa UndefInitializer)
            chunks = filter(x -> x <= length(samples), get_chunk(dynamic_filter, i))
            if length(chunks) > 0
                new_track[chunks] = filt(f, samples[chunks])
            end
        end
    end

    new_track
end

##############################################################
##############################################################
##############################################################
