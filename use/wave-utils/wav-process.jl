
using DSP, WAV

"""
    noise_gate!(samples; level = 0.005, release = 30)

Apply noise gate to samples.
"""
function noise_gate!(samples::Vector{T}; level::Real = 0.005, release::Integer = 30) where T<:Number
	for i in 1:(length(samples) - (release - 1))
		v_samples = @view samples[(max(1, i - floor(Int, release/2))):(min(i + (floor(Int, release/2)), length(samples)))]
		samples[i] = (
			sum(abs, v_samples) / length(v_samples) <= level
		) ? 0.0 : samples[i]
	end
	samples
end

"""
    normalize!(samples; level = 1.0)

Normalize samples to `level`.
"""
function normalize!(samples::Vector{T}; level::Real = 1.0) where T<:Number
	max_peak = maximum(abs, samples)
	apply_padd!(val::Real) = clamp((val/max_peak) * level, -1.0, 1.0)
	samples .= apply_padd!.(samples)
    samples
end

"""
    trim_wav!(samples; level = 1.0)

Trim all trailing samples in `samples` below `level` threshold.
"""
function trim_wav!(samples::Vector{T}; level::Real = 0.0) where T<:Number

	@assert maximum(abs, samples) >= level "Level $(level) is too high or wav file is empty"

	before = 1
	after = length(samples)
	while before < length(samples) && abs(samples[before]) <= level
		before = before + 1
	end
	while after > 1 && abs(samples[after]) <= level
		after = after - 1
	end
	if after != length(samples)
		splice!(samples, (after+1):length(samples))
	end
	if before != 1
		splice!(samples, 1:(before-1))
	end

	samples
end

"""
    remove_long_silences!(samples; level = 0.005, considered_long = 200)

Remove long silences from WAV.

NB: It is a good idea to use [`trim_wav!`](@ref) first.
"""
function remove_long_silences!(samples::Vector{T}; level::Real = 0.005, considered_long::Integer = 200) where T<:Number
    under_threshold_values::Vector{Int64} = findall(s -> s <= level, samples)

    i = 1
    current_series::Vector{Int64} = []
    intervals::Vector{UnitRange{Int64}} = []
    while i <= length(under_threshold_values)
        if length(current_series) > 1
            if under_threshold_values[i] != current_series[end] + 1
                if length(current_series) >= considered_long
                    push!(intervals, current_series[1]:current_series[end])
                end
                empty!(current_series)
            end

        end

        push!(current_series, under_threshold_values[i])

        i += 1
    end

    for interval in reverse(intervals)
        deleteat!(samples, interval)
    end

    samples
end

"""
    splitwav(samples, samplerate; wintime = 0.05, steptime = 0.025, preprocess = [ noise_gate!, normalize! ], preprocess_kwargs = [ (level = 0.01,), (level = 1.0,) ], postprocess = [ normalize!, trim_wav! ], postprocess_kwargs = [ (level = 1.0,), (level = 0.0,) ])
    splitwav(filepath; kwargs...)

Split wav in multiple instances using a RMS-Energy based algorithm.

A head will read the RMS-Energy representation of the wave
trying to understand where the peaks are located and then
cutting the samples accordingly.

The RMS-Energy representation of the wave is built using
overlapping frames using `wintime` and `steptime` parameters.

The head sensibility can be controlled using a few parameters:

* `use_last_non_peak_rms` if `true` the algorithm will try to
determine if the peak is finished using the RMS of the last
non-peak section of the wave.
* `head_sensibility` controls how many RMS-frames will be
analyzed to determine if the current peak is finished (when the
next peak is starting).
* `head_strictness` will determine how many RMS-sample have to
respect the sensibility of the head.
* `minimum_time` tells the head that a cut can not be made
before `minimum_time` seconds (note: the granularity of this
time depends on the `wintime` parameter).

The wave can be transformed using functions:

* `preprocess` (parameterizable using `preprocess_kwargs`)
is a vector of effects that will be applied to the original
wave before any cutting is performed.
* `postprocess` (parameterizable using `postprocess_kwargs`)
is a vector of effects that will be applied to the resulting
waves, after the cutting.
"""
function splitwav(
            samples               :: Vector{T},
            samplerate            :: Real;
            wintime               :: Real                 = 0.05,
            steptime              :: Real                 = wintime / 2,
            use_last_non_peak_rms :: Bool                 = false,
            head_sensibility      :: Int64                = 10,
            head_strictness       :: Real                 = 0.7,
            minimum_time          :: Real                 = 0.15,
            preprocess            :: Vector{Function}     = Vector{Function}([ noise_gate!, normalize! ]),
            preprocess_kwargs     :: Vector{NamedTuple}   = Vector{NamedTuple}([ (level = 0.005,), (level = 1.0,) ]),
            postprocess           :: Vector{Function}     = Vector{Function}([ noise_gate!, normalize!, trim_wav!, remove_long_silences! ]),
            postprocess_kwargs    :: Vector{NamedTuple}   = Vector{NamedTuple}([ (level = 0.01,), (level = 1.0,), (level = 0.0,), (level = 0.005,) ])
        )::Tuple{Vector{Vector{T}}, Real} where {T<:AbstractFloat}

    if head_sensibility < 2
        @warn "'head_sensibility` less than 2 doesn't make any sense: automatically setting it to 2."
        head_sensibility = 2
    end

    if length(preprocess) > length(preprocess_kwargs)
        append!(preprocess_kwargs, fill(NamedTuple(), length(preprocess) - length(preprocess_kwargs)))
    elseif length(preprocess) < length(preprocess_kwargs)
        splice!(preprocess_kwargs, (length(preprocess_kwargs) - length(preprocess)):length(preprocess_kwargs))
    end

    if length(postprocess) > length(postprocess_kwargs)
        append!(postprocess_kwargs, fill(NamedTuple(), length(postprocess) - length(postprocess_kwargs)))
    elseif length(postprocess) < length(postprocess_kwargs)
        splice!(postprocess_kwargs, (length(postprocess_kwargs) - length(postprocess)):length(postprocess_kwargs))
    end

    Threads.@threads for (pre_proc, pre_proc_kwargs) in collect(zip(preprocess, preprocess_kwargs))
        pre_proc(samples; pre_proc_kwargs...)
    end

    frames::Vector{Vector{T}} = framesample(
                        samples,
                        samplerate;
                        wintime = wintime,
                        steptime = steptime,
                        moving_average_size = 1,
                        moving_average_step = 1
                    )

    frames_rms::Vector{T} = [ rms(f) for f in frames ]
    rms_f = rms(frames_rms)

    cut_points::Vector{Integer} = [ 1 ]
    head_status::Symbol = :initial
    last_read::Vector{T} = Vector{T}()
    last_non_peak_values::Vector{T} = []
    last_non_peak_rms::T = 0.0
    first_peak_found::Int64 = 1
    for (i, f) in enumerate(frames_rms)
        if head_status == :initial
            push!(last_non_peak_values, f)
            if f > rms_f
                last_non_peak_rms = rms(last_non_peak_values[1:max(1,length(last_non_peak_values)-1)])
                # now the head is reading from a peak
                first_peak_found = i
                head_status = :on_peak
            end
        elseif head_status == :on_peak
            if f < rms_f && (!use_last_non_peak_rms || f < last_non_peak_rms)
                # peak is over
                head_status = :right_after_peak
            end
            # still on peak
        elseif head_status == :right_after_peak
            if f > rms_f
                # false alarm: the peak is not yet over
                head_status = :on_peak
                empty!(last_read)
                continue
            end
            # start saving 'f` only if minimum_time is reached or was not set
            if (i-first_peak_found) * steptime > minimum_time
            # if (i-cut_points[end]) * steptime > minimum_time
                push!(last_read, f)
                if length(last_read) > head_sensibility
                    popat!(last_read, 1)
                end
                if length(last_read) >= head_sensibility && isnondecreasing(last_read; approx = head_strictness)
                    # rms was raising for `head_sensibility` samples: time to cut
                    push!(cut_points, i - head_sensibility)
                    # reset head status
                    head_status = :initial
                    empty!(last_read)
                    empty!(last_non_peak_values)
                    last_non_peak_rms = 0.0
                end
            end
        end
    end

    # if the head was reading a peak keep last piece too
    if head_status == :on_peak || head_status == :right_after_peak
        push!(cut_points, length(samples))
    end

    results::Vector{Vector{T}} = Vector{Vector{T}}(undef, length(cut_points)-1)
    Threads.@threads for i in 2:(length(cut_points))
        # assumption: always good idea to cut first frame
        prev_cp = cut_points[i-1] + round(Int64, head_sensibility * head_strictness)
        curr_cp = cut_points[i]
        left = max(1, frame2points(prev_cp, wintime, steptime, samplerate).start)
        right = min(frame2points(curr_cp, wintime, steptime, samplerate).stop, length(samples))
        results[i-1] = deepcopy(samples[left:right])
    end

    Threads.@threads for samps in results
        for (post_proc, post_proc_kwargs) in collect(zip(postprocess, postprocess_kwargs))
            post_proc(samps; post_proc_kwargs...)
        end
    end

    results, samplerate
end
function splitwav(samples::Matrix{T}, samplerate::Real; kwargs...)::Tuple{Vector{Vector{T}}, Real} where T<:AbstractFloat
    splitwav(merge_channels(samples), samplerate; kwargs...)
end
function splitwav(samples_and_samplerate::Tuple{Vector{T}, Real}; kwargs...)::Tuple{Vector{Vector{T}}, Real} where T<:AbstractFloat
    splitwav(samples_and_samplerate...; kwargs...)
end
function splitwav(filepath::String; kwargs...)::Tuple{Vector{Vector{AbstractFloat}}, Real}
    samples, samplerate = wavread(filepath)
    splitwav(samples, samplerate; kwargs...)
end

"""
    dataset_from_wav_paths(paths, labels; nbands = 40, audio_kwargs = (), modal_args = (), data_modal_args = (), max_points = -1, ma_size = -1, ma_step = -1, dataset_form = :stump_with_memoization, save_dataset = false)
    dataset_from_wav_paths(path, label; kwargs...)

Create a dataset from a list of WAV files.

Default values for `audio_kwargs`, `modal_args` and `data_modal_args` are:

    default_audio_kwargs = (
        wintime      = 0.025,
        steptime     = 0.010,
        fbtype       = :mel,
        window_f     = DSP.triang,
        pre_emphasis = 0.97,
        nbands       = nbands,
        sumpower     = false,
        dither       = false,
    )

    default_modal_args = (;
        initConditions = DecisionTree.startWithRelationGlob,
        useRelationGlob = false,
    )

    default_data_modal_args = (;
        ontology = getIntervalOntologyOfDim(Val(1)),
        test_operators = [ TestOpGeq_80, TestOpLeq_80 ]
    )

Possible values for `dataset_form` are:

* `:dimensional` - create the dataset in the form of `OntologicalDataset`
* `:fmd` - create the dataset in the form of `FeatModalDataset`
* `:stump` - create the dataset in the form of `StumpFeatModalDataset`
* `:stump_with_memoization` - create the dataset in the form of `StumpFeatModalDatasetWithMemoization`

Possible values for `timing` are:

* `:none` - no timing
* `:time` - use the macro [`@time`](@ref) while creating the dataset
* `:btime` - use the macro [`@btime`](@ref) while creating the dataset
"""
function dataset_from_wav_paths(
            paths                  :: Vector{String},
            labels                 :: Vector{S};
            nbands                 :: Int64            = 40,
            audio_kwargs           :: NamedTuple       = NamedTuple(),
            modal_args             :: NamedTuple       = NamedTuple(),
            data_modal_args        :: NamedTuple       = NamedTuple(),
            preprocess_sample      :: Vector{Function} = Vector{Function}(),
            max_points             :: Int64            = -1,
            ma_size                :: Int64            = -1,
            ma_step                :: Int64            = -1,
            timing                 :: Symbol           = :none,
            dataset_form           :: Symbol           = :stump_with_memoization,
            save_dataset           :: Bool             = false
        ) where S

    # TODO: a lot of assumptions here! add more options for more fine tuning
    @assert length(paths) == length(labels) "File number and labels number mismatch: $(length(paths)) != $(length(labels))"

    function compute_X(max_timepoints, n_unique_freqs, timeseries, expected_length)
        @assert expected_length == length(timeseries)
        X = zeros((max_timepoints, n_unique_freqs, length(timeseries)))
        for (i,ts) in enumerate(timeseries)
            X[1:size(ts, 1),:,i] = ts
        end
        X
    end

    default_audio_kwargs = (
        wintime      = 0.025,
        steptime     = 0.010,
        fbtype       = :mel,
        window_f     = DSP.triang,
        pre_emphasis = 0.97,
        nbands       = nbands,
        sumpower     = false,
        dither       = false,
    )

    default_modal_args = (;
        initConditions = DecisionTree.startWithRelationGlob,
        useRelationGlob = false,
    )

    default_data_modal_args = (;
        ontology = getIntervalOntologyOfDim(Val(1)),
        test_operators = [ TestOpGeq_80, TestOpLeq_80 ]
    )

    audio_kwargs = merge(default_audio_kwargs, audio_kwargs)
    modal_args = merge(default_modal_args, modal_args)
    data_modal_args = merge(default_data_modal_args, data_modal_args)

    audio_kwargs = merge(default_audio_kwargs, (nbands = nbands,))

    tss = []
    for filepath in paths
        curr_ts = wav2stft_time_series(filepath, audio_kwargs; preprocess_sample = preprocess_sample, use_full_mfcc = false)

        curr_ts = @views curr_ts[2:end,:]

        if ma_size > 0 && ma_step > 0
            curr_ts = moving_average(curr_ts, ma_size, ma_step)
        end

        if max_points > 0 && size(curr_ts,1) > max_points
            curr_ts = curr_ts[1:max_points,:]
        end

        push!(tss, curr_ts)
    end

    max_timepoints = maximum(size(ts, 1) for ts in tss)
    n_unique_freqs = unique(size(ts,  2) for ts in tss)
    @assert length(n_unique_freqs) == 1 "length(n_unique_freqs) != 1: {$n_unique_freqs} != 1"
    n_unique_freqs = n_unique_freqs[1]

    global timing_mode
    global data_savedir
    timing_mode = timing
    data_savedir = "/tmp/DecisionTree.jl_cache/"
    mkpath(data_savedir)

    X = compute_X(max_timepoints, n_unique_freqs, tss, length(paths))
    X = X_dataset_c("test", data_modal_args, [X], modal_args, save_dataset, dataset_form, false)

    X, labels
end
dataset_from_wav_paths(filepath::String, label::S where S; kwargs...) = dataset_from_wav_paths([ filepath ], [ label ]; kwargs...)

