
using DSP, WAV

"""
    noise_gate!(samples; level = 0.005, release = 30)

Apply noise gate to samples.
"""
function noise_gate!(sample::AbstractVector; level::Float64 = 0.005, release::Int = 30)
	for i in 1:(length(sample) - (release - 1))
		v_sample = @view sample[(max(1, i - floor(Int, release/2))):(min(i + (floor(Int, release/2)), length(sample)))]
		sample[i] = (
			sum(abs, v_sample) / length(v_sample) <= level
		) ? 0.0 : sample[i]
	end
	sample
end

"""
    normalize!(samples; level = 1.0)

Normalize samples to `level`.
"""
function normalize!(sample::AbstractVector; level::Float64 = 1.0)
	max_peak = maximum(abs, sample)
	apply_padd!(val::Float64) = clamp((val/max_peak) * level, -1.0, 1.0)
	sample .= apply_padd!.(sample)
end

"""
    trim_wav!(samples; level = 1.0)

Trim all trailing samples in `samples` below `level` threshold.
"""
function trim_wav!(sample::AbstractVector; level::Float64 = 0.0)

	@assert maximum(abs, sample) >= level "Level $(level) is too high or wav file is empty"

	before = 1
	after = length(sample)
	while before < length(sample) && abs(sample[before]) <= level
		before = before + 1
	end
	while after > 1 && abs(sample[after]) <= level
		after = after - 1
	end
	if after != length(sample)
		splice!(sample, (after+1):length(sample))
	end
	if before != 1
		splice!(sample, 1:(before-1))
	end

	sample
end

"""
    splitwav(samples, samplerate; wintime = 0.05, steptime = 0.025, preprocess = [ noise_gate!, normalize! ], preprocess_kwargs = [ (level = 0.01,), (level = 1.0,) ], postprocess = [ normalize!, trim_wav! ], postprocess_kwargs = [ (level = 1.0,), (level = 0.0,) ])
    splitwav(filepath; kwargs...)

Split wav in multiple instances using a RMS-Energy based algorithm.
"""
function splitwav(
            samples             :: Vector{T},
            samplerate          :: Real;
            wintime             :: Real                 = 0.05,
            steptime            :: Real                 = wintime / 2,
            head_sensibility    :: Int64                = 10,
            preprocess          :: Vector{Function}     = Vector{Function}([ noise_gate!, normalize! ]),
            preprocess_kwargs   :: Vector{NamedTuple}   = Vector{NamedTuple}([ (level = 0.01,), (level = 1.0,) ]),
            postprocess         :: Vector{Function}     = Vector{Function}([ normalize!, trim_wav! ]),
            postprocess_kwargs  :: Vector{NamedTuple}   = Vector{NamedTuple}([ (level = 1.0,), (level = 0.0,) ])
        )::Tuple{Vector{Vector{T}}, Real} where {T<:AbstractFloat}

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

    for (pre_proc, pre_proc_kwargs) in collect(zip(preprocess, preprocess_kwargs))
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
    last_read::T = Inf
    last_non_peak_values::Vector{T} = []
    last_non_peak_rms::T = 0.0
    for (i, f) in enumerate(frames_rms)
        if head_status == :initial
            push!(last_non_peak_values, f)
            if f > rms_f
                last_non_peak_rms = rms(last_non_peak_values)
                # now the head is reading from a peak
                head_status = :on_peak
            end
        elseif head_status == :on_peak
            if f < rms_f && f < last_non_peak_rms
                # peak is over
                head_status = :right_after_peak
            end
            # still on peak
        elseif head_status == :right_after_peak
            if f > rms_f
                # false alarm: the peak is not yet over
                head_status = :on_peak
                last_read = Inf
                continue
            end
            if last_read < f
                # rms is start raising: time to cut
                push!(cut_points, i)
                # reset head status
                head_status = :initial
                last_read = Inf
                last_non_peak_values = []
                last_non_peak_rms = 0.0
                continue
            end
            last_read = f
        end
    end

    # if the head was reading a peak keep last piece too
    if head_status == :on_peak || head_status == :right_after_peak
        push!(cut_points, length(samples))
    end

    results::Vector{Vector{T}} = Vector{Vector{T}}(undef, length(cut_points)-1)
    for i in 2:(length(cut_points))
        prev_cp = cut_points[i-1]
        curr_cp = cut_points[i]
        left = max(1, frame2points(prev_cp, wintime, steptime, samplerate).start)
        right = min(frame2points(curr_cp, wintime, steptime, samplerate).stop, length(samples))
        results[i-1] = deepcopy(samples[left:right])
    end

    for samps in results
        for (post_proc, post_proc_kwargs) in collect(zip(postprocess, postprocess_kwargs))
            post_proc(samps; post_proc_kwargs...)
        end
    end

    results, samplerate
end
function splitwav(samples::Matrix{T}, samplerate::Real; kwargs...)::Tuple{Vector{Vector{T}}, Real} where T<:AbstractFloat
    splitwav(merge_channels(samples), samplerate; kwargs...)
end
function splitwav(filepath::String; kwargs...)::Tuple{Vector{Vector{AbstractFloat}}, Real}
    samples, samplerate = wavread(filepath)
    splitwav(samples, samplerate; kwargs...)
end

"""
    dataset_from_wav_paths(paths, labels; nbands = 40, audio_kwargs = (), modal_args = (), data_modal_args = (), max_points = -1, ma_size = -1, ma_step = -1, dataset_form = :stump_with_memoization, save_dataset = false)
    dataset_from_wav_paths(path, label; kwargs...)

Create a dataset from a list of WAV files.

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

"""
function dataset_from_wav_paths(
            paths                  :: Vector{String},
            labels                 :: Vector{S};
            nbands                 :: Int64 = 40,
            audio_kwargs           :: NamedTuple = NamedTuple(),
            modal_args             :: NamedTuple = NamedTuple(),
            data_modal_args        :: NamedTuple = NamedTuple(),
            preprocess_sample      :: Vector{Function} = Vector{Function}(),
            max_points             :: Int64 = -1,
            ma_size                :: Int64 = -1,
            ma_step                :: Int64 = -1,
            dataset_form           :: Symbol = :stump_with_memoization,
            save_dataset           :: Bool = false
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
    timing_mode = :none
    data_savedir = "/tmp/DecisionTree.jl_cache/"
    mkpath(data_savedir)

    X = compute_X(max_timepoints, n_unique_freqs, tss, length(paths))
    X = X_dataset_c("test", data_modal_args, [X], modal_args, save_dataset, dataset_form, false)

    X, labels
end
dataset_from_wav_paths(filepath::String, label::S where S; kwargs...) = dataset_from_wav_paths([ filepath ], [ label ]; kwargs...)

