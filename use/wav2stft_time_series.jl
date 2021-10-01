# ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

# 1. Augmentation：augmented data for task2 and task3 are also provided in health* and asthma* files;

# function mel(filepath,FRAME_LENGTH,FRAME_INTERVAL)
# 	samps, sr = wavread(filepath)
# 	samps = vec(samps)
# 	frames = powspec(samps, sr; wintime=FRAME_LENGTH, steptime=FRAME_INTERVAL)
# 	energies = log.(sum(frames', dims=2))
# 	fbanks = audspec(frames, sr; nfilts=40, fbtype=:mel)'
# 	fbanks = hcat(fbanks, energies)
# 	# fbank_deltas = deltas(fbanks)
# 	# fbank_deltadeltas = deltas(fbank_deltas)
# 	# attributes = hcat(fbanks, fbank_deltas, fbank_deltadeltas)
# 	attributes = hcat(fbanks)
# end
# mel(filepath,FRAME_LENGTH,FRAME_INTERVAL)

# Typical frame sizes in speech processing range from 20 ms to 40 ms with 50% (+/-10%) overlap between consecutive frames.

using DSP
using WAV
using MFCC
using MFCC: fft2barkmx, fft2melmx


function hz2semitone(f::Vector{T}, base_freq) where {T<:AbstractFloat}
	12 .* log2.(f./base_freq)
end
hz2semitone(f::AbstractFloat, base_freq)  = hz2semitone([f], base_freq)[1]

function semitone2hz(z::Vector{T}, base_freq) where {T<:AbstractFloat}
	# z .= 12 .* log2.(f./base_freq)
	# z ./ 12 .= log2.(f./base_freq)
	# 2 .^ (z ./ 12) .= (f./base_freq)
	base_freq .* (2 .^ (z ./ 12))
end
semitone2hz(f::AbstractFloat, base_freq)  = semitone2hz([f], base_freq)[1]


TODOMMMMMM = 0.1

function my_fft2semitonemx(nfft::Int, nfilts::Int; sr=8000.0, width=1.0, minfreq=TODOMMMMMM, maxfreq=sr/2, base_freq=nothing)
	wts=zeros(nfilts, nfft)
	# Center freqs of each DFT bin
	fftfreqs = collect(0:nfft-1) / nfft * sr;
	# 'Center freqs' of semitone bands - uniformly spaced between limits
	minsemitone = hz2semitone(minfreq, base_freq);
	maxsemitone = hz2semitone(maxfreq, base_freq);
	binfreqs = semitone2hz(minsemitone .+ collect(0:(nfilts+1)) / (nfilts+1) * (maxsemitone-minsemitone), base_freq);

	# println(minsemitone)
	# println(maxsemitone)

	for i in 1:nfilts
		fs = binfreqs[i .+ (0:2)]
		# scale by width
		fs = fs[2] .+ (fs .- fs[2])width
		# lower and upper slopes for all bins
		loslope = (fftfreqs .- fs[1]) / (fs[2] - fs[1])
		hislope = (fs[3] .- fftfreqs) / (fs[3] - fs[2])
		# then intersect them with each other and zero
		wts[i,:] = max.(0, min.(loslope, hislope))
	end

	# Make sure 2nd half of DFT is zero
	wts[:, (nfft>>1)+1:nfft] .= 0.
	return wts
end


function my_powspec(x::Vector{T}, sr::Real=8000.0; wintime=0.025, steptime=0.01, dither=true, window_f::Function) where {T<:AbstractFloat}
	nwin = round(Integer, wintime * sr)
	nstep = round(Integer, steptime * sr)

	nfft = 2 .^ Integer((ceil(log2(nwin))))
	window = window_f(nwin)      # overrule default in specgram which is hamming in Octave
	noverlap = nwin - nstep

	y = spectrogram(x .* (1<<15), nwin, noverlap, nfft=nfft, fs=sr, window=window, onesided=true).power
	## for compability with previous specgram method, remove the last frequency and scale
	y = y[1:end-1, :] ##  * sumabs2(window) * sr / 2
	y .+= dither * nwin / (sum(abs2, window) * sr / 2) ## OK with julia 0.5, 0.6 interpretation as broadcast!

	return y
end

# audspec tested against octave with simple vectors for all fbtypes
function my_audspec(x::Matrix{T}, sr::Real=16000.0; nfilts=ceil(Int, hz2bark(sr/2)), fbtype=:bark,
				 minfreq=0., maxfreq=sr/2, sumpower=true, bwidth=1.0, base_freq=nothing) where {T<:AbstractFloat}
	nfreqs, nframes = size(x)
	nfft = 2(nfreqs-1)
	wts =
		if fbtype == :bark
			fft2barkmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq)
		elseif fbtype == :mel
			fft2melmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq)
		elseif fbtype == :htkmel
			fft2melmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq,
				htkmel=true, constamp=true)
		elseif fbtype == :fcmel
			fft2melmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq,
				htkmel=true, constamp=false)
		elseif fbtype == :semitone
			my_fft2semitonemx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq, base_freq=base_freq)
		else
			error("Unknown filterbank type: $(fbtype)")
		end
	wts = wts[:, 1:nfreqs]
	if sumpower
		return wts * x
	else
		return (wts * sqrt.(x)).^2
	end
end

function my_stft(x::Vector{T}, sr::Real=16000.0; wintime=0.025, steptime=0.01,
			  sumpower=false, pre_emphasis=0.97, dither=false, minfreq=0.0, maxfreq=sr/2,
			  nbands=20, bwidth=1.0, fbtype=:htkmel,
			  usecmp=false, window_f=hamming,
			  base_freq=nothing,
			  # , do_log = false
			  ) where {T<:AbstractFloat}
	if (pre_emphasis != 0)
		x = filt(PolynomialRatio([1., -pre_emphasis], [1.]), x)
	end
	pspec = my_powspec(x, sr, wintime=wintime, steptime=steptime, dither=dither, window_f=window_f)
	
	# println(size(pspec))
	# heatmap(pspec)
	# display(plot!())

	aspec = my_audspec(pspec, sr, nfilts=nbands, fbtype=fbtype, minfreq=minfreq, maxfreq=maxfreq, sumpower=sumpower, bwidth=bwidth, base_freq=base_freq)
	# if do_log log.(aspec) else aspec end
	# log.(aspec)
end

merge_channels(samps) = vec(sum(samps, dims=2)/size(samps, 2))

function wav2stft_time_series(
		filepath,
		kwargs;
		preprocess_sample::AbstractVector = [],
		use_full_mfcc = false,
		ignore_samples_with_sr_less_than = -Inf,
	)
	samps, sr = wavread(filepath)

	if sr < ignore_samples_with_sr_less_than
		println("Ignoring file \"$(filepath)\" due to sampling rate constraint ($(sr) < $(ignore_samples_with_sr_less_than))")
		return nothing
	end

	samps = merge_channels(samps)

	for pps in preprocess_sample
		pps(samps)
	end
	
	if ! (maximum(abs, samps) > 0.0)
		println("ERROR: File $(filepath) has max peak 0!")
		return nothing
	end
	if any(isnan.(samps))
		println("ERROR: File $(filepath) has a NaN value!")
		return nothing
	end

	# wintime = 0.025 # ms
	# steptime = 0.010 # ms
	# fbtype=:mel # [:mel, :htkmel, :fcmel]

	# #window_f = (nwin)->tukey(nwin, 0.25)
	# window_f = hamming

	ts = if use_full_mfcc
		mfcc(samps, sr; kwargs...)[1]
	else
		my_stft(samps, sr; kwargs...)'
	end
	# print(size(ts))
	ts
end

# wav2stft_time_series("../datasets/KDD/asthmaandroidwithcough/breath/breaths_2aSAZx0fOr_1586937599109.wav")

# DSP.Periodograms.stft(samps, div(length(samps), 8), div(n, 2); onesided=true, nfft=nextfastfft(n), fs=1, window=nothing)

function noise_gate!(sample::AbstractVector; level::Float64 = 0.005, release::Int = 30)
	for i in 1:(length(sample) - (release - 1))
		v_sample = @view sample[(max(1, i - floor(Int, release/2))):(min(i + (floor(Int, release/2)), length(sample)))]
		sample[i] = (
			sum(abs, v_sample) / length(v_sample) <= level
		) ? 0.0 : sample[i]
	end
	sample
end

function normalize!(sample::AbstractVector; level::Float64 = 1.0)
	max_peak = maximum(abs, sample)
	apply_padd!(val::Float64) = clamp((val/max_peak) * level, -1.0, 1.0)
	sample .= apply_padd!.(sample)
end

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
