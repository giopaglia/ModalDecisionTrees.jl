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

function my_fft2semitonemx(nfft::Int, nfilts::Int; sr=8000.0, width=1.0, minfreq=20, maxfreq=sr/2, base_freq=nothing)
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
				 minfreq=0., maxfreq=sr/2, sumpower=true, bwidth=1.0,
				 base_freq=:fft, base_freq_min = 200, base_freq_max = 700,
			 ) where {T<:AbstractFloat}
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
			my_fft2semitonemx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq, base_freq=base_freq, base_freq_min = base_freq_min, base_freq_max = base_freq_max,)
		else
			throw_n_log("Unknown filterbank type: $(fbtype)")
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
			  base_freq=:fft, base_freq_min = 200, base_freq_max = 700,
			  # , do_log = false
			  ) where {T<:AbstractFloat}
	if (pre_emphasis != 0)
		x = filt(PolynomialRatio([1., -pre_emphasis], [1.]), x)
	end
	pspec = my_powspec(x, sr, wintime=wintime, steptime=steptime, dither=dither, window_f=window_f)
	
	# println(size(pspec))
	# heatmap(pspec)
	# display(plot!())

	if fbtype == :semitone
		base_freq = calc_F0(x, sr; min_freq = base_freq_min, max_freq = base_freq_max, method = base_freq)
	end
	
	aspec = my_audspec(pspec, sr, nfilts=nbands, fbtype=fbtype, minfreq=minfreq, maxfreq=maxfreq, sumpower=sumpower, bwidth=bwidth,
		base_freq=base_freq, base_freq_min = base_freq_min, base_freq_max = base_freq_max,
	)
	# if do_log log.(aspec) else aspec end
	# log.(aspec)
end


function F0_autocor(samples, sr; center_clip = maximum(samples)*.3)
	# println(center_clip)
	clip_(s) = begin
		if abs(s) < center_clip
			0
		elseif s > center_clip
			s - center_clip
		elseif s < -center_clip
			s + center_clip
		else
			s
		end
	end
	y = StatsBase.autocor(clip_.(samples), );
	x = 0:(length(y)-1);
	(reverse((sr ./ x)), reverse(y))
end

function F0_fft(samples, sr)
	n = length(samples)
	p = DSP.fft(samples)

	nUniquePts = ceil(Int, (n+1)/2)
	p = p[1:nUniquePts]
	p = abs.(p)

	p = p / n #scale
	p = p.^2  # square it
	# odd nfft excludes Nyquist point
	if n % 2 > 0
    p[2:length(p)] = p[2:length(p)]*2 # we've got odd number of   points fft
	else 
    p[2: (length(p) -1)] = p[2: (length(p) -1)]*2 # we've got even number of points fft
	end

	freqArray = (0:(nUniquePts-1)) * (sr / n)
	freqArray, p
end

function calc_F0(args...; method = :fft, kwargs...)
	d = Dict(
		:autocor => F0_autocor,
		:fft     => F0_fft,
		# :esprit  => F0_esprit,
		# :world   => F0_world,
	)
	calc_F0(args...; method_fun = d[method], kwargs...)
end


function calc_F0(samples, sr; min_freq = 200, max_freq = 700, method_fun = F0_autocor, return_all = false)
	freqArray, p = method_fun(samples, sr)
	sp = sortperm(freqArray)
	freqArray = freqArray[sp]
	p = p[sp]
	fake_p = p[:]
	x1 = findfirst((a)->a>=min_freq,freqArray)
	x2 = findfirst((a)->a>=max_freq,freqArray)
	
	println()
	println(x1)
	println(collect(freqArray)[max(x1-1,1):min(x1+1, length(freqArray))])
	println(p[max(x1-1,1):min(x1+1, length(freqArray))])
	println()
	println(x2)
	println(collect(freqArray)[max(x2-1,1):min(x2+1, length(freqArray))])
	println(p[max(x2-1,1):min(x2+1, length(freqArray))])

	fake_p[1:x1-1] .= -Inf
	fake_p[x2:end] .= -Inf
	peak_pos = argmax(moving_average_same(fake_p,1))
	peak_ampl = maximum(fake_p)
	F0 = freqArray[peak_pos]
	(return_all ? (freqArray, p, F0) : F0)
end

# TODO use mono: https://github.com/JuliaMusic/MusicProcessing.jl/blob/master/src/audio.jl
merge_channels(samps) = vec(sum(samps, dims=2)/size(samps, 2))

function wav2stft_time_series(
		filepath,
		kwargs;
		preprocess_sample::AbstractVector = [],
		use_full_mfcc = false,
		ignore_samples_with_sr_less_than = -Inf,
	)
	println(filepath)
	
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

	ts =
		if use_full_mfcc
			mfcc(samps, sr; kwargs...)[1]
		else
			my_stft(samps, sr; kwargs...)'
		end

	println("SIZE: ", size(ts,1))
	ts
end

# wav2stft_time_series("../datasets/KDD/asthmaandroidwithcough/breath/breaths_2aSAZx0fOr_1586937599109.wav")

# DSP.Periodograms.stft(samps, div(length(samps), 8), div(n, 2); onesided=true, nfft=nextfastfft(n), fs=1, window=nothing)
