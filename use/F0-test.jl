using Pkg
Pkg.activate("..")

using WAV
using StatsBase
using Plots: plot, plot!, vline!
using DSP
using PlotlyJS
using WORLD
using SignalOperators
using SignalOperators.Units: Hz, kHz, dB, s # allows the use of dB, Hz, s etc... as unitful values

include("datasets.jl")
include("wav2stft_time_series.jl")

moving_average_same(vs,n) = [StatsBase.mean(@view vs[max((i-n),1):min((i+n),length(vs))]) for i in 1:length(vs)]

function playtone(freq)
	if iszero(freq) println("Can't play zero frequency!"); return end
	Signal(sin,ω=freq * Hz) |> Until(1s) |> Ramp |> Normpower |> Amplify(-20dB) |>  ToFramerate(44.1kHz) |> sink("tmp-playtone().wav")
	wavplay("tmp-playtone().wav")
end


function F0_esprit(samples, sr)
	# ESPRIT

	N = length(samples)
	M = ceil(Int, (N+1)/3)

	p = 10

	sy = DSP.Estimation.esprit(samples, 10, p, fs)

	push!(esprit_res, sy[1])
	# push!(esprit_res, abs(sy[1]))
	println(sy)
	x = 1:length(sy);
	# DSP.Estimation.esprit(samples, M, p, fs)

	println(filepath)
	plot!(x, sy)
	# plot!(x, sy, label=filepath)

	display(plot!())
end

function F0_world(samples, sr)
	# opt = HarvestOption(71.0, 800.0, 20)
	opt = HarvestOption(71.0, 800.0, 50)
	f0, timeaxis = harvest(samples, sr, opt)

	plot!(timeaxis, f0;
		xaxis="Time",
		yaxis="Frequency (Hz)",
	)
	# plot!(x, y, label=filepath)

	display(plot!())
end


files = KDD_getSamplesList(;
	dir = "../datasets/KDD-norm-partitioned-v1",
	rel_path = false,
	only_version = 1,
);

println("$(length(files)) files")

(F0s = []; display(plot()); for (i,filepath) in enumerate(files)
	samples, sr = wavread(filepath);
	samples = merge_channels(samples)
	println(filepath)
	# println(size(samples))
	samples = samples[2:end]
	
	min_freq = 200
	max_freq = 700

	(x, p, F0_a) = calc_F0(samples, sr; min_freq = min_freq, max_freq = max_freq, method_fun = F0_autocor, return_all = true)
	(x, p, F0_f) = calc_F0(samples, sr; min_freq = min_freq, max_freq = max_freq, method_fun = F0_fft,     return_all = true)
	# F0_esprit
	# F0_world
	
	F0 = F0_a
	push!(F0s, F0)
	
	println("autocor: $(F0_a)")
	println("fft: $(F0_f)")

	# plot!(x, 10*log10.(p);
	plot!(x, p;
	# plot!(x, moving_average_same(10*log10.(p),7);
		xaxis="Frequency (Hz)",
		yaxis="Power (dB)",
		axis_zeroline=false,
		axis_showline=true,
		axis_mirror=true,
		xlim=(0, max_freq),
		# xlim=(min_freq, max_freq),
	)
	vline!([F0], color = :black)
	# display(plot())

	# plot!((12 .* log2.(x/F0)), moving_average_same(10*log10.(p),7); yaxis="Power (dB)",
	# plot!((12 .* log2.(x/F0)), moving_average_same(p./peak_ampl, 0);
	# plot!((12 .* log2.(x/F0)), moving_average_same(p./peak_ampl, 4);
	# plot!((12 .* log2.(x/F0)), [p./peak_ampl, ];
	# plot!((12 .* log2.(x/F0)), [, fake_p./peak_ampl];
	# plot!((12 .* log2.(x/F0)), 10*log10.(p./peak_ampl);
	# 	label="",
	# 	xaxis="Frequency (semitone, relative to F0)",
	# 	axis_zeroline=false,
	# 	axis_showline=true,
	# 	axis_mirror=true,
	# 	# xlim=(12 .* log2.(min_freq/F0), 12 .* log2.(max_freq/F0)),
	# 	xlim=(-12*3, 12*3),
	# 	# ylim=(0, 1),
	# 	# ylim=(0, 0),
	# )

	# plot!((sr ./ x) ./ 1000, 10*log10.(p+1),
	# plot!((sr ./ x) ./ 1000, p,
	# 	xlim=(0, 1.8),
	# )
	# plot!(x, y, label=filepath)

	display(plot!())

	# display(plot!())

	wavplay(samples, sr)
	playtone(F0_a)
	playtone(F0_f)
	wavplay(samples, sr)
	playtone(F0_a)
	wavplay(samples, sr)
	playtone(F0_f)

	# wavplay(samples, sr)
	# playtone(F0)
	# wavplay(samples, sr)
	# playtone(F0)
	
	readline()

	# break
end)

playtone(0.15)
playtone(0.5)

display(plot())

max_sample_rate,nbands,fbtype = 8000,30,:semitone
# max_sample_rate,nbands,fbtype = 8000,20,:mel

audio_kwargs_partial_mfcc(max_sample_rate,nbands,fbtype,base_freq) = (
	wintime = 0.025, # in ms          # 0.020-0.040
	steptime = 0.010, # in ms         # 0.010-0.015
	fbtype = fbtype, # :mel                   # [:mel, :htkmel, :fcmel]
	base_freq = base_freq,
	# window_f = DSP.hamming, # [DSP.hamming, (nwin)->DSP.tukey(nwin, 0.25)]
	window_f = DSP.triang,
	pre_emphasis = 0.97,              # any, 0 (no pre_emphasis)
	nbands = nbands,                      # any, (also try 20)
	sumpower = false,                 # [false, true]
	dither = false,                   # [false, true]
	# bwidth = 1.0,                   # 
	minfreq = (fbtype != :semitone ? 0.0 : 0.1),
	maxfreq = max_sample_rate/2,
	# usecmp = false,
)

for (i,filepath) in enumerate(files)
	# i=1
	# filepath = files[i]
	
	base_freq = F0s[i]

	# base_freq = base_freq/1000
	println(filepath)
	println(base_freq)
	ts = wav2stft_time_series(filepath, audio_kwargs_partial_mfcc(max_sample_rate,nbands,fbtype,base_freq); preprocess_sample = [noise_gate!, normalize!], use_full_mfcc = false)
	# readline()
	display(ts)
	println(size(ts))
	heatmap(1:size(ts,1), 1:size(ts,2), ts)

	# break
	display(plot!())
	readline()
end
