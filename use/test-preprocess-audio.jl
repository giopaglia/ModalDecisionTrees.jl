
include("wav2stft_time_series.jl")

using Plots
gr()

samps, sr = wavread("../datasets/KDD/healthyandroidwithcough/cough/cough_9me0RMtVww_1586943699308.wav_aug_noise1.wav")
samps = merge_channels(samps)

#plot(samps, 1, length(samps))
range = 8000:13000

println("plot(1:length(samps), samps)\nnoise_gate!(samps)\nnormalize!(samps)\nplot!(1:length(samps), samps)"