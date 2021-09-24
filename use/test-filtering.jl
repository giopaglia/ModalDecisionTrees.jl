
import Dates
using FileIO

include("wav-filtering.jl")

gr()
# gaston() # this uses GNUPLOT but led to some problem generating the gif

# SETTINGS
outpath = "filtering-results"
cache_dir = outpath * "/cache"
if !isdir(outpath) mkpath(outpath) end
if !isdir(cache_dir) mkpath(cache_dir) end

selected_range = :whole

inputfile_healthy = "../datasets/KDD/healthyandroidwithcough/cough/cough_9me0RMtVww_1586943699308.wav_aug_noise1.wav"
outfile_healthy = "filtered_healthy.wav"
totaloutpath_healthy = outpath * "/" * outfile_healthy

inputfile_covid = "../datasets/KDD/covidwebwithcough/2020-04-17-19_23_25_139694/audio_file_cough.wav"
outfile_covid = "filtered_covid.wav"
totaloutpath_covid = outpath * "/" * outfile_covid
outfile_covid_single_band(feat::Int64) = string("A", feat, "_filtered_covid.wav")
totaloutpath_covid_single_band(feat::Int64) = string(outpath, "/", outfile_covid_single_band(feat))

gifout = outpath * "/" * "gif.gif"
heatmap_png_path = outpath * "/" * "spectrogram.png"

original_copy_healty = outpath * "/" * "original_healthy.wav"
original_copy_covid = outpath * "/" * "original_covid.wav"

selected_features = [ 16, 17, 24, 25 ]
nbands = 40

animation_fps = 30
features_colors = [ RGB(1 - (1 * (i/(nbands - 1))), 0, 1 * (i/(nbands - 1))) for i in 0:(nbands-1) ]

# DATASET SETTINGS
max_sample_rate = 8_000

n_task = 2
n_version = 1
use_aug = true
use_full_mfcc = false
audio_kwargs = (
    wintime = 0.025,
    steptime = 0.010,
    fbtype = :mel,                     
    window_f = DSP.hamming,
    pre_emphasis = 0.97,
    nbands = nbands,
    sumpower = false,
    dither = false,
    maxfreq = max_sample_rate/2,
)
dataset_kwargs = (
    ma_size = 120,
    ma_step = 100
)
preprocess_wavs = []

dataset_func_params = (
    (n_task,n_version),
    audio_kwargs
)
dataset_func_kwparams = (
    dataset_kwargs...,
    return_filepaths = true,
    use_augmentation_data = use_aug,
    preprocess_wavs = preprocess_wavs,
    use_full_mfcc = use_full_mfcc,
    force_monolithic_dataset = :train_n_test
)

modal_args = (;
	initConditions = DecisionTree.startWithRelationGlob,
	useRelationGlob = false,
)

data_modal_args = (;
	ontology = getIntervalOntologyOfDim(Val(1)),
    test_operators = [TestOpGeq_80, TestOpLeq_80]
)

save_datasets = true
dataset_form = :stump_with_memoization
data_savedir = cache_dir
timing_mode = :none

# TREE SETTINGS
tree_path = "covid-august/trees"
tree_hash = "d5cce8625a82b7c1e5360a4d055175425302db80dc57e55695e32d4d782c6ac5"

# TEST APPLY_TREE_TO_WAV
tree = JLD2.load(tree_path * "/tree_$(tree_hash).jld")["T"]

(X, Y, filepaths), (n_pos, n_neg) = @cache "dataset" cache_dir dataset_func_params dataset_func_kwparams KDDDataset_not_stratified
X_modal = X_dataset_c("test", data_modal_args, X, modal_args, save_datasets, dataset_form, false)

apply_tree_to_datasets_wavs(
        tree_hash,
        tree,
        X_modal,
        filepaths[1],
        Y;
        filter_kwargs = (nbands = nbands, maxfreq = max_sample_rate / 2),
        remove_from_path = "../datasets/KDD/",
        draw_anim_for_instances = [ findfirst(isequal("../datasets/KDD/healthyandroidwithcough/cough/cough_9me0RMtVww_1586943699308.wav"), filepaths[1]) ]
    )

# # DRAW MEL-FILTERS
# d40_8000 = draw_synthetic_mel_filters_graph(; nbands = 40, minfreq = 0.0, maxfreq = 8_000.0 / 2)
# d60_8000 = draw_synthetic_mel_filters_graph(; nbands = 60, minfreq = 0.0, maxfreq = 8_000.0 / 2)
# d40_16000 = draw_synthetic_mel_filters_graph(; nbands = 40, minfreq = 0.0, maxfreq = 16_000.0 / 2)
# d60_16000 = draw_synthetic_mel_filters_graph(; nbands = 60, minfreq = 0.0, maxfreq = 16_000.0 / 2)
# savefig(d40_8000, outpath * "/" * "synthetic_filters_40_8000.png")
# savefig(d60_8000, outpath * "/" * "synthetic_filters_60_8000.png")
# savefig(d40_16000, outpath * "/" * "synthetic_filters_40_16000.png")
# savefig(d60_16000, outpath * "/" * "synthetic_filters_60_16000.png")
# real_d40_8000 = draw_mel_filters_graph(8_000.0, triang; nbands = 40)
# real_d60_8000 = draw_mel_filters_graph(8_000.0, triang; nbands = 60)
# real_d40_16000 = draw_mel_filters_graph(16_000.0, triang; nbands = 40)
# real_d60_16000 = draw_mel_filters_graph(16_000.0, triang; nbands = 60)
# savefig(real_d40_8000, outpath * "/" * "real_filters_40_8000.png")
# savefig(real_d60_8000, outpath * "/" * "real_filters_60_8000.png")
# savefig(real_d40_16000, outpath * "/" * "real_filters_40_16000.png")
# savefig(real_d60_16000, outpath * "/" * "real_filters_60_16000.png")

# # READ INPUT
# samps_healthy, sr_healthy, nbits_healthy = wavread(inputfile_healthy)
# samps_healthy = merge_channels(samps_healthy)
# noise_gate!(samps_healthy)
# trim_wav!(samps_healthy)

# samps_covid, sr_covid, nbits_covid = wavread(inputfile_covid)
# samps_covid = merge_channels(samps_covid)
# noise_gate!(samps_covid)
# trim_wav!(samps_covid)

# # COMPUTE
# filter_healthy = multibandpass_digitalfilter_mel(selected_features, sr_healthy, hamming; nbands = nbands)
# filtered_healthy = filt(filter_healthy, samps_healthy)

# filter_covid = multibandpass_digitalfilter_mel(selected_features, sr_covid, hamming; nbands = nbands)
# filtered_covid = filt(filter_covid, samps_covid)

# single_band_filters = [ multibandpass_digitalfilter_mel([ feat ], sr_covid, hamming; nbands = nbands) for feat in selected_features ]
# single_band_wavs = [ (feat, totaloutpath_covid_single_band(feat), filt(single_band_filters[i], samps_covid)) for (i, feat) in enumerate(selected_features) ]

# # OUTPUT
# print("Copying original healthy WAV to $(original_copy_healty)...")
# wavwrite(samps_healthy, original_copy_healty; Fs = sr_healthy)
# println(" done")
# print("Generating filtered healthy WAV to $(totaloutpath_healthy)...")
# wavwrite(filtered_healthy, totaloutpath_healthy; Fs = sr_healthy)
# println(" done")

# print("Copying original covid WAV to $(original_copy_covid)...")
# wavwrite(samps_covid, original_copy_covid; Fs = sr_covid)
# println(" done")
# print("Generating filtered covid WAV to $(totaloutpath_covid)...")
# wavwrite(filtered_covid, totaloutpath_covid; Fs = sr_covid)
# println(" done")
# # No need to write in files the single_band_wavs samples
# # for (feat, path, wav) in single_band_wavs
# #     print("Generating filtered covid WAV to $(path)...")
# #     wavwrite(wav, path; Fs = sr_covid)
# #     println(" done")
# # end

# # GENERATE SPECTROGRAM
# hm_orig = draw_spectrogram(samps_covid, sr_covid; title = "Original", melbands = (draw = true, nbands = 40, minfreq = 0.0, maxfreq = sr_covid / 2, htkmel = false))
# hm_filt = draw_spectrogram(filtered_covid, sr_covid; title = "Filtered", melbands = (draw = true, nbands = 40, minfreq = 0.0, maxfreq = sr_covid / 2, htkmel = false))
# plot(hm_orig, hm_filt, layout = (1, 2))
# savefig(heatmap_png_path)

# # GENERATE GIF
# draw_audio_anim(
#     [ (samps_covid, sr_covid), (filtered_covid, sr_covid), [ (single_band_wavs[i][3], sr_covid) for i in 1:length(single_band_wavs) ]... ],
#     labels = [ "Original", "Filtered", [ string("A", single_band_wavs[i][1]) for i in 1:length(single_band_wavs) ]... ],
#     colors = [ RGB(.3, .3, 1), RGB(1, .3, .3), features_colors[selected_features]...],
#     outfile = gifout,
#     fps = animation_fps,
#     selected_range = selected_range
# )

# # GENERATE VIDEOS
# generate_video(gifout, [ original_copy_covid, totaloutpath_covid ])
