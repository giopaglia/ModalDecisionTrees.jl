
function usage(io::IO)
    println(io, "\nusage:\n\t$(PROGRAM_FILE) <TYPE> [LABEL] <WAV_FILE_PATH>\n")
    println(io, "TYPE")
    println(io, "\t-b\tbreath")
    println(io, "\t-c\tcough")
    println(io, "LABEL (optional)")
    println(io, "\t-p\tpositive")
    println(io, "\t-n\tnegative")
    println(io, "WAV_FILE_PATH")
    println(io, "\tThe path to the wav file\n")
end
usage() = usage(stdout)

args = deepcopy(ARGS)

############################
# SCRIPT ARGS
############################

type =
    if "-b" in args
        :breath
    elseif "-c" in args
        :cough
    else
        println(stderr, "ERROR: Need to pass at least one between '-b` (breath) or '-c` (cough)")
        usage(stderr)
        exit(1)
    end
args = filter(x -> !(x == "-b" || x == "-c"), args)

label =
    if "-p" in args
        "YES"
    elseif "-n" in args
        "NO_CLEAN_HISTORY_AND_LOW_PROBABILITY"
    else
        nothing
        # println(stderr, "ERROR: Need to pass at least one between '-p` (positive) or '-n` (negative)")
        # usage(stderr)
        # exit(1)
    end
args = filter(x -> !(x == "-p" || x == "-n"), args)

if length(args) == 0
    println(stderr, "ERROR: No filepath passed")
    usage(stderr)
    exit(1)
end

if length(args) > 1
    println(stderr, "ERROR: A single instance must be passed")
    usage(stderr)
    exit(1)
end

filepath = args[1]

if !isfile(filepath)
    println(stderr, "ERROR: File $(filepath) can not be found")
    usage(stderr)
    exit(1)
end

############################
############################
############################

import Dates
using FileIO

include("wav-filtering.jl")
include("paper-trees.jl")
include("load-models.jl")

gr()

# SETTINGS
outpath = "filtering-results-rai"
cache_dir = outpath * "/cache"
tmp_dir = "/tmp/filtering_tmp_dir"
if !isdir(outpath) mkpath(outpath) end
if !isdir(cache_dir) mkpath(cache_dir) end
if !isdir(tmp_dir) mkpath(tmp_dir) end

save_datasets = true
dataset_form = :stump_with_memoization
data_savedir = cache_dir
timing_mode = :none
filtered_destination_dir = outpath * "/filtered"

tmp_ffmpeg_output_file = tmp_dir * "/ffmpeg.out"
tmp_ffmpeg_error_output_file = tmp_dir * "/ffmpeg.out"

# convert file to wav
if !endswith(filepath, ".wav")
    ext = "$(split(filepath, '.')[end])"
    println("File format $(ext) is not supported: converting to WAV...")
    total_output_path = tmp_dir * "/" * replace(basename(filepath), ".$(ext)" => ".wav")
    run(pipeline(`ffmpeg -i $filepath -y $total_output_path`, stdout = tmp_ffmpeg_output_file, stderr = tmp_ffmpeg_error_output_file))
    filepath = total_output_path
end

# CONFIGS
max_sample_rate = 8_000
use_full_mfcc = false
max_points = 30
ignore_max_points = false

# hash, tree, nbands, preprocess_wavs, max_points, ma_size, ma_step
# cough =  ("τ1", τ1, 60, [ normalize! ], 30, 75, 50)
        #("τ2", τ2, 40, [], 30, 75, 50)
# breath = ("τ3", τ3, 40, [], 30, 45, 30)
# breath = ("τ3", τ3, 40, [ normalize! ], 30, 45, 30)

hashes_1_1 = ( # task1 v1
    "rf_9e40eb67b60e5cc0be523f44eb16f7eb1382dd990a53d7f614f1caa078938366",
    "rf_a97311a51cf36c73b9968e31edaf9216b58030120207a3c952ff37372e13a196",
    "rf_58adc2ea33cbd06880efbea70232200b4681a4171ca4831e3b07ead7f64aa4dd",
    "rf_a0832de68f0af8f56ce045ec6dcd1fd752fa8b850610975000e140c38a680d4f",
    "rf_a52e0602b0b6b9ae1398a685434c34c054c6a4c7d567a7218373e5c19fa30f66",
    "rf_eac548a16e4725cf720573ed763cc4b75db78091b62a650d59fbf8c6dbf92e6e",
    "rf_6b6adf67414880a58ab6a79fd7a900594f2fa02ff3be09e99b8cd0078c93d13a",
    "rf_8cc8e656eaef64f21a47b776cf881dbdacba30f4716a84958646c369a4260ed9",
    "rf_0f9abb51ebfb35f0c40afafe6a7a1a7ee1798c007a93b67ae3b629d11c1ed1e6",
    "rf_166e0030857042f443af2148c2ea62d2760bb6b20b5cbfe41339a85d2f539856"
)
hashes_2_1 = ( # task2 v1
    "rf_629d85da7d43a3aa65a0a6c3c70be41ec0a45aee3b7499e4d1a8e8ab38bf2cc1",
    "rf_22ead53c1fa4d10399f48fb1a566e89ca5a6e41af14180e1ba349dd0ced22998",
    "rf_2fb23746b85ce0ebc3a499a153a97d4741f57208b24e7def92095a4e775386b4",
    "rf_8e811bb3bf4893e7b55481eaa3ab658577b60032f425a9a1c9af9777dd4fe146",
    "rf_9a9802c59b300fd136ec151d0598357f3b97ba6b8bdf4c2ea5c7c7be51e85c5b"
)

cough_model_hash = hashes_1_1[1] # TODO:
breath_model_hash = "" # TODO:
model_save_dir = "models/trees"


# Dataseed	use_training_form	n_task_use_aug	n_version	nbands	dataset_kwargs	use_full_mfcc	preprocess_wavs	test_operators
# 10	stump_with_memoization	(2,true)	1	40	(100,75)	FALSE	["NG","Normalize"]	TestOp_80
# hash, tree, nbands, preprocess_wavs, max_points, ma_size, ma_step
cough =  (cough_model_hash, load_model(cough_model_hash, model_save_dir), 40, [ noise_gate!, normalize! ], 30, 100, 75)
# cough =  (cough_model_hash, load_model(cough_model_hash, model_save_dir), 40, [ noise_gate!, normalize! ], 30, 100, 75)
# TODO: select proper parameters for selected models for breath
breath = (breath_model_hash, τ3, 40, [], 30, 45, 30)
# breath = (breath_model_hash, load_model(breath_model_hash, model_save_dir), 40, [], 30, 45, 30)

tree_hash, tree, nbands, preprocess_wavs, max_points, ma_size, ma_step = type == :breath ? breath : cough

audio_kwargs_partial_mfcc = (
	wintime      = 0.025,
	steptime     = 0.010,
	fbtype       = :mel,
	window_f     = DSP.triang,
	pre_emphasis = 0.97,
	nbands       = 40,
	sumpower     = false,
	dither       = false,
	maxfreq      = max_sample_rate / 2,
)

modal_args = (;
    initConditions = DecisionTree.startWithRelationGlob,
    useRelationGlob = false,
)

data_modal_args = (;
    ontology = getIntervalOntologyOfDim(Val(1)),
    test_operators = [TestOpGeq_80, TestOpLeq_80]
)

#####################
#####################
#####################

compute_X(max_timepoints, n_unique_freqs, timeseries, expected_length) = begin
    @assert expected_length == length(timeseries)
    X = zeros((max_timepoints, n_unique_freqs, length(timeseries)))
    for (i,ts) in enumerate(timeseries)
        # println(size(ts))
        X[1:size(ts, 1),:,i] = ts
    end
    X
end

audio_kwargs = merge(audio_kwargs_partial_mfcc, (nbands = nbands,))

ts = wav2stft_time_series(filepath, audio_kwargs; preprocess_sample = preprocess_wavs, use_full_mfcc = use_full_mfcc)

ts = @views ts[2:end,:]
# original_ts_length = size(ts,1)
ts = moving_average(ts, ma_size, ma_step)
# moving_average_ts_length = size(ts,1)
if !ignore_max_points && max_points != -1 && size(ts,1) > max_points
    ts = ts[1:max_points,:]
end

max_timepoints = maximum(size(ts, 1))
n_unique_freqs = unique(size(ts,  2))
@assert length(n_unique_freqs) == 1 "length(n_unique_freqs) != 1: {$n_unique_freqs} != 1"
n_unique_freqs = n_unique_freqs[1]

X = compute_X(max_timepoints, n_unique_freqs, [ ts ], 1)
X = X_dataset_c("test", data_modal_args, [X], modal_args, false, dataset_form, false)

result = apply_model(tree, X)[1]
if isnothing(label)
    label = result
end

result_str = startswith(result, "YES") ? "POSITIVE" : "NEGATIVE"

banner = "##########################################"
pre_post_banner = floor(Int64, (length(banner) - length(result_str) - 2) / 2)
println(banner)
println("#"^pre_post_banner, " ", result_str, " ", "#"^pre_post_banner)
println(banner)

# results, spectrograms = apply_tree_to_datasets_wavs(
#     tree_hash,
#     tree,
#     X,
#     [ filepath ],
#     [ label ];
#     postprocess_wavs = Vector{Function}(),
#     remove_from_path = dirname(filepath),
#     filter_kwargs = (nbands = nbands, maxfreq = max_sample_rate / 2),
#     destination_dir = filtered_destination_dir,
#     generate_spectrogram = true,
#     spectrograms_kwargs = (melbands = (draw = true, nbands = nbands, maxfreq = max_sample_rate / 2), spectrogram_plot_options = (ylims = (0, max_sample_rate / 2),)),
# )

# samples, samplerate = wavread(filepath)
# samples = merge_channels(samples)
# winsize = round(Int64, (audio_kwargs_partial_mfcc.wintime * ma_size * samplerate))
# stepsize = round(Int64, (audio_kwargs_partial_mfcc.steptime * ma_step * samplerate))

# wav_descriptor, points, seconds = get_points_and_seconds_from_worlds(results[1].path[end].worlds, winsize, stepsize, length(samples), samplerate)

# println("Points: ", points)
# println("Seconds: ", seconds)

# println("● File: ", filepath)
# println("  ├─WAV duration: ", round(length(samples) / samplerate, digits = 2), "s")
# println("  └─Worlds duration: ", round(length(findall(wav_descriptor)) / samplerate, digits = 2), "s")

# draw_tree_anim(wav_descriptor, "tree-anim/t_frame_blank.png", "tree-anim/t_frame_neg2.png", samplerate)

# run(`ffmpeg -y -i tree-anim/tree-anim.gif tree-anim/tree-anim.mkv`)
# TODO: add call to "get_points_and_seconds_from_worlds" in "apply_tree_to_datasets_wavs"
# GOOD => "filtering-results-paper/filtered/τ3-maxfreq-4000/Y_{1}_⟨G⟩_(A32_⪴₈₀_7.781613442411969e-5)/N_{1}_A38_⪳₈₀_0.00016212943898189937/healthyandroidnosymp/breath/breaths_2pYxTRaHvl_1586931806240.orig.wav"
# "/home/ferdiu/Julia Projects/DT/use/filtering-results-paper/filtered/τ3-maxfreq-4000/Y_{1}_⟨G⟩_(A32_⪴₈₀_7.781613442411969e-5)/N_{1}_A38_⪳₈₀_0.00016212943898189937/healthywebnosymp/2020-04-07-09_19_14_441629/audio_file_breathe.orig.wav"