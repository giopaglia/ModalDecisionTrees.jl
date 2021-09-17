
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

# hash, tree, nbands, preprocess_wavs, max_points, ma_size, ma_step
cough =  ("τ1", τ1, 60, [ normalize! ], 30, 75, 50)
        #("τ2", τ2, 40, [], 30, 75, 50)
breath = ("τ3", τ3, 40, [], 30, 45, 30)

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
ts = moving_average(ts, ma_size, ma_step)
if max_points != -1 && size(ts,1)>max_points
    ts = ts[1:max_points,:]
end

max_timepoints = maximum(size(ts, 1))
n_unique_freqs = unique(size(ts,  2))
@assert length(n_unique_freqs) == 1 "length(n_unique_freqs) != 1: {$n_unique_freqs} != 1"
n_unique_freqs = n_unique_freqs[1]

X = compute_X(max_timepoints, n_unique_freqs, [ ts ], 1)
X = X_dataset_c("test", data_modal_args, [X], modal_args, false, dataset_form, false)

result = apply_tree(tree, X)[1]
if isnothing(label)
    label = result
end
println(result)

apply_tree_to_datasets_wavs(
    tree_hash,
    tree,
    X,
    [ filepath ],
    [ label ];
    postprocess_wavs = [],
    postprocess_wavs_kwargs = [],
    remove_from_path = dirname(filepath),
    filter_kwargs = (nbands = nbands, maxfreq = max_sample_rate / 2),
    destination_dir = filtered_destination_dir,
    generate_spectrogram = true,
    spectrograms_kwargs = (melbands = (draw = false, nbands = nbands, maxfreq = max_sample_rate / 2), spectrogram_plot_options = (ylims = (0, max_sample_rate / 2),)),
)
