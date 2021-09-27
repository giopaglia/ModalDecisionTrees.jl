
import Dates
using FileIO

include("wav-filtering.jl")
include("paper-trees.jl")

gr()

# SETTINGS
outpath = "filtering-results-paper-dynamic-filters"
cache_dir = outpath * "/cache"
filtered_destination_dir = outpath * "/filtered"
if !isdir(outpath) mkpath(outpath) end
if !isdir(cache_dir) mkpath(cache_dir) end
if !isdir(filtered_destination_dir) mkpath(filtered_destination_dir) end

save_datasets = true
dataset_form = :stump_with_memoization
data_savedir = cache_dir
timing_mode = :none
ignore_max_points = true

spectrogram_size = (1000, 500)

# VIDEO
video_kwargs = (
    video_codec              = "libx264", # "libx265"
    output_ext               = "mkv",
    audio_codec              = "copy", #libmp3lame"
    additional_ffmpeg_args_v = "-crf 0 -preset ultrafast -pix_fmt yuv422p", # "-crf 0 -preset ultrafast -x265-params lossless=1"
    additional_ffmpeg_args_a = "" #"-b:a 320k"
)

# TREES
tree_configs = [
    # (tree_hash = "τ1", tree = τ1, n_task = 1, n_version = 1, nbands = 60, preprocess_wavs = [ normalize! ], max_points = 30, ma_size = 75, ma_step = 50, max_sample_rate = nothing), 
    # (tree_hash = "τ2", tree = τ2, n_task = 1, n_version = 1, nbands = 40, preprocess_wavs = Vector{Function}(), max_points = 30, ma_size = 75, ma_step = 50, max_sample_rate = nothing), 
    (tree_hash = "τ3", tree = τ3, n_task = 1, n_version = 2, nbands = 40, preprocess_wavs = Vector{Function}(), max_points = 30, ma_size = 45, ma_step = 30, max_sample_rate = 8_000)
]

results_dict = Dict{String,Any}()

selected_wavs = [
    "../datasets/KDD/covidandroidnocough/breath/breaths_CNz7PwFNQz_1589436648020.wav",
    "../datasets/KDD/healthyandroidnosymp/breath/breaths_Pf3lZHDYTV_1587122029533.wav"
]

for tree_config in tree_configs

    tree_hash, tree, n_task, n_version, nbands, preprocess_wavs, max_points, ma_size, ma_step, max_sample_rate = tree_config

    audio_kwargs = (
        wintime = 0.025,
        steptime = 0.010,
        fbtype = :mel,                     
        window_f = DSP.hamming,
        pre_emphasis = 0.97,
        nbands = nbands,
        sumpower = false,
        dither = false,
        # maxfreq = max_sample_rate/2,
    )

    if !isnothing(max_sample_rate)
        audio_kwargs = merge(audio_kwargs, (maxfreq = max_sample_rate / 2,))
    end

    dataset_kwargs = (
        ma_size = ma_size,
        ma_step = ma_step
    )

    if !ignore_max_points && !isnothing(max_points) 
        dataset_kwargs = merge(dataset_kwargs, (max_points = max_points,))
    end

    # dataset_func_params = (
    #     (n_task,n_version),
    #     audio_kwargs
    # )
    # dataset_func_kwparams = (
    #     dataset_kwargs...,
    #     return_filepaths = true,
    #     use_augmentation_data = true,
    #     preprocess_wavs = preprocess_wavs,
    #     use_full_mfcc = false,
    #     force_monolithic_dataset = :train_n_test
    # )

    modal_args = (;
        initConditions = DecisionTree.startWithRelationGlob,
        useRelationGlob = false,
    )

    data_modal_args = (;
        ontology = getIntervalOntologyOfDim(Val(1)),
        test_operators = [TestOpGeq_80, TestOpLeq_80]
    )

    X_modal, Y = dataset_from_wav_paths(
        selected_wavs,
        [ "YES", "NO_CLEAN_HISTORY_AND_LOW_PROBABILITY" ];
        nbands = nbands,
        audio_kwargs = audio_kwargs,
        modal_args = modal_args,
        data_modal_args = data_modal_args,
        preprocess_sample = preprocess_wavs,
        ma_size = ma_size,
        ma_step = ma_step
    )
    filepaths = [ selected_wavs ]

    # (X, Y, filepaths), (n_pos, n_neg) = @cache "dataset" cache_dir dataset_func_params dataset_func_kwparams KDDDataset_not_stratified
    # X_modal = X_dataset_c("test", data_modal_args, X, modal_args, save_datasets, dataset_form, false)

    draw_anim_for_instances::Vector{Int64} =
        if length(selected_wavs) > 0
            filter(x -> isa(x, Integer), [ findfirst(isequal(p), filepaths[1]) for p in selected_wavs ])
        else
            collect(1:length(filepaths[1]))
        end

    Xslice, Yslice, filepaths_slice, draw_insts = 
        if length(selected_wavs) > 0 && length(draw_anim_for_instances) > 0 && length(draw_anim_for_instances) != length(filepaths[1])
            ModalLogic.slice_dataset(X_modal, draw_anim_for_instances), Y[draw_anim_for_instances], [ filepaths[1][draw_anim_for_instances] ], collect(1:length(draw_anim_for_instances))
        else
            X_modal, Y, filepaths, draw_anim_for_instances
        end

    add_tree_hash::String, filter_kwargs::NamedTuple, spectrograms_kwargs::NamedTuple =
        if isnothing(max_sample_rate)
            "-no-maxfreq",
            (nbands = nbands,),
            (melbands = (draw = true, nbands = nbands,), spectrogram_plot_options = (size = spectrogram_size,))
        else
            "-maxfreq-" * string(round(Int64, max_sample_rate / 2)),
            (nbands = nbands, maxfreq = max_sample_rate / 2),
            (melbands = (draw = true, nbands = nbands, maxfreq = max_sample_rate / 2), spectrogram_plot_options = (ylims = (0, max_sample_rate / 2), size = spectrogram_size))
        end

    println("Using max samplerate: ", max_sample_rate, " (maxfreq: ", audio_kwargs.maxfreq, ")")
    apply_tree_to_datasets_wavs(
            tree_hash * add_tree_hash,
            tree,
            Xslice,
            filepaths_slice[1],
            Yslice;
            only_files = length(selected_wavs) == 0 ? Vector{String}() : selected_wavs,
            postprocess_wavs = Vector{Function}(),
            filter_kwargs = filter_kwargs,
            remove_from_path = "../datasets/KDD/",
            spectrograms_kwargs = spectrograms_kwargs,
            destination_dir = filtered_destination_dir,
            anim_kwargs = (fps = 30,),
            normalize_before_draw_anim = true,
            video_kwargs = video_kwargs,
            draw_anim_for_instances = draw_insts,
            wintime = audio_kwargs.wintime,
            steptime = audio_kwargs.steptime,
            movingaverage_size = dataset_kwargs.ma_size,
            movingaverage_step = dataset_kwargs.ma_step
        )
end

println("DONE!")
