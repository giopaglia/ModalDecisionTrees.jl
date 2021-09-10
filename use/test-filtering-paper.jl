
import Dates
using FileIO

include("wav-filtering.jl")
include("paper-trees.jl")

gr()

# SETTINGS
outpath = "filtering-results-paper"
cache_dir = outpath * "/cache"
if !isdir(outpath) mkpath(outpath) end
if !isdir(cache_dir) mkpath(cache_dir) end

save_datasets = true
dataset_form = :stump_with_memoization
data_savedir = cache_dir
timing_mode = :none
filtered_destination_dir = outpath * "/filtered"

# TREES
tree_configs = [ 
    (tree_hash = "τ1", tree = τ1, n_task = 1, n_version = 1, nbands = 60, preprocess_wavs = [ normalize! ], max_points = 30, ma_size = 75, ma_step = 50), 
    (tree_hash = "τ2", tree = τ2, n_task = 1, n_version = 1, nbands = 40, preprocess_wavs = [], max_points = 30, ma_size = 75, ma_step = 50), 
    (tree_hash = "τ3", tree = τ3, n_task = 1, n_version = 2, nbands = 40, preprocess_wavs = [], max_points = 30, ma_size = 45, ma_step = 30)
]

for tree_config in tree_configs
    # max_sample_rate = 8_000
    max_sample_rate = 16000
    tree_hash, tree, n_task, n_version, nbands, preprocess_wavs, max_points, ma_size, ma_step = tree_config

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

    dataset_kwargs = (
        max_points = max_points,
        ma_size = ma_size,
        ma_step = ma_step
    )

    dataset_func_params = (
        (n_task,n_version),
        audio_kwargs
    )
    dataset_func_kwparams = (
        dataset_kwargs...,
        return_filepaths = true,
        use_augmentation_data = true,
        preprocess_wavs = preprocess_wavs,
        use_full_mfcc = false,
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

    (X, Y, filepaths), (n_pos, n_neg) = @cache "dataset" cache_dir dataset_func_params dataset_func_kwparams KDDDataset_not_stratified
    X_modal = X_dataset_c("test", data_modal_args, X, modal_args, save_datasets, dataset_form, false)

    apply_tree_to_datasets_wavs(
            tree_hash,
            tree,
            X_modal,
            filepaths[1],
            Y;
            postprocess_wavs = [],
            postprocess_wavs_kwargs = [],
            filter_kwargs = (nbands = nbands,),#, maxfreq = max_sample_rate / 2),
            remove_from_path = "../datasets/KDD/",
            destination_dir = filtered_destination_dir,
    #        draw_anim_for_instances = [ findfirst(isequal("../datasets/KDD/healthyandroidwithcough/cough/cough_9me0RMtVww_1586943699308.wav"), filepaths[1]) ]
        )
end

println("DONE!")
exit(0)
