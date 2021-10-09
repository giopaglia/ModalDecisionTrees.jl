
include("wav-filtering.jl")

# found = map(str -> replace(str, r"^../datasets/KDD/" => ""),cat(searchdir(data_dir * "KDD", ".wav"; recursive = true, results_limit = 10), searchdir(data_dir * "KDD", ".wav"; exclude = "breath", recursive = true, results_limit = 10); dims = 1))

# process_dataset("KDD", found; partition_instances = true)

process_dataset("KDD", KDD_getSamplesList(; only_version = "c", rel_path = true);
    out_dataset_dir_name = "KDD-norm-partitioned",
    partition_instances = true,
    partitioning_kwargs = (
        cut_original        = Val(true),
        preprocess          = Function[noise_gate!, normalize!],
        # preprocess_kwargs   = NamedTuple[ (level = 0.005,), (level = 1.0,) ],
        postprocess         = Function[],
    )
)

process_dataset("KDD", KDD_getSamplesList(; only_version = "b", rel_path = true);
    out_dataset_dir_name = "KDD-norm-partitioned",
    partition_instances = true,
    partitioning_kwargs = (
        cut_original        = Val(true),
        preprocess          = Function[normalize!],
        # preprocess_kwargs   = NamedTuple[ (level = 0.005,), (level = 1.0,) ],
        postprocess         = Function[],
    )
)

draw_wavs_for_partitioned_dataset("KDD", "KDD-norm-partitioned")

# generate_splitted_wavs_dataset("../datasets/KDD"; exclude = ["aug","mono","pitch"], draw_wavs = true, limit = 10)
# generate_splitted_wavs_dataset("../datasets/KDD"; exclude = ["aug","mono","pitch","breath"], draw_wavs = true, limit = 20)
