
include("wav-filtering.jl")

# found = map(str -> replace(str, r"^../datasets/KDD/" => ""),cat(searchdir(data_dir * "KDD", ".wav"; recursive = true, results_limit = 10), searchdir(data_dir * "KDD", ".wav"; exclude = "breath", recursive = true, results_limit = 10); dims = 1))

# process_dataset("KDD", found; partition_instances = true)

draw_wavs_for_partitioned_dataset("KDD", "KDD-partitioned-PREPROC[]-POSTPROC[]")

# generate_splitted_wavs_dataset("../datasets/KDD"; exclude = ["aug","mono","pitch"], draw_wavs = true, limit = 10)
# generate_splitted_wavs_dataset("../datasets/KDD"; exclude = ["aug","mono","pitch","breath"], draw_wavs = true, limit = 20)
