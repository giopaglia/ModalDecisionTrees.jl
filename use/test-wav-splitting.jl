
include("wav-filtering.jl")

generate_splitted_wavs_dataset("../datasets/KDD"; exclude = ["aug","mono","pitch"], draw_wavs = true, limit = 10)
generate_splitted_wavs_dataset("../datasets/KDD"; exclude = ["aug","mono","pitch","breath"], draw_wavs = true, limit = 20)
