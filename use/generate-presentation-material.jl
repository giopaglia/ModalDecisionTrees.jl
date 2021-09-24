
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
presentation_output_suffix = "/presentation"
presentation_total_output = filtered_destination_dir * presentation_output_suffix
if !isdir(presentation_total_output) mkpath(presentation_total_output) end

# CONFIGS
max_sample_rate = 8_000
nbands = 40

# ANIM
anim_fps = 30

# SPECTROGRAMS
spectrogram_path = presentation_total_output * "/" * "spectrograms.png"
spec_mosaic_margin = 2mm
spec_mosaic_size = (1000, 1000)
spectrogram_plot_options = (ylims = (0, max_sample_rate / 2), left_margin = spec_mosaic_margin, right_margin = spec_mosaic_margin, top_margin = spec_mosaic_margin, bottom_margin = spec_mosaic_margin)

# VIDEO
video_codec = "libx264"
video_output_extension = "mp4"
audio_codec = "libmp3lame"
additional_ffmpeg_args_v = "-crf 0 -preset ultrafast -pix_fmt yuv422p"
additional_ffmpeg_args_a = "-b:a 320k"


###########################
########## STATIC #########
###########################

pos_path = filtered_destination_dir * presentation_output_suffix * "/τ3-maxfreq-4000/Y_{1}_⟨G⟩_(A32_⪴₈₀_7.781613442411969e-5)/Y_{1}_A38_⪳₈₀_0.00016212943898189937/covidandroidnocough/breath"
pos_prefix = "breaths_8PmvbJ4U3o_1588144326476"
pos_id = "pos"
pos_string = "positive"

neg_path = filtered_destination_dir * presentation_output_suffix * "/τ3-maxfreq-4000/Y_{1}_⟨G⟩_(A32_⪴₈₀_7.781613442411969e-5)/N_{1}_A38_⪳₈₀_0.00016212943898189937/healthyandroidnosymp/breath"
neg_prefix = "breaths_VN8n8tjozE_1589473637538"
neg_id = "neg"
neg_string = "negative"

pos = (
    id = pos_id,
    string = pos_string,
    input = (
        wav_orig = pos_path * "/" * pos_prefix * ".orig" * ".wav",
        wav_filt = pos_path * "/" * pos_prefix * ".filt" * ".wav",
        wav_A32 = pos_path * "/" * pos_prefix * ".A32" * ".wav",
        wav_A38 = pos_path * "/" * pos_prefix * ".A38" * ".wav",
    ),
    output = (
        gif_original_path = presentation_total_output * "/" * "original-" * pos_id * ".gif",
        gif_filtered_path = presentation_total_output * "/" * "filtered-" * pos_id * ".gif",
        vid_original_path = presentation_total_output * "/" * "original-" * pos_id * "." * video_output_extension,
        vid_filtered_path = presentation_total_output * "/" * "filtered-" * pos_id * "." * video_output_extension,
        spectrogram = presentation_total_output * "/" * pos_id * "-spectrogram.png"
    )
)

neg = (
    id = neg_id,
    string = neg_string,
    input = (
        wav_orig = neg_path * "/" * neg_prefix * ".orig" * ".wav",
        wav_filt = neg_path * "/" * neg_prefix * ".filt" * ".wav",
        wav_A32 = neg_path * "/" * neg_prefix * ".A32" * ".wav",
        wav_A38 = neg_path * "/" * neg_prefix * ".A38" * ".wav",
    ),
    output = (
        gif_original_path = presentation_total_output * "/" * "original-" * neg_id * ".gif",
        gif_filtered_path = presentation_total_output * "/" * "filtered-" * neg_id * ".gif",
        vid_original_path = presentation_total_output * "/" * "original-" * neg_id * "." * video_output_extension,
        vid_filtered_path = presentation_total_output * "/" * "filtered-" * neg_id * "." * video_output_extension,
        spectrogram = presentation_total_output * "/" * neg_id * "-spectrogram.png"
    )
)

spec_dict = Dict{String,Plots.Plot}()



######################
####### START ########
######################

for inst in (pos, neg)
    # 0) normalize wavs
    original_samp, original_sr = wavread(inst.input.wav_orig)
    original_samp = merge_channels(original_samp)

    orig_norm = normalize!(deepcopy(original_samp))
    norm_rate = maximum(abs, orig_norm) / maximum(abs, original_samp)

    filtered_samp, filtered_sr = wavread(inst.input.wav_filt)
    A32_samp, A32_sr = wavread(inst.input.wav_A32)
    A38_samp, A38_sr = wavread(inst.input.wav_A38)

    filtered_samp = merge_channels(filtered_samp)
    A32_samp = merge_channels(A32_samp)
    A38_samp = merge_channels(A38_samp)

    filtered_samp .*= norm_rate
    A32_samp .*= norm_rate
    A38_samp .*= norm_rate

    # 1) original's video
    # draw_audio_anim(
    #     [ (orig_norm, original_sr) ];
    #     outfile = inst.output.gif_original_path,
    #     colors = [ RGB(0.3, 0.3, 1) ],
    #     resample_at_rate = max_sample_rate,
    #     fps = anim_fps
    # )
    generate_video(
        inst.output.gif_original_path,
        [ inst.input.wav_orig ];
        outpath = dirname(inst.output.vid_original_path),
        outputfile_name = basename(inst.output.vid_original_path),
        video_codec = video_codec,
        audio_codec = audio_codec,
        output_ext = video_output_extension,
        additional_ffmpeg_args_v = additional_ffmpeg_args_v,
        additional_ffmpeg_args_a = additional_ffmpeg_args_a,
    )
    # 2) filtered videos
    # draw_audio_anim(
    #     [ (A38_samp, original_sr) ];
    #     labels = [ "Filtered " * inst.string ],
    #     outfile = inst.output.gif_filtered_path,
    #     colors = [ RGB(1, 0.3, 0.3), RGB(0.1, 0.1, 1), RGB(0.2, 0.2, 1) ],
    #     resample_at_rate = max_sample_rate,
    #     fps = anim_fps
    # )
    generate_video(
        inst.output.gif_filtered_path,
        [ inst.input.wav_A38 ];
        outpath = dirname(inst.output.vid_filtered_path),
        outputfile_name = basename(inst.output.vid_filtered_path),
        video_codec = video_codec,
        audio_codec = audio_codec,
        output_ext = video_output_extension,
        additional_ffmpeg_args_v = additional_ffmpeg_args_v,
        additional_ffmpeg_args_a = additional_ffmpeg_args_a,
    )
    # 3.0) spectrograms
    # spec_dict[inst.id * "_original"] = draw_spectrogram(
    #     inst.input.wav_orig;
    #     title = "Original " * inst.string,
    #     spectrogram_plot_options = spectrogram_plot_options,
    #     melbands = (draw = true, nbands = nbands, maxfreq = max_sample_rate / 2)
    # )
    # spec_dict[inst.id * "_filtered"] = draw_spectrogram(
    #     inst.input.wav_filt;
    #     title = "Filtered " * inst.string,
    #     spectrogram_plot_options = spectrogram_plot_options,
    #     melbands = (draw = true, nbands = nbands, maxfreq = max_sample_rate / 2)
    # )
    # 3.1) spectrograms normalized
    # spec_dict[inst.id * "_original_normalized"] = draw_spectrogram(
    #     orig_norm, original_sr;
    #     title = "Original " * inst.string,
    #     spectrogram_plot_options = spectrogram_plot_options,
    #     melbands = (draw = true, nbands = nbands, maxfreq = max_sample_rate / 2)
    # )
    # spec_dict[inst.id * "_filtered_normalized"] = draw_spectrogram(
    #     filtered_samp, original_sr;
    #     title = "Filtered " * inst.string,
    #     spectrogram_plot_options = spectrogram_plot_options,
    #     melbands = (draw = true, nbands = nbands, maxfreq = max_sample_rate / 2)
    # )
    # 3.2) spectrograms just A32 A38
    # spec_dict[inst.id * "_original_just_band"] = draw_spectrogram(
    #     orig_norm, original_sr;
    #     title = "Original " * inst.string,
    #     spectrogram_plot_options = spectrogram_plot_options
    # )
    # spec_dict[inst.id * "_filtered_just_band"] = draw_spectrogram(
    #     filtered_samp, original_sr;
    #     title = "Filtered " * inst.string,
    #     spectrogram_plot_options = spectrogram_plot_options
    # )

    # melscale = get_mel_bands(nbands, 0.0, max_sample_rate / 2)
    # mA32, mA38 = melscale[32], melscale[38]

    # freqs = Vector{Float64}([ 0.0, mA32.peak, mA38.peak, (max_sample_rate/2) ])
    # labels = Vector{String}([ "0", "A32\n" * string(round(Int64, mA32.peak)), "A38\n" * string(round(Int64, mA38.peak)), string(round(Int64, max_sample_rate/2)) ])
    # yticks!(spec_dict[inst.id * "_original_just_band"], freqs, labels)
    # yticks!(spec_dict[inst.id * "_filtered_just_band"], freqs, labels)

    # for band in (mA32, mA38) hline!(spec_dict[inst.id * "_original_just_band"], [ band.left, band.right ], line = (1, :white), leg = false) end
    # for band in (mA32, mA38) hline!(spec_dict[inst.id * "_filtered_just_band"], [ band.left, band.right ], line = (1, :white), leg = false) end
end

# 5.1) compose spectrograms mosaic
# plts = (
#     spec_dict[pos_id * "_original"], spec_dict[pos_id * "_filtered"],
#     spec_dict[neg_id * "_original"], spec_dict[neg_id * "_filtered"]
# )

# plot(plts..., layout = (2, 2), size = spec_mosaic_size)
# savefig(spectrogram_path)

# 5.2) compose normalized spectrograms mosaic
# plts = (
#     spec_dict[pos_id * "_original_normalized"], spec_dict[pos_id * "_filtered_normalized"],
#     spec_dict[neg_id * "_original_normalized"], spec_dict[neg_id * "_filtered_normalized"]
# )

# plot(plts..., layout = (2, 2), size = spec_mosaic_size)
# savefig(replace(spectrogram_path, ".png" => ".normalized.png"))

# 5.3) compose just A32 and A38 spectrograms mosaic
# plts = (
#     spec_dict[pos_id * "_original_just_band"], spec_dict[pos_id * "_filtered_just_band"],
#     spec_dict[neg_id * "_original_just_band"], spec_dict[neg_id * "_filtered_just_band"]
# )

# plot(plts..., layout = (2, 2), size = spec_mosaic_size)
# savefig(replace(spectrogram_path, ".png" => ".just.A32.A38.png"))

# 6) generate example tex file
orig_ratio = 1000 / 150
filt_ratio = 1000 / 150

selected_width = 325 # normal article body A4 is ~426pt
width_measure_unit = "pt"

dim_string_orig = "width=$(selected_width)$(width_measure_unit),height=$(selected_width/filt_ratio)$(width_measure_unit)"
dim_string_filt = "width=$(selected_width)$(width_measure_unit),height=$(selected_width/orig_ratio)$(width_measure_unit)"

main_tex_content = """
\\documentclass{article}
\\usepackage{tikz}
\\usepackage{multimedia}

\\begin{document}

\\centering
\\noindent
\\movie[showcontrols=true]{\\includegraphics[$(dim_string_filt)]{$(replace(basename(pos.output.gif_original_path), ".gif" => ".png"))}}{$(basename(pos.output.vid_original_path))}
\\movie[showcontrols=true]{\\includegraphics[$(dim_string_filt)]{$(replace(basename(neg.output.gif_original_path), ".gif" => ".png"))}}{$(basename(neg.output.vid_original_path))}
\\movie[showcontrols=true]{\\includegraphics[$(dim_string_orig)]{$(replace(basename(pos.output.gif_filtered_path), ".gif" => ".png"))}}{$(basename(pos.output.vid_filtered_path))}
\\movie[showcontrols=true]{\\includegraphics[$(dim_string_orig)]{$(replace(basename(neg.output.gif_filtered_path), ".gif" => ".png"))}}{$(basename(neg.output.vid_filtered_path))}
\\newpage
\\resizebox{\\textwidth}{!}{\\includegraphics{$(basename(spectrogram_path))}}
\\resizebox{\\textwidth}{!}{\\includegraphics{$(basename(replace(spectrogram_path, ".png" => ".normalized.png")))}}
\\resizebox{\\textwidth}{!}{\\includegraphics{$(basename(replace(spectrogram_path, ".png" => ".just.A32.A38.png")))}}

\\end{document}
"""

main_tex_file = "main.tex"

f = open(presentation_total_output * "/" * main_tex_file, "w+")
write(f, main_tex_content)
close(f)

cd(presentation_total_output)

# 7) extract first frame from gifs to use it as poster
# for f in map(basename, [ pos.output.gif_original_path, neg.output.gif_original_path, pos.output.gif_filtered_path, neg.output.gif_filtered_path ])
#     o = replace(f, ".gif" => ".png")
#     try
#         run(`convert "$f[0]" $o`)
#     catch
#         error("An error occurred while extracting posters from gifs")
#     end
# end

println("DONE!")

run(`pdflatex $main_tex_file`)
pdf_name = replace(main_tex_file, ".tex" => ".pdf")
run(`evince $pdf_name`)

exit(0)
