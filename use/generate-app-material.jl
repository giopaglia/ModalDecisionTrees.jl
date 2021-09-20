
import Dates
using FileIO

include("wav-filtering.jl")
include("paper-trees.jl")
# include("generate-app-placeholders.jl")

gr()

# SETTINGS
input_dir = "filtering-results-paper/filtered"
app_dir = homedir() * "/" * "Julia Projects/covid-detector/"
outpath = app_dir * "/generated-media"
cache_dir = outpath * "/cache"
filtered_destination_dir = outpath * "/filtered"
presentation_output_suffix = "/presentation"
presentation_total_output = filtered_destination_dir * presentation_output_suffix
if !isdir(outpath) mkpath(outpath) end
if !isdir(cache_dir) mkpath(cache_dir) end
if !isdir(presentation_total_output) mkpath(presentation_total_output) end

# DATASET CONSTANTS
save_datasets = true
dataset_form = :stump_with_memoization
data_savedir = cache_dir
timing_mode = :none

ma_size = 45
ma_step = 30


# CONFIGS
max_sample_rate = 8_000
nbands = 40
wintime = 0.025
steptime = 0.010

# ANIM
blank_tree = app_dir * "/" * "tree/anim/t_frame_blank.png"
anim_fps = 30

# ANIM (TREE)
tree_anim_size = (500, 250)

# SPECTROGRAMS
spec_margin = 4mm
spec_size = (1000, 500)
spectrogram_plot_options = (ylims = (0, max_sample_rate / 2), left_margin = spec_margin, right_margin = spec_margin, top_margin = spec_margin, bottom_margin = spec_margin)
spec_melbands_arg = (draw = true, nbands = nbands, maxfreq = max_sample_rate / 2)

# VIDEO
video_codec = "libx264"
video_output_extension = "mkv"
audio_codec = "copy"
additional_ffmpeg_args_v = "-crf 0 -preset ultrafast -pix_fmt yuv422p"
additional_ffmpeg_args_a = "" # "-b:a 320k"

# AUDIO
filtered_normalization_level = 0.5

###########################
########## STATIC #########
###########################
pos_path = input_dir * "/τ3-maxfreq-4000/Y_{1}_⟨G⟩_(A32_⪴₈₀_7.781613442411969e-5)/Y_{1}_A38_⪳₈₀_0.00016212943898189937/covidandroidnocough/breath"
pos_prefix = "breaths_CNz7PwFNQz_1589436648020"
pos_id = "pos"
pos_string = "positive"

neg_path = input_dir * "/τ3-maxfreq-4000/Y_{1}_⟨G⟩_(A32_⪴₈₀_7.781613442411969e-5)/N_{1}_A38_⪳₈₀_0.00016212943898189937/healthyandroidnosymp/breath"
neg_prefix = "breaths_Pf3lZHDYTV_1587122029533"
neg_id = "neg"
neg_string = "negative"

###########################
######### DATASET #########
###########################
X, Y = dataset_from_wav_paths(
    [ pos_path * "/" * pos_prefix * ".orig" * ".wav", neg_path * "/" * neg_prefix * ".orig" * ".wav" ],
    [ "YES", "NO_CLEAN_HISTORY_AND_LOW_PROBABILITY" ];
    ma_size = ma_size,
    ma_step = ma_step,
    nbands = nbands,
    audio_kwargs = (maxfreq = max_sample_rate / 2,),
    save_dataset = save_datasets
)

res = get_path_in_tree(τ3, X)
res_pos = res[1]
res_neg = res[2]

println(res_pos)
println(res_neg)

pos_samples, pos_sr = wavread(pos_path * "/" * pos_prefix * ".orig" * ".wav")
neg_samples, neg_sr = wavread(neg_path * "/" * neg_prefix * ".orig" * ".wav")

winsize_pos = round(Int64, (wintime * ma_size * pos_sr))
stepsize_pos = round(Int64, (steptime * ma_step * pos_sr))

winsize_neg = round(Int64, (wintime * ma_size * neg_sr))
stepsize_neg = round(Int64, (steptime * ma_step * neg_sr))

wav_descriptor_pos, points_pos, seconds_pos = get_points_and_seconds_from_worlds(res_pos[1].worlds, winsize_pos, stepsize_pos, length(pos_samples), pos_sr)
wav_descriptor_neg, points_neg, seconds_neg = get_points_and_seconds_from_worlds(res_neg[1].worlds, winsize_neg, stepsize_neg, length(neg_samples), neg_sr)

println("Positive intervals (seconds): ", seconds_pos)
println("Negative intervals (seconds): ", seconds_neg)

pos = (
    id = pos_id,
    string = pos_string,
    input = (
        wav_orig = pos_path * "/" * pos_prefix * ".orig" * ".wav",
        wav_filt = pos_path * "/" * pos_prefix * ".filt" * ".wav",
        wav_A32 = pos_path * "/" * pos_prefix * ".A32" * ".wav",
        wav_A38 = pos_path * "/" * pos_prefix * ".A38" * ".wav",
        highlighted_tree = app_dir * "/" * "tree/anim/t_frame_pos.png"
    ),
    output = (
        gif_original_path = presentation_total_output * "/" * "original-" * pos_id * ".gif",
        gif_filtered_path = presentation_total_output * "/" * "filtered-" * pos_id * ".gif",
        vid_original_path = presentation_total_output * "/" * "original-" * pos_id * "." * video_output_extension,
        vid_filtered_path = presentation_total_output * "/" * "filtered-" * pos_id * "." * video_output_extension,
        spectrogram = presentation_total_output * "/" * "spectrogram-" * pos_id * ".png",
        tree_anim = presentation_total_output * "/" * "tree-anim-" * pos_id * ".gif",
        tree_anim_video = presentation_total_output * "/" * "tree-anim-" * pos_id * "." * video_output_extension,
        tree_and_wave_vid = presentation_total_output * "/" * "animation-video-" * pos_id * "." * video_output_extension
    ),
    wav_descriptor = wav_descriptor_pos,
)

neg = (
    id = neg_id,
    string = neg_string,
    input = (
        wav_orig = neg_path * "/" * neg_prefix * ".orig" * ".wav",
        wav_filt = neg_path * "/" * neg_prefix * ".filt" * ".wav",
        wav_A32 = neg_path * "/" * neg_prefix * ".A32" * ".wav",
        wav_A38 = neg_path * "/" * neg_prefix * ".A38" * ".wav",
        highlighted_tree = app_dir * "/" * "tree/anim/t_frame_neg2.png"
    ),
    output = (
        gif_original_path = presentation_total_output * "/" * "original-" * neg_id * ".gif",
        gif_filtered_path = presentation_total_output * "/" * "filtered-" * neg_id * ".gif",
        vid_original_path = presentation_total_output * "/" * "original-" * neg_id * "." * video_output_extension,
        vid_filtered_path = presentation_total_output * "/" * "filtered-" * neg_id * "." * video_output_extension,
        spectrogram = presentation_total_output * "/" * "spectrogram-" * neg_id * ".png",
        tree_anim = presentation_total_output * "/" * "tree-anim-" * neg_id * ".gif",
        tree_anim_video = presentation_total_output * "/" * "tree-anim-" * neg_id * "." * video_output_extension,
        tree_and_wave_vid = presentation_total_output * "/" * "animation-video-" * neg_id * "." * video_output_extension
    ),
    wav_descriptor = wav_descriptor_neg,
)


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

    A38_exagerated = normalize!(deepcopy(A38_samp); level = filtered_normalization_level)
    A38_only_worlds = [ Int64(inst.wav_descriptor[i]) * s for (i, s) in enumerate(A38_exagerated) ]

    filtered_samp .*= norm_rate
    A32_samp .*= norm_rate
    A38_samp .*= norm_rate

    exagerated_tmp = "/tmp/DecisionTree.jl_tmp/A38_exagerated.wav"
    mkpath(dirname(exagerated_tmp))
    wavwrite(A38_exagerated, exagerated_tmp)

    # 1) original's video
    draw_audio_anim(
        [ (orig_norm, original_sr) ];
        outfile = inst.output.gif_original_path,
        colors = [ RGB(0.3, 0.3, 1) ],
        resample_at_rate = max_sample_rate,
        fps = anim_fps
    )
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
    draw_audio_anim(
        [ (A38_exagerated, original_sr), (A38_only_worlds, original_sr) ];
        # labels = [ "Filtered " * inst.string ],
        outfile = inst.output.gif_filtered_path,
        colors = [ RGB(1, 0.3, 0.3), RGB(0.3, 0.3, 1.0) ],
        resample_at_rate = max_sample_rate,
        fps = anim_fps,
        single_graph = true
    )
    generate_video(
        inst.output.gif_filtered_path,
        [ exagerated_tmp ];
        outpath = dirname(inst.output.vid_filtered_path),
        outputfile_name = basename(inst.output.vid_filtered_path),
        video_codec = video_codec,
        audio_codec = audio_codec,
        output_ext = video_output_extension,
        additional_ffmpeg_args_v = additional_ffmpeg_args_v,
        additional_ffmpeg_args_a = additional_ffmpeg_args_a,
    )
    rm(exagerated_tmp)
    # 3) tree videos and join them
    draw_tree_anim(
        inst.wav_descriptor,
        blank_tree,
        inst.input.highlighted_tree,
        original_sr;
        outfile = inst.output.tree_anim,
        size = tree_anim_size
    )
    generate_video(
        inst.output.tree_anim,
        [ inst.input.wav_A38 ];
        outpath = dirname(inst.output.tree_anim_video),
        outputfile_name = basename(inst.output.tree_anim_video),
        video_codec = video_codec,
        audio_codec = audio_codec,
        output_ext = video_output_extension,
        additional_ffmpeg_args_v = additional_ffmpeg_args_v,
        additional_ffmpeg_args_a = additional_ffmpeg_args_a,
    )
    join_tree_video_and_waveform_video(
        inst.output.tree_anim,
        tree_anim_size,
        inst.output.vid_filtered_path,
        (1000, 150),
        output_file = inst.output.tree_and_wave_vid
    )
    # 4) spectrograms
    spec_orig = draw_spectrogram(
        orig_norm, original_sr;
        title = "Original",
        only_bands = [ 32, 38 ],
        spectrogram_plot_options = spectrogram_plot_options,
        melbands = spec_melbands_arg
    )
    spec_filt = draw_spectrogram(
        filtered_samp, original_sr;
        title = "Filtered",
        only_bands = [ 32, 38 ],
        spectrogram_plot_options = spectrogram_plot_options,
        melbands = spec_melbands_arg
    )
    plot(spec_orig, spec_filt, layout = (1, 2), size = spec_size)
    savefig(inst.output.spectrogram)
end

# 5) extract first frame from gifs to use it as poster
for f in [
            pos.output.gif_original_path,
            neg.output.gif_original_path,
            pos.output.gif_filtered_path,
            neg.output.gif_filtered_path,
            pos.output.tree_anim,
            neg.output.tree_anim
        ]
    o = replace(f, ".gif" => ".png")
    try
        run(`convert "$f[0]" $o`)
    catch
        error("An error occurred while extracting posters from gifs")
    end
end

println("DONE!")
