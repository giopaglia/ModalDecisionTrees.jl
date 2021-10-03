using JSON
using Glob
using ThreadsX
# include("../wav2stft_time_series.jl")

################################################################################
################################################################################
################################################################################
# task 1: YES/NO_CLEAN_HISTORY_AND_LOW_PROBABILITY
#   ( 66 user (141 sample) / 220 users (298 samples) in total)
# - v1: USING COUGH
# - v2: USING BREATH
# - v3: USING COUGH + BREATH
# task 2: YES_WITH_COUGH/NO_CLEAN_HISTORY_AND_LOW_PROBABILITY
#   ( 23 user (54 sample) / 29 users (32 samples) in total)
# - v1: USING COUGH
# - v2: USING BREATH
# - v3: USING COUGH + BREATH
# task 3: YES_WITH_COUGH/NO_CLEAN_HISTORY_AND_LOW_PROBABILITY_WITH_ASTHMA_AND_COUGH_REPORTED
#   ( 23 user (54 sample) / 18 users (20 samples) in total)
# - v1: USING COUGH
# - v2: USING BREATH
# - v3: USING COUGH + BREATH
################################################################################
################################################################################
################################################################################

kdd_data_dir = data_dir * "KDD/"

kdd_has_subfolder_structure = ["asthmawebwithcough", "covidwebnocough", "covidwebwithcough", "healthywebnosymp", "healthywebwithcough"]

kdd_nonexisting_files = [
	"asthmawebwithcough/2020-04-07-18_49_21_155697/audio_file_breathe.wav_aug_amp1.wav",  # Missing
	"asthmawebwithcough/2020-04-07-18_49_21_155697/audio_file_cough.wav_aug_amp1.wav",    # Ignore to square up
	# These are unacceptable
	"healthywebnosymp/2020-04-07-22_28_51_823683/audio_file_breathe.wav",
	"healthywebnosymp/2020-04-07-22_28_51_823683/audio_file_cough.wav",
	"healthywebnosymp/2020-04-08-07_10_48_484176/audio_file_breathe.wav",
	"healthywebnosymp/2020-04-08-07_10_48_484176/audio_file_cough.wav",
	"healthywebnosymp/2020-04-16-20_22_54_947910/audio_file_breathe.wav",
	"healthywebnosymp/2020-04-16-20_22_54_947910/audio_file_cough.wav",
	"healthywebnosymp/2020-04-16-22_37_29_854249/audio_file_breathe.wav",
	"healthywebnosymp/2020-04-16-22_37_29_854249/audio_file_cough.wav",
	"healthywebnosymp/2020-04-17-18_08_20_326685/audio_file_breathe.wav",
	"healthywebnosymp/2020-04-17-18_08_20_326685/audio_file_cough.wav",
	"healthywebwithcough/2020-04-11-02_14_39_384740/audio_file_breathe.wav",
	"healthywebwithcough/2020-04-11-02_14_39_384740/audio_file_cough.wav",
	"healthywebwithcough/2020-04-07-22_29_27_236889/audio_file_breathe.wav",
]
kdd_files_to_ignore(n_version) = begin
	[
		kdd_nonexisting_files...,
		# Initially, preprocess functions (preprocess_wavs == [noise_gate!, normalize!])
		#  revealed that there was something wrong on these *breath* samples.
		#  They are indeed flawed, so I'll remove them, and also remove the corresponding coughs in n_version=3
		(
			# Flawed cough samples
			n_version in [1,3] ?
			[
				"healthywebnosymp/2020-04-07-16_07_32_459037/audio_file_cough.wav", # This is actually a breath sample
				"healthywebnosymp/2020-04-07-16_07_32_459037/audio_file_breathe.wav",
			] : []
		)...,
		(
			# Flawed breath samples
			n_version in [2,3] ?
			[
				"healthywebnosymp/2020-04-07-12_07_01_639904/audio_file_breathe.wav", # empty file
				"healthywebnosymp/2020-04-07-12_07_01_639904/audio_file_cough.wav",
				# asthmawebwithcough/2020-04-09-13_30_09_391043
				"healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav",                     # semi-empty file
				"healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav_aug_amp1.wav",        # semi-empty file
				"healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav_aug_amp2.wav",        # semi-empty file
				"healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav_aug_noise1.wav",      # semi-empty file
				"healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav_aug_noise2.wav",      # semi-empty file
				"healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav_aug_pitchspeed1.wav", # semi-empty file
				"healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav_aug_pitchspeed2.wav", # semi-empty file
				"healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav",                     
				"healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav_aug_amp1.wav",        
				"healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav_aug_amp2.wav",        
				"healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav_aug_noise1.wav",      
				"healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav_aug_noise2.wav",      
				"healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav_aug_pitchspeed1.wav", 
				"healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav_aug_pitchspeed2.wav", 
				# asthmawebwithcough/2020-04-07-20_46_20_561555
				"asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav",                      # semi-empty file
				"asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav_aug_amp1.wav",         # semi-empty file
				"asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav_aug_amp2.wav",         # semi-empty file
				"asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav_aug_noise1.wav",       # semi-empty file
				"asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav_aug_noise2.wav",       # semi-empty file
				"asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav_aug_pitchspeed1.wav",  # semi-empty file
				"asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav_aug_pitchspeed2.wav",  # semi-empty file
				"asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav",                      
				"asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav_aug_amp1.wav",         
				"asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav_aug_amp2.wav",         
				"asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav_aug_noise1.wav",       
				"asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav_aug_noise2.wav",       
				"asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav_aug_pitchspeed1.wav",  
				"asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav_aug_pitchspeed2.wav",  
				#
				"healthywebnosymp/2020-04-07-14_22_21_780211/audio_file_breathe.wav", # music in the background (what the heck?)
				"healthywebnosymp/2020-04-07-14_22_21_780211/audio_file_cough.wav",
			] : []
		)...
	]
end


kdd_has_augmentation_data = [
	"healthyandroidwithcough",
	"healthywebwithcough",
	"asthmaandroidwithcough",
	"asthmawebwithcough",
]
kdd_augmentation_file_suffixes = [
	"_aug_amp1.wav",
	"_aug_amp2.wav",
	"_aug_noise1.wav",
	"_aug_noise2.wav",
	"_aug_pitchspeed1.wav",
	"_aug_pitchspeed2.wav",
]

kdd_task_to_folders = [
	[
		["covidandroidnocough",  "covidandroidwithcough", "covidwebnocough", "covidwebwithcough"],
		["healthyandroidnosymp", "healthywebnosymp"],
		["YES", "NO_CLEAN_HISTORY_AND_LOW_PROBABILITY"]
	],
	[
		["covidandroidwithcough",   "covidwebwithcough"],
		["healthyandroidwithcough", "healthywebwithcough"],
		["YES_WITH_COUGH", "NO_CLEAN_HISTORY_AND_LOW_PROBABILITY"]
	],
	[
		["covidandroidwithcough",  "covidwebwithcough"],
		["asthmaandroidwithcough", "asthmawebwithcough"],
		["YES_WITH_COUGH", "NO_CLEAN_HISTORY_AND_LOW_PROBABILITY_WITH_ASTHMA_AND_COUGH_REPORTED"]
	],
]

# Obtain the list of all files considered throughout all tasks and versions
KDD_getSamplesList(; files_to_ignore = kdd_nonexisting_files, dir = kdd_data_dir, rel_path = false) = begin
	all_folders = Iterators.flatten([task_folders[1:2] for task_folders in kdd_task_to_folders]) |> collect
	Iterators.flatten(
		ThreadsX.collect([KDD_getSamplesList(folder, n_version, true; files_to_ignore = files_to_ignore, dir = dir, rel_path = rel_path, separate_base_n_aug = false)
				for folder in all_folders
					for n_version in 1:2])
	) |> collect |> Vector{String}
end

# Obtain the list files considered for a given task and version
KDD_getSamplesList(n_task::Integer, n_version::Integer, use_augmentation_data; files_to_ignore = kdd_nonexisting_files, dir = kdd_data_dir, rel_path = false) = begin
	global kdd_task_to_folders
	folders_Y, folders_N, class_labels = kdd_task_to_folders[n_task]

	vcat(
		KDD_getSamplesList(folders_Y, n_version, use_augmentation_data; files_to_ignore = files_to_ignore, dir = dir, rel_path = rel_path, separate_base_n_aug = false),
		KDD_getSamplesList(folders_N, n_version, use_augmentation_data; files_to_ignore = files_to_ignore, dir = dir, rel_path = rel_path, separate_base_n_aug = false),
	)
end

# Obtain the list files inside a given folder (note: n_version is needed for computing the files to ignore)
KDD_getSamplesList(folders::AbstractVector{<:AbstractString}, n_version::Integer, use_augmentation_data; files_to_ignore = kdd_nonexisting_files, dir = kdd_data_dir, rel_path = false, separate_base_n_aug = true) = begin
	global kdd_has_augmentation_data
	global kdd_has_subfolder_structure
	global kdd_augmentation_file_suffixes
	
	subfolder, file_suffix, file_prefix = 
		if n_version == 1
			("cough","cough","cough_")
		elseif n_version == 2
			("breath","breathe","breaths_")
		else
			error("Unknown n_version: $(n_version)")
		end

	files_map = JSON.parsefile(kdd_data_dir * "files.json")
	# println(folders)
	timeseries     = [] # Vector{Vector{Array{Float64, 2}}}[]
	aug_timeseries = []
	for folder in folders

		# Groups of filepaths
		base_file_paths = files_map[folder]
		
		# Augmented data (if present)
		aug_file_paths =
			if use_augmentation_data && folder in kdd_has_augmentation_data && ! (folder in kdd_has_subfolder_structure)
				map((file_path_arr)->["$(file_path)$(suff)" for suff in filter(x->x != "", kdd_augmentation_file_suffixes) for file_path in file_path_arr], base_file_paths)
			else
				[]
			end

		all_file_paths = vcat(collect(zip(Iterators.repeated(false),base_file_paths)), collect(zip(Iterators.repeated(true),aug_file_paths)))

		cur_folder_timeseries     = Dict{Integer,Vector{String}}()
		cur_folder_aug_timeseries = Dict{Integer,Vector{String}}()

		# println()
		# println()
		# println("base_file_paths:     $(base_file_paths)\t$(length(base_file_paths))\t$(map(length, base_file_paths))")
		# println("aug_file_paths: $(aug_file_paths)\t$(length(aug_file_paths))\t$(map(length, aug_file_paths))")

		# collect is necessary because the threads macro only supports arrays
		# https://stackoverflow.com/questions/57633477/multi-threading-julia-shows-error-with-enumerate-iterator
		# Threads.@threads (TODO right now can't because of GLPK when using augmentation data)
		for (i_samples, (is_aug,these_samples)) in collect(enumerate(all_file_paths))
			
			# Correct folder/subfolder structure
			these_samples =
				if folder in kdd_has_subfolder_structure
					map((subfoldname)->"$folder/$subfoldname/audio_file_$(file_suffix).wav", these_samples)
				else
					filter!((filepath)->startswith(filepath,file_prefix), these_samples)
					map((filepath)->"$folder/$subfolder/$filepath", these_samples)
				end
				
			# Derive augmentation data or just take this as augmentation data
			base_samples, aug_samples =
				if is_aug
					[], these_samples
				else
					base_samples = these_samples
					aug_samples =
						if use_augmentation_data && folder in kdd_has_augmentation_data && folder in kdd_has_subfolder_structure
							["$(file_path)$(suff)" for suff in kdd_augmentation_file_suffixes for file_path in base_samples]
						else
							[]
						end
					base_samples, aug_samples
				end

			# println(base_samples)

			# Filter out files that are known to be missing
			filter!((filepath)->! (filepath in files_to_ignore), base_samples)
			filter!((filepath)->! (filepath in files_to_ignore), aug_samples)

			# Interpret the filepath as a prefix
			base_samples = [sort(glob("$(filepath)*", dir)) for filepath in base_samples] |> Iterators.flatten |> collect
			aug_samples  = [sort(glob("$(filepath)*", dir)) for filepath in aug_samples]  |> Iterators.flatten |> collect
			
			removeprefix(s::AbstractString, prefix::AbstractString) = startswith(s, prefix) ? s[length(prefix)+1:end] : s
			removesuffix(s::AbstractString, suffix::AbstractString) = endswith(s, prefix) ? s[end-length(prefix):end] : s

			if rel_path
				base_samples = [removeprefix(filepath, dir) for filepath in base_samples]
				aug_samples  = [removeprefix(filepath, dir) for filepath in aug_samples]
			end
			# now we have base_samples & aug_samples
			# println("$(length(base_samples)) base_samples: $(base_samples)")
			# println("$(length(aug_samples)) aug_samples: $(aug_samples)")

			cur_folder_timeseries[i_samples]     = base_samples
			cur_folder_aug_timeseries[i_samples] = aug_samples
			# break
		end
		cur_folder_timeseries     = [cur_folder_timeseries[i]     for i in 1:length(cur_folder_timeseries)]
		cur_folder_aug_timeseries = [cur_folder_aug_timeseries[i] for i in 1:length(cur_folder_aug_timeseries)]

		append!(timeseries,     [j for i in cur_folder_timeseries     for j in i])
		append!(aug_timeseries, [j for i in cur_folder_aug_timeseries for j in i])
	end
	# timeseries[1:5]
	# println(length(timeseries))
	# println(length(aug_timeseries))
	# println(map(length, timeseries))
	# println(map(length, aug_timeseries))
	if separate_base_n_aug
		timeseries, aug_timeseries
	else
		vcat(timeseries, aug_timeseries)
	end
end

function KDDDataset((n_task,n_version),
		audio_kwargs;
		ma_size = 1,
		ma_step = 1,
		max_points = -1,
		use_full_mfcc = false,
		preprocess_wavs = [],
		use_augmentation_data = true,
		return_filepaths = false,
		force_monolithic_dataset = false,
		ignore_samples_with_sr_less_than = -Inf,
		dir = kdd_data_dir,
	)
	@assert n_task    in [1,2,3] "KDDDataset: invalid n_task:    {$n_task}"
	@assert n_version in [1,2,3,"c","b","c+b"] "KDDDataset: invalid n_version: {$n_version}"
	
	global kdd_task_to_folders

	n_version =
		if n_version == "c"         1
		elseif n_version == "b"     2
		elseif n_version == "c+b"   3
		else                        n_version
		end

	# TASK

	folders_Y, folders_N, class_labels = kdd_task_to_folders[n_task]

	# VERSION

	files_to_ignore = kdd_files_to_ignore(n_version)

	n_version =
		if n_version == 1
			(1,)
		elseif n_version == 2
			(2,)
		elseif n_version == 3
			(1, 2)
		else
			error("Unknown n_version: $(n_version)")
		end

	##############################################################################
	##############################################################################
	##############################################################################
	
	loadSample(filepath, ts_lengths_dict, ts_with_ma_lengths_dict, ts_cut_lengths_dict) = begin
		# println(filepath)
		ts = wav2stft_time_series(filepath, audio_kwargs; preprocess_sample = preprocess_wavs, use_full_mfcc = use_full_mfcc, ignore_samples_with_sr_less_than = ignore_samples_with_sr_less_than)
		
		# Some breath samples are empty or semi-empty. As such, for tasks 2 and 3 we need to ignore them, and also ignore the paired cough samples
		if isnothing(ts)
			println("Warning: could not read file \"$(filepath)\"")
			nothing
		else
			# Ignore instances with NaN (careful! this may leave with just a few instances)
			# if any(isnan.(ts))
			# 	@warn "Instance with NaN values was ignored"
			# 	continue
			# end

			# println("ts_length:	$(size(ts,1))	$(filepath)")
			original_length = size(ts,1)

			# Drop first point
			ts = @views ts[2:end,:]
			ts = moving_average(ts, ma_size, ma_step)

			if size(ts,1) == 0
				println("Warning! Moving average ends up killing sample \"$(filepath)\" ($(original_length) points, ma_size = $(ma_size), ma_step = $(ma_step))")
				nothing
			end

			# println("ts_with_ma_length:	$(size(ts,1))	$(filepath)")
			after_ma_length = size(ts,1)

			# println(size(ts))
			if max_points != -1 && size(ts,1)>max_points
				ts = ts[1:max_points,:]
			end

			# println("ts_cut_length:	$(size(ts,1))	$(filepath)")
			after_cut_length = size(ts,1)

			ts_lengths_dict[filepath] = original_length
			ts_with_ma_lengths_dict[filepath] = after_ma_length
			ts_cut_lengths_dict[filepath] = after_cut_length

			# println(size(ts))
			# readline()
			ts
		end
	end

	compute_X(max_timepoints, n_unique_freqs, timeseries, expected_length) = begin
		@assert expected_length == length(timeseries)
		X = zeros((max_timepoints, n_unique_freqs, length(timeseries)))
		for (i,ts) in enumerate(timeseries)
			# println(size(ts))
			X[1:size(ts, 1),:,i] = ts
		end
		X
	end
	
	##############################################################################
	##############################################################################
	##############################################################################

	getTimeSeries(n_subver, return_filepaths) = begin
		pos_filepaths, pos_aug_filepaths = KDD_getSamplesList(folders_Y, n_subver, use_augmentation_data; files_to_ignore = files_to_ignore, dir = dir)
		neg_filepaths, neg_aug_filepaths = KDD_getSamplesList(folders_N, n_subver, use_augmentation_data; files_to_ignore = files_to_ignore, dir = dir)
		
		ts_lengths_dict         = Dict{String,Integer}()
		ts_with_ma_lengths_dict = Dict{String,Integer}()
		ts_cut_lengths_dict     = Dict{String,Integer}()

		# Threads.@threads (TODO right now can't because of GLPK when using augmentation data)
		pos      = Dict(ThreadsX.collect([f => loadSample(f, ts_lengths_dict, ts_with_ma_lengths_dict, ts_cut_lengths_dict) for f in pos_filepaths]))
		pos_aug  = Dict(ThreadsX.collect([f => loadSample(f, ts_lengths_dict, ts_with_ma_lengths_dict, ts_cut_lengths_dict) for f in pos_aug_filepaths]))
		neg      = Dict(ThreadsX.collect([f => loadSample(f, ts_lengths_dict, ts_with_ma_lengths_dict, ts_cut_lengths_dict) for f in neg_filepaths]))
		neg_aug  = Dict(ThreadsX.collect([f => loadSample(f, ts_lengths_dict, ts_with_ma_lengths_dict, ts_cut_lengths_dict) for f in neg_aug_filepaths]))

		# Some samples are nothing: filter them out and sort filepaths
		pos_filepaths     = sort(collect(keys(filter( (x)->!isnothing(x[2]), pos))))
		pos_aug_filepaths = sort(collect(keys(filter( (x)->!isnothing(x[2]), pos_aug))))
		neg_filepaths     = sort(collect(keys(filter( (x)->!isnothing(x[2]), neg))))
		neg_aug_filepaths = sort(collect(keys(filter( (x)->!isnothing(x[2]), neg_aug))))

		# 
		# sort(collect(keys(filter( (x)->!isnothing(x[2]), Dict([1=>2, 2=>nothing])))))

		# Finally derive samples
		pos      = [pos[s]     for s in pos_filepaths]
		pos_aug  = [pos_aug[s] for s in pos_aug_filepaths]
		neg      = [neg[s]     for s in neg_filepaths]
		neg_aug  = [neg_aug[s] for s in neg_aug_filepaths]

		println("Base: POS={$(length(pos))}, NEG={$(length(neg))}\tAug: POS={$(length(pos_aug))}, NEG={$(length(neg_aug))}")

		# Stratify
		# timeseries = vec(hcat(pos,neg)')
		# Y = vec(hcat(ones(Int,length(pos)),zeros(Int,length(neg)))')

		# println(typeof(pos))
		timeseries     = [pos...,     neg...]
		timeseries_aug = [pos_aug..., neg_aug...]
		# println(typeof(timeseries))
		# println(typeof([[p for p in pos]...,     [n for n in neg]...]))
		# println(typeof([[p for p in pos_aug]..., [n for n in neg_aug]...]))
		# println(size(timeseries))
		# println(size(timeseries_aug))
		# print(size(timeseries[1]))
		Y     = String[fill(class_labels[1], length(pos))...,     fill(class_labels[2], length(neg))...]
		Y_aug = String[fill(class_labels[1], length(pos_aug))..., fill(class_labels[2], length(neg_aug))...]
		# print(size(Y))

		ts_lengths          = values(ts_lengths_dict)
		ts_with_ma_lengths  = values(ts_with_ma_lengths_dict)
		ts_cut_lengths      = values(ts_cut_lengths_dict)

		@assert length(ts_lengths) == length(ts_with_ma_lengths) == length(ts_cut_lengths) "$(length(ts_lengths)), $(length(ts_with_ma_lengths)), $(length(ts_cut_lengths))"

		println("$(length(ts_lengths)) samples")
		println("ts_lengths         = (max = $(StatsBase.maximum(ts_lengths)        ), min = $(StatsBase.minimum(ts_lengths)        ), mean = $(StatsBase.mean(ts_lengths)        ), std = $(StatsBase.std(ts_lengths)))")
		println("ts_with_ma_lengths = (max = $(StatsBase.maximum(ts_with_ma_lengths)), min = $(StatsBase.minimum(ts_with_ma_lengths)), mean = $(StatsBase.mean(ts_with_ma_lengths)), std = $(StatsBase.std(ts_with_ma_lengths)))")
		println("ts_cut_lengths     = (max = $(StatsBase.maximum(ts_cut_lengths)    ), min = $(StatsBase.minimum(ts_cut_lengths)    ), mean = $(StatsBase.mean(ts_cut_lengths)    ), std = $(StatsBase.std(ts_cut_lengths)))")

		# println([size(ts, 1) for ts in timeseries])
		max_timepoints = maximum(size(ts, 1) for ts in [timeseries..., timeseries_aug...])
		n_unique_freqs = unique(size(ts,  2) for ts in [timeseries..., timeseries_aug...])
		@assert length(n_unique_freqs) == 1 "KDDDataset: length(n_unique_freqs) != 1: {$n_unique_freqs} != 1"
		n_unique_freqs = n_unique_freqs[1]
		
		class_counts     = (length(pos),     length(neg))
		class_counts_aug = (length(pos_aug), length(neg_aug))
		println("Class counts: $(class_counts); Aug class counts: $(class_counts_aug); # points: $(max_timepoints)")

		X     = compute_X(max_timepoints, n_unique_freqs, timeseries,     sum(class_counts))
		X_aug = compute_X(max_timepoints, n_unique_freqs, timeseries_aug, sum(class_counts_aug))
		
		@assert n_samples(X)     == length(Y)     "$(n_samples(X))     != $(length(Y))"
		@assert n_samples(X_aug) == length(Y_aug) "$(n_samples(X_aug)) != $(length(Y_aug))"

		if return_filepaths
			timeseries_filepaths     = [pos_filepaths...,     neg_filepaths...]
			timeseries_aug_filepaths = [pos_aug_filepaths..., neg_aug_filepaths...]
			@assert n_samples(X)     == length(timeseries_filepaths)     "$(n_samples(X))     != $(length(timeseries_filepaths))"
			@assert n_samples(X_aug) == length(timeseries_aug_filepaths) "$(n_samples(X_aug)) != $(length(timeseries_aug_filepaths))"
			# ((X,Y,timeseries_filepaths), length(pos), length(neg))
			(
				train_n_test  = ((X,     Y    , timeseries_filepaths),     class_counts),
				only_training = ((X_aug, Y_aug, timeseries_aug_filepaths), class_counts_aug),
			)
		else
			# ((X,Y), length(pos), length(neg))
			(
				train_n_test  = ((X,     Y),     class_counts),
				only_training = ((X_aug, Y_aug), class_counts_aug),
			)
		end
	end

	datasets = Vector(undef, length(n_version))
	n_pos    = Vector(undef, length(n_version))
	n_neg    = Vector(undef, length(n_version))
	
	datasets_aug = Vector(undef, length(n_version))
	n_pos_aug    = Vector(undef, length(n_version))
	n_neg_aug    = Vector(undef, length(n_version))

	for (i, n_subver) in enumerate(n_version)
		cur_frame_dataset = getTimeSeries(n_subver, return_filepaths)
		
		datasets[i],     (n_pos[i],     n_neg[i])     = deepcopy(cur_frame_dataset.train_n_test)
		datasets_aug[i], (n_pos_aug[i], n_neg_aug[i]) = deepcopy(cur_frame_dataset.only_training)

		@assert datasets[1][2] == datasets[i][2] "mismatching classes:\n\tdatasets[1][2] = $(datasets[1][2])\n\tdatasets[i][2] = $(datasets[i][2])"
		@assert n_pos[1] == n_pos[i] "n_pos mismatching class count across frames:\n\tn_pos[1] = $(n_pos[1]) != n_pos[i] = $(n_pos[i]))"
		@assert n_neg[1] == n_neg[i] "n_neg mismatching class count across frames:\n\tn_neg[1] = $(n_neg[1]) != n_neg[i] = $(n_neg[i]))"

		@assert datasets_aug[1][2] == datasets_aug[i][2] "mismatching classes:\n\tdatasets_aug[1][2] = $(datasets_aug[1][2])\n\tdatasets_aug[i][2] = $(datasets_aug[i][2])"
		@assert n_pos_aug[1] == n_pos_aug[i] "n_pos_aug mismatching class count across frames:\n\tn_pos_aug[1] = $(n_pos_aug[1]) != n_pos_aug[i] = $(n_pos_aug[i]))"
		@assert n_neg_aug[1] == n_neg_aug[i] "n_neg_aug mismatching class count across frames:\n\tn_neg_aug[1] = $(n_neg_aug[1]) != n_neg_aug[i] = $(n_neg_aug[i]))"
	end

	d =
		if return_filepaths
			# ((getindex.(datasets, 1), datasets[1][2], getindex.(datasets, 3)), (n_pos[1], n_neg[1]))
			NamedTuple{(:train_n_test,:only_training,)}((;
				train_n_test  = ((getindex.(datasets,     1), datasets[1][2],     getindex.(datasets,     3)), (n_pos[1],     n_neg[1])),
				only_training = ((getindex.(datasets_aug, 1), datasets_aug[1][2], getindex.(datasets_aug, 3)), (n_pos_aug[1], n_neg_aug[1])),
			))
		else
			# ((getindex.(datasets, 1),datasets[1][2]), (n_pos[1], n_neg[1]))
			NamedTuple{(:train_n_test,:only_training,)}((;
				train_n_test  = ((getindex.(datasets,     1), datasets[1][2]),     (n_pos[1],     n_neg[1])),
				only_training = ((getindex.(datasets_aug, 1), datasets_aug[1][2]), (n_pos_aug[1], n_neg_aug[1])),
			))
		end

	# println()
	# println("train_n_test")
	# println(typeof(d.train_n_test[1]))
	# println(size.(d.train_n_test[1][1]))
	# println(size(d.train_n_test[1][2]))
	# println(size.(d.train_n_test[1][3]))

	# println()
	# println("only_training")
	# println(typeof(d.only_training[1]))
	# println(size.(d.only_training[1][1]))
	# println(size(d.only_training[1][2]))
	# println(size.(d.only_training[1][3]))

	if force_monolithic_dataset == true
		error("TODO account for class ordering when using concat_labeled_datasets")
		d = (concat_labeled_datasets(d.train_n_test[1], d.only_training[1]), (d.train_n_test[2] .+ d.only_training[2]))
	elseif force_monolithic_dataset == :train_n_test
		d = d.train_n_test
	end

	d
end
################################################################################
################################################################################
################################################################################
