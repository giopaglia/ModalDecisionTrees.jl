using JSON
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

function KDDDataset_not_stratified((n_task,n_version),
		audio_kwargs;
		ma_size = 1,
		ma_step = 1,
		max_points = -1,
		use_full_mfcc = false,
		preprocess_wavs = [],
		use_augmentation_data = true,
		return_filepaths = false,
		force_monolithic_dataset = false,
	)
	@assert n_task    in [1,2,3] "KDDDataset: invalid n_task:    {$n_task}"
	@assert n_version in [1,2,3,"c","b","c+b"] "KDDDataset: invalid n_version: {$n_version}"
	
	n_version =
		if n_version == "c"         1
		elseif n_version == "b"     2
		elseif n_version == "c+b"   3
		else                        n_version
		end

	kdd_data_dir = data_dir * "KDD/"
	
	task_to_folders = [
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

	has_subfolder_structure = ["asthmawebwithcough", "covidwebnocough", "covidwebwithcough", "healthywebnosymp", "healthywebwithcough"]

	files_to_ignore = [
		# Missing
		"asthmawebwithcough/2020-04-07-18_49_21_155697/audio_file_breathe.wav_aug_amp1.wav",
		# Ignore to square up
		"asthmawebwithcough/2020-04-07-18_49_21_155697/audio_file_cough.wav_aug_amp1.wav"
	]

	has_augmentation_data = [
		"healthyandroidwithcough",
		"healthywebwithcough",
		"asthmaandroidwithcough",
		"asthmawebwithcough",
	]
	augmentation_file_suffixes = [
		"_aug_amp1.wav",
		"_aug_amp2.wav",
		"_aug_noise1.wav",
		"_aug_noise2.wav",
		"_aug_pitchspeed1.wav",
		"_aug_pitchspeed2.wav",
	]

	cough = ("cough","cough","cough_")
	breath = ("breath","breathe","breaths_")
	dir_infos =
		if n_version == 1
			(cough,)
		elseif n_version == 2
			(breath,)
		elseif n_version == 3
			(cough, breath)
		else
			error("Unknown n_version: $(n_version)")
		end

	folders_Y, folders_N, class_labels = task_to_folders[n_task]

	##############################################################################
	##############################################################################
	##############################################################################
	# n_samples = 0
	
	loadSamples(samples_filepaths, return_filepaths) = begin
		cur_file_timeseries =
			if return_filepaths
				Dict{Integer,String}()
			else
				Dict{Integer,Array{Float64, 2}}()
			end
		# Threads.@threads (TODO right now can't because of GLPK when using augmentation data)
		for (i_filename, filename) in collect(enumerate(samples_filepaths))
			# println(filename)
			filepath = kdd_data_dir * "$filename"
			ts = wav2stft_time_series(filepath, audio_kwargs; preprocess_sample = preprocess_wavs, use_full_mfcc = use_full_mfcc)
			
			# Some breath samples are empty or semi-empty. As such, for tasks 2 and 3 we need to ignore them, and also ignore the paired cough samples
			if !isnothing(ts) && ! (
                                          (n_version == 2 ||
                                          n_version == 3) &&
                                          # Initially, preprocess functions revealed that there was something wrong on these breath samples.
                                          # But they are indeed flawed, so no need to consider them. ever.
                                          # preprocess_wavs == [noise_gate!, normalize!] &&
                                          any(endswith.(filepath,
                                          [
                                  "healthywebnosymp/2020-04-07-12_07_01_639904/audio_file_breathe.wav", # empty file
                                  "healthywebnosymp/2020-04-07-12_07_01_639904/audio_file_cough.wav",

                                  # asthmawebwithcough/2020-04-09-13_30_09_391043
                                  
                                  # "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav", # empty file
                                  # "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav",
                                  # "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav_aug_pitchspeed1.wav", # semi-empty file
                                  # "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav_aug_pitchspeed1.wav",
                                  # "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav_aug_pitchspeed2.wav", # semi-empty file
                                  # "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav_aug_pitchspeed2.wav",
                                  
                                  "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav",                     # semi-empty file
                                  "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav_aug_amp1.wav",        # semi-empty file
                                  "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav_aug_amp2.wav",        # semi-empty file
                                  "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav_aug_noise1.wav",      # semi-empty file
                                  "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav_aug_noise2.wav",      # semi-empty file
                                  "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav_aug_pitchspeed1.wav", # semi-empty file
                                  "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_breathe.wav_aug_pitchspeed2.wav", # semi-empty file
                                  "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav",                     # semi-empty file
                                  "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav_aug_amp1.wav",        # semi-empty file
                                  "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav_aug_amp2.wav",        # semi-empty file
                                  "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav_aug_noise1.wav",      # semi-empty file
                                  "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav_aug_noise2.wav",      # semi-empty file
                                  "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav_aug_pitchspeed1.wav", # semi-empty file
                                  "healthywebwithcough/2020-04-09-13_30_09_391043/audio_file_cough.wav_aug_pitchspeed2.wav", # semi-empty file

                                  # asthmawebwithcough/2020-04-07-20_46_20_561555

                                  # "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav_aug_pitchspeed1.wav", # semi-empty file
                                  # "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav_aug_pitchspeed1.wav",
                                  # "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav_aug_pitchspeed2.wav", # semi-empty file
                                  # "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav_aug_pitchspeed2.wav",

                                  "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav",                      # semi-empty file
                                  "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav_aug_amp1.wav",         # semi-empty file
                                  "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav_aug_amp2.wav",         # semi-empty file
                                  "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav_aug_noise1.wav",       # semi-empty file
                                  "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav_aug_noise2.wav",       # semi-empty file
                                  "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav_aug_pitchspeed1.wav",  # semi-empty file
                                  "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_breathe.wav_aug_pitchspeed2.wav",  # semi-empty file
                                  "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav",                      # semi-empty file
                                  "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav_aug_amp1.wav",         # semi-empty file
                                  "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav_aug_amp2.wav",         # semi-empty file
                                  "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav_aug_noise1.wav",       # semi-empty file
                                  "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav_aug_noise2.wav",       # semi-empty file
                                  "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav_aug_pitchspeed1.wav",  # semi-empty file
                                  "asthmawebwithcough/2020-04-07-20_46_20_561555/audio_file_cough.wav_aug_pitchspeed2.wav",  # semi-empty file

                                  ])))
				cur_file_timeseries[i_filename] =
					if return_filepaths
						filepath
					else
						# Ignore instances with NaN (careful! this may leave with just a few instances)
						# if any(isnan.(ts))
						# 	@warn "Instance with NaN values was ignored"
						# 	continue
						# end
						# Drop first point
						ts = @views ts[2:end,:]
						ts = moving_average(ts, ma_size, ma_step)
						# println(size(ts))
						if max_points != -1 && size(ts,1)>max_points
							ts = ts[1:max_points,:]
						end
						# println(size(ts))
						# readline()
						# println(size(wav2stft_time_series(filepath, audio_kwargs)))
						ts
					end
				# n_samples += 1
			end
		end
		[cur_file_timeseries[i] for i in 1:length(cur_file_timeseries)]
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
	
	readFiles_no_aug(folders, subfolder, file_suffix, file_prefix, return_filepaths) = begin
		files_map = JSON.parsefile(kdd_data_dir * "files.json")
		# println(folders)
		# timeseries = Vector{Vector{Array{Float64, 2}}}[]
		timeseries = []
		for folder in folders

			# Groups of filepaths
			file_paths = files_map[folder]
			
			# Correct with augmentated data (if present)
			file_paths = 
				if use_augmentation_data && folder in has_augmentation_data && ! (folder in has_subfolder_structure)
					map((file_path_arr)->["$(file_path)$(suff)" for suff in augmentation_file_suffixes for file_path in file_path_arr], file_paths)
				else
					file_paths
				end

			cur_folder_timeseries = return_filepaths ? Vector{Vector{String}}(undef, length(file_paths)) : Vector{Vector{Array{Float64, 2}}}(undef, length(file_paths))

			# collect is necessary because the threads macro only supports arrays
			# https://stackoverflow.com/questions/57633477/multi-threading-julia-shows-error-with-enumerate-iterator
			# Threads.@threads (TODO right now can't because of GLPK when using augmentation data)
			for (i_samples, samples) in collect(enumerate(file_paths))

				# Correct with augmentated data (if present)
				samples = 
					if folder in has_subfolder_structure
						samples = map((subfoldname)->"$folder/$subfoldname/audio_file_$(file_suffix).wav", samples)
						if use_augmentation_data && folder in has_augmentation_data
							samples = ["$(file_path)$(suff)" for suff in augmentation_file_suffixes for file_path in samples]
						end
						samples
					else
						# println(samples)
						filter!((filename)->startswith(filename,file_prefix), samples)
						map((filename)->"$folder/$subfolder/$filename", samples)
					end
				# println(samples)

				# Filter out files that are known to be missing
				filter!((filename)->! (filename in files_to_ignore), samples)

				cur_folder_timeseries[i_samples] = loadSamples(samples, return_filepaths)
				# break
			end
			append!(timeseries, [j for i in cur_folder_timeseries for j in i])
		end
		# timeseries[1:5]
		timeseries
	end

	getTimeSeries_no_aug(folders::NTuple{N,AbstractVector{String}}, dir_infos::NTuple{3,String}, return_filepaths) where N = begin
		subfolder,file_suffix,file_prefix = dir_infos
		pos = readFiles_no_aug(folders[1], subfolder, file_suffix, file_prefix, false)
		neg = readFiles_no_aug(folders[2], subfolder, file_suffix, file_prefix, false)

		if return_filepaths
			pos_filepaths = readFiles_no_aug(folders[1], subfolder, file_suffix, file_prefix, true)
			@assert length(pos) == length(pos_filepaths) "mismatch length(pos) == length(pos_filepaths) : $(length(pos)) != $(length(pos_filepaths))"
			neg_filepaths = readFiles_no_aug(folders[2], subfolder, file_suffix, file_prefix, true)
			@assert length(neg) == length(neg_filepaths) "mismatch length(neg) == length(neg_filepaths) : $(length(neg)) != $(length(neg_filepaths))"
			timeseries_filepaths = [pos_filepaths..., neg_filepaths...]
		end

		println("POS={$(length(pos))}, NEG={$(length(neg))}")
		#n_per_class = min(length(pos), length(neg))

		#println("Balanced -> {$n_per_class}+{$n_per_class}")

		# Stratify
		# timeseries = vec(hcat(pos,neg)')
		# Y = vec(hcat(ones(Int,length(pos)),zeros(Int,length(neg)))')

		# print(size(pos))
		# print(size(neg))
		timeseries = [pos..., neg...]
		# print(size(timeseries))
		# print(size(timeseries[1]))
		# Y = [ones(Int, length(pos))..., zeros(Int, length(neg))...]
		# Y = [zeros(Int, length(pos))..., ones(Int, length(neg))...]
		Y = [fill(class_labels[1], length(pos))..., fill(class_labels[2], length(neg))...]
		# print(size(Y))

		# println([size(ts, 1) for ts in timeseries])
		max_timepoints = maximum(size(ts, 1) for ts in timeseries)
		println("max_timepoints: $(max_timepoints)")
		n_unique_freqs = unique(size(ts, 2) for ts in timeseries)
		@assert length(n_unique_freqs) == 1 "KDDDataset: length(n_unique_freqs) != 1: {$n_unique_freqs} != 1"
		n_unique_freqs = n_unique_freqs[1]
		
		X = compute_X(max_timepoints, n_unique_freqs, timeseries, length(Y))

		if return_filepaths
			((X, Y, timeseries_filepaths), (length(pos), length(neg)))
		else
			((X, Y), (length(pos), length(neg)))
		end
	end

	readFiles(folders, subfolder, file_suffix, file_prefix, return_filepaths) = begin
		files_map = JSON.parsefile(kdd_data_dir * "files.json")
		# println(folders)
		timeseries     = [] # Vector{Vector{Array{Float64, 2}}}[]
		aug_timeseries = []
		for folder in folders

			# Groups of filepaths
			file_paths = files_map[folder]
			
			# Augmentated data (if present)
			aug_file_paths =
				if use_augmentation_data && folder in has_augmentation_data && ! (folder in has_subfolder_structure)
					map((file_path_arr)->["$(file_path)$(suff)" for suff in filter(x->x != "", augmentation_file_suffixes) for file_path in file_path_arr], file_paths)
				else
					[]
				end

			all_file_paths = vcat(collect(zip(Iterators.repeated(false),file_paths)), collect(zip(Iterators.repeated(true),aug_file_paths)))

			cur_folder_timeseries     = return_filepaths ? Dict{Integer,Vector{String}}() : Dict{Integer,Vector{Array{Float64, 2}}}()
			cur_folder_aug_timeseries = return_filepaths ? Dict{Integer,Vector{String}}() : Dict{Integer,Vector{Array{Float64, 2}}}()

			# println()
			# println()
			# println("file_paths:     $(file_paths)\t$(length(file_paths))\t$(map(length, file_paths))")
			# println("aug_file_paths: $(aug_file_paths)\t$(length(aug_file_paths))\t$(map(length, aug_file_paths))")

			# collect is necessary because the threads macro only supports arrays
			# https://stackoverflow.com/questions/57633477/multi-threading-julia-shows-error-with-enumerate-iterator
			# Threads.@threads (TODO right now can't because of GLPK when using augmentation data)
			for (i_samples, (is_aug,these_samples)) in collect(enumerate(all_file_paths))
				
				# Correct folder/subfolder structure
				these_samples =
					if folder in has_subfolder_structure
						map((subfoldname)->"$folder/$subfoldname/audio_file_$(file_suffix).wav", these_samples)
					else
						filter!((filename)->startswith(filename,file_prefix), these_samples)
						map((filename)->"$folder/$subfolder/$filename", these_samples)
					end
					
				# Derive augmentation data or just take this as augmentation data
				samples, aug_samples =
					if is_aug
						[], these_samples
					else
						samples = these_samples
						aug_samples = if use_augmentation_data && folder in has_augmentation_data && folder in has_subfolder_structure
							["$(file_path)$(suff)" for suff in filter(x->x != "", augmentation_file_suffixes) for file_path in samples]
						else
							[]
						end
						samples, aug_samples
					end

				# println(samples)

				# Filter out files that are known to be missing
				filter!((filename)->! (filename in files_to_ignore), samples)
				filter!((filename)->! (filename in files_to_ignore), aug_samples)

				# now we have samples & aug_samples
				# println("$(length(samples)) samples: $(samples)")
				# println("$(length(aug_samples)) aug_samples: $(aug_samples)")

				cur_folder_timeseries[i_samples]     = loadSamples(samples,     return_filepaths)
				cur_folder_aug_timeseries[i_samples] = loadSamples(aug_samples, return_filepaths)
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
		timeseries, aug_timeseries
	end

	getTimeSeries(folders::NTuple{N,AbstractVector{String}}, dir_infos::NTuple{3,String}, return_filepaths) where N = begin
		subfolder,file_suffix,file_prefix = dir_infos
		pos, pos_aug = readFiles(folders[1], subfolder, file_suffix, file_prefix, false)
		neg, neg_aug = readFiles(folders[2], subfolder, file_suffix, file_prefix, false)

		if return_filepaths
			pos_filepaths, pos_aug_filepaths = readFiles(folders[1], subfolder, file_suffix, file_prefix, true)
			@assert length(pos)     == length(pos_filepaths)     "mismatch length(pos)     == length(pos_filepaths)     : $(length(pos))     != $(length(pos_filepaths))"
			@assert length(pos_aug) == length(pos_aug_filepaths) "mismatch length(pos_aug) == length(pos_aug_filepaths) : $(length(pos_aug)) != $(length(pos_aug_filepaths))"
			neg_filepaths, neg_aug_filepaths = readFiles(folders[2], subfolder, file_suffix, file_prefix, true)
			@assert length(neg)     == length(neg_filepaths)     "mismatch length(neg)     == length(neg_filepaths)     : $(length(neg))     != $(length(neg_filepaths))"
			@assert length(neg_aug) == length(neg_aug_filepaths) "mismatch length(neg_aug) == length(neg_aug_filepaths) : $(length(neg_aug)) != $(length(neg_aug_filepaths))"
			timeseries_filepaths     = [pos_filepaths...,     neg_filepaths...]
			timeseries_aug_filepaths = [pos_aug_filepaths..., neg_aug_filepaths...]
		end

		println("POS={$(length(pos))}, NEG={$(length(neg))}\tAug: POS={$(length(pos_aug))}, NEG={$(length(neg_aug))}")

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
		# print(size(timeseries))
		# print(size(timeseries[1]))
		# Y = [ones(Int, length(pos))..., zeros(Int, length(neg))...]
		# Y = [zeros(Int, length(pos))..., ones(Int, length(neg))...]
		Y     = String[fill(class_labels[1], length(pos))...,     fill(class_labels[2], length(neg))...]
		Y_aug = String[fill(class_labels[1], length(pos_aug))..., fill(class_labels[2], length(neg_aug))...]
		# print(size(Y))

		# println([size(ts, 1) for ts in timeseries])
		max_timepoints = maximum(size(ts, 1) for ts in [timeseries..., timeseries_aug...])
		n_unique_freqs = unique(size(ts,  2) for ts in [timeseries..., timeseries_aug...])
		@assert length(n_unique_freqs) == 1 "KDDDataset: length(n_unique_freqs) != 1: {$n_unique_freqs} != 1"
		n_unique_freqs = n_unique_freqs[1]
		
		# println("ts_lengths         = (max = $(StatsBase.maximum(ts_lengths)        ), min = $(StatsBase.minimum(ts_lengths)        ), mean = $(StatsBase.mean(ts_lengths)        ), std = $(StatsBase.std(ts_lengths)))")
		# println("ts_with_ma_lengths = (max = $(StatsBase.maximum(ts_with_ma_lengths)), min = $(StatsBase.minimum(ts_with_ma_lengths)), mean = $(StatsBase.mean(ts_with_ma_lengths)), std = $(StatsBase.std(ts_with_ma_lengths)))")
		# println("ts_cut_lengths     = (max = $(StatsBase.maximum(ts_cut_lengths)    ), min = $(StatsBase.minimum(ts_cut_lengths)    ), mean = $(StatsBase.mean(ts_cut_lengths)    ), std = $(StatsBase.std(ts_cut_lengths)))")

		class_counts     = (length(pos),     length(neg))
		class_counts_aug = (length(pos_aug), length(neg_aug))
		println("Class counts: $(class_counts); Aug class counts: $(class_counts_aug); # points: $(max_timepoints)")

		X     = compute_X(max_timepoints, n_unique_freqs, timeseries,     sum(class_counts))
		X_aug = compute_X(max_timepoints, n_unique_freqs, timeseries_aug, sum(class_counts_aug))
		
		@assert n_samples(X)     == length(Y)     "$(n_samples(X))     != $(length(Y))"
		@assert n_samples(X_aug) == length(Y_aug) "$(n_samples(X_aug)) != $(length(Y_aug))"

		if return_filepaths
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

	if !use_augmentation_data # TODO remove this obsolete OLD code (now keeping for retro-compatibility)

		datasets = Vector(undef, length(dir_infos))
		n_pos    = Vector(undef, length(dir_infos))
		n_neg    = Vector(undef, length(dir_infos))

		for (i, dir_info) in enumerate(dir_infos)
			datasets[i], (n_pos[i], n_neg[i]) = getTimeSeries_no_aug((folders_Y, folders_N), dir_info, return_filepaths)
			@assert datasets[1][2] == datasets[i][2] "mismatching classes:\n\tY1 = $(datasets[1][2])\n\tY2 = $(datasets[i][2])"
			@assert n_pos[1] == n_pos[i] "n_pos mismatching class count across frames:\n\tn_pos[1] = $(n_pos[1]) != n_pos[i] = $(n_pos[i]))"
			@assert n_neg[1] == n_neg[i] "n_neg mismatching class count across frames:\n\tn_neg[1] = $(n_neg[1]) != n_neg[i] = $(n_neg[i]))"
		end

		if return_filepaths
			((getindex.(datasets, 1), datasets[1][2], getindex.(datasets, 3)), (n_pos[1], n_neg[1]))
		else
			((getindex.(datasets, 1), datasets[1][2]), (n_pos[1], n_neg[1]))
		end

	else

		datasets = Vector(undef, length(dir_infos))
		n_pos    = Vector(undef, length(dir_infos))
		n_neg    = Vector(undef, length(dir_infos))
		
		datasets_aug = Vector(undef, length(dir_infos))
		n_pos_aug    = Vector(undef, length(dir_infos))
		n_neg_aug    = Vector(undef, length(dir_infos))

		for (i, dir_info) in enumerate(dir_infos)
			cur_frame_dataset = getTimeSeries((folders_Y, folders_N), dir_info, return_filepaths)
			
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
end
################################################################################
################################################################################
################################################################################
