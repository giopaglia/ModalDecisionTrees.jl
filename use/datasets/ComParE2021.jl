using CSV
using DataFrames
using StatsBase

function ComParE2021Dataset(;
		subchallenge = "CCS",
		use_lowSR = true,
		mode = :development,
		include_static_data = false,
		treat_as_single_attribute_2D_context = false,
		#
		audio_kwargs,
		ma_size = 1,
		ma_step = 1,
		max_points = -1,
		use_full_mfcc = false,
		preprocess_wavs = [],
	)
	dataset_dir = data_dir * "ComParE2021-CCS-CSS-Data/"

	@assert mode in [:development, :testing]
	@assert subchallenge in ["CCS", "CSS"] "Unknown subchallenge: $(subchallenge)"
	
	@assert subchallenge == "CCS" "Currently, only subchalleng=CCS is supported."
	@assert include_static_data == false || subchallenge == "CCS" "Currently, static data is only supported for subchallenge = CCS"

	records = CSV.read("$(dataset_dir)metaData_$(subchallenge).csv", DataFrame)
	samples_folder = "$(dataset_dir)ComParE2021_$(subchallenge)/dist/wav/"

	to_ignore = use_lowSR ? [] : [row.filename_x for row in CSV.File("$(dataset_dir)lowSR_$(subchallenge).list")]

	is_to_ignore(filename) = begin
		filename in to_ignore ||
		(mode == :testing && (startswith(filename, "devel_") || startswith(filename, "train_"))) ||
		(mode == :development && startswith(filename, "test_"))
	end

	records = records[map(!,(is_to_ignore.(records.filename))), :]

	sort!(records, [:label], alg=Base.Sort.InsertionSort)

	# display(records)
	class_names = unique(map((record)->record[:label], eachrow(records)))

	class_counts = [class_name=>0 for class_name in class_names] |> Dict

	if include_static_data
		static_data = Dict{Integer,Vector{Int64}}()
	end

	timeseries = Dict{Integer,Array{Float64, 2}}()
	labels = Dict{Integer,String}()

	ts_lengths         = Dict{Integer,Integer}()
	ts_with_ma_lengths = Dict{Integer,Integer}()
	ts_cut_lengths     = Dict{Integer,Integer}()

	# println(records |> typeof)
	# println(eachrow(records))
	# println(enumerate(eachrow(records)))
	Threads.@threads for (i_record,record) in collect(enumerate(eachrow(records)))
		# println(i_record)
		# println(record)
		# readline()

		if include_static_data &&
			(record[:Age] == "pnts" ||
			record[:Sex] == "pnts" ||
			record[:Medhistory] == "pnts" ||
			record[:Symptoms] == "pnts" ||
			record[:Smoking] == "pnts" ||
			record[:Hospitalized] == "pnts" ||
			record[:Sex] == "Other")
			continue
		end

		filepath = "$(samples_folder)$(record[:filename])"
		label = record[:label]
		
		class_counts[label] +=1
		# println(filepath, label)

		ts = wav2stft_time_series(filepath, audio_kwargs; preprocess_sample = preprocess_wavs, use_full_mfcc = use_full_mfcc)

		ts_lengths[i_record] = size(ts,1)

		# display(ts)
		# ts = @views ts[:,2:end]
		# display(ts)
		ts = moving_average(ts, ma_size, ma_step)
		# display(ts)

		ts_with_ma_lengths[i_record] = size(ts,1)

		if max_points != -1 && size(ts,1)>max_points
			ts = ts[1:max_points,:]
		end
		# display(ts)
		# display(size(ts,1))
		
		ts_cut_lengths[i_record] = size(ts,1)
		
		if include_static_data
			# "							
			
			age_dict = Dict([
				"0-19"  => 0,
				"16-19" => 0,
				"20-29" => 20,
				"30-39" => 30,
				"40-49" => 40,
				"50-59" => 50,
				"60-69" => 60,
				"70-79" => 70,
				"80-89" => 80,
			])
			
			sex_dict = Dict([
				"Female"  => 0,
				"Male"    => 1,
			])

			smoking_dict = Dict([
				"never"   => -100,
				"ex"      => -50,
				"ltOnce"  => 0,
				"1to10"   => 1,
				"11to20"  => 11,
				"21+"      => 21,
			])

			hospitalized_dict = Dict([
				"no"   => -1,
				"yes"  => 1,
			])

			Medhistory_domain = [
				"angina",
				"asthma",
				"cancer",
				"copd",
				"cystic",
				"diabetes",
				"hbp",
				"heart",
				"hiv",
				"long",
				"longterm",
				"lung",
				"otherHeart",
				"pulmonary",
				"valvular",
			]

			Symptoms_domain = [
				"fever",
				"drycough",
				"tightness",
				"headache",
				"sorethroat",
				"chills",
				"muscleache",
				"dizziness",
				"pnts",
				"shortbreath",
				"smelltasteloss",
				"wetcough",
			]

			features = Int64[]

			push!(features, age_dict[record[:Age]])
			push!(features, sex_dict[record[:Sex]])
			
			features = vcat(features, smoking_dict[record[:Smoking]])
			features = vcat(features, hospitalized_dict[record[:Hospitalized]])

			Medhistory_arr = filter(!isempty, split((record[:Medhistory] == "None" ? "" : record[:Medhistory]), ','))
			features = vcat(features, [Int(v in Medhistory_arr) for v in Medhistory_domain])

			Symptoms_arr = filter(!isempty, split((record[:Symptoms] == "None" ? "" : record[:Symptoms]), ','))
			features = vcat(features, [Int(v in Symptoms_arr) for v in Symptoms_domain])

			# TODO use attribute "Covid-Tested"?
			# record[Symbol("Covid-Tested")]
			# "negativeNever",
			# "positiveLast14",
			# "negativeOver14",
			# "yes",
			
			static_data[i_record] = features

		end

		timeseries[i_record] = ts
		labels[i_record] = label
		# readline()
	end

	idxs = sort(collect(keys(timeseries)))
	if include_static_data
		static_data = [static_data[i] for i in idxs]
	end
	timeseries  = [timeseries[i]  for i in idxs]
	labels      = [labels[i]      for i in idxs]

	ts_lengths          = [ts_lengths[i]         for i in idxs]
	ts_with_ma_lengths  = [ts_with_ma_lengths[i] for i in idxs]
	ts_cut_lengths      = [ts_cut_lengths[i]     for i in idxs]

	if include_static_data
		@assert length(static_data) == length(timeseries) "length(static_data) == length(timeseries). $(length(static_data)), $(length(timeseries))"
	end
	@assert length(timeseries) == length(labels) "length(timeseries) == length(labels). $(length(timeseries)), $(length(labels))"

	max_timepoints = maximum(size(ts, 1) for ts in timeseries)
	n_unique_freqs = unique(size(ts,  2) for ts in timeseries)
	@assert length(n_unique_freqs) == 1 "KDDDataset: length(n_unique_freqs) != 1: {$n_unique_freqs} != 1"
	n_unique_freqs = n_unique_freqs[1]
	X = zeros((max_timepoints, n_unique_freqs, length(timeseries)))
	for (i,ts) in enumerate(timeseries)
		# println(size(ts))
		X[1:size(ts, 1),:,i] = ts
	end

	if include_static_data
		X_static = zeros((length(static_data[1]), length(static_data)))
		for (i,features) in enumerate(static_data)
			# println(size(ts))
			X_static[:,i] .= features
		end
	end

	if treat_as_single_attribute_2D_context
		X = reshape(X, (size(X)[1], size(X)[2], 1, size(X)[3]))
	end

	class_counts = [class_counts[label] for label in class_names] |> Tuple
	
	println("ts_lengths         = (max = $(StatsBase.maximum(ts_lengths)        ), min = $(StatsBase.minimum(ts_lengths)        ), mean = $(StatsBase.mean(ts_lengths)        ), std = $(StatsBase.std(ts_lengths)))")
	println("ts_with_ma_lengths = (max = $(StatsBase.maximum(ts_with_ma_lengths)), min = $(StatsBase.minimum(ts_with_ma_lengths)), mean = $(StatsBase.mean(ts_with_ma_lengths)), std = $(StatsBase.std(ts_with_ma_lengths)))")
	println("ts_cut_lengths     = (max = $(StatsBase.maximum(ts_cut_lengths)    ), min = $(StatsBase.minimum(ts_cut_lengths)    ), mean = $(StatsBase.mean(ts_cut_lengths)    ), std = $(StatsBase.std(ts_cut_lengths)))")

	println("Class counts: $(class_counts); # points: $(max_timepoints)")


	if include_static_data
		([X, X_static], labels), class_counts
	else
		(X, labels), class_counts
	end
end
