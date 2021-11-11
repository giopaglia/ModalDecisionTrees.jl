using CSV
using DataFrames
using Glob
using Dates
using Random

siemens_trip_d = "/media/gio/Rover/"
tripvib_data_dir = "$(siemens_trip_d)Trip_vibration_data/"
new_tripvib_data_dir = "$(siemens_trip_d)Trip_vibration_data-new/"
data_request_file = tripvib_data_dir * "New_data_request_2021_09_29.csv"

function get_filenames_n_caps(pattern)
	filepaths = glob(pattern, tripvib_data_dir)
	filepaths = removeprefix.(filepaths, tripvib_data_dir)
	# https://discourse.julialang.org/t/sorting-strings-containing-numbers-so-that-a2-a10/5372/5
	function naturalsort(x::Vector{String})
	    f = text -> all(isnumeric, text) ? Char(parse(Int, text)) : text
	    sorter = key -> join(f(m.match) for m in eachmatch(r"[0-9]+|[^0-9]+", key))
	    sort(x, by=sorter)
	end
	filenames = naturalsort(filepaths)
	filenames_n_caps = [[f, (match(r"(\d+)_(\d+).csv_(\d+)_(\d+)_(\d+).csv.gz", f).captures .|> (x)->parse(Int,x))...] for f in filenames]
	filenames_n_caps = hcat(filenames_n_caps...) |> permutedims
end

function ProcessTripVibDataset(;
		out_dir = new_tripvib_data_dir,
		consistency_checks = false,
		limit_csv_rows = nothing,
		days_before_trip_threshold = 30,
	)
	
	mkpath(out_dir)

	tag_desc_2_tag_ids_datasources_dict, tag_id_2_tag_desc_idx_dict, tag_descs = begin
		
		datasource_tag_id_2_tag_desc_dict = begin
			request_df = CSV.read("$(data_request_file)", DataFrame)
			datasource_tag_id_2_tag_desc_df = unique(request_df[:,[:datasource,:tag_id,:tag_desc]], [1,2,3])
			Dict((r[:datasource], r[:tag_id]) => r[:tag_desc] for r in eachrow(datasource_tag_id_2_tag_desc_df))
		end

		tag_desc_2_tag_ids_datasources_dict = Dict(k => [] for k in values(datasource_tag_id_2_tag_desc_dict))
		# tag_id_2_datasource_dict = Dict()
		tag_id_2_tag_desc_dict = Dict()
		for ((datasource,tag_id),tag_desc) in datasource_tag_id_2_tag_desc_dict
			push!(tag_desc_2_tag_ids_datasources_dict[tag_desc],(datasource,tag_id))
			# push!(tag_desc_2_tag_ids_datasources_dict[tag_desc],datasource)
			# push!(tag_desc_2_tag_ids_datasources_dict[tag_desc],tag_id)
			# tag_id_2_datasource_dict[tag_id] = datasource
			tag_id_2_tag_desc_dict[tag_id] = tag_desc
		end
		for l in values(tag_desc_2_tag_ids_datasources_dict)
			sort!(l)
		end

		tag_descs = sort(collect(keys(tag_desc_2_tag_ids_datasources_dict)))

		tag_id_2_tag_desc_idx_dict = Dict()
		for (tag_id, tag_desc) in tag_id_2_tag_desc_dict
			tag_id_2_tag_desc_idx_dict[tag_id] = findfirst((x)->x==tag_desc, tag_descs)
		end

		# @assert length(unique([length(x) for x in values(tag_desc_2_tag_ids_datasources_dict)])) == 1 "$(tag_desc_2_tag_ids_datasources_dict)"
		tag_desc_2_tag_ids_datasources_dict, tag_id_2_tag_desc_idx_dict, tag_descs
	end

	filenames_n_caps = get_filenames_n_caps("*.csv.gz")

	datasources_ids = filenames_n_caps[:,2] |> unique |> sort

	datasource_df = nothing
	data = nothing

	datasources_ids = [42817]
	for datasources_id in datasources_ids
		filepaths = get_filenames_n_caps("$(datasources_id)_*.csv.gz")[:,1]
		datasource_df = DataFrame()
		datasource_df_l = ReentrantLock() # create lock variable
		# Threads.@threads for (i_filepath,filepath) in collect(enumerate(filepaths))
		# TODO remove
		filepaths = filepaths[Random.randperm(length(filepaths))]
		println(filepaths)
		
		for (i_filepath,filepath) in collect(enumerate(filepaths))
			println("$(i_filepath+1)/$(length(filepaths)) $(filepath)")
			tmp_filepath = "$(tripvib_data_dir)$(filepath)-tmp.csv"
			println(tmp_filepath)
			if !isfile(tmp_filepath)
				run(pipeline(`gzip -k -d -c $(tripvib_data_dir)$(filepath)`, stdout="$(tmp_filepath)-tmp"))
				mv("$(tmp_filepath)-tmp", "$(tmp_filepath)", force = true)
			end
			data = CSV.read("$(tmp_filepath)", DataFrame;
				header = [:datasource, :tag_id, :unk1, :timestamp, :value, :unk2, :unk3, :next_trip],
				limit = limit_csv_rows,
				dateformat=Dates.DateFormat("yyyy-mm-dd HH:MM:SS.sss"));

			if consistency_checks
				@assert length(unique(data[:,:unk3]))       == 1 "$(unique(data[:,:unk3]))"
				@assert length(unique(data[:,:datasource])) == 1 "$(unique(data[:,:datasource]))"
				@assert length(unique(data[:,:tag_id]))     == 1 "$(unique(data[:,:tag_id]))"
			end

			# Base.summarysize(data) |> println
			# Base.summarysize(data[[:tag_id, :timestamp, :value, :next_trip]]) |> println

			milliseconds_in_a_day = (DateTime(Day(2)) - DateTime(Day(1))).value
			
			# println(data[1,:next_trip])
			# println(data[1,:timestamp])
			# println((data[1,:next_trip]-data[1,:timestamp]).value)
			# println(days_before_trip_threshold*milliseconds_in_a_day)
			# readline()
			
			filter!((row)->(row[:next_trip]-row[:timestamp]).value < days_before_trip_threshold*milliseconds_in_a_day, data)
			
			data[!,:time_to_trip] = [(row[:timestamp]-row[:next_trip]).value for row in eachrow(data)]

			data = data[:,[:tag_id, :time_to_trip, :value, :next_trip]]

			# println(nrow(data))
			# println(data[1,:])
			println("$(nrow(data)) rows")

			# transform!(data, :, [:tag_id] => (x)->tag_id_2_tag_desc_idx_dict[x])
			# println(data)
			# break
			lock(datasource_df_l) # need to lock for append!
			datasource_df = vcat(datasource_df, data)
			unlock(datasource_df_l)
			println("$(nrow(datasource_df)) rows")
			if nrow(datasource_df) > 10
				break
			end
		end
		
		# Create :trip column as categorical version of :next_trip
		trips = unique(datasource_df[:,:next_trip])
		datasource_df[!,:trip] = [findfirst((x)->x==next_trip, trips) for row in eachrow(datasource_df)]
		# transform!(datasource_df,:next_trip => (next_trip)->findfirst((x)->x==next_trip, trips) => :next_trip)
		datasource_df = datasource_df[:,[:tag_id, :time_to_trip, :value, :trip]]

		CSV.write("$(out_dir)$(datasources_id).csv", dateformat=Dates.DateFormat("yyyy-mm-dd HH:MM:SS.sss"), datasource_df)
		
		DataFrame([1:length(trips),trips], [:id,:trip])
		CSV.write("$(out_dir)$(datasources_id)-trips.csv", dateformat=Dates.DateFormat("yyyy-mm-dd HH:MM:SS.sss"), datasource_df)

		break
	end

	datasource_df
end


# TODO: TripVibDataset(;consistency_checks = true)

function TripVibDataset(; consistency_checks = false, limit_csv_rows = nothing)

	# for tag_desc in tag_descs
	# 	tag_ids = tag_desc_2_tag_ids_datasources_dict[tag_desc]
	# 	# println(tag_ids, length(tag_ids))
	# 	println("$(length(tag_ids))\t$(tag_desc)")
	# end

	# filenames_n_caps = get_filenames_n_caps("*.csv.gz")

	# for row in eachrow(unique(DataFrame(filenames_n_caps)[:,[2,3]]))
	# 	datasource_tag_id = Tuple(row)
	# 	tag_desc = datasource_tag_id_2_tag_desc_dict[datasource_tag_id]
	# 	(datasource_tag_id,tag_desc) |> println
	# end

end

# for f in {
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_0_0_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_0_0_1.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_0_1_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_0_1_1.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_0_2_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_0_3_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_0_4_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_0_4_1.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_0_5_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_0_6_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_0_6_1.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_0_7_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_0_7_1.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_1_0_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_1_1_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_1_2_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_1_3_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_1_4_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_1_5_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_1_6_0.csv.gz,
# /media/gio/Rover/Trip_vibration_data/28171_535203.csv_1_7_0.csv.gz}; do
# 	gzip -k -d -c $f > tmp
# 	head -n 1 tmp
# done
