import JLD2
using Dates

_default_table_file_name(type::String) = "$(type)_cached.tsv"
_default_jld_file_name(type::String, hash::String) = string(type * "_" * hash * ".jld")

function _infos_to_dict(infos::NamedTuple)::Dict
    Dict([String(k) => v for (k,v) in zip(keys(infos),values(infos))])
end

function cached_obj_exists(type::String, common_cache_dir::String, hash::String)::Bool
	isdir(common_cache_dir) && isfile(common_cache_dir * "/" * _default_jld_file_name(type, hash))
end
cached_obj_exists(type::String, common_cache_dir::String, infos::Dict)::Bool = cached_obj_exists(type, common_cache_dir, get_hash_sha256(infos))
cached_obj_exists(type::String, common_cache_dir::String, infos::NamedTuple)::Bool = cached_obj_exists(type, common_cache_dir, _infos_to_dict(infos))

function cache_obj(type::String, common_cache_dir::String, obj::Any, hash::String, args_string::String; column_separator::String = "\t", time_spent::Dates.Millisecond, use_serialize::Bool = false)
	total_save_path = common_cache_dir * "/" * _default_jld_file_name(type, hash)
	mkpath(dirname(total_save_path))

	cache_table_exists = isfile(common_cache_dir * "/" * _default_table_file_name(type))

    table_file = open(common_cache_dir * "/" * _default_table_file_name(type), "a+")
    if !cache_table_exists
        write(table_file, string("TIMESTAMP$(column_separator)FILE NAME$(column_separator)COMP TIME$(column_separator)ARGS$(column_separator)\n"))
    end
	write(table_file, string(
            Dates.format(Dates.now(), "dd/mm/yyyy HH:MM:SS"), column_separator,
            _default_jld_file_name(type, hash), column_separator,
            human_readable_time(time_spent), column_separator,
            args_string, column_separator,
			"\n"))
	close(table_file)

	checkpoint_stdout("Saving $(type) to file $(total_save_path)...")
	if use_serialize
		Serialization.serialize(total_save_path, obj)
	else
		JLD2.@save total_save_path obj
	end
end
cache_obj(type::String, common_cache_dir::String, obj::Any, infos::Dict; kwargs...) = cache_obj(type, common_cache_dir, obj, get_hash_sha256(infos), string(infos); kwargs...)
cache_obj(type::String, common_cache_dir::String, obj::Any, infos::NamedTuple; kwargs...) = cache_obj(type, common_cache_dir, obj, _infos_to_dict(infos))

function load_cached_obj(type::String, common_cache_dir::String, hash::String)
	total_load_path = common_cache_dir * "/" * _default_jld_file_name(type, hash)

	checkpoint_stdout("Loading $(type) from file $(total_load_path)...")
	obj = nothing

	# TODO use magic number check instead of try/catch
	try
		JLD2.@load total_load_path obj
	catch e
		try
			obj = Serialization.deserialize(total_load_path)
		catch e
			throw_n_log("File $(total_load_path) is neither in JLD2 format nor a Serialized object", ArgumentError)
		end
	end

	obj
end
load_cached_obj(type::String, common_cache_dir::String, infos::NamedTuple) = load_cached_obj(type, common_cache_dir, _infos_to_dict(infos))
load_cached_obj(type::String, common_cache_dir::String, infos::Dict) = load_cached_obj(type, common_cache_dir, get_hash_sha256(infos))

macro cache(type, common_cache_dir, args, kwargs, compute_function)
	# TODO type check
	# hyigene
	type = esc(type)
	common_cache_dir = esc(common_cache_dir)
	args = esc(args)
	kwargs = esc(kwargs)
	compute_function = esc(compute_function)

	return quote
		_info_dict = _infos_to_dict($(kwargs))
		_hash = get_hash_sha256(($(args), _info_dict))
		if cached_obj_exists($(type), $(common_cache_dir), _hash)
			load_cached_obj($(type), $(common_cache_dir), _hash)
		else
			checkpoint_stdout("Computing " * $(type) * "...")

			_started = Dates.now()
			_result_value = $(compute_function)($(args)...; $(kwargs)...)
			_finish_time = (Dates.now() - _started)

			checkpoint_stdout("Computed " * $(type) * " in " * human_readable_time_s(_finish_time) * " seconds (" * human_readable_time(_finish_time) * ")")

			cache_obj($(type), $(common_cache_dir), _result_value, _hash, string($(args), "_", _info_dict), time_spent = _finish_time)
			_result_value
		end
	end
end
macro cache(type, common_cache_dir, args, compute_function)
	# TODO type check
	# hyigene
	type = esc(type)
	common_cache_dir = esc(common_cache_dir)
	args = esc(args)
	compute_function = esc(compute_function)

	return quote
		_hash = get_hash_sha256($(args))
		if cached_obj_exists($(type), $(common_cache_dir), _hash)
			load_cached_obj($(type), $(common_cache_dir), _hash)
		else
			checkpoint_stdout("Computing " * $(type) * "...")

			_started = Dates.now()
			_result_value = $(compute_function)($(args)...)
			_finish_time = (Dates.now() - _started)

			checkpoint_stdout("Computed " * $(type) * " in " * human_readable_time_s(_finish_time) * " seconds (" * human_readable_time(_finish_time) * ")")

			cache_obj($(type), $(common_cache_dir), _result_value, _hash, string($(args)), time_spent = _finish_time)
			_result_value
		end
	end
end
# TODO dry this code some way
macro cachefast(type, common_cache_dir, args, kwargs, compute_function)
	# TODO type check
	# hyigene
	type = esc(type)
	common_cache_dir = esc(common_cache_dir)
	args = esc(args)
	kwargs = esc(kwargs)
	compute_function = esc(compute_function)

	return quote
		_info_dict = _infos_to_dict($(kwargs))
		_hash = get_hash_sha256(($(args), _info_dict))
		if cached_obj_exists($(type), $(common_cache_dir), _hash)
			load_cached_obj($(type), $(common_cache_dir), _hash)
		else
			checkpoint_stdout("Computing " * $(type) * "...")

			_started = Dates.now()
			_result_value = $(compute_function)($(args)...; $(kwargs)...)
			_finish_time = (Dates.now() - _started)

			checkpoint_stdout("Computed " * $(type) * " in " * human_readable_time_s(_finish_time) * " seconds (" * human_readable_time(_finish_time) * ")")

			cache_obj($(type), $(common_cache_dir), _result_value, _hash, string($(args), "_", _info_dict), time_spent = _finish_time, use_serialize = true)
			_result_value
		end
	end
end
macro cachefast(type, common_cache_dir, args, compute_function)
	# TODO type check
	# hyigene
	type = esc(type)
	common_cache_dir = esc(common_cache_dir)
	args = esc(args)
	compute_function = esc(compute_function)

	return quote
		_hash = get_hash_sha256($(args))
		if cached_obj_exists($(type), $(common_cache_dir), _hash)
			load_cached_obj($(type), $(common_cache_dir), _hash)
		else
			checkpoint_stdout("Computing " * $(type) * "...")

			_started = Dates.now()
			_result_value = $(compute_function)($(args)...)
			_finish_time = (Dates.now() - _started)

			checkpoint_stdout("Computed " * $(type) * " in " * human_readable_time_s(_finish_time) * " seconds (" * human_readable_time(_finish_time) * ")")

			cache_obj($(type), $(common_cache_dir), _result_value, _hash, string($(args)), time_spent = _finish_time, use_serialize = true)
			_result_value
		end
	end
end
