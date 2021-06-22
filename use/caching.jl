
_default_table_file_name(type::String) = "$(type)_cached.csv"
_default_jld_file_name(type::String, hash::String) = string(type * "_" * hash * ".jld")

function _infos_to_dict(infos::NamedTuple)::Dict
    Dict([String(k) => v for (k,v) in zip(keys(infos),values(infos))])
end

function cached_obj_exists(type::String, infos::Dict, common_cache_dir::String)::Bool
	isdir(common_cache_dir) && isfile(common_cache_dir * "/" * _default_jld_file_name(type, get_hash_sha256(infos)))
end
cached_obj_exists(type::String, infos::NamedTuple, common_cache_dir::String; kwargs...) = cached_obj_exists(type, _infos_to_dict(infos), common_cache_dir)

function cache_obj(type::String, obj::Any, infos::Dict, common_cache_dir::String; column_separator::String = ";")
	info_hash = get_hash_sha256(infos)

	total_save_path = common_cache_dir * "/" * _default_jld_file_name(type, info_hash)
	mkpath(dirname(total_save_path))

    should_write_header::Bool = ! isfile(common_cache_dir * "/" * _default_table_file_name(type))

    table_file = open(common_cache_dir * "/" * _default_table_file_name(type), "a+")
    if should_write_header
        write(table_file, string("TIMESTAMP$(column_separator)INFOS$(column_separator)FILE NAME$(column_separator)\n"))
    end
	write(table_file, string(
            Dates.format(Dates.now(), "dd/mm/yyyy HH:MM:SS"), column_separator,
            _default_jld_file_name(type, info_hash), column_separator,
            infos, column_separator, "\n"))
	close(table_file)

	checkpoint_stdout("Saving $(type) to file $(total_save_path)...")
    # saving obj_infos dict allows the user to inspect a cached file later
	JLD2.@save total_save_path obj infos
end
cache_obj(type::String, obj::Any, infos::NamedTuple, common_cache_dir::String; kwargs...) = cache_obj(type, obj, _infos_to_dict(infos), common_cache_dir)

function load_cached_obj(type::String, infos::Dict, common_cache_dir::String)
	info_hash = get_hash_sha256(infos)

	total_load_path = common_cache_dir * "/" * _default_jld_file_name(type, info_hash)

	checkpoint_stdout("Loading $(type) from file $(total_load_path)...")
	JLD2.@load total_load_path obj

	obj
end
load_cached_obj(type::String, infos::NamedTuple, common_cache_dir::String) = load_cached_obj(type, _infos_to_dict(infos), common_cache_dir)