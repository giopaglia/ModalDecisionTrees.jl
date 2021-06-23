
import JSON

function init_new_history(file_path::String, params_names::Vector{String}, exec_ranges::Vector)
	history = []
	save_history(file_path, history)
	import_history(file_path)
end

function save_history(file_path::String, dicts::AbstractVector)
	mkpath(dirname(file_path))
	file = open(file_path, "w+")
	write(file, JSON.json(dicts))
	close(file)
end

function import_history(file_path::String)
	file = open(file_path)
	dicts = JSON.parse(file)
	close(file)
	dicts
end

function append_in_file(file_name::String, text::String)
	mkpath(dirname(file_name))
	file = open(file_name, "a+")
	write(file, text)
	close(file)
end

function _are_the_same(obj1::Any, obj2::Any)::Bool
	# Tuple -> Array
	obj1 = obj1 isa Tuple ? [obj1...] : obj1
	obj2 = obj2 isa Tuple ? [obj2...] : obj2

	# NamedTuple -> Dict
	obj1 = obj1 isa NamedTuple ? 
		Dict{String, Any}([String(k) => v for (k,v) in zip(keys(obj1),values(obj1))]) : obj1

	obj2 = obj2 isa NamedTuple ? 
		Dict{String, Any}([String(k) => v for (k,v) in zip(keys(obj2),values(obj2))]) : obj2

	if obj1 isa Dict && obj2 isa Dict
		for (key1,val1) in obj1
			if !haskey(obj2, key1) || !_are_the_same(val1, obj2[key1])
				return false
			end
		end
		return length(obj1) == length(obj2)
	elseif obj1 isa Array && obj2 isa Array
		for (val1,val2) in zip(obj1,obj2)
			if !_are_the_same(val1, val2)
				return false
			end
		end
		return true
	else
		isequal(obj1, obj2)
	end
end

function iteration_in_history(history, nt)::Bool
	for item in history
		if _are_the_same(nt, item)
			return true
		end
	end
	return false
end

function push_iteration_to_history!(history, nt)
	push!(history, nt)
end

function _match_filter(test_parameters, filters)::Bool
	for filter in filters
		for (i, k) in enumerate(keys(filter))
			# TODO: handle test_parameters has no key "k"
			if filter[k] == test_parameters[k]
				if i == length(filter)
					# if was it was the last key then there is a match
					return true
				else
					# if key has same value continue cycling through filter keys
					continue
				end
			else
				# if there is a key not matching go to next filter
				break
			end
		end
	end

	return false
end

# note: filters may contain less keys than test_parameters
function is_whitelisted_test(test_parameters, filters = [])::Bool
	# if filters is empty no whitelisting is applied
	if length(filters) == 0
		return true
	end

	return _match_filter(test_parameters, filters)
end

function is_blacklisted_test(test_parameters, filters = [])::Bool
	# if filters is empty no blacklisting is applied
	if length(filters) == 0
		return false
	end

	return _match_filter(test_parameters, filters)
end

function load_or_create_history(file_path::String, args...)
	if isfile(file_path)
		println("Loading history file \"$(file_path)\"...")
		import_history(file_path)
	else
		println("Creating history file \"$(file_path)\"...")
		init_new_history(file_path, args...)
	end
end

# https://discourse.julialang.org/t/how-to-make-a-named-tuple-from-a-dictionary/10899/3
dictkeys(d::Dict) = (collect(keys(d))...,)
dictvalues(d::Dict) = (collect(values(d))...,)
namedtuple(d::Dict{Symbol,T}) where {T} = NamedTuple{dictkeys(d)}(dictvalues(d))

