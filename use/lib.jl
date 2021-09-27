using SHA
using Serialization
import JLD2
import Dates

function get_hash_sha256(var)::String
	io = IOBuffer();
	serialize(io, var)
	result = bytes2hex(sha256(take!(io)))
	close(io)

	result
end

function checkpoint_stdout(string::String)
	println("● ", Dates.format(Dates.now(), "[ dd/mm/yyyy HH:MM:SS ] "), string)
	flush(stdout)
end

function human_readable_time(ms::Dates.Millisecond)::String
	result = ms.value / 1000
	seconds = round(Int64, result % 60)
	result /= 60
	minutes = round(Int64, result % 60)
	result /= 60
	hours = round(Int64, result % 24)
	string(string(hours; pad=2), ":", string(minutes; pad=2), ":", string(seconds; pad=2))
end

function human_readable_time_s(ms::Dates.Millisecond)::String
	string(ms.value / 1000)
end


function printprogress(io::Base.TTY, string::String)
    print(io, "\033[2K\033[1000D" * string)
end
function printprogress(io::IO, string::String)
    println(io, string)
end
function printprogress(string::String)
    printprogress(stdout, string)
end

_test_function_trend(vec::Vector{<:Number}, op::Function; approx::Real = 1.0)::Bool =
	length(vec) > 1 ? length(findall([ op((vec[i+1] - vec[i]), 0)  for i in 1:(length(vec)-1) ])) >= round(Int64, (length(vec)-1)*clamp(approx,0,1)) : true

isnondecreasing(vec::Vector{<:Number}; approx::Real = 1.0)::Bool = _test_function_trend(vec, >=; approx = approx)
isdecreasing(vec::Vector{<:Number}; approx::Real = 1.0)::Bool = _test_function_trend(vec, <; approx = approx)
isnonincreasing(vec::Vector{<:Number}; approx::Real = 1.0)::Bool = _test_function_trend(vec, <=; approx = approx)
isincreasing(vec::Vector{<:Number}; approx::Real = 1.0)::Bool = _test_function_trend(vec, >; approx = approx)
