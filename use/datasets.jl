using MAT
import Base: getindex, values
import Random
using JSON

include("local.jl")
include("wav2stft_time_series.jl")

# include("datasets/dummy.jl")
# include("datasets/eduard.jl")
# include("datasets/siemens-0.jl")

# https://stackoverflow.com/questions/59562325/moving-average-in-julia
moving_average(vs::AbstractArray{T,1},n,st=1) where {T} = [sum(@view vs[i:(i+n-1)])/n for i in 1:st:(length(vs)-(n-1))]
moving_average(vs::AbstractArray{T,2},n,st=1) where {T} = mapslices((x)->(@views moving_average(x,n,st)), vs, dims=1)
# (sum(w) for w in partition(1:9, 3, 2))
# moving_average_np(vs,num_out_points,st) = moving_average(vs,length(vs)-num_out_points*st+1,st)
# moving_average_np(vs,num_out_points,o) = (w = length(vs)-num_out_points*(1-o/w)+1; moving_average(vs,w,1-o/w))
# moving_average_np(vs,t,o) = begin
# 	N = length(vs);
# 	s = floor(Int, (N+1)/(t+(1/(1-o))))
# 	w = ceil(Int, s/(1-o))
# 	# moving_average(vs,w,1-ceil(Int, o/w))
# end


include("datasets/siemens.jl")
include("datasets/KDD.jl")
include("datasets/land-cover.jl")
include("datasets/ComParE2021.jl")
#include("datasets/LoadModalDataset.jl")
