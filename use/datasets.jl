using MAT
import Base: getindex, values
import Random
using JSON

include("local.jl")
include("wav2stft_time_series.jl")

# include("datasets/dummy.jl")
# include("datasets/eduard.jl")
# include("datasets/siemens-0.jl")

include("datasets/siemens.jl")
include("datasets/KDD.jl")
include("datasets/land-cover.jl")
