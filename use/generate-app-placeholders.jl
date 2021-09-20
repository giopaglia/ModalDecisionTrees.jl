
import Dates
using FileIO

include("wav-filtering.jl")
include("paper-trees.jl")

gr()

covid_detector_dir = homedir() * "/" * "Julia Projects/covid-detector/"
filename = "empty-spectrogram.png"
spec_size = (1000, 500)
spec_mosaic_margin = 4mm
samplerate = 44100
duration_seconds = 5.0
samples = Vector{Float64}(undef, round(Int64, samplerate * duration_seconds))
samples .= 0.0

spec = draw_spectrogram(samples, samplerate; spectrogram_plot_options = (ylims = (0, 8_000), size = spec_size),)
yticks!(spec, [ 0, 8_000 ], [ "", "" ])
xticks!(spec, [ 0 ], [ "" ])
spec2 = deepcopy(spec)
title!(spec, "Original")
title!(spec2, "Filtered")
plot(spec, spec2, layout = (1, 2), left_margin = spec_mosaic_margin, right_margin = spec_mosaic_margin, top_margin = spec_mosaic_margin, bottom_margin = spec_mosaic_margin)

savefig(covid_detector_dir * "/" * filename)

l = 5000
orig = (
    color = RGB(0.3, 0.3, 1),
    size = (500, 75),
    dest = "original-empty-track.png"
)
filt = (
    color = RGB(1, 0.3, 0.3),
    size = (1000, 150),
    dest = "filtered-empty-track.png"
)

for f in (orig, filt)
    plot(
        collect(0:(l - 1)),
        fill(0, l),
        xlims = (0, length(l)),
        ylims = (-1, 1),
        framestyle = :zerolines,       # show axis at zeroes
        fill = 0,                      # show area under curve
        leg = false,                   # hide legend
        yshowaxis = false,             # hide y axis
        grid = false,                  # hide y grid
        ticks = false,                 # hide y ticks
        tick_direction = :none,
        linecolor = f.color,
        fillcolor = f.color,
        size = f.size,
        margin = 0mm,
        left_margin = 0mm,
        right_margin = 0mm,
        top_margin = 0mm,
        bottom_margin = 0mm
    )
    savefig(covid_detector_dir * "/" * f.dest)
end