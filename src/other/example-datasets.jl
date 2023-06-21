using MLJ
using SoleData
using DataFrames

macro load_japanesevowels()
    quote
        X = DataFrame(MLJ.OpenML.load(375)) # Load JapaneseVowels https://www.openml.org/search?type=data&status=active&id=375
        names(X)

        take_col = []
        take = 1
        prev_frame = nothing
        prev_utterance = nothing
        for row in eachrow(X)
            cur_frame = Float64(row.frame)
            cur_utterance = row.utterance
            if !isnothing(prev_frame) && cur_frame == 1.0
                take += 1
            end
            prev_frame = cur_frame
            prev_utterance = cur_utterance
            push!(take_col, take)
        end

        # combine(groupby(X, [:speaker, :take, :utterance]), :coefficient1 => Base.vect)
        # combine(groupby(X, [:speaker, :take, :utterance]), Base.vect)
        # combine(groupby(X, [:speaker, :take, :utterance]), All() .=> Base.vect)
        # combine(groupby(X, [:speaker, :take, :utterance]), :coefficient1 => Ref)

        # countmap(take_col)
        X[:,:take] = take_col

        X = combine(DataFrames.groupby(X, [:speaker, :take, :utterance]), Not([:speaker, :take, :utterance, :frame]) .=> Ref; renamecols=false)

        Y = X[:,:speaker]
        # select!(X, Not([:speaker, :take, :utterance]))

        # Force uniform size across instances by capping series
        minimum_n_points = minimum(collect(Iterators.flatten(eachrow(length.(X[:,Not([:speaker, :take, :utterance])])))))
        new_X = (x->x[1:minimum_n_points]).(X[:,Not([:speaker, :take, :utterance])])

        new_X, varnames = SoleData.dataframe2cube(new_X)

        new_X, Y
    end
end
