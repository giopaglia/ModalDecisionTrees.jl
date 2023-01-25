using MLJ
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

        # dataframe2cube(new_X)
        # instances, n_attrs, minimum_n_points = begin
        #     instances = [hcat([collect(attr) for attr in instance]...) for instance in eachrow(new_X)]
        #     n_attrs = unique((x->x[2]).(size.(instances)))
        #     @assert length(n_attrs) == 1
        #     n_attrs = n_attrs[1]
        #     minimum_n_points = minimum(first.(size.(instances)))
        #     instances, n_attrs, minimum_n_points
        # end

        # X_cube = Array{Float64, 3}(undef, minimum_n_points, n_attrs, length(instances))

        # for (i_instance, instance) in enumerate(instances)
        #     X_cube[:,:,i_instance] = instance[1:minimum_n_points,:]
        # end

        new_X, Y
    end
end
