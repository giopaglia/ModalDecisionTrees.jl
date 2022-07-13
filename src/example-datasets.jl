using MLJ
using DataFrames

macro load_japanesevowels()
    quote
        X = DataFrame(OpenML.load(375)) # Load JapaneseVowels https://www.openml.org/search?type=data&status=active&id=375
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

        X = combine(groupby(X, [:speaker, :take, :utterance]), Not([:speaker, :take, :utterance, :frame]) .=> Ref; renamecols=false)

        Y = X[:,:speaker]
        select!(X, Not([:speaker, :take, :utterance]))

        # Force uniform size across instances by capping series
        minimum_n_points = minimum(collect(Iterators.flatten(eachrow(length.(X)))))
        X = (x->x[1:minimum_n_points]).(X)

        # instances = []
        # Y = eltype(X[:,:speaker])[]
        # for group in groupby(X, [:speaker, :take, :utterance])
        #     push!(Y, group[1,:speaker])
        #     instance = hcat([collect(row) for row in eachrow(select!(sort!(group[:,setdiff(names(X), ["speaker", "take", "utterance"])], :frame), Not(:frame)))]...)'
        #     push!(instances, instance)
        # end
        # n_attrs = unique((x->x[2]).(size.(instances)))
        # @assert length(n_attrs) == 1
        # n_attrs = n_attrs[1]
        # minimum_n_points = minimum(first.(size.(instances)))

        # X_cube = Array{Float64, 3}(undef, minimum_n_points, n_attrs, length(instances))

        # for (i_instance, instance) in enumerate(instances)
        #     X_cube[:,:,i_instance] = instance[1:minimum_n_points,:]
        # end

        # X_cube, Y
        # minimum(values(countmap(X[:,:speaker])))

        (X, Y)
    end
end
