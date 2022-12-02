
############################################################################################
# Multi-frame dataset
# 
# Multi-modal learning in this context is allowed by defining learning functions on so-called
#  `multi-frame datasets`. These are essentially vectors of modal datasets
############################################################################################
export get_world_types

struct MultiFrameModalDataset{MD<:ModalDataset}
    frames  :: Vector{<:MD}
    function MultiFrameModalDataset{MD}(X::MultiFrameModalDataset{MD}) where {MD<:ModalDataset}
        MultiFrameModalDataset{MD}(X.frames)
    end
    function MultiFrameModalDataset{MD}(Xs::AbstractVector) where {MD<:ModalDataset}
        Xs = collect(Xs)
        @assert length(Xs) > 0 && length(unique(nsamples.(Xs))) == 1 "Can't create an empty MultiFrameModalDataset or with mismatching number of samples (nframes: $(length(Xs)), frame_sizes: $(nsamples.(Xs)))."
        new{MD}(Xs)
    end
    function MultiFrameModalDataset{MD}() where {MD<:ModalDataset}
        new{MD}(MD[])
    end
    function MultiFrameModalDataset{MD}(X::MD) where {MD<:ModalDataset}
        MultiFrameModalDataset{MD}(MD[X])
    end
    function MultiFrameModalDataset(Xs::AbstractVector{<:MD}) where {MD<:ModalDataset}
        MultiFrameModalDataset{MD}(Xs)
    end
    function MultiFrameModalDataset(X::MD) where {MD<:ModalDataset}
        MultiFrameModalDataset{MD}(X)
    end
end

frames(X::MultiFrameModalDataset) = X.frames

Base.iterate(X::MultiFrameModalDataset, state=1)                    = state > length(X) ? nothing : (get_instance(X, state), state+1)
Base.length(X::MultiFrameModalDataset)                              = nsamples(X)
Base.size(X::MultiFrameModalDataset)                                = map(size, frames(X))
get_frame(X::MultiFrameModalDataset, i_frame::Integer)              = nframes(X) > 0 ? frames(X)[i_frame] : error("MultiFrameModalDataset has no frame!")
nframes(X::MultiFrameModalDataset)                                 = length(frames(X))
nsamples(X::MultiFrameModalDataset)                                = nsamples(get_frame(X, 1))::Int64
Base.push!(X::MultiFrameModalDataset, f::ModalDataset) = push!(frames(X), f)

# max_channel_size(X::MultiFrameModalDataset) = map(max_channel_size, frames(X)) # TODO: figure if this is useless or not. Note: channel_size doesn't make sense at this point. Only the accessibles_funs[i] functions.
nfeatures(X::MultiFrameModalDataset) = map(nfeatures, frames(X)) # Note: used for safety checks in fit_tree.jl
# nrelations(X::MultiFrameModalDataset) = map(nrelations, frames(X)) # TODO: figure if this is useless or not
nfeatures(X::MultiFrameModalDataset,  i_frame::Integer) = nfeatures(get_frame(X, i_frame))
nrelations(X::MultiFrameModalDataset, i_frame::Integer) = nrelations(get_frame(X, i_frame))
world_type(X::MultiFrameModalDataset,  i_frame::Integer) = world_type(get_frame(X, i_frame))
get_world_types(X::MultiFrameModalDataset) = Vector{Type{<:World}}(world_type.(frames(X)))

get_instance(X::MultiFrameModalDataset,  i_frame::Integer, idx_i::Integer, args...)  = get_instance(get_frame(X, i_frame), idx_i, args...)
# slice_dataset(X::MultiFrameModalDataset, i_frame::Integer, inds::AbstractVector{<:Integer}, args...)  = slice_dataset(get_frame(X, i_frame), inds, args...; kwargs...)

# get_instance(X::MultiFrameModalDataset, idx_i::Integer, args...)  = get_instance(get_frame(X, i), idx_i, args...) # TODO should slice across the frames!
slice_dataset(X::MultiFrameModalDataset{MD}, inds::AbstractVector{<:Integer}, args...; kwargs...) where {MD<:ModalDataset} = 
    MultiFrameModalDataset{MD}(Vector{MD}(map(frame->slice_dataset(frame, inds, args...; kwargs...), frames(X))))

function display_structure(Xs::MultiFrameModalDataset; indent_str = "")
    out = "$(typeof(Xs))" # * "\t\t\t$(Base.summarysize(Xs) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
    for (i_frame, X) in enumerate(frames(Xs))
        if i_frame == nframes(Xs)
            out *= "\n$(indent_str)└ "
        else
            out *= "\n$(indent_str)├ "
        end
        out *= "[$(i_frame)] "
        # \t\t\t$(Base.summarysize(X) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t(world_type: $(world_type(X)))"
        out *= display_structure(X; indent_str = indent_str * (i_frame == nframes(Xs) ? "   " : "│  ")) * "\n"
    end
    out
end

hasnans(Xs::MultiFrameModalDataset) = any(hasnans.(frames(Xs)))

isminifiable(::MultiFrameModalDataset) = true

function minify(Xs::MultiFrameModalDataset)
    if !any(map(isminifiable, frames(Xs)))
        if !all(map(isminifiable, frames(Xs)))
            @error "Cannot perform minification with frames of types $(typeof.(frames(Xs))). Please use a minifiable format (e.g., ExplicitModalDatasetS)."
        else
            @warn "Cannot perform minification on some of the frames provided. Please use a minifiable format (e.g., ExplicitModalDatasetS) ($(typeof.(frames(Xs))) were used instead)."
        end
    end
    Xs, backmap = zip([!isminifiable(X) ? minify(X) : (X, identity) for X in frames(Xs)]...)
    Xs, backmap
end
