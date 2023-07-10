using SoleBase: movingwindow

DOWNSIZE_MSG = "If this process gets killed, please downsize your dataset beforehand."

function make_downsizing_function(channelsize::NTuple)
    return function downsize(X)
        return moving_average(X, channelsize)
    end
end

function make_downsizing_function(::TreeModel)
    function downsize(X)
        channelsize = SoleData.channelsize(X)
        nvariables = SoleData.nvariables(X)
        channelndims = length(channelsize)
        if channelndims == 1
            n_points = channelsize[1]
            if nvariables > 30 && n_points > 100
                @warn "Downsizing series $(n_points) points to $(100) points ($(nvariables) variables). $DOWNSIZE_MSG"
                X = moving_average(X, 100)
            elseif n_points > 150
                @warn "Downsizing series $(n_points) points to $(150) points ($(nvariables) variables). $DOWNSIZE_MSG"
                X = moving_average(X, 150)
            end
        elseif channelndims == 2
            if nvariables > 30 && prod(channelsize) > prod((7,7),)
                new_channelsize = min.(channelsize, (7,7))
                @warn "Downsizing image of size $(channelsize) to $(new_channelsize) pixels ($(nvariables) variables). $DOWNSIZE_MSG"
                X = moving_average(X, new_channelsize)
            elseif prod(channelsize) > prod((10,10),)
                new_channelsize = min.(channelsize, (10,10))
                @warn "Downsizing image of size $(channelsize) to $(new_channelsize) pixels ($(nvariables) variables). $DOWNSIZE_MSG"
                X = moving_average(X, new_channelsize)
            end
        end
        X
    end
end

function make_downsizing_function(::ForestModel)
    function downsize(X)
        channelsize = SoleData.channelsize(X)
        nvariables = SoleData.nvariables(X)
        channelndims = length(channelsize)
        if channelndims == 1
            n_points = channelsize[1]
            if nvariables > 30 && n_points > 100
                @warn "Downsizing series $(n_points) points to $(100) points ($(nvariables) variables). $DOWNSIZE_MSG"
                X = moving_average(X, 100)
            elseif n_points > 150
                @warn "Downsizing series $(n_points) points to $(150) points ($(nvariables) variables). $DOWNSIZE_MSG"
                X = moving_average(X, 150)
            end
        elseif channelndims == 2
            if nvariables > 30 && prod(channelsize) > prod((4,4),)
                new_channelsize = min.(channelsize, (4,4))
                @warn "Downsizing image of size $(channelsize) to $(new_channelsize) pixels ($(nvariables) variables). $DOWNSIZE_MSG"
                X = moving_average(X, new_channelsize)
            elseif prod(channelsize) > prod((7,7),)
                new_channelsize = min.(channelsize, (7,7))
                @warn "Downsizing image of size $(channelsize) to $(new_channelsize) pixels ($(nvariables) variables). $DOWNSIZE_MSG"
                X = moving_average(X, new_channelsize)
            end
        end
        X
    end
end

function moving_average(
    X::AbstractArray{T,1};
    kwargs...
) where {T}
    npoints = length(X)
    return [mean(X[idxs]) for idxs in movingwindow(npoints; kwargs...)]
end

function moving_average(
    X::AbstractArray{T,1},
    nwindows::Integer,
    relative_overlap::AbstractFloat = .5,
) where {T}
    npoints = length(X)
    return [mean(X[idxs]) for idxs in movingwindow(npoints; nwindows = nwindows, relative_overlap = relative_overlap)]
end

function moving_average(
    X::AbstractArray{T,3},
    nwindows::Integer,
    relative_overlap::AbstractFloat = .5,
) where {T}
    npoints, n_variables, n_instances = size(X)
    new_X = similar(X, (nwindows, n_variables, n_instances))
    for i_instance in 1:n_instances
        for i_variable in 1:n_variables
            new_X[:, i_variable, i_instance] .= [mean(X[idxs, i_variable, i_instance]) for idxs in movingwindow(npoints; nwindows = nwindows, relative_overlap = relative_overlap)]
        end
    end
    return new_X
end

function moving_average(
    X::AbstractArray{T,4},
    new_channelsize::Tuple{Integer,Integer},
    relative_overlap::AbstractFloat = .5,
) where {T}
    n_X, n_Y, n_variables, n_instances = size(X)
    windows_1 = movingwindow(n_X; nwindows = new_channelsize[1], relative_overlap = relative_overlap)
    windows_2 = movingwindow(n_Y; nwindows = new_channelsize[2], relative_overlap = relative_overlap)
    new_X = similar(X, (new_channelsize..., n_variables, n_instances))
    for i_instance in 1:n_instances
        for i_variable in 1:n_variables
            new_X[:, :, i_variable, i_instance] .= [mean(X[idxs1, idxs2, i_variable, i_instance]) for idxs1 in windows_1, idxs2 in windows_2]
        end
    end
    return new_X
end
