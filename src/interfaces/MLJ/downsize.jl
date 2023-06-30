DOWNSIZE_MSG = "If this process gets killed, please downsize your dataset beforehand."

function tree_downsizing_function(_X)
    channelsize = SoleData.channelsize(_X)
    ninstances = SoleData.ninstances(_X)
    channelndims = length(channelsize)
    if channelndims == 1
        n_points = channelsize[1]
        if ninstances > 300 && n_points > 100
            @warn "Downsizing series $(n_points) points to $(100) points ($(ninstances) instances). $DOWNSIZE_MSG"
            _X = moving_average(_X, 100)
        elseif n_points > 150
            @warn "Downsizing series $(n_points) points to $(150) points ($(ninstances) instances). $DOWNSIZE_MSG"
            _X = moving_average(_X, 150)
        end
    elseif channelndims == 2
        if ninstances > 300 && prod(channelsize) > prod((7,7),)
            new_channelsize = min.(channelsize, (7,7))
            @warn "Downsizing image of size $(channelsize) to $(new_channelsize) pixels ($(ninstances) instances). $DOWNSIZE_MSG"
            _X = moving_average(_X, new_channelsize)
        elseif prod(channelsize) > prod((10,10),)
            new_channelsize = min.(channelsize, (10,10))
            @warn "Downsizing image of size $(channelsize) to $(new_channelsize) pixels ($(ninstances) instances). $DOWNSIZE_MSG"
            _X = moving_average(_X, new_channelsize)
        end
    end
    _X
end

function forest_downsizing_function(_X)
    channelsize = SoleData.channelsize(_X)
    ninstances = SoleData.ninstances(_X)
    channelndims = length(channelsize)
    if channelndims == 1
        n_points = channelsize[1]
        if ninstances > 300 && n_points > 100
            @warn "Downsizing series $(n_points) points to $(100) points ($(ninstances) instances). $DOWNSIZE_MSG"
            _X = moving_average(_X, 100)
        elseif n_points > 150
            @warn "Downsizing series $(n_points) points to $(150) points ($(ninstances) instances). $DOWNSIZE_MSG"
            _X = moving_average(_X, 150)
        end
    elseif channelndims == 2
        if ninstances > 300 && prod(channelsize) > prod((4,4),)
            new_channelsize = min.(channelsize, (4,4))
            @warn "Downsizing image of size $(channelsize) to $(new_channelsize) pixels ($(ninstances) instances). $DOWNSIZE_MSG"
            _X = moving_average(_X, new_channelsize)
        elseif prod(channelsize) > prod((7,7),)
            new_channelsize = min.(channelsize, (7,7))
            @warn "Downsizing image of size $(channelsize) to $(new_channelsize) pixels ($(ninstances) instances). $DOWNSIZE_MSG"
            _X = moving_average(_X, new_channelsize)
        end
    end
    _X
end

function __moving_window_without_overflow_fixed_num(
    npoints::Integer;
    nwindows::Integer,
    relative_overlap::AbstractFloat,
)::AbstractVector{UnitRange{Int}}
    # Code by Giovanni Pagliarini (@giopaglia) & Federico Manzella (@ferdiu)
    #
    # start = 1+half_context
    # stop = npoints-half_context
    # step = (stop-start+1)/nwindows
    # half_context = step*relative_overlap/2

    # half_context = relative_overlap * (npoints-1) / (2* nwindows+2*relative_overlap)
    half_context = relative_overlap * npoints / (2* nwindows+2*relative_overlap)
    start = 1+half_context
    stop = npoints-half_context
    step = (stop-start+1)/nwindows

    # _w = floor(Int, step+2*half_context)
    # _w = floor(Int, ((stop-start+1)/nwindows)+2*half_context)
    # _w = floor(Int, ((npoints-half_context)-(1+half_context)+1)/nwindows+2*half_context)
    # _w = floor(Int, (npoints-2*half_context)/nwindows+2*half_context)
    _w = floor(Int, (npoints-2*half_context)/nwindows + 2*half_context)

    # println("step: ($(stop)-$(start)+1)/$(nwindows) = ($(stop-start+1)/$(nwindows) = $(step)")
    # println("half_context: $(half_context)")
    # first_points = range(start=start, stop=stop, length=nwindows+1)[1:end-1]
    first_points = range(start=start, stop=stop, length=nwindows+1)[1:end-1] # TODO needs Julia 1.7: warn user
    first_points = map((x)->x-half_context, first_points)
    @assert isapprox(first_points[1], 1.0)
    # println("first_points: $(collect(first_points))")
    # println("window: $(step)+$(2*half_context) = $(step+2*half_context)")
    # println("windowi: $(_w)")
    first_points = map((x)->round(Int, x), first_points)
    # first_points .|> (x)->(x+step/2) .|> (x)->(x-size/2,x+size/2)
    # first_points .|> (x)->(max(1.0,x-half_context),min(x+step+half_context,npoints))
    # first_points .|> (x)->(x-half_context,x+step+half_context)
    first_points .|> (xi)->(xi:xi+_w-1)
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
            new_X[:, i_variable, i_instance] .= [mean(X[idxs, i_variable, i_instance]) for idxs in __moving_window_without_overflow_fixed_num(npoints; nwindows = nwindows, relative_overlap = relative_overlap)]
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
    windows_1 = __moving_window_without_overflow_fixed_num(n_X; nwindows = new_channelsize[1], relative_overlap = relative_overlap)
    windows_2 = __moving_window_without_overflow_fixed_num(n_Y; nwindows = new_channelsize[2], relative_overlap = relative_overlap)
    new_X = similar(X, (new_channelsize..., n_variables, n_instances))
    for i_instance in 1:n_instances
        for i_variable in 1:n_variables
            new_X[:, :, i_variable, i_instance] .= [mean(X[idxs1, idxs2, i_variable, i_instance]) for idxs1 in windows_1, idxs2 in windows_2]
        end
    end
    return new_X
end
