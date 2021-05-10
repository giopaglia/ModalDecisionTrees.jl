
using Pkg
Pkg.activate("..")
using Revise
using DecisionTree

include("table-printer.jl")

import Statistics
tables_path = "./tables"
destination = "./tables"
primary_file_name = tables_path * "/full_columns.csv"
mkpath(tables_path)

seeds = [
    "1",
    "2",
    "3",
    "4",
    "5"
]

convert_col_name = Dict{String,String}(
    "sensitivity" => "sens",
    "specificity" => "spec",
    "precision" => "prec",
    "accuracy" => "acc"
)

#seed_dictionary = Dict{String,String}(zip(seeds, "s" * string(i) for i in 1:length(seeds)))

function write_to_file(file_name, string::String)
    file = open(file_name, "w")
    write(file, string)
    close(file)
end

T = extract_model(primary_file_name, "T")
RF = extract_model(primary_file_name, "RF")

# ROW
function get_task(row_label)
    return split(row_label, ",", limit = 4)[3]
end

function get_row_ref_no_seed(row_label)
    return split(row_label, ",", limit = 2)[2]
end

function get_nbands(row_label)
    return split(row_label, ",", limit = 5)[4]
end

function get_seed(row_label)
    return split(row_label, ",", limit = 2)[1]
end

function is_normalized(row_label)::Bool
    return contains(row_label, "Normalize")
end

# HEADER
function get_model_type(h)
    return split(h, "(", limit = 2)[1]
end
function get_column_var(h)
    return split(h, ")", limit = 2)[2]
end
function get_n_trees(h)
    return split(split(h, "(", limit = 2)[2], ",")[1]
end
function get_model(h)
    return lstrip(split(split(h, get_n_trees(h), limit = 2)[2], ")")[1], ',')
end

function table_to_dict(model)
    seed_order = Vector{String}()
    task_order = Vector{String}()

    for (i, row) in enumerate(model)
        if i == 1
            continue
        end
        seed = get_seed(row[1])
        if !(seed in seed_order)
            push!(seed_order, seed)
        end
        task = get_row_ref_no_seed(row[1])
        if !(task in task_order)
            push!(task_order, task)
        end
    end

    dict = Dict{String,Dict{String,AbstractVector}}()
    for (i, row) in enumerate(model)
        if i == 1
            continue
        end
        task = get_row_ref_no_seed(row[1])
        seed = get_seed(row[1])

        if !haskey(dict, task)
            dict[task] = Dict{String,AbstractVector}()
        end

        dict[task][seed] = model[i]
    end

    dict, model[1], seed_order, task_order
end

function dict_to_desired_format_table(dict, header, seed_order, task_order; average_row_decorate = "", use_only_seeds = [], beautify_latex = false)
    average_string = "avg"
    std_dev_string = "std"
    function avg_std_row(dict; exclude_conditions::Union{Nothing,AbstractVector} = nothing)::Vector{Any}
        avg_row::Vector{Any} = [average_row_decorate * average_string]
        std_row::Vector{Any} = [average_row_decorate * std_dev_string]
        rows = collect(values(dict))
        for i in 2:length(rows[1])
            # TODO: there is a better way to do this, I know it
            vals::Vector{Real} = []
            for r in rows
                if length(use_only_seeds) != 0 && !(get_seed(r[1]) in use_only_seeds)
                    continue
                end
                if !isempty(findall([ exclude_conditions[i](r[1]) for i in 1:length(exclude_conditions) ]))
                    continue
                end
                push!(vals, parse(Float64, r[i]))
            end
            push!(avg_row, string(round(Statistics.mean(vals), digits=2)))
            push!(std_row, string(round(Statistics.std(vals), digits=2)))
        end
        [ avg_row, std_row ]
    end

    function pretty_header(header)::Vector{Vector{Any}}
        type = get_model_type(header[2])

        models::Vector{String} = []
        for i in 2:length(header)
            mod = ""
            if type == "RF"
                mod = string(get_n_trees(header[i]), ",", get_model(header[i]))
            else
                mod = get_model(header[i])
            end
            if !(mod in models)
                push!(models, mod)
            end
        end

        vars::Vector{String} = []
        for i in 2:length(header)
            mod = ""
            if type == "RF"
                mod = string(get_n_trees(header[i]), ",", get_model(header[i]))
            else
                mod = get_model(header[i])
            end
            if mod == models[1]
                push!(vars, get_column_var(header[i]))
            end
        end
        vars = map(str -> convert_col_name[str], vars)

        nh::Vector{Vector{Any}} = []
        if type == "RF"
            # first row
            new_row_1::Vector{Any} = [ "\\multirow{2}{*}{\\textit{$(type)}}" ]
            for (j, m) in enumerate(models)
                res = "\\multicolumn{$(length(vars))}{c"
                if j != length(models)
                    res *= "|"
                end
                res *= "}{$(m)}"
                push!(new_row_1, res)
            end
            push!(nh, new_row_1)
            # second row
            new_row_2::Vector{Any} = [ "" ]
            for i in 1:length(models)
                for v in vars
                    push!(new_row_2, v)
                end
            end
            push!(nh, new_row_2)
        elseif type == "T"
            new_row::Vector{Any} = [ "\\textit{TDT}" ]
            for i in 1:length(models)
                for v in vars
                    push!(new_row, v)
                end
            end
            push!(nh, new_row)
        end

        nh
    end

    function prettify_first_column(table, header_size)::Vector{Vector{Any}}
        new_table::Vector{Vector{Any}} = []
        for i in 1:header_size
            if i > 1
                splice!(table[i], 1:0, [""])
            end
            push!(new_table, table[i])
        end
        new_table[1][1] = "\\multicolumn{2}{c|}{$(new_table[1][1])}"

        task_already_inserted::Vector{Any} = []

        for i in (header_size+1):length(table)
            if startswith(table[i][1], average_row_decorate * average_string)
                table[i][1] = average_row_decorate * "\\multicolumn{2}{r|}{$(replace(table[i][1], average_row_decorate => ""))}"
            elseif startswith(table[i][1], average_row_decorate * std_dev_string)
                table[i][1] = average_row_decorate * "\\multicolumn{2}{r|}{$(replace(table[i][1], average_row_decorate => ""))}"
            else
                seed = get_seed(table[i][1])
                task = (get_task(table[i][1]) == 1 ? "C" : "B") * # task id
                       "," * get_nbands(table[i][1]) * ",(" *     # nbands
                       split(table[i][1], "(30,")[2]              # dataset kwargs
                if is_normalized(table[i][1])
                    task *= ",N"
                end
                table[i][1] = get_seed(table[i][1])
                if !(task in task_already_inserted)
                    k = i
                    row_count = 1
                    while !startswith(table[k][1], average_row_decorate * average_string)
                        k += 1
                        row_count += 1
                    end
                    splice!(table[i], 1:0, ["\\multirow{$(row_count-1)}{*}{\\begin{sideways}$(task)\\end{sideways}}"])
                    push!(task_already_inserted, task)
                else
                    splice!(table[i], 1:0, [""])
                end
            end
            push!(new_table, table[i])
        end

        new_table
    end

    table::Vector{Vector{Any}} = []
    new_header::Vector{Vector{Any}} = []
    if beautify_latex
        append!(new_header, pretty_header(header))
    else
        append!(new_header, [header])
    end
    header_size = length(new_header)
    append!(table, new_header)
    for task in task_order
        if !haskey(dict, task)
            continue
        end
        # Not Normalized
        n_row_added = 0
        for seed in seed_order
            if is_normalized(dict[task][seed][1])
                continue
            end
            if length(use_only_seeds) != 0 && !(seed in use_only_seeds)
                continue
            end
            if !haskey(dict[task], seed)
                continue
            end
            n_row_added += 1
            push!(table, dict[task][seed])
        end
        if n_row_added > 2
            append!(table, avg_std_row(dict[task]; exclude_conditions = [ is_normalized ]))
        end
        # Normalized
        n_row_added = 0
        for seed in seed_order
            if !is_normalized(dict[task][seed][1])
                continue
            end
            if length(use_only_seeds) != 0 && !(seed in use_only_seeds)
                continue
            end
            if !haskey(dict[task], seed)
                continue
            end
            n_row_added += 1
            push!(table, dict[task][seed])
        end
        if n_row_added > 2
            append!(table, avg_std_row(dict[task]; exclude_conditions = [ !is_normalized ]))
        end
    end

    row_size = length(table[header_size+1])
    
    new_t::Vector{Vector{Any}} = []
    if beautify_latex
        new_t = prettify_first_column(table, header_size)
    else
        new_t = table
    end

    first_column_size = (length(table[header_size+1])-row_size) + 1

    new_t, header_size, first_column_size
end

# Create dicts
Ttuple = table_to_dict(T)
RFtuple = table_to_dict(RF)

# Arrange table in proper way (CSV)
Ttable_csv = dict_to_desired_format_table(Ttuple..., use_only_seeds = seeds, beautify_latex = false)
RFtable_csv = dict_to_desired_format_table(RFtuple..., use_only_seeds = seeds, beautify_latex = false)

write_to_file(destination * "/results.csv", string(
    string_table_csv(Ttable_csv[1]), "\n",
    string_table_csv(RFtable_csv[1]),
))

color_row = "\\rowcolor{lightgray!50}"
# Arrange table in proper way (LaTeX)
Ttable_latex = dict_to_desired_format_table(Ttuple...; average_row_decorate = color_row, beautify_latex = true, use_only_seeds = seeds)
RFtable_latex = dict_to_desired_format_table(RFtuple...; average_row_decorate = color_row, beautify_latex = true, use_only_seeds = seeds)

latex_t = string_table_latex(Ttable_latex[1], header_size = Ttable_latex[2], first_column_size = Ttable_latex[3], v_lin_every_cloumn = 4, scale = 1.0) * "\n"
latex_rf = string_table_latex(RFtable_latex[1], header_size = RFtable_latex[2], first_column_size = RFtable_latex[3], v_lin_every_cloumn = 4, scale = 1.0) * "\n"

write_to_file(destination * "/tree.tex", latex_t)
write_to_file(destination * "/rf.tex", latex_rf)
