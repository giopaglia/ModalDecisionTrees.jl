

using Tables
using ARFFFiles
using DataFrames
using StatsBase
import MLJModelInterface: fit
using HTTP
using ZipFile

variable_names_latex = [
"\\text{hand tip}_X^L",
"\\text{hand tip}_Y^L",
"\\text{hand tip}_Z^L",
"\\text{hand tip}_X^R",
"\\text{hand tip}_Y^R",
"\\text{hand tip}_Z^R",
"\\text{elbow}_X^L",
"\\text{elbow}_Y^L",
"\\text{elbow}_Z^L",
"\\text{elbow}_X^R",
"\\text{elbow}_Y^R",
"\\text{elbow}_Z^R",
"\\text{wrist}_X^L",
"\\text{wrist}_Y^L",
"\\text{wrist}_Z^L",
"\\text{wrist}_X^R",
"\\text{wrist}_Y^R",
"\\text{wrist}_Z^R",
"\\text{thumb}_X^L",
"\\text{thumb}_Y^L",
"\\text{thumb}_Z^L",
"\\text{thumb}_X^R",
"\\text{thumb}_Y^R",
"\\text{thumb}_Z^R",
]

function load_arff_dataset(dataset_name, path = "http://www.timeseriesclassification.com/Downloads/$(dataset_name).zip")
# function load_arff_dataset(dataset_name, path = "../datasets/Multivariate_arff/$(dataset_name)")
    df_train, df_test = begin
        if(any(startswith.(path, ["https://", "http://"])))
            r = HTTP.get(path);
            z = ZipFile.Reader(IOBuffer(r.body))
            (
                ARFFFiles.load(DataFrame, z.files[[f.name == "$(dataset_name)_TRAIN.arff" for f in z.files]][1]),
                ARFFFiles.load(DataFrame, z.files[[f.name == "$(dataset_name)_TEST.arff" for f in z.files]][1]),
            )
        else
            (
                ARFFFiles.load(DataFrame, "$(path)/$(dataset_name)_TRAIN.arff"),
                ARFFFiles.load(DataFrame, "$(path)/$(dataset_name)_TEST.arff"),
            )
        end
    end

    @assert dataset_name == "NATOPS" "This code is only for showcasing. Need to expand code to comprehend more datasets."
    variable_names = [
        "Hand tip left, X coordinate",
        "Hand tip left, Y coordinate",
        "Hand tip left, Z coordinate",
        "Hand tip right, X coordinate",
        "Hand tip right, Y coordinate",
        "Hand tip right, Z coordinate",
        "Elbow left, X coordinate",
        "Elbow left, Y coordinate",
        "Elbow left, Z coordinate",
        "Elbow right, X coordinate",
        "Elbow right, Y coordinate",
        "Elbow right, Z coordinate",
        "Wrist left, X coordinate",
        "Wrist left, Y coordinate",
        "Wrist left, Z coordinate",
        "Wrist right, X coordinate",
        "Wrist right, Y coordinate",
        "Wrist right, Z coordinate",
        "Thumb left, X coordinate",
        "Thumb left, Y coordinate",
        "Thumb left, Z coordinate",
        "Thumb right, X coordinate",
        "Thumb right, Y coordinate",
        "Thumb right, Z coordinate",
    ]


    variable_names_latex = [
    "\\text{hand tip l}_X",
    "\\text{hand tip l}_Y",
    "\\text{hand tip l}_Z",
    "\\text{hand tip r}_X",
    "\\text{hand tip r}_Y",
    "\\text{hand tip r}_Z",
    "\\text{elbow l}_X",
    "\\text{elbow l}_Y",
    "\\text{elbow l}_Z",
    "\\text{elbow r}_X",
    "\\text{elbow r}_Y",
    "\\text{elbow r}_Z",
    "\\text{wrist l}_X",
    "\\text{wrist l}_Y",
    "\\text{wrist l}_Z",
    "\\text{wrist r}_X",
    "\\text{wrist r}_Y",
    "\\text{wrist r}_Z",
    "\\text{thumb l}_X",
    "\\text{thumb l}_Y",
    "\\text{thumb l}_Z",
    "\\text{thumb r}_X",
    "\\text{thumb r}_Y",
    "\\text{thumb r}_Z",
    ]
    X_train, Y_train = fix_dataframe(df_train, variable_names)
    X_test,  Y_test  = fix_dataframe(df_test, variable_names)

    class_names = [
        "I have command",
        "All clear",
        "Not clear",
        "Spread wings",
        "Fold wings",
        "Lock wings",
    ]

    fix_class_names(y) = class_names[round(Int, parse(Float64, y))]

    Y_train = map(fix_class_names, Y_train)
    Y_test  = map(fix_class_names, Y_test)

    @assert nrow(X_train) == length(Y_train) "$(nrow(X_train)), $(length(Y_train))"

    ((X_train, Y_train), (X_test,  Y_test))
end

function fix_dataframe(df, variable_names = nothing)
    s = unique(size.(df[:,:relationalAtt]))
    @assert length(s) == 1 "$(s)"
    n = unique(names.(df[:,:relationalAtt]))
    @assert length(n) == 1 "$(n)"
    nvars, npoints = s[1]
    old_var_names = n[1]
    X = Dict()

    if isnothing(variable_names)
        variable_names = ["V$(i_var)" for i_var in 1:nvars]
    end

    @assert nvars == length(variable_names)

    for (i_var,var) in enumerate(variable_names)
        X[Symbol(var)] = [collect(instance[i_var,old_var_names]) for instance in (df[:,:relationalAtt])]
    end

    X = DataFrame(X)
    Y = df[:,end]

    X, string.(Y)
    # X, Y
end



function show_latex(tree; file_suffix = "", variable_names = nothing, silent = true)
    include("../results/utils/print-tree-to-latex.jl")

    savedir = "latex"


    additional_dict = Dict{String, String}(
        "predictionI have command" => "\\fcolorbox{black}{pastel1}{\\ \\ I have command\\ \\ }",
        "predictionAll clear"      => "\\fcolorbox{black}{pastel2}{\\ \\ All clear\\ \\ }",
        "predictionNot clear"      => "\\fcolorbox{black}{pastel3}{\\ \\ Not clear\\ \\ }",
        "predictionSpread wings"   => "\\fcolorbox{black}{pastel4}{\\ \\ Spread wings\\ \\ }",
        "predictionFold wings"     => "\\fcolorbox{black}{pastel5}{\\ \\ Fold wings\\ \\ }",
        "predictionLock wings"     => "\\fcolorbox{black}{pastel6}{\\ \\ Lock wings\\ \\ }",
    )
    common_kwargs = (
        conversion_dict = additional_dict,
        # threshold_scale_factor = 3,
        threshold_show_decimals = 2,    
        hide_modality_ids = true,
        variable_names_map = variable_names,
        # replace_dict = Dict([
        # #     "\\{1\\}" => "",
        # #     "{1}" => "",
        #     "NN" => "N",
        # ]),
        scale = 1.,
        height = ["25em", "25em", "20em", "22em", "22em", "22em"],
        decisions_at_nodes = false,
        edges_textsize = [:Large, :small, :Large, :small, :small, :small],
        tree_names = ["t_minmax_static", "t_minmax_temporal", "t_neuro_static", "t_minmax_neuro_temporal", "t_minmax_neuro_static", "t_neuro_temporal"],
        latex_preamble = """
    \\definecolor{pastel1}{RGB}{161, 201, 244}
    \\definecolor{pastel2}{RGB}{255, 180, 130}
    \\definecolor{pastel3}{RGB}{141, 229, 161}
    \\definecolor{pastel4}{RGB}{255, 159, 155}
    \\definecolor{pastel5}{RGB}{208, 187, 255}
    \\definecolor{pastel6}{RGB}{222, 187, 155}
    \\definecolor{pastel7}{RGB}{250, 176, 228}
    \\definecolor{pastel8}{RGB}{207, 207, 207}
    \\definecolor{pastel9}{RGB}{255, 254, 163}
    \\definecolor{pastel10}{RGB}{185, 242, 240}
    """,
        # space_unit = (2.2, 3.9)./.2,
        space_unit = (2.2, 3.9)./.75,
        # min_n_inst = 
    )

    main_tex_file = "tree$(file_suffix == "" ? "" : "-$(file_suffix)").tex"
    save_tree_latex(
        [tree],
        savedir;
        main_tex_file = main_tex_file,
        common_kwargs...,
    )
    
    cd(savedir)
    if !silent
        run(`pdflatex $(main_tex_file)`);
    else
        # run(`bash -c "echo 2"`);
        # run(`bash -c "echo 2 2\\\>\\&1 \\\> /dev/null"`);
        run(`bash -c "pdflatex $(main_tex_file)  2\\\>\\&1 \\\> /dev/null"`);
    end
    pdf_name = replace(main_tex_file, ".tex" => ".pdf")
    run(`evince $pdf_name`);
    cd("..")
end
