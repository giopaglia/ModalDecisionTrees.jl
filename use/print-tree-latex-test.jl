
using Pkg
Pkg.activate("..")
using Revise

using DecisionTree
using DecisionTree.ModalLogic

include("paper-trees.jl")

print_tree_latex_out_dir = "print-tree-latex"
main_tex_file = "main.tex"

main_tex_content = """
\\documentclass{article}
\\usepackage{tikz}

\\begin{document}

\\begin{figure}
\\include{t1}
\\end{figure}

\\begin{figure}
\\include{t2}
\\end{figure}

\\begin{figure}
\\include{t3}
\\end{figure}

\\end{document}
"""

mkpath(print_tree_latex_out_dir)

if !isfile(print_tree_latex_out_dir * "/" * main_tex_file)
    f = open(print_tree_latex_out_dir * "/" * main_tex_file, "w+")
    write(f, main_tex_content)
    close(f)
end

# OPTIONS

additional_dict = Dict{String, String}("YES" => "pos", "NO_CLEAN_HISTORY_AND_LOW_PROBABILITY" => "neg")
common_kwargs = (conversion_dict = additional_dict, threshold_scale_factor = 3, threshold_show_decimals = 2, show_frame_number = false)

ft1 = open(print_tree_latex_out_dir * "/" * "t1.tex", "w+")
write(ft1, print_tree_latex(τ1; tree_name = "τ_1", common_kwargs...))
close(ft1)

ft2 = open(print_tree_latex_out_dir * "/" * "t2.tex", "w+")
write(ft2, print_tree_latex(τ2; tree_name = "τ_2", common_kwargs...))
close(ft2)

ft3 = open(print_tree_latex_out_dir * "/" * "t3.tex", "w+")
write(ft3, print_tree_latex(τ3; tree_name = "τ_3", common_kwargs...))
close(ft3)

cd(print_tree_latex_out_dir)
run(`pdflatex $main_tex_file`)
pdf_name = replace(main_tex_file, ".tex" => ".pdf")
run(`evince $pdf_name`)
