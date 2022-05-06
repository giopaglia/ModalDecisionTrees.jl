################################################################################
# Print tree latex
################################################################################

export print_tree_latex

default_conversion_dict_latex = Dict{String, String}(
    "τ" => "\\tau ",
    "⫹" => "\\leq ",
    "⫺" => "\\geq ",
    "⪳" => "\\preceqq ",
    "⪴" => "\\succeqq ",
    "⪵" => "\\precneqq ",
    "⪶" => "\\succneqq ",
    "⟨" => "\\langle ",
    "⟩" => "\\rangle ",
    "A̅" => "\\overline{A}",
    "L̅" => "\\overline{L}",
    "B̅" => "\\overline{B}",
    "E̅" => "\\overline{E}",
    "D̅" => "\\overline{D}",
    "O̅" => "\\overline{O}",
)

const NodeCoord = Tuple{Real,Real}

import Base: +, -

+(coord1::NodeCoord, coord2::NodeCoord)::NodeCoord = (coord1[1] + coord2[1], coord1[2] + coord2[2])
-(coord1::NodeCoord, coord2::NodeCoord)::NodeCoord = (coord1[1] - coord2[1], coord1[2] - coord2[2])

function _attr_to_latex(str::String)::String
    matched = match(r"\bA[0-9]+\b", str)
    if !isnothing(matched)
        str = replace(str, matched.match => "A_{" * replace(matched.match, "A" => "") * "}")
    end
    str
end

function _latex_string(
        obj::Any;
        conversion_dict::Dict{String,String} = default_conversion_dict_latex,
        add_dollars::Bool = true,
        show_test_operator_alpha::Bool = true,
        show_frame_number::Bool = true
    )::String

    subscript_replace = Dict{String,String}(
        "₀" => "0",
        "₁" => "1",
        "₂" => "2",
        "₃" => "3",
        "₄" => "4",
        "₅" => "5",
        "₆" => "6",
        "₇" => "7",
        "₈" => "8",
        "₉" => "9",
        "ₑ" => "e",
        "․" => ".",
        "․" => ".",
        "₋" => "-"
    )

    result = string(obj)
    if !show_test_operator_alpha
        for k in keys(subscript_replace)
            result = replace(result, k => "")
        end
    end

    # WARN: assumption: Global relation is actually Later
    result = replace(result, "G" => "L")

    if show_frame_number
        # Escape {frame}
        result = replace(result, "{" => "\\{", count = 1)
        result = replace(result, "}" => "\\}", count = 1)
    else
        # Remove {frame}
        matched = match(r"^\{[0-9]+\}", result)
        if !isnothing(matched)
            result = replace(result, matched.match => "", count = 1)
        end
    end

    result = _attr_to_latex(result)

    subscript_num_regex = Regex("\\b[" * join(keys(subscript_replace)) * "]+\\b")
    matched = match(subscript_num_regex, result)
    while !isnothing(matched)
        m = matched.match
        for (k, v) in subscript_replace
            m = replace(m, k => v)
        end
        result = replace(result, matched.match => "_{" * m * "}")
        matched = match(subscript_num_regex, result)
    end

    for (k, v) in conversion_dict
        result = replace(result, k => v)
    end

    if add_dollars
        result = "\$" * result * "\$"
        if result == "\$\$"
            result = ""
        end
    end

    result
end

_node_content(leaf::DTLeaf; kwargs...)::String = _latex_string(prediction(leaf); kwargs...)
_node_content(node::DTInternal; kwargs...)::String = ""

function _print_tree_latex(
    leaf                      :: DTLeaf,
    previous_node_index       :: String,
    previous_node_position    :: NodeCoord,
    space_unit                :: NodeCoord,
    nodes_margin              :: NodeCoord,
    conversion_dict           :: Dict{String,String},
    add_dollars               :: Bool,
    print_test_operator_alpha :: Bool,
    show_frame_number         :: Bool,
    t_display_func            :: Function,
    nodes_script_size         :: Symbol,
    edges_script_size         :: Symbol
    )::String
    ""
end
function _print_tree_latex(
        node                      :: DTInternal,
        previous_node_index       :: String,
        previous_node_position    :: NodeCoord,
        space_unit                :: NodeCoord,
        nodes_margin              :: NodeCoord,
        conversion_dict           :: Dict{String,String},
        add_dollars               :: Bool,
        print_test_operator_alpha :: Bool,
        show_frame_number         :: Bool,
        t_display_func            :: Function,
        nodes_script_size         :: Symbol,
        edges_script_size         :: Symbol
    )::String

    # use tree height to determine the horizontal-spacing between the nodes
    h = height(node)

    # TODO: calculate proper position
    left_node_pos = previous_node_position + (-abs(space_unit[1])*h, -abs(space_unit[2])) + (-abs(nodes_margin[1]), -abs(nodes_margin[2]))
    right_node_pos = previous_node_position + (abs(space_unit[1])*h, -abs(space_unit[2])) + (abs(nodes_margin[1]), -abs(nodes_margin[2]))

    result = "\\" * string(nodes_script_size) * "\n"
    # add left node
    result *= "\\node ($(previous_node_index)0) at $left_node_pos {$(_node_content(node.left; conversion_dict = conversion_dict, add_dollars = add_dollars))};\n"
    # add right node
    result *= "\\node ($(previous_node_index)1) at $right_node_pos {$(_node_content(node.right; conversion_dict = conversion_dict, add_dollars = add_dollars))};\n"

    result *= "\\" * string(edges_script_size) * "\n"
    # add left edge
    result *= "\\path ($previous_node_index) edge[sloped,above] node {$(_latex_string(display_decision(node; threshold_display_method = t_display_func); conversion_dict = conversion_dict, add_dollars = add_dollars, show_test_operator_alpha = print_test_operator_alpha, show_frame_number = show_frame_number))} ($(previous_node_index)0);\n"
    # add right edge
    result *= "\\path ($previous_node_index) edge[sloped,above] node {$(_latex_string(display_decision_inverse(node; threshold_display_method = t_display_func); conversion_dict = conversion_dict, add_dollars = add_dollars, show_test_operator_alpha = print_test_operator_alpha, show_frame_number = show_frame_number))} ($(previous_node_index)1);\n"
    # recursive calls
    result *= _print_tree_latex(node.left, previous_node_index * "0", left_node_pos, space_unit, nodes_margin, conversion_dict, add_dollars, print_test_operator_alpha, show_frame_number, t_display_func, nodes_script_size, edges_script_size)
    result *= _print_tree_latex(node.right, previous_node_index * "1", right_node_pos, space_unit, nodes_margin, conversion_dict, add_dollars, print_test_operator_alpha, show_frame_number, t_display_func, nodes_script_size, edges_script_size)

    result
end
# :Huge         = \Huge
# :huge         = \huge
# :LARGE        = \LARGE
# :Large        = \Large
# :large        = \large
# :normalsize   = \normalsize
# :small        = \small
# :footnotesize = \footnotesize
# :scriptsize   = \scriptsize
# :tiny         = \tiny
function print_tree_latex(
        tree                               :: DTree;
        tree_name                          :: String                             = "τ",
        conversion_dict                    :: Union{Nothing,Dict{String,String}} = nothing,
        first_node_idx                     :: String                             = "0",
        first_node_position                :: NodeCoord                          = (0, 0),
        space_unit                         :: NodeCoord                          = (0.5, 2.0),
        nodes_margin                       :: NodeCoord                          = (1.8, 0),
        merge_conversion_dict_with_default :: Bool                               = true,
        wrap_in_tikzpicture_block          :: Bool                               = true,
        add_dollars                        :: Bool                               = true,
        print_test_operator_alpha          :: Bool                               = true,
        show_frame_number                  :: Bool                               = true,

        threshold_scale_factor             :: Integer                            = 0,
        threshold_show_decimals            :: Union{Symbol,Integer}              = :all,

        tree_name_script_size              :: Symbol                             = :large,
        nodes_script_size                  :: Symbol                             = :normalsize,
        edges_script_size                  :: Symbol                             = :footnotesize
    )::String

    function threshold_display_func(threshold::Real, scale_factor::Integer, show_decimals::Union{Symbol,Integer})::Real
        result = threshold * (10^scale_factor)
        if isa(show_decimals, Integer)
            result = round(result, digits = show_decimals)
        end
        result
    end

    if merge_conversion_dict_with_default
        if isnothing(conversion_dict)
            conversion_dict = deepcopy(default_conversion_dict_latex)
        else
            merge!(conversion_dict, default_conversion_dict_latex)
        end
    else
        if isnothing(conversion_dict)
            conversion_dict = Dict{String,String}()
        end
    end

    print_tree_comment = replace(string(tree), "\n" => "\n% ")

        result = "\$\$\n"
        result *= "% packages needed: tikz, amssymb, newtxmath\n"
        result *= "% " * tree_name * "\n"
        result *= "% " * print_tree_comment * "\n"
        result *= wrap_in_tikzpicture_block ? "\\begin{tikzpicture}\n" : ""
        result *= "\\" * string(tree_name_script_size) * "\n"
        result *= "\\node ($first_node_idx) at $first_node_position [above] {$(_latex_string(tree_name; conversion_dict = conversion_dict, add_dollars = add_dollars))};\n"
        result *= _print_tree_latex(
        tree.root,
        first_node_idx,
        first_node_position,
        space_unit,
        nodes_margin,
        conversion_dict,
        add_dollars,
        print_test_operator_alpha,
        show_frame_number,
        x -> threshold_display_func(x, threshold_scale_factor, threshold_show_decimals),
        nodes_script_size,
        edges_script_size
    )
    result *= wrap_in_tikzpicture_block ? "\\end{tikzpicture}\n" : ""
    result *= "\$\$\n"

    result
end
