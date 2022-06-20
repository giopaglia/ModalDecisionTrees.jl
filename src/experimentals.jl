################################################################################
# Experimental features
################################################################################
module experimentals

using ..ModalDecisionTrees
using ..ModalDecisionTrees.ModalLogic

MDT = ModalDecisionTrees
ML  = ModalLogic

# TODO extend so that we can accept those operators using these characters
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

################################################################################
# Parse Trees
################################################################################

function parse_tree(tree_str::String; check_format = true, _depth = 0, offset = 0, worldTypes = Type{ML.World}[], init_conditions = MDT.InitCondition[])
    worldTypes = Type{<:ML.World}[worldTypes...]
    init_conditions = MDT.InitCondition[init_conditions...]
    root = _parse_tree(tree_str; check_format = check_format, _depth = _depth, offset = offset)
    DTree(root, worldTypes, init_conditions)
end

function _parse_tree(tree_str::String; check_format = true, _depth = 0, offset = 0)

    ########################################################################################
    ########################################################################################
    ########################################################################################
    
    # _threshold_ex = "[-+]?(?:[0-9]+(\.[0-9]*)?|\.[0-9]+)" # TODO use smarter regex (e.g., https://www.oreilly.com/library/view/regular-expressions-cookbook/9781449327453/ch06s10.html )
    _threshold_ex = "[^\\)\\s)]+" # TODO use smarter regex (e.g., https://www.oreilly.com/library/view/regular-expressions-cookbook/9781449327453/ch06s10.html )
    _indentation_ex = "[ │]*[✔✘]"
    _metrics_ex = "\\(\\S*.*\\)"
    _feature_ex             = "(?:\\S+)\\s+(?:(?:⫹|⫺|⪳|⪴|⪵|⪶|↗|↘|>|<|=|≤|≥|<=|>=))"
    _normal_feature_ex_capturing    = "^(\\S+)\\(A(\\d+)\\)\\s+((?:>|<|=|≤|≥|<=|>=))$"
    _special_feature_ex_capturing   = "^A(\\d+)\\s+((?:⫹|⫺|⪳|⪴|⪵|⪶|↗|↘))$"
    _decision_ex            = "$(_feature_ex)\\s+(?:$(_threshold_ex))"
    _decision_ex__capturing = "($(_feature_ex))\\s+($(_threshold_ex))"
    
    leaf_ex            = "(?:\\S+)\\s+:\\s+\\d+/\\d+(?:\\s+(?:$(_metrics_ex)))?"
    leaf_ex__capturing = "(\\S+)\\s+:\\s+(\\d+)/(\\d+)(?:\\s+($(_metrics_ex)))?"
    decision_ex            = "(?:⟨(?:\\S+)⟩\\s*)?(?:$(_decision_ex)|\\(\\s*$(_decision_ex)\\s*\\))"
    decision_ex__capturing = "(?:⟨(\\S+)⟩\\s*)?\\(?\\s*$(_decision_ex__capturing)\\s*\\)?"
    
    # TODO default frame to 1
    # split_ex = "(?:\\s*{(\\d+)}\\s+)?($(decision_ex))(?:\\s+($(leaf_ex)))?"
    split_ex = "\\s*{(\\d+)}\\s+($(decision_ex))(?:\\s+($(leaf_ex)))?"
    
    blank_line_regex = Regex("^\\s*\$")
    split_line_regex = Regex("^($(_indentation_ex)\\s+)?$(split_ex)\\s*\$")
    leaf_line_regex  = Regex("^($(_indentation_ex)\\s+)?$(leaf_ex)\\s*\$") 

    function _parse_simple_real(x)
        x = parse(Float64, x)
        x = isinteger(x) ? Int(x) : x
    end 

    function _parse_decision((i_this_line, decision_str)::Tuple{<:Integer,<:AbstractString},)
        function _parse_relation(relation_str)
            rel_d = Dict([
                "G" => ML.RelationGlob,
                "A" => ML.IA_A,
                "L" => ML.IA_L,
                "B" => ML.IA_B,
                "E" => ML.IA_E,
                "D" => ML.IA_D,
                "O" => ML.IA_O,
                "Ai" => ML.IA_Ai,        " ̅A" => ML._IA_Ai,
                "Li" => ML.IA_Li,        " ̅L" => ML._IA_Li,
                "Bi" => ML.IA_Bi,        " ̅B" => ML._IA_Bi,
                "Ei" => ML.IA_Ei,        " ̅E" => ML._IA_Ei,
                "Di" => ML.IA_Di,        " ̅D" => ML._IA_Di,
                "Oi" => ML.IA_Oi,        " ̅O" => ML._IA_Oi,
            ])
            if isnothing(relation_str)
                RelationId
            else
                rel_d[relation_str]
            end
        end

        function _parse_feature_test_operator(feature_str)
            
            m_normal  = match(Regex(_normal_feature_ex_capturing), feature_str)
            m_special = match(Regex(_special_feature_ex_capturing), feature_str) 

            if !isnothing(m_normal) && length(m_normal) == 3
                feature_fun, i_attribute, test_operator = m_normal
                function eval_feature_fun_constructor(fun_str)
                    if     fun_str == "min" MDT.SingleAttributeMin
                    elseif fun_str == "max" MDT.SingleAttributeMax
                    else
                        try
                            fun = eval(Symbol(fun_str))
                            (i_attribute)->MDT.SingleAttributeGenericFeature(i_attribute, fun)
                        catch
                            (i_attribute)->MDT.SingleAttributeNamedFeature(i_attribute, fun_str)
                        end
                    end
                end
                feature_type = eval_feature_fun_constructor(feature_fun)
                i_attribute = parse(Int, i_attribute)
                test_operator = eval(Symbol(test_operator))
                feature_type(i_attribute), test_operator
            elseif !isnothing(m_special) && length(m_special) == 2
                i_attribute, feature_fun_test_operator = m_special
                feature_fun_test_operator_d = Dict([
                    "⪴"   => (i_attribute)->(ML.SingleAttributeMin(i_attribute), ≥),
                    "⪴₈₀" => (i_attribute)->(ML.SingleAttributeSoftMin(i_attribute, 80), ≥),
                    "⪳₈₀" => (i_attribute)->(ML.SingleAttributeSoftMax(i_attribute, 80), ≤),
                    "⪳"   => (i_attribute)->(ML.SingleAttributeMax(i_attribute), ≤),
                    "↘"   => (i_attribute)->(ML.SingleAttributeMin(i_attribute), ≤),
                    "↘"   => (i_attribute)->(ML.SingleAttributeMax(i_attribute), ≥),
                ])
                feature_fun_test_operator = feature_fun_test_operator_d[feature_fun_test_operator]
                i_attribute = parse(Int, i_attribute)
                feature_fun_test_operator(i_attribute)
            else
                error("Unexpected format encountered on line $(i_this_line+offset) when parsing feature: \"$(feature_str)\". Matches $(m_normal), $(m_special)")
            end
        end 

        print(repeat(" ", _depth))
        m = match(Regex(decision_ex), decision_str)
        @assert !isnothing(m) "Unexpected format encountered on line $(i_this_line+offset) when parsing decision: \"$(decision_str)\". Matches: $(m)" 

        m = match(Regex(decision_ex__capturing), decision_str)
        @assert !isnothing(m) && length(m) == 3 "Unexpected format encountered on line $(i_this_line+offset) when parsing decision: \"$(decision_str)\". Matches: $(m) Expected matches = 3"
        # print(repeat(" ", _depth))
        # println(m) 

        relation, feature_test_operator, threshold = m
        relation = _parse_relation(relation)
        feature, test_operator = _parse_feature_test_operator(feature_test_operator)
        threshold = _parse_simple_real(threshold) 

        Decision(relation, feature, test_operator, threshold)
    end
    function _parse_leaf((i_this_line, leaf_str)::Tuple{<:Integer,<:AbstractString},)
        m = match(Regex(leaf_ex__capturing), leaf_str)
        @assert !isnothing(m) && length(m) == 4 "Unexpected format encountered on line $(i_this_line+offset) when parsing leaf: \"$(leaf_str)\". Matches: $(m) Expected matches = 4"
        # print(repeat(" ", _depth))
        # println(m)
        class, n_good, n_tot, metrics = m
        class = String(class)
        n_good = parse(Int, n_good)
        n_tot  = parse(Int, n_tot)
        # println(class, n_good, n_tot, metrics)
        # Note: metrics are not used
        DTLeaf(class, String[fill(class, n_good)..., fill("NO_$(class)", n_tot-n_good)...])
    end

    ########################################################################################
    ########################################################################################
    ########################################################################################
    
    # Can't do this because then i_line is misaligned
    # tree_str = strip(tree_str)
    lines = enumerate(split(tree_str, "\n")) |> collect
    
    if check_format
        for (i_line, line) in lines
            !isempty(strip(line)) || continue
            _line = line 

            blank_match = match(blank_line_regex, _line)
            split_match = match(split_line_regex, _line)
            leaf_match  = match(leaf_line_regex,  _line)
            is_blank = !isnothing(blank_match)
            is_split = !isnothing(split_match)
            is_leaf  = !isnothing(leaf_match)
            
            # DEBUG
            # println(match(Regex("($(_indentation_ex)\\s+)?"), _line))
            # println(match(Regex("^\\s*($(_indentation_ex)\\s+)?({(\\d+)}\\s+)?"), _line))
            # println(match(Regex("^\\s*($(_indentation_ex)\\s+)?({(\\d+)}\\s+)?A(\\d+)"), _line))
            # println(match(Regex("^\\s*($(_indentation_ex)\\s+)?({(\\d+)}\\s+)?(\\S+\\s+)?A(\\d+)"), _line))
            # println(match(Regex("^\\s*($(_indentation_ex)\\s+)?({(\\d+)}\\s+)?(\\S+\\s+)?A(\\d+)\\s+([⫹⫺⪳⪴⪵⪶↗↘])"), _line))
            # println(match(Regex("^\\s*($(_indentation_ex)\\s+)?({(\\d+)}\\s+)?$(_decision_ex)"), _line))
            # println(match(Regex("^\\s*($(_indentation_ex)\\s+)?({(\\d+)}\\s+)?$(decision_ex)"), _line))
            # println(match(Regex("^\\s*($(_indentation_ex)\\s+)?({(\\d+)}\\s+)?$(decision_ex)\\s+$(leaf_ex)"), _line))
            
            @assert xor(is_blank, is_split, is_leaf) "Couldn't parse line $(i_line+offset): \"$(line)\". $((is_blank, is_split, is_leaf))"
        end
    end 

    _lines = filter(((i_line, line),)->(!isempty(strip(line))), lines) 

    if length(_lines) == 1 # a leaf
        _parse_leaf(_lines[1])
    else # a split
        
        this_line, yes_line, no_line = begin
            this_line = nothing
            yes_line = -Inf
            no_line = Inf 

            for (i_line, line) in lines
                !isempty(strip(line)) || continue
                _line = line 

                if !isnothing(match(r"^\s*{.*$", _line))
                    @assert isnothing(this_line) "Can't have more than one row beginning with '{'"
                    this_line = i_line
                    yes_line = i_line + 1
                    # print(repeat(" ", _depth))
                    # println("First: $(_line)")
                elseif i_line == yes_line
                    @assert startswith(_line, "✔") "Line $(i_line+offset) \"$(_line)\" should start with '✔'"
                elseif no_line > i_line > yes_line
                    if !startswith(_line, "│")
                        @assert startswith(_line, "✘") "Line $(i_line+offset) \"$(_line)\" should start with '✘'"
                        no_line = i_line-1
                    end
                else
                    @assert startswith(_line, " ") "Line $(i_line+offset) \"$(_line)\" should start with ' '"
                end
            end
            this_line, yes_line, no_line
        end
        
        function clean_lines(lines)
            join([(isempty(strip(line)) ? line : begin
                    begin_ex = Regex("^([ │]|[✔✘]\\s+)(.*)\$")
                    match(begin_ex, line)[2]
                end) for (i_line, line) in lines], "\n")
        end
        left_tree_str, right_tree_str = clean_lines(lines[yes_line:no_line]), clean_lines(lines[no_line+1:end])
        i_this_line, this_line = lines[this_line]
        
        print(repeat(" ", _depth))
        m = match(Regex(split_ex), this_line)
        @assert !isnothing(m) && length(m) == 3 "Unexpected format encountered on line $(i_this_line+offset) : \"$(this_line)\". Matches: $(m) Expected matches = 3"
        # println(m)
        i_frame_str, decision_str, leaf_str = m 

        i_frame = parse(Int, i_frame_str)
        decision = _parse_decision((i_this_line, decision_str),) 

        # println(clean_lines(lines[yes_line:no_line]))
        # println("\n")
        # println(clean_lines(lines[no_line+1:end]))
        left  = _parse_tree(left_tree_str;  offset = yes_line-1, check_format = false, _depth = _depth + 1)
        right = _parse_tree(right_tree_str; offset = no_line-1,  check_format = false, _depth = _depth + 1)
        
        if isnothing(leaf_str)
            DTInternal(i_frame, decision, left, right)
        else
            this = _parse_leaf((i_this_line, leaf_str),)
            DTInternal(i_frame, decision, this, left, right)
        end
    end
end 

################################################################################
# Print Trees as latex (TikZ)
################################################################################


end
