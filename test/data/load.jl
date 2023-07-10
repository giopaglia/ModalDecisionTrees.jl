
function load_digits()
    data_path = joinpath(dirname(pathof(ModalDecisionTrees)), "..", "test/data/")

    f = open(joinpath(data_path, "digits.csv"))
    data = readlines(f)[2:end]
    data = [[parse(Float32, i) for i in split(row, ",")] for row in data]
    data = hcat(data...)
    y = Int.(data[1, 1:end]) .+ 1
    X = convert(Matrix, transpose(data[2:end, 1:end]))
    return X, y
end
