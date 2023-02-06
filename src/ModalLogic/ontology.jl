world2frametype = Dict([
    OneWorld => FullDimensionalFrame{0,OneWorld,Bool},
    Interval => FullDimensionalFrame{1,Interval,Bool},
    Interval2D => FullDimensionalFrame{2,Interval2D,Bool},
])

# An ontology is a pair `world type` + `set of relations`, and represents the kind of
#  modal frame that underlies a certain logic
struct Ontology{W<:AbstractWorld}

    relations :: AbstractVector{<:AbstractRelation}

    function Ontology{W}(_relations::AbstractVector) where {W<:AbstractWorld}
        _relations = collect(unique(_relations))
        for relation in _relations
            @assert goeswith(world2frametype[W], relation) "Can't instantiate Ontology{$(W)} with relation $(relation)!"
        end
        if W == OneWorld && length(_relations) > 0
          _relations = similar(_relations, 0)
          @warn "Instantiating Ontology{$(W)} with empty set of relations!"
        end
        new{W}(_relations)
    end

    Ontology(worldType::Type{<:AbstractWorld}, relations) = Ontology{worldType}(relations)
end

worldtype(::Ontology{WT}) where {WT<:AbstractWorld} = WT
relations(o::Ontology) = o.relations

Base.show(io::IO, o::Ontology{WT}) where {WT<:AbstractWorld} = begin
    if o == OneWorldOntology
        print(io, "OneWorldOntology")
    else
        print(io, "Ontology{")
        show(io, WT)
        print(io, "}(")
        if issetequal(relations(o), SoleLogics.IARelations)
            print(io, "IA")
        elseif issetequal(relations(o), SoleLogics.IARelations_extended)
            print(io, "IA_extended")
        elseif issetequal(relations(o), SoleLogics.IA2DRelations)
            print(io, "IA²")
        elseif issetequal(relations(o), SoleLogics.IA2D_URelations)
            print(io, "IA²_U")
        elseif issetequal(relations(o), SoleLogics.IA2DRelations_extended)
            print(io, "IA²_extended")
        elseif issetequal(relations(o), SoleLogics.RCC8Relations)
            print(io, "RCC8")
        elseif issetequal(relations(o), SoleLogics.RCC5Relations)
            print(io, "RCC5")
        else
            show(io, relations(o))
        end
        print(io, ")")
    end
end
