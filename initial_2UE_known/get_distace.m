function distance = get_distace(A,B)
    distance = sum((A-B).*(A-B))^0.5;
end