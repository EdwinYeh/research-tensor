function X = createSparseMatrix_multiple(prefix, domainNameList, numDomain, dataType)
outputCell = cell(1,2);
for i = 1:numDomain
    fprintf('Create sparse matrix: %s\n', [prefix, domainNameList{i}]);
    data = load([prefix, domainNameList{i}]);
    if dataType == 1
        x = data(:, 1);
        y = data(:, 2);
        vals = data(:, 3);
        outputCell{i} = sparse(x, y, vals);
    elseif dataType == 2
        outputCell{i} = sparse(data);
    end
end
X = outputCell;
clear x y vals data outputCell;
end