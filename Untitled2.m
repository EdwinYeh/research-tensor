sampleInstanceIndex = randperm(3000, 100);
X = load('../20-newsgroup/source1.csv');
X = X(sampleInstanceIndex, :);
allLabel = load('../20-newsgroup/source1_label.csv');
label = allLabel(sampleInstanceIndex, :);
Y = zeros(100, 3);
for j = 1: 100
    Y(j, label(j)) = 1;
end