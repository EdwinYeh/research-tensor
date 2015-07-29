prefix = 'origin/';
datasetId = 20;
sourceOutputFile = fopen(sprintf('%ssource%d.csv', prefix, datasetId), 'w');
targetOutputFile = fopen(sprintf('%starget%d.csv', prefix, datasetId), 'w');
%setting1 = [896 666 666 666 84], sum = 2978, id = 17, 33%
%setting2 = [1344 1000 1000 1000 126], sum = 4470, id = 18, 50%
%setting4 = [2016 1500 1500 1500 189], sum = 6705, id = 19, 75%
%datasetid = 20, [17422 4470]
M1 = load([prefix 'source10_original.csv']);
M2 = load([prefix 'source13_original.csv']);
M3 = load([prefix 'source14_original.csv']);
M4 = load([prefix 'source15_original.csv']);
M5 = load([prefix 'source16_original.csv']);
source_ori = [M1(1:17422,:) M2(1:17422,:) M3(1:17422,:) M4(1:17422,:) M5(1:17422,:)];

[col, row] = size(source_ori);
numSample = row/2;
sample = randperm(row, numSample);
fprintf('source%d/[instance, feature] = [%d %d]\n', datasetId, col, numSample);
source_ori = source_ori(:, sample);
for i = 1:col
    for j = 1:numSample
        fprintf(sourceOutputFile, sprintf('%d,%d,%f\n', i, j, source_ori(i,j)));
    end
end

M1 = load([prefix 'target10_original.csv']);
M2 = load([prefix 'target13_original.csv']);
M3 = load([prefix 'target14_original.csv']);
M4 = load([prefix 'target15_original.csv']);
M5 = load([prefix 'target16_original.csv']);
target_ori = [M1 M2 M3 M4 M5];
[col, row] = size(target_ori);
numSample = row/2;
sample = randperm(row, numSample);
fprintf('target%d/[instance, feature] = [%d %d]\n', datasetId, col, numSample);
target_ori = target_ori(:, sample);
for i = 1:col
    for j = 1:numSample
        fprintf(targetOutputFile, sprintf('%d,%d,%f\n', i, j, target_ori(i,j)));
    end
end

fclose(sourceOutputFile);
fclose(targetOutputFile);