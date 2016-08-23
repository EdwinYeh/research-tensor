if strncmp(system_dependent('getos'),'Linux',5)
	parentPath = sprintf('/');
	inputPath = sprintf('datasets/');
    outputPath = sprintf('/');
    addpath([parentPath 'baselines/']);
    addpath([parentPath 'baselines/OKM/']);
	addpath([parentPath 'baselines/MOC/']);
	addpath([parentPath 'evaluate/']);
else
    parentPath = sprintf('');
	inputPath = sprintf('datasets/');
    outputPath = sprintf('');
    addpath([parentPath 'baselines/']);
	addpath([parentPath 'baselines/OKM/']);
	addpath([parentPath 'baselines/MOC/']);
	addpath([parentPath 'evaluate/']);
end