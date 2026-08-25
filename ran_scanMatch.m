% Remember to first add scanMatch_struct to path!
% Loop through each field in the struct

% --- Input directory (uncomment the one you want to use) ---
directory = '\\polygon.ucsd.edu\kiyonaga\yueying\gazeObject_YueyingDong\results\variables\scanMatchVariables\360to400_stim2delay_1sdelay\';
% directory = 'Z:\yueying\gazeObject_YueyingDong\results\variables\scanMatchVariables\360to400_stim2delay_2sdelay\';
% directory = 'C:\Users\yud070\Desktop\gazeObject_YueyingZoe\variables\scanMatchVariables\';  % Parent directory

filePattern = fullfile(directory, '*.mat');
files = dir(filePattern);

for i = 1:2:length(files)

    j = load([directory files(i).name]);      % delay
    q = load([directory files(i+1).name]);     % uncomment to run stim vs delay
    % q = load([directory files(i).name]);     % uncomment to run stim vs stim

    thiSubj = files(i).name(1:3);

    fieldNames = fieldnames(q);
    similarityArr = zeros(numel(fieldNames), numel(fieldNames));
    dd = 0;
    % diary('');
    % ScanMatchInfo = ScanMatch_struct();

    for k = 1:numel(fieldNames)   % for each trial, get its gaze sequence

        fieldNameStim = fieldNames{k};
        currentFieldThis_stim = q.(fieldNameStim);   % get the current field's data

        % get one trial from stim
        try
            testThis_stim = ScanMatch_FixationToSequence(currentFieldThis_stim, ScanMatchInfo);
        catch
            dd = dd + 1;
            testThis_stim = '';
        end

        % test it against every other trial in delay
        for kk = 1:numel(fieldNames)

            fieldNameDelay = fieldNames{kk};
            currentFieldThis_delay = j.(fieldNameDelay);

            % if nan, input 9999
            try
                testThis_delay = ScanMatch_FixationToSequence(currentFieldThis_delay, ScanMatchInfo);
                similarityArr(k, kk) = ScanMatch(testThis_stim, testThis_delay, ScanMatchInfo);
            catch
                similarityArr(k, kk) = 9999;
            end
        end
    end

    % Define the path components
    basePath = '\\polygon.ucsd.edu\kiyonaga\yueying\gazeObject_YueyingDong\results\scanMatch_similarity\360to400_stim2delay_1sdelay\';
    % basePath = "C:\Users\yud070\Desktop\gazeObject_YueyingZoe\results\scanMatch_similarity\";
    fileName = thiSubj + "_similarityMatrix.mat";

    % Concatenate the path and file name
    fullPath = basePath + fileName;
    save(fullPath, 'similarityArr');
end

diary off
