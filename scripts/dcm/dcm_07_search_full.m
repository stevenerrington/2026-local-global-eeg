%% dcm_07_search_full.m — add per-connection BMR to the 0–600 ms surprise/interaction DCMs (dcm_05)
%
% dcm_06_early.m reports, for every connection, the posterior probability of being present after a
% per-connection Bayesian model reduction (spm_dcm_peb_bmc without a model space). dcm_05_surprise.m
% did not, so its connections.csv only has the sign probability of the family-level BMA, which is ~1
% for almost everything. This script runs the same search on the saved 0–600 ms PEBs, so both windows
% are judged by the same statistic. No DCMs are refitted: it takes seconds.
%
% Needs  : derivatives/dcm/surprise_coded/<coding>/PEB_BMR.mat (dcm_05_surprise.m)
% Updates: derivatives/dcm/surprise_coded/<coding>/connections.csv (adds Pp_sign, Pp_family,
%          Ep_search, Pp_search; the old Pp column is kept) and PEB_BMR.mat (adds BMAs)

clear; close all;
ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if isempty(ROOT) || ~exist(fullfile(ROOT, 'derivatives', 'dcm'), 'dir')
    ROOT = '/Volumes/Mnemosyne/Codespace/2026-local-global-eeg';
end
spm('defaults', 'eeg');
for coding = {'surprise', 'interaction'}
    d = fullfile(ROOT, 'derivatives', 'dcm', 'surprise_coded', coding{1});
    S = load(fullfile(d, 'PEB_BMR.mat'));
    BMAs = spm_dcm_peb_bmc(S.PEB);                       % per-connection search
    T = readtable(fullfile(d, 'connections.csv'), 'TextType', 'string');
    assert(height(T) == numel(S.PEB.Pnames), 'connections.csv does not match PEB parameters');
    T.Pp_sign = T.Pp;
    T.Pp_family = full(S.BMA.Pw(:));
    T.Ep_search = full(BMAs.Ep(:));
    T.Pp_search = full(BMAs.Pp(:));
    writetable(T, fullfile(d, 'connections.csv'));
    save(fullfile(d, 'PEB_BMR.mat'), '-struct', 'S');
    save(fullfile(d, 'PEB_BMR.mat'), 'BMAs', '-append');
    fprintf('%s: %d of %d connections present (Pp_search > .95)\n', coding{1}, sum(T.Pp_search > .95), height(T));
end
