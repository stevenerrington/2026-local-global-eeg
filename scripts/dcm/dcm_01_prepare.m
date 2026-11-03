%% dcm_01_prepare.m — convert exported ERPs to SPM MEEG files and build an EEG BEM head model.
%
% Input : derivatives/dcm/<sub>_erp.mat   (from scripts/21_dcm_export.py)
% Output: derivatives/dcm/spm/<sub>.mat/.dat  (SPM evoked file, 4 conditions, template head model)
%
% Steps per subject
%   1  FieldTrip-style struct -> SPM MEEG (spm_eeg_ft2spm); conditions LSGS, LDGS, LSGD, LDGD
%   2  default EEG sensor positions + fiducials (SPM template cap, matched by channel name)
%   3  head model: SPM template MRI mesh, fiducial coregistration, 'EEG BEM' forward model
% Data are average-referenced (as exported); SPM's forward model is re-referenced to the same montage.
%
% Requires SPM12 on the MATLAB path. Run from anywhere.

clear; close all;
ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % project root (scripts/dcm/ -> ..)
if isempty(ROOT) || ~exist(fullfile(ROOT, 'derivatives', 'dcm'), 'dir')   % run as a section / pasted code
    ROOT = '/Volumes/Mnemosyne/Codespace/2026-local-global-eeg';
end
assert(exist(fullfile(ROOT, 'derivatives', 'dcm'), 'dir') == 7, 'Project not found at %s', ROOT);
fprintf('Project root: %s\n', ROOT);
IN_DIR  = fullfile(ROOT, 'derivatives', 'dcm');
OUT_DIR = fullfile(IN_DIR, 'spm');
CONDS   = {'LSGS', 'LDGS', 'LSGD', 'LDGD'};
if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end
spm('defaults', 'eeg');

files = dir(fullfile(IN_DIR, 'sub-*_erp.mat'));
assert(~isempty(files), 'No exported ERPs in %s — run scripts/21_dcm_export.py first', IN_DIR);
for s = 1:numel(files)
    sub = erase(files(s).name, '_erp.mat');
    E   = load(fullfile(IN_DIR, files(s).name));

    % ---- 1  FieldTrip struct -> SPM
    ft = [];
    ft.label  = cellstr(E.label);
    ft.time   = E.time(:)';
    ft.trial  = permute(E.data, [3 1 2]);            % rpt x chan x time
    ft.dimord = 'rpt_chan_time';
    D = spm_eeg_ft2spm(ft, fullfile(OUT_DIR, sub));
    D = type(D, 'evoked');
    D = conditions(D, ':', CONDS);
    D = chantype(D, ':', 'EEG');
    D = units(D, ':', 'uV');
    D = repl(D, ':', double(E.n_trials(:)'));        % number of trials averaged per condition
    save(D);

    % ---- 2  default sensor positions + fiducials (matched by label)
    S = []; S.D = D; S.task = 'defaulteegsens'; S.save = 1;
    D = spm_eeg_prep(S);
    sens = D.sensors('EEG');
    missing = setdiff(D.chanlabels(D.indchantype('EEG')), sens.label);
    if ~isempty(missing)
        warning('%s: no template position for %s -> marked bad', sub, strjoin(missing, ', '));
        D = badchannels(D, D.indchannel(missing), 1);
        save(D);
    end

    % ---- 3  template head model, EEG BEM (direct calls rather than the batch, so errors are visible)
    D = spm_eeg_load(fullfile(D));
    D.val = 1;
    D.inv = {struct('comment', {'template EEG BEM'}, 'date', {datestr(now)})};
    D = spm_eeg_inv_mesh_ui(D, 1, 1, 2);                 % template MRI, normal mesh resolution
    meegfid = D.fiducials;
    mrifid  = D.inv{1}.mesh.fid;
    fprintf('  sensor fiducials: %s | MRI fiducials: %s\n', ...
        strjoin(meegfid.fid.label(:)', ', '), strjoin(mrifid.fid.label(:)', ', '));
    want = {'nas', 'lpa', 'rpa'};
    norm = @(c) regexprep(lower(c(:)'), '^spm', '');    % sensor fiducials are 'spmnas' etc.
    [ok1, i1] = ismember(want, norm(meegfid.fid.label));
    [ok2, i2] = ismember(want, norm(mrifid.fid.label));
    assert(all(ok1) && all(ok2), '%s: fiducials nas/lpa/rpa not found (see labels above)', sub);
    meegfid.fid.pnt   = meegfid.fid.pnt(i1, :);  meegfid.fid.label = want(:);
    mrifid.fid.pnt    = mrifid.fid.pnt(i2, :);   mrifid.fid.label  = want(:);
    D = spm_eeg_inv_datareg_ui(D, 1, meegfid, mrifid, 0);   % fiducial coregistration, no head shape
    D.inv{1}.forward = struct('voltype', 'EEG BEM');
    D = spm_eeg_inv_forward(D);
    save(D);
    fprintf('%s: converted, %d channels, head model done\n', sub, numel(D.indchantype('EEG', 'GOOD')));
end
