%% dcm_03_peb.m — group inference: which connections do local and global deviance modulate?
%
% PEB over subjects on the B parameters of the full models (dcm_02_fit.m), then Bayesian model
% reduction over a 4 x 4 model space:
%   local  effect (B{1}) on: none | forward only | backward only | forward + backward
%   global effect (B{2}) on: none | forward only | backward only | forward + backward
% Family comparison per factor answers the a-priori questions:
%   Q1  local  deviance — forward-dominant?   (prediction: feedforward prediction error)
%   Q2  global deviance — needs backward?     (prediction: top-down / feedback signalling)
% Bayesian model average of the B parameters (Pp > .95 shown) for the connection-level picture.
% Families are defined on connection type, so they pool frontal and parietal routes; the BMA shows
% which specific connections (e.g. IPL -> STG vs IFG -> STG) carry the global effect.
%
% Outputs: derivatives/dcm/peb/{PEB_B.mat, BMR.mat, family_local.mat, family_global.mat},
%          figures/dcm_families.png, figures/dcm_bma_B.png

clear; close all;
ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % project root (scripts/dcm/ -> ..)
if isempty(ROOT) || ~exist(fullfile(ROOT, 'derivatives', 'dcm'), 'dir')   % run as a section / pasted code
    ROOT = '/Volumes/Mnemosyne/Codespace/2026-local-global-eeg';
end
assert(exist(fullfile(ROOT, 'derivatives', 'dcm'), 'dir') == 7, 'Project not found at %s', ROOT);
fprintf('Project root: %s\n', ROOT);
FIT_DIR = fullfile(ROOT, 'derivatives', 'dcm', 'fits');
PEB_DIR = fullfile(ROOT, 'derivatives', 'dcm', 'peb');
FIG_DIR = fullfile(ROOT, 'figures');
if ~exist(PEB_DIR, 'dir'), mkdir(PEB_DIR); end
spm('defaults', 'eeg');

files = dir(fullfile(FIT_DIR, 'DCM_sub-*_full.mat'));
assert(~isempty(files), 'No fitted DCMs in %s — run dcm_02_fit.m first (and let it finish)', FIT_DIR);
GCM   = cellfun(@(f) fullfile(FIT_DIR, f), {files.name}', 'UniformOutput', false);
GCM   = spm_dcm_load(GCM);
N     = numel(GCM);
fprintf('%d subjects\n', N);

% ---------------------------------------------------------------- PEB (group mean only)
M = struct();
M.X = ones(N, 1);
M.Xnames = {'mean'};
M.Q = 'all';
PEB = spm_dcm_peb(GCM, M, {'B'});
save(fullfile(PEB_DIR, 'PEB_B.mat'), 'PEB');

% ---------------------------------------------------------------- reduced model space (4 x 4)
% Specified as a logical matrix over the PEB parameters (models x parameters), built from the
% parameter names ('B{k}(to,from)'). Passing template DCMs instead fails in some SPM12 versions:
% the templates inherit the full model's prior covariance, so every template looks like the full model.
F  = GCM{1}.A{1} > 0;                          % forward mask  (to, from)
Bk = GCM{1}.A{2} > 0;                          % backward mask
names = {'none', 'forward', 'backward', 'fwd+bwd'};
Np  = numel(PEB.Pnames);
eff = zeros(Np, 1); isF = false(Np, 1); isB = false(Np, 1);
for p = 1:Np
    tok = regexp(PEB.Pnames{p}, 'B\{(\d+)\}\((\d+),(\d+)\)', 'tokens', 'once');
    assert(~isempty(tok), 'Unexpected parameter name %s', PEB.Pnames{p});
    k = str2double(tok{1}); r = str2double(tok{2}); c = str2double(tok{3});
    eff(p) = k; isF(p) = F(r, c); isB(p) = Bk(r, c);
end
fprintf('%d B parameters: %d forward, %d backward (local %d, global %d)\n', Np, sum(isF), sum(isB), ...
    sum(eff == 1), sum(eff == 2));
use = @(o, k) (eff == k) & ((o == 2 & isF) | (o == 3 & isB) | (o == 4 & (isF | isB)));   % o: 1 none .. 4 both
K = false(16, Np);
fam_local = zeros(1, 16); fam_global = zeros(1, 16);
m = 0;
for i = 1:4                                    % local
    for j = 1:4                                % global
        m = m + 1;
        K(m, :) = (use(i, 1) | use(j, 2))';
        fam_local(m) = i; fam_global(m) = j;
    end
end
[BMA, BMR] = spm_dcm_peb_bmc(PEB, K);
save(fullfile(PEB_DIR, 'BMR.mat'), 'BMA', 'BMR');

% ---------------------------------------------------------------- family comparison
[~, famL] = spm_dcm_peb_bmc_fam(BMA, BMR, fam_local,  'NONE');
[~, famG] = spm_dcm_peb_bmc_fam(BMA, BMR, fam_global, 'NONE');
save(fullfile(PEB_DIR, 'family_local.mat'),  'famL');
save(fullfile(PEB_DIR, 'family_global.mat'), 'famG');
fprintf('\nLOCAL  deviance — family posterior: %s\n', strjoin(compose('%s %.3f', string(names'), famL.family.post(:)), ' | '));
fprintf('GLOBAL deviance — family posterior: %s\n', strjoin(compose('%s %.3f', string(names'), famG.family.post(:)), ' | '));

% ---------------------------------------------------------------- figures
f = figure('Color', 'w', 'Position', [100 100 640 260]);
subplot(1, 2, 1); bar(famL.family.post, 'FaceColor', [0.77 0.31 0.32]);
set(gca, 'XTickLabel', names, 'Box', 'off'); ylim([0 1]); ylabel('Family posterior'); title('Local deviance');
subplot(1, 2, 2); bar(famG.family.post, 'FaceColor', [0.51 0.45 0.70]);
set(gca, 'XTickLabel', names, 'Box', 'off'); ylim([0 1]); title('Global deviance');
exportgraphics(f, fullfile(FIG_DIR, 'dcm_families.png'), 'Resolution', 300);

spm_dcm_peb_review(BMA, GCM);                  % interactive review of the averaged B parameters
