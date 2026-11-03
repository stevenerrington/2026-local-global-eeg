%% dcm_02_fit.m — fit one full DCM for ERPs per subject (2 x 2 local-global design).
%
% Network: Garrido et al. (2008, 2009) MMN architecture, both hemispheres, extended with an inferior
% parietal node so the model can generate the parietal P3b (the global effect is a parietal
% positivity; without a parietal source it would have to be produced by frontotemporal sources).
% MNI coordinates (mm):
%   1 lA1  [-42 -22  7]    2 rA1  [ 46 -14  8]     primary auditory cortex (inputs)
%   3 lSTG [-61 -32  8]    4 rSTG [ 59 -25  8]     superior temporal gyrus
%   5 lIFG [-46  20  8]    6 rIFG [ 46  20  8]     inferior frontal gyrus
%   7 lIPL [-48 -50 42]    8 rIPL [ 48 -50 42]     inferior parietal lobule / supramarginal (TPJ)
%   forward : A1 -> STG -> IFG,  STG -> IPL          backward : the reverse of each forward connection
%   lateral : IFG <-> IPL within each hemisphere (same hierarchical level; fixed, not modulated)
% IPL coordinates are a-priori (TPJ/IPL, the region most consistently implicated in the P3b); the
% eLORETA analysis did not provide a reliable focal estimate, so data-driven coordinates were not used.
%
% Trials (order fixed by the export): 1 LSGS, 2 LDGS, 3 LSGD, 4 LDGD
% Between-trial effects (xU.X, coded relative to LSGS):
%   B{1} local  deviance  [0 1 0 1]
%   B{2} global deviance  [0 0 1 1]
% Full model: both effects may modulate every forward and backward connection. Reduced models
% (which connection types each effect needs) are compared afterwards with PEB + Bayesian model
% reduction (dcm_03_peb.m), so only this one model is inverted per subject.
%
% Window 0–600 ms (covers the MMN and most of the P3b), 8 spatial modes, ECD sources, ERP neural
% mass model. Output: derivatives/dcm/fits/DCM_<sub>_full.mat
%
% Each inversion takes minutes; parfor over subjects if the Parallel Computing Toolbox is available.

clear; close all;
ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % project root (scripts/dcm/ -> ..)
if isempty(ROOT) || ~exist(fullfile(ROOT, 'derivatives', 'dcm'), 'dir')   % run as a section / pasted code
    ROOT = '/Volumes/Mnemosyne/Codespace/2026-local-global-eeg';
end
assert(exist(fullfile(ROOT, 'derivatives', 'dcm'), 'dir') == 7, 'Project not found at %s', ROOT);
fprintf('Project root: %s\n', ROOT);
SPM_DIR = fullfile(ROOT, 'derivatives', 'dcm', 'spm');
FIT_DIR = fullfile(ROOT, 'derivatives', 'dcm', 'fits');
if ~exist(FIT_DIR, 'dir'), mkdir(FIT_DIR); end
spm('defaults', 'eeg');

% ---------------------------------------------------------------- model specification
Sname = {'lA1', 'rA1', 'lSTG', 'rSTG', 'lIFG', 'rIFG', 'lIPL', 'rIPL'};
Lpos  = [-42 -22 7; 46 -14 8; -61 -32 8; 59 -25 8; -46 20 8; 46 20 8; -48 -50 42; 48 -50 42]';
n     = numel(Sname);
F = zeros(n);                                                         % forward (to, from)
F(3,1) = 1; F(4,2) = 1;                                               % A1 -> STG
F(5,3) = 1; F(6,4) = 1;                                               % STG -> IFG
F(7,3) = 1; F(8,4) = 1;                                               % STG -> IPL
Bk = F';                                                              % backward = reverse of forward
L = zeros(n); L(5,7) = 1; L(7,5) = 1; L(6,8) = 1; L(8,6) = 1;         % lateral IFG <-> IPL
A  = {F, Bk, L};
B  = {F | Bk, F | Bk};                                                % full: both effects on all fwd/bwd connections
C  = [1 1 0 0 0 0 0 0]';

files = dir(fullfile(SPM_DIR, 'sub-*.mat'));
assert(~isempty(files), 'No SPM files in %s — run dcm_01_prepare.m first', SPM_DIR);
parfor s = 1:numel(files)
    sub = erase(files(s).name, '.mat');
    out = fullfile(FIT_DIR, ['DCM_' sub '_full.mat']);
    if exist(out, 'file'), continue; end
    DCM = [];
    DCM.xY.Dfile = fullfile(SPM_DIR, files(s).name);
    DCM.xY.modality = 'EEG';
    DCM.options.analysis = 'ERP';
    DCM.options.model    = 'ERP';
    DCM.options.spatial  = 'ECD';
    DCM.options.trials   = [1 2 3 4];
    DCM.options.Tdcm     = [0 600];
    DCM.options.Nmodes   = 8;
    DCM.options.h        = 1;
    DCM.options.onset    = 60;
    DCM.options.dur      = 16;
    DCM.options.D        = 1;
    DCM.options.han      = 1;
    DCM.options.lock     = 0;
    DCM.options.location = 0;
    DCM.options.symmetry = 0;
    DCM.options.Nmax     = 256;
    DCM = spm_dcm_erp_data(DCM);
    DCM.Sname = Sname;
    DCM.Lpos  = Lpos;
    DCM.A = A; DCM.B = B; DCM.C = C;
    DCM.xU.X    = [0 1 0 1; 0 0 1 1]';
    DCM.xU.name = {'local', 'global'};
    DCM = spm_dcm_erp_dipfit(DCM);
    DCM.name = out;
    DCM = spm_dcm_erp(DCM);                                           % saves to DCM.name
    fprintf('%s: F = %.1f\n', sub, DCM.F);
end
