%% dcm_05_surprise.m — which connections track local vs global SURPRISE (not the 0/1 condition labels)?
%
% Why: with additive 0/1 coding (dcm_02_fit.m) and no interaction, the larger MMN to local deviants in
% repeat-rule blocks (LD/GD, where the deviant sound is rare) can only be absorbed by the "global"
% effect, which then appears to strengthen A1 -> STG. Two re-analyses separate the levels properly:
%
%   surprise     between-condition effects = each participant's condition-mean surprise from the
%                fitted learners (M2 local, M3 global; scripts/25_dcm_surprise_export.py), relative to
%                LSGS and scaled to a maximum of 1. Context-dependent local surprise now loads on the
%                local effect; the global effect carries only rule-level surprise.
%   interaction  0/1 labels plus a local x global term [0 0 0 1] (standard check).
%
% Network, priors, window (0–600 ms), spatial model (ECD) and group analysis exactly as dcm_02/03.
% For each coding: one full DCM per participant, PEB on B, BMR over which connection types (none /
% forward / backward / both) each effect modulates, family posteriors, and the Bayesian model average
% of every connection with its posterior probability (Pp).
%
% Needs: derivatives/dcm/spm/<sub>.mat (dcm_01_prepare.m) and
%        derivatives/dcm/surprise_by_condition.csv (run scripts/25_dcm_surprise_export.py first).
% Outputs: derivatives/dcm/surprise_coded/<coding>/{DCM_<sub>.mat, PEB_BMR.mat}, connections.csv
% Time: about 30–60 min per coding with parfor (15 participants).

clear; close all;
ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if isempty(ROOT) || ~exist(fullfile(ROOT, 'derivatives', 'dcm'), 'dir')
    ROOT = '/Volumes/Mnemosyne/Codespace/2026-local-global-eeg';
end
fprintf('Project root: %s\n', ROOT);
SPM_DIR = fullfile(ROOT, 'derivatives', 'dcm', 'spm');
OUT_DIR = fullfile(ROOT, 'derivatives', 'dcm', 'surprise_coded');
SUR_CSV = fullfile(ROOT, 'derivatives', 'dcm', 'surprise_by_condition.csv');
assert(exist(SUR_CSV, 'file') == 2, 'Missing %s — run scripts/25_dcm_surprise_export.py first', SUR_CSV);
spm('defaults', 'eeg');

% ---------------------------------------------------------------- network (as dcm_02_fit.m)
Sname = {'lA1', 'rA1', 'lSTG', 'rSTG', 'lIFG', 'rIFG', 'lIPL', 'rIPL'};
Lpos  = [-42 -22 7; 46 -14 8; -61 -32 8; 59 -25 8; -46 20 8; 46 20 8; -48 -50 42; 48 -50 42]';
n = numel(Sname);
F = zeros(n); F(3,1) = 1; F(4,2) = 1; F(5,3) = 1; F(6,4) = 1; F(7,3) = 1; F(8,4) = 1;
Bk = F';
L = zeros(n); L(5,7) = 1; L(7,5) = 1; L(6,8) = 1; L(8,6) = 1;
A = {F, Bk, L}; C = [1 1 0 0 0 0 0 0]';
CONDS = {'LSGS', 'LDGS', 'LSGD', 'LDGD'};

S = readtable(SUR_CSV, 'TextType', 'string');
files = dir(fullfile(SPM_DIR, 'sub-*.mat'));
assert(~isempty(files), 'No SPM files in %s — run dcm_01_prepare.m first', SPM_DIR);

codings = {'surprise', 'interaction'};
for cdx = 1:numel(codings)
    coding = codings{cdx};
    out_dir = fullfile(OUT_DIR, coding);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    fprintf('\n===== coding: %s =====\n', coding);

    % ---- design per participant
    X = cell(numel(files), 1);
    for s = 1:numel(files)
        sub = erase(files(s).name, '.mat');
        if strcmp(coding, 'surprise')
            T = S(S.subject == sub, :);
            [~, ord] = ismember(CONDS, T.condition);
            assert(all(ord > 0), 'Missing conditions for %s in %s', sub, SUR_CSV);
            X{s} = [T.x_loc(ord), T.x_glob(ord)];
        else
            X{s} = [0 1 0 1; 0 0 1 1; 0 0 0 1]';
        end
    end
    neff = size(X{1}, 2);
    effnames = {'local', 'global', 'interaction'};
    B = repmat({F | Bk}, 1, neff);

    % ---- fit
    parfor s = 1:numel(files)
        sub = erase(files(s).name, '.mat');
        outf = fullfile(out_dir, ['DCM_' sub '.mat']);
        if exist(outf, 'file'), continue; end
        DCM = [];
        DCM.xY.Dfile = fullfile(SPM_DIR, files(s).name);
        DCM.xY.modality = 'EEG';
        DCM.options = struct('analysis', 'ERP', 'model', 'ERP', 'spatial', 'ECD', 'trials', [1 2 3 4], ...
            'Tdcm', [0 600], 'Nmodes', 8, 'h', 1, 'onset', 60, 'dur', 16, 'D', 1, 'han', 1, 'lock', 0, ...
            'location', 0, 'symmetry', 0, 'Nmax', 256);
        DCM = spm_dcm_erp_data(DCM);
        DCM.Sname = Sname; DCM.Lpos = Lpos;
        DCM.A = A; DCM.B = B; DCM.C = C;
        DCM.xU.X = X{s}; DCM.xU.name = effnames(1:neff);
        DCM = spm_dcm_erp_dipfit(DCM);
        DCM.name = outf;
        DCM = spm_dcm_erp(DCM);
    end

    % ---- group: PEB + BMR (each effect: none / forward / backward / both)
    f = dir(fullfile(out_dir, 'DCM_sub-*.mat'));
    GCM = spm_dcm_load(cellfun(@(x) fullfile(out_dir, x), {f.name}', 'UniformOutput', false));
    M = struct('X', ones(numel(GCM), 1), 'Xnames', {{'mean'}}, 'Q', 'all');
    PEB = spm_dcm_peb(GCM, M, {'B'});
    Np = numel(PEB.Pnames); eff = zeros(Np, 1); isF = false(Np, 1); isB = false(Np, 1); rc = zeros(Np, 2);
    for p = 1:Np
        tok = regexp(PEB.Pnames{p}, 'B\{(\d+)\}\((\d+),(\d+)\)', 'tokens', 'once');
        eff(p) = str2double(tok{1}); rc(p, :) = [str2double(tok{2}) str2double(tok{3})];
        isF(p) = F(rc(p,1), rc(p,2)); isB(p) = Bk(rc(p,1), rc(p,2));
    end
    use = @(o, k) (eff == k) & ((o == 2 & isF) | (o == 3 & isB) | (o == 4 & (isF | isB)));
    combos = dec2base(0:4^neff - 1, 4) - '0' + 1;              % every combination of 4 options per effect
    K = false(size(combos, 1), Np);
    for m = 1:size(combos, 1)
        for k = 1:neff, K(m, :) = K(m, :) | use(combos(m, k), k)'; end
    end
    [BMA, BMR] = spm_dcm_peb_bmc(PEB, K);
    fam = cell(1, neff);
    for k = 1:neff
        [~, fam{k}] = spm_dcm_peb_bmc_fam(BMA, BMR, combos(:, k)', 'NONE');
        fprintf('family %-11s (none/fwd/bwd/both): %s\n', effnames{k}, mat2str(round(fam{k}.family.post(:)', 3)));
    end
    save(fullfile(out_dir, 'PEB_BMR.mat'), 'PEB', 'BMA', 'BMR', 'fam', 'X');

    % ---- connections (Pp from the BMA; Cp may be stored as variances or as a covariance matrix)
    Ep = full(BMA.Ep(:)); Cp = full(BMA.Cp); if isvector(Cp), v = Cp(:); else, v = diag(Cp); end
    Pp = spm_Ncdf(abs(Ep) ./ max(sqrt(max(v, 0)), eps));
    conn = arrayfun(@(p) sprintf('%s->%s', Sname{rc(p,2)}, Sname{rc(p,1)}), (1:Np)', 'UniformOutput', false);
    T = table(repmat({coding}, Np, 1), effnames(eff)', conn, Ep, Pp, ...
        'VariableNames', {'coding', 'effect', 'connection', 'Ep', 'Pp'});
    writetable(T, fullfile(out_dir, 'connections.csv'));
    disp(T(T.Pp > 0.95, :));
end
fprintf('\nDone. Results in %s\n', OUT_DIR);
