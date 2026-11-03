%% dcm_06_early.m — surprise- and interaction-coded DCM restricted to the MMN window (0–250 ms)
%
% Why: in dcm_05_surprise.m (0–600 ms) global deviance still strengthened A1 -> STG. With the only
% driving input at A1, the late P3b in the global-deviant conditions has to pass through A1 -> STG, so
% that effect may be forced by the model structure rather than by an early global effect. Fitting
% 0–250 ms (the conventional MMN window for DCM for ERP; e.g. Garrido et al. 2007, 2008) removes the
% P3b from the data:
%   - global still modulates A1 -> STG at 0–250 ms  -> a genuine early global effect
%   - it disappears                                 -> the 0–600 ms effect came from the late response
%
% Identical to dcm_05_surprise.m (network, priors, ECD spatial model, regressors, PEB, family BMR)
% except: Tdcm = [0 250] and, in addition, a per-connection search (spm_dcm_peb_bmc without a model
% space) that prunes connections individually and gives each one a posterior probability of being
% present. That probability is the standard per-connection statistic. The family-level K matrix gives
% every connection Pp ~ 1 once a family wins.
%
% Needs: derivatives/dcm/spm/<sub>.mat (dcm_01_prepare.m), derivatives/dcm/surprise_by_condition.csv
% Outputs: derivatives/dcm/early_window/<coding>/{DCM_<sub>.mat, PEB_BMR.mat, connections.csv}
%   connections.csv columns: coding, effect, connection, Ep (family BMA), Pp_sign (P that Ep has
%   this sign), Pp_family (P present, family BMR), Ep_search, Pp_search (per-connection BMR)
% Time: about 30–45 min per coding with parfor (15 participants). Re-running skips finished fits.

clear; close all;
ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if isempty(ROOT) || ~exist(fullfile(ROOT, 'derivatives', 'dcm'), 'dir')
    ROOT = '/Volumes/Mnemosyne/Codespace/2026-local-global-eeg';
end
fprintf('Project root: %s\n', ROOT);
SPM_DIR = fullfile(ROOT, 'derivatives', 'dcm', 'spm');
OUT_DIR = fullfile(ROOT, 'derivatives', 'dcm', 'early_window');
SUR_CSV = fullfile(ROOT, 'derivatives', 'dcm', 'surprise_by_condition.csv');
assert(exist(SUR_CSV, 'file') == 2, 'Missing %s — run scripts/25_dcm_surprise_export.py first', SUR_CSV);
spm('defaults', 'eeg');

TWIN = [0 250];                                   % ms — the only change from dcm_05

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
    fprintf('\n===== coding: %s, window %d–%d ms =====\n', coding, TWIN);

    % ---- between-condition design per participant
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

    % ---- fit one full DCM per participant
    parfor s = 1:numel(files)
        sub = erase(files(s).name, '.mat');
        outf = fullfile(out_dir, ['DCM_' sub '.mat']);
        if exist(outf, 'file'), continue; end
        DCM = [];
        DCM.xY.Dfile = fullfile(SPM_DIR, files(s).name);
        DCM.xY.modality = 'EEG';
        DCM.options = struct('analysis', 'ERP', 'model', 'ERP', 'spatial', 'ECD', 'trials', [1 2 3 4], ...
            'Tdcm', TWIN, 'Nmodes', 8, 'h', 1, 'onset', 60, 'dur', 16, 'D', 1, 'han', 1, 'lock', 0, ...
            'location', 0, 'symmetry', 0, 'Nmax', 256);
        DCM = spm_dcm_erp_data(DCM);
        DCM.Sname = Sname; DCM.Lpos = Lpos;
        DCM.A = A; DCM.B = B; DCM.C = C;
        DCM.xU.X = X{s}; DCM.xU.name = effnames(1:neff);
        DCM = spm_dcm_erp_dipfit(DCM);
        DCM.name = outf;
        DCM = spm_dcm_erp(DCM);
    end

    % ---- group: PEB on B, then family BMR (each effect: none / forward / backward / both)
    f = dir(fullfile(out_dir, 'DCM_sub-*.mat'));
    GCM = spm_dcm_load(cellfun(@(x) fullfile(out_dir, x), {f.name}', 'UniformOutput', false));
    % fit quality: variance explained in the projected data (data = prediction H + residual R)
    r2 = cellfun(@(d) 100 * (1 - sum(spm_vec(d.R).^2) / sum((spm_vec(d.H) + spm_vec(d.R)).^2)), GCM);
    fprintf('variance explained per participant (%%): %s\n', mat2str(round(r2(:)')));
    M = struct('X', ones(numel(GCM), 1), 'Xnames', {{'mean'}}, 'Q', 'all');
    PEB = spm_dcm_peb(GCM, M, {'B'});
    Np = numel(PEB.Pnames); eff = zeros(Np, 1); isF = false(Np, 1); isB = false(Np, 1); rc = zeros(Np, 2);
    for p = 1:Np
        tok = regexp(PEB.Pnames{p}, 'B\{(\d+)\}\((\d+),(\d+)\)', 'tokens', 'once');
        eff(p) = str2double(tok{1}); rc(p, :) = [str2double(tok{2}) str2double(tok{3})];
        isF(p) = F(rc(p,1), rc(p,2)); isB(p) = Bk(rc(p,1), rc(p,2));
    end
    use = @(o, k) (eff == k) & ((o == 2 & isF) | (o == 3 & isB) | (o == 4 & (isF | isB)));
    combos = dec2base(0:4^neff - 1, 4) - '0' + 1;
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

    % ---- per-connection BMR: greedy search prunes each connection on its own merits
    BMAs = spm_dcm_peb_bmc(PEB);
    save(fullfile(out_dir, 'PEB_BMR.mat'), 'PEB', 'BMA', 'BMR', 'fam', 'X', 'BMAs', 'TWIN');

    % ---- table
    Ep = full(BMA.Ep(:)); Cp = full(BMA.Cp); if isvector(Cp), v = Cp(:); else, v = diag(Cp); end
    Pp_sign = spm_Ncdf(abs(Ep) ./ max(sqrt(max(v, 0)), eps));
    Pp_family = full(BMA.Pw(:));
    Ep_search = full(BMAs.Ep(:));
    if isfield(BMAs, 'Pp'), Pp_search = full(BMAs.Pp(:)); else, Pp_search = nan(Np, 1); end
    conn = arrayfun(@(p) sprintf('%s->%s', Sname{rc(p,2)}, Sname{rc(p,1)}), (1:Np)', 'UniformOutput', false);
    T = table(repmat({coding}, Np, 1), effnames(eff)', conn, Ep, Pp_sign, Pp_family, Ep_search, Pp_search, ...
        'VariableNames', {'coding', 'effect', 'connection', 'Ep', 'Pp_sign', 'Pp_family', 'Ep_search', 'Pp_search'});
    writetable(T, fullfile(out_dir, 'connections.csv'));
    disp(T(T.Pp_search > 0.95, :));
end
fprintf('\nDone. Results in %s\n', OUT_DIR);
