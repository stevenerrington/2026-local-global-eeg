%% dcm_04_robustness.m — do the DCM conclusions survive a shorter window and a different IPL site?
%
% Variants (everything else identical to dcm_02_fit.m / dcm_03_peb.m):
%   win400   time window 0–400 ms instead of 0–600 ms
%   iplsup   IPL nodes moved ~15 mm posterior-superior (angular gyrus / superior IPL, MNI ±36 -64 48)
%            instead of supramarginal/TPJ (±48 -50 42)
%   h0       no removal of the window mean (options.h = 0), so the sustained P3b is fitted as it is
%            (h = 1 removes each condition's 0–600 ms mean, which reshapes slow components)
%   img      cortical-patch sources with orientation fixed by the anatomy (options.spatial = 'IMG')
%            instead of free-orientation dipoles: a temporal source can no longer mimic a parietal
%            topography, so the parietal P3b has to come from the IPL nodes if it comes from there
% Also printed per variant: each source's share of the modelled global (300–600 ms) and local
% (80–140 ms) difference waves — in the main analysis the P3b came almost entirely from right STG.
% For each variant: one full DCM per participant, PEB on B, Bayesian model reduction over the same
% 4 x 4 forward/backward model space, family posteriors, and the BMA estimates of every connection.
% The comparison with the main analysis is printed at the end (key result: the sign reversal of
% left IPL -> STG between local (+) and global (-) deviance).
%
% Outputs: derivatives/dcm/robustness/<variant>/{DCM_<sub>.mat, PEB_BMR.mat}, robustness_summary.csv
% Requires SPM12; uses parfor if the Parallel Computing Toolbox is available (otherwise runs serially).

clear; close all;
ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if isempty(ROOT) || ~exist(fullfile(ROOT, 'derivatives', 'dcm'), 'dir')
    ROOT = '/Volumes/Mnemosyne/Codespace/2026-local-global-eeg';
end
fprintf('Project root: %s\n', ROOT);
SPM_DIR = fullfile(ROOT, 'derivatives', 'dcm', 'spm');
ROB_DIR = fullfile(ROOT, 'derivatives', 'dcm', 'robustness');
spm('defaults', 'eeg');

% ---------------------------------------------------------------- base specification (as dcm_02_fit.m)
Sname = {'lA1', 'rA1', 'lSTG', 'rSTG', 'lIFG', 'rIFG', 'lIPL', 'rIPL'};
Lpos0 = [-42 -22 7; 46 -14 8; -61 -32 8; 59 -25 8; -46 20 8; 46 20 8; -48 -50 42; 48 -50 42]';
n = numel(Sname);
F = zeros(n); F(3,1) = 1; F(4,2) = 1; F(5,3) = 1; F(6,4) = 1; F(7,3) = 1; F(8,4) = 1;
Bk = F';
L = zeros(n); L(5,7) = 1; L(7,5) = 1; L(6,8) = 1; L(8,6) = 1;
A = {F, Bk, L}; B = {F | Bk, F | Bk}; C = [1 1 0 0 0 0 0 0]';

variants = struct('name',    {'win400', 'iplsup', 'h0',    'img'}, ...
                  'Tdcm',    {[0 400],  [0 600],  [0 600], [0 600]}, ...
                  'Lpos',    {Lpos0,    Lpos0,    Lpos0,   Lpos0}, ...
                  'h',       {1,        1,        0,       1}, ...
                  'spatial', {'ECD',    'ECD',    'ECD',   'IMG'});
variants(2).Lpos(:, 7) = [-36 -64 48]';
variants(2).Lpos(:, 8) = [ 36 -64 48]';

files = dir(fullfile(SPM_DIR, 'sub-*.mat'));
assert(~isempty(files), 'No SPM files in %s — run dcm_01_prepare.m first', SPM_DIR);
rows = {};
for v = 1:numel(variants)
    V = variants(v);
    out_dir = fullfile(ROB_DIR, V.name);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    fprintf('\n===== variant %s =====\n', V.name);

    % ---- fit
    parfor s = 1:numel(files)
        sub = erase(files(s).name, '.mat');
        out = fullfile(out_dir, ['DCM_' sub '.mat']);
        if exist(out, 'file'), continue; end
        DCM = [];
        DCM.xY.Dfile = fullfile(SPM_DIR, files(s).name);
        DCM.xY.modality = 'EEG';
        DCM.options = struct('analysis', 'ERP', 'model', 'ERP', 'spatial', V.spatial, 'trials', [1 2 3 4], ...
            'Tdcm', V.Tdcm, 'Nmodes', 8, 'h', V.h, 'onset', 60, 'dur', 16, 'D', 1, 'han', 1, 'lock', 0, ...
            'location', 0, 'symmetry', 0, 'Nmax', 256);
        DCM = spm_dcm_erp_data(DCM);
        DCM.Sname = Sname; DCM.Lpos = V.Lpos;
        DCM.A = A; DCM.B = B; DCM.C = C;
        DCM.xU.X = [0 1 0 1; 0 0 1 1]'; DCM.xU.name = {'local', 'global'};
        DCM = spm_dcm_erp_dipfit(DCM);
        DCM.name = out;
        DCM = spm_dcm_erp(DCM);
    end

    % ---- PEB + BMR (same model space as dcm_03_peb.m)
    f = dir(fullfile(out_dir, 'DCM_sub-*.mat'));
    GCM = spm_dcm_load(cellfun(@(x) fullfile(out_dir, x), {f.name}', 'UniformOutput', false));
    M = struct('X', ones(numel(GCM), 1), 'Xnames', {{'mean'}}, 'Q', 'all');
    PEB = spm_dcm_peb(GCM, M, {'B'});
    Np = numel(PEB.Pnames); eff = zeros(Np, 1); isF = false(Np, 1); isB = false(Np, 1);
    rc = zeros(Np, 2);
    for p = 1:Np
        tok = regexp(PEB.Pnames{p}, 'B\{(\d+)\}\((\d+),(\d+)\)', 'tokens', 'once');
        eff(p) = str2double(tok{1}); rc(p, :) = [str2double(tok{2}) str2double(tok{3})];
        isF(p) = F(rc(p,1), rc(p,2)); isB(p) = Bk(rc(p,1), rc(p,2));
    end
    use = @(o, k) (eff == k) & ((o == 2 & isF) | (o == 3 & isB) | (o == 4 & (isF | isB)));
    K = false(16, Np); famL = zeros(1, 16); famG = zeros(1, 16); m = 0;
    for i = 1:4, for j = 1:4
        m = m + 1; K(m, :) = (use(i, 1) | use(j, 2))'; famL(m) = i; famG(m) = j;
    end, end
    [BMA, BMR] = spm_dcm_peb_bmc(PEB, K);
    [~, fL] = spm_dcm_peb_bmc_fam(BMA, BMR, famL, 'NONE');
    [~, fG] = spm_dcm_peb_bmc_fam(BMA, BMR, famG, 'NONE');
    save(fullfile(out_dir, 'PEB_BMR.mat'), 'PEB', 'BMA', 'BMR', 'fL', 'fG');
    fprintf('families (none/fwd/bwd/both)  local: %s | global: %s\n', ...
        mat2str(round(fL.family.post(:)', 3)), mat2str(round(fG.family.post(:)', 3)));

    % ---- connection estimates
    Ep = full(BMA.Ep(:)); Cp = full(BMA.Cp); if isvector(Cp), v = Cp(:); else, v = diag(Cp); end   % Cp may be stored as variances
    sd = sqrt(max(v, 0));
    Pp = spm_Ncdf(abs(Ep) ./ max(sd, eps));
    nm = {'local', 'global'};
    for p = 1:Np
        rows(end + 1, :) = {V.name, nm{eff(p)}, sprintf('%s->%s', Sname{rc(p,2)}, Sname{rc(p,1)}), ...
            Ep(p), Pp(p)}; %#ok<SAGROW>
    end
    k1 = find(eff == 1 & rc(:,1) == 3 & rc(:,2) == 7); k2 = find(eff == 2 & rc(:,1) == 3 & rc(:,2) == 7);
    fprintf('left IPL -> STG   local %+.2f (Pp %.3f)   global %+.2f (Pp %.3f)\n', Ep(k1), Pp(k1), Ep(k2), Pp(k2));

    % ---- which sources generate the modelled effects (group mean difference wave per source)
    try
        ns = numel(Sname); dl = []; dg = [];
        for s = 1:numel(GCM)
            J = full(GCM{s}.Eg.J); js = find(J); S = cell(1, 4);
            for i = 1:4
                Kt = full(GCM{s}.K{i}); S{i} = zeros(size(Kt, 1), ns);
                for q = 1:numel(js), S{i} = S{i} + J(js(q)) * Kt(:, (q - 1) * ns + (1:ns)); end
            end
            dl(:, :, s) = (S{2} + S{4}) / 2 - (S{1} + S{3}) / 2; %#ok<SAGROW>   LD - LS
            dg(:, :, s) = (S{3} + S{4}) / 2 - (S{1} + S{2}) / 2; %#ok<SAGROW>   GD - GS
        end
        pst = GCM{1}.xY.pst;
        wl = pst >= 80 & pst <= 140; wg = pst >= 300 & pst <= min(600, V.Tdcm(2));
        shl = mean(abs(mean(dl(wl, :, :), 3)), 1); shl = shl / sum(shl);
        shg = mean(abs(mean(dg(wg, :, :), 3)), 1); shg = shg / sum(shg);
        fprintf('source share  local : %s\n', strjoin(compose('%s %.0f%%', string(Sname), 100 * shl), ', '));
        fprintf('source share  global: %s\n', strjoin(compose('%s %.0f%%', string(Sname), 100 * shg), ', '));
        for k = 1:ns
            rows(end + 1, :) = {V.name, 'share_local',  Sname{k}, shl(k), NaN}; %#ok<SAGROW>
            rows(end + 1, :) = {V.name, 'share_global', Sname{k}, shg(k), NaN}; %#ok<SAGROW>
        end
    catch err
        fprintf('source shares not computed for %s: %s\n', V.name, err.message);
    end
    R2 = cellfun(@(D) 1 - sum(cellfun(@(r) sum(r(:).^2), D.R)) / ...
        sum(cellfun(@(h, r) sum((h(:) + r(:)).^2), D.H, D.R)), GCM);
    fprintf('variance explained: median %.2f (range %.2f-%.2f)\n', median(R2), min(R2), max(R2));
end
T = cell2table(rows, 'VariableNames', {'variant', 'effect', 'connection', 'Ep', 'Pp'});
writetable(T, fullfile(ROB_DIR, 'robustness_summary.csv'));
fprintf('\nSummary -> %s\n', fullfile(ROB_DIR, 'robustness_summary.csv'));
