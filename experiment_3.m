% Experiment 3
%
% This script is used to run the feature extraction experiment in the
% paper.
%
%Note that this function requires some files from the tr-als-sampled
%repo which is available at: https://github.com/OsmanMalik/tr-als-sampled

vec = @(x) x(:);
[X, class_array] = data_loader('coil-100');

%method = "cp_als";
%no_trials = 10;
ACC = zeros(no_trials, 1);
ER = zeros(no_trials, 1);
TIME = zeros(no_trials, 1);
alpha = 1e-2; % Regularizer coefficient, if regularization needed.

sz = size(X);
ranks = [5,5,5,5];
CP_rank = 25;
tol = 0;
maxiters = 40;
for tr = 1:no_trials
    exp3_tic = tic;
    switch method
        case "cp_als"
            CPD = cp_als(tensor(X), CP_rank, 'tol', tol, 'maxiters', maxiters);
            factor_mats = CPD.U;
            lambda = CPD.lambda;
        case "cpd_als" % Tensorlab
            options.Algorithm = @cpd_als;
            options.MaxIter = maxiters;
            factor_mats = cpd(X, CP_rank, options);
            lambda = ones(CP_rank,1);
        case "cpd_minf" % Tensorlab
            options.Algorithm = @cpd_minf;
            factor_mats = cpd(X, CP_rank, options);
            lambda = ones(CP_rank,1);
        case "cpd_nls" % Tensorlab
            options.Algorithm = @cpd_nls;
            options.MaxIter = maxiters;
            factor_mats = cpd(X, CP_rank, options);
            lambda = ones(CP_rank,1);
        case "cpd3_sd" % Tensorlab -- only 3-way tensors
            options.Algorithm = @cpd3_sd;
            options.MaxIter = maxiters;
            factor_mats = cpd(X, CP_rank, options);
            lambda = ones(CP_rank,1);
        case "cpd3_sgsd" % Tensorlab -- only 3-way tensors
            options.Algorithm = @cpd3_sgsd;
            options.MaxIter = maxiters;
            factor_mats = cpd(X, CP_rank, options);
            lambda = ones(CP_rank,1);
        case "cpd_rbs" % Tensorlab
            options.Algorithm = @cpd_rbs;
            options.MaxIter = maxiters;
            factor_mats = cpd(X, CP_rank, options);
            lambda = ones(CP_rank,1);
        case "cp_arls_lev"
            J = 2000;
            factor_mats = cp_arls_lev(X, CP_rank, J*ones(1,4), 'maxiters', maxiters);
            lambda = ones(CP_rank,1);
        case "cp_als_es"
            J1 = 10000;
            J2 = 2000;        
            factor_mats = cp_als_es(X, CP_rank, J1*ones(1,4), J2*ones(1,4), 'maxiters', maxiters, 'permute_for_speed', true);
            lambda = ones(CP_rank,1);
        case "tns_cp"
            J = 2000;        
            factor_mats = TNS_CP(X, CP_rank, J*ones(1,4), 'maxiters', maxiters, 'permute_for_speed', true);
            lambda = ones(CP_rank,1);
        case "tr_als"
            cores = tr_als(X, ranks, 'tol', tol, 'conv_crit', 'relative error', 'maxiters', maxiters, 'verbose', true);
        case "rtr_als"
            K = [20 20 3 200];
            cores = rtr_als(X, ranks, K, 'tol', tol, 'conv_crit', 'relative error', 'maxiters', maxiters, 'verbose', true);
        case "tr_als_sampled"
            J = 1000;
            cores = tr_als_sampled(X, ranks, J*ones(size(sz)), 'tol', tol, 'conv_crit', 'relative error', 'maxiters', maxiters, 'resample', true, 'verbose', true, 'alpha', alpha);
        case "tr_svd"
            cores = TRdecomp_ranks(X, ranks);
        case "tr_svd_rand"
            oversamp = 10;
            cores = tr_svd_rand(X, ranks, oversamp);
        case "tr_als_es"
            J1 = 10000;
            J2 = 1000;
            cores = tr_als_es(X, ranks, J1, J2*ones(size(sz)), 'tol', tol, 'permute_for_speed', true, 'conv_crit', 'relative error', 'maxiters', maxiters, 'verbose', true, 'alpha', alpha);
        case "tns_tr"
            J = 1000;
            cores = TNS_TR(X, ranks, J*ones(size(sz)), 'tol', tol, 'permute_for_speed', true, 'conv_crit', 'relative error', 'maxiters', maxiters, 'verbose', true, 'alpha', alpha);
        otherwise
            error('Invalid method.')
    end
    exp3_toc = toc(exp3_tic);
    TIME(tr) = exp3_toc;
    
    % The if statement below makes sure that the 4th core/factor matrix is
    % reshaped appropriately.
    if contains(method, "tr") % TR method
        feat_mat = reshape(permute(cores{end}, [2 1 3]), 7200, ranks(end-1)*ranks(end));
        Y = cores_2_tensor(cores);
    else % CP method
        feat_mat = factor_mats{end};
        Y = double(tensor(ktensor(lambda, factor_mats)));    
    end

    Mdl = fitcknn(feat_mat, class_array, 'numneighbors', 1);
    cvmodel = crossval(Mdl);
    loss = kfoldLoss(cvmodel, 'lossfun', 'classiferror');
    accuracy = 1 - loss;
    ACC(tr) = accuracy;
    er = norm(vec(X - Y))/norm(vec(Y));
    ER(tr) = er;

    fprintf('Accuracy is %.4f\n', accuracy);
    fprintf('Error is %.4f\n', er);
    fprintf('Decomposition time was %.4f\n', exp3_toc);
end

fprintf('Mean accuracy is %.4f\n', mean(ACC));
fprintf('Mean error is %.4f\n', mean(ER));
fprintf('Mean decomposition time was %.4f\n', mean(TIME));
%{
%save("experiment_3_" + method + "_results.mat", 'accuracy', 'exp3_toc');
%}
