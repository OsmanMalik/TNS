function [cores, varargout] = TNS_TR(X, ranks, embedding_dims, varargin)
%tr_als_es Compute tensor ring decomposition via efficiently sampled ALS
%
%WARNING: This function is an adaption of the tr_als_sampled.m function in
%the repo https://github.com/OsmanMalik/tr-als-sampled. I have not tested
%all the various optional inputs to make sure that they still function as
%they should with the new sampling procedure used in tr_als_es.m, so
%proceed with caution.
%
%For loading from file: It is assumed that the tensor is stored in a
%variable Y in the mat file.
%
%cores = tr_als_es(X, ranks, embedding_dims) computes a tensor ring (TR) 
%decomposition of the input N-dimensional array X by sampling the LS
%problems using sketch sizes for each dimension given in embedding_dims.
%Ranks is a length-N vector containing the target ranks. The output cores
%is a cell containing the N cores tensors, each represented as a 3-way
%array.
%
%cores = tr_als_es(___, 'conv_crit', conv_crit) is an optional parameter
%used to control which convergence criterion is used. Set to either
%'relative error' or 'norm' to terminate when change in relative error or
%norm of TR-tensor is below the tolerance in tol. Default is that no
%convergence criterion is used.
%
%cores = tr_als_es(___, 'tol', tol) is an optional argument that controls
%the termination tolerance. If the change in the relative error is less
%than tol at the conclusion of a main loop iteration, the algorithm
%terminates. Default is 1e-3.
%
%cores = tr_als_es(___, 'maxiters', maxiters) is an optional argument that
%controls the maximum number of main loop iterations. The default is 50.
%
%cores = tr_als_es(___, 'verbose', verbose) is an optional argument that
%controls the amount of information printed to the terminal during
%execution. Setting verbose to true will result in more print out. Default
%is false.
%
%cores = tr_als_es(___, 'no_mat_inc', no_mat_inc) is used to control how
%input tensors read from file are sliced up to save RAM. We never use this
%in our experiments, and I may eventually remove this functionality.
%
%cores = tr_als_es(___, 'breakup', breakup) is an optional length-N vector
%input that can be used to break up the LS problems with multiple right
%hand sides that come up into pieces so that not all problems are solved at
%the same time. This is useful when a tensor dimension is particularly
%large.
%
%cores = tr_als_es(___, 'alpha', alpha) alpha is an optional parameter
%which controls how much Tikhonov regularization is added in LS problems.
%We found that this helped avoid ill-conditioning on certain datasets.
%
%cores = tr_als_es(___, 'permute_for_speed', permute_for_speed)
%permute_for_speed is an optional parameter which can be set to true in
%order to permute the tensor modes so that the largest mode is the first
%one. This can help speed up the sampling process since all the first-mode
%indices can be drawn at the same time from the same distribution, thus
%making it beneficial to do this for the mode with the largest dimension.
%
%cores = tr_als_es(___, 'init', init) can be used to set how the core
%tensors are initialized. If init is "randn" (the default), then all
%entries of the cores drawn independently from a standard normal
%distribution. init can also be a cell array containing initializations for
%the factor matrices.

%% Handle inputs 

% Optional inputs
params = inputParser;
addParameter(params, 'conv_crit', 'none');
addParameter(params, 'tol', 1e-3, @isscalar);
addParameter(params, 'maxiters', 50, @(x) isscalar(x) & x > 0);
addParameter(params, 'verbose', false, @isscalar);
addParameter(params, 'no_mat_inc', false);
addParameter(params, 'breakup', false);
addParameter(params, 'alpha', 0);
addParameter(params, 'permute_for_speed', false);
addParameter(params, 'init', "randn")
parse(params, varargin{:});

conv_crit = params.Results.conv_crit;
tol = params.Results.tol;
maxiters = params.Results.maxiters;
verbose = params.Results.verbose;
no_mat_inc = params.Results.no_mat_inc;
breakup = params.Results.breakup;
alpha = params.Results.alpha;
permute_for_speed = params.Results.permute_for_speed;
init = params.Results.init;

% Check if X is path to mat file on disk
%   X_mat_flag is a flag that keeps track of if X is an array or path to
%   a mat file on disk. In the latter case, X_mat will be a matfile that
%   can be used to access elements of the mat file.
if isa(X, 'char') || isa(X, 'string')
    X_mat_flag = true;
    X_mat = matfile(X, 'Writable', false);
else
    X_mat_flag = false;
end

% Permute modes of X for increased speed
if permute_for_speed
    sz = size(X);
    N = length(sz);
    [~, max_idx] = max(sz);
    perm_vec = mod((max_idx : max_idx+N-1) - 1, N) + 1;
    inv_perm_vec(perm_vec) = 1:N;
    X = permute(X, perm_vec);
    ranks = ranks(perm_vec);
end

%% Initialize cores, sampling probabilities and sampled cores

if X_mat_flag
    sz = size(X_mat, 'Y');
    N = length(sz);
    col_cell = cell(1,N);
    for n = 1:N
        col_cell{n} = ':';
    end
    
    % If value for no_mat_inc is provided, make sure it is a properly shaped
    % vector.
    if no_mat_inc(1)
        if ~(size(no_mat_inc,1)==1 && size(no_mat_inc,2)==N)
            no_mat_inc = no_mat_inc(1)*ones(1,N);
        end
    end
else
    sz = size(X);
    N = length(sz);
end

% Appropriately initialize cores
if iscell(init)
    cores = init;    
elseif strcmp(init, "randn")
    cores = initialize_cores(sz, ranks);
end

core_samples = cell(1, N);
if ~breakup(1)
    breakup = ones(1,N);
end

slow_idx = cell(1,N);
sz_shifted = [1 sz(1:end-1)];
idx_prod = cumprod(sz_shifted);
sz_pts = cell(1,N);
for n = 1:N
    sz_pts{n} = round(linspace(0, sz(n), breakup(n)+1));
    slow_idx{n} = cell(1,breakup(n));
    for brk = 1:breakup(n)
        J = embedding_dims(n);
        samples_lin_idx_2 = prod(sz_shifted(1:n))*(sz_pts{n}(brk):sz_pts{n}(brk+1)-1).';
        slow_idx{n}{brk} = repelem(samples_lin_idx_2, J, 1);
    end
end

if nargout > 1 && tol > 0 && (strcmp(conv_crit, 'relative error') || strcmp(conv_crit, 'norm'))
    conv_vec = zeros(1, maxiters);
end

% Precompute vectors with order in which to multiply cores.
% For some reason, reinitializing the vector below ends up taking some
% non-negligible amount of time if it's done inside the main loop below, so
% predefining them here instead to avoid that.
idx_order = cell(N,1);
for n = 1:N
    idx_order{n} = [n+1:N 1:n-1];
end


%% Main loop
% Iterate until convergence, for a maximum of maxiters iterations

er_old = Inf;
for it = 1:maxiters
    
    % Inner for loop
    for n = 1:N
        
        % Construct sketch and sample cores
        J2 = embedding_dims(n);
        %PsiG_2 = recursive_sketch_TR(cores, n, J1);
        [samples, sqrt_p] = draw_samples_TNS_TR(cores, n, J2);
        for m = 1:N
            if m ~= n
                core_samples{m} = cores{m}(:, samples(:,m), :);
            end
        end
        
        % Compute the row rescaling factors
        rescaling = ones(J2, 1) ./ (sqrt_p * sqrt(J2));
        
        % Construct sketched design matrix
        idx = idx_order{n}; % Order in which to multiply cores
        G_sketch = permute(core_samples{idx(1)}, [1 3 2]);
        for m = 2:N-1
            permuted_core = permute(core_samples{idx(m)}, [1 3 2]);
            G_sketch = pagemtimes(G_sketch, permuted_core);
        end
        G_sketch = permute(G_sketch, [3 2 1]);
        G_sketch = reshape(G_sketch, J2, numel(G_sketch)/J2);
        G_sketch = rescaling .* G_sketch;
        if breakup(n) > 1
            if alpha > 0
                [L, U, p] = lu(G_sketch.'*G_sketch + alpha*eye(size(G_sketch,2)), 'vector');
            else 
                [L, U, p] = lu(G_sketch, 'vector');
            end
            ZT = zeros(size(G_sketch,2), sz(n));
        end
        
        % Sample right hand side
        for brk = 1:breakup(n)
          
            no_cols = sz_pts{n}(brk+1)-sz_pts{n}(brk);
            samples_lin_idx_1 = 1 + (samples(:, idx)-1) * idx_prod(idx).';
            samples_lin_idx = repmat(samples_lin_idx_1, no_cols, 1) + slow_idx{n}{brk};
            X_sampled = X(samples_lin_idx);
            Xn_sketch = reshape(X_sampled, J2, no_cols);

            % Rescale right hand side
            Xn_sketch = rescaling .* Xn_sketch;
            
            if breakup(n) > 1
                if alpha > 0
                    ZT(:, sz_pts{n}(brk)+1:sz_pts{n}(brk+1)) = U \ (L \ G_sketch(:,p).'*Xn_sketch);
                else
                    ZT(:, sz_pts{n}(brk)+1:sz_pts{n}(brk+1)) = U \ (L \ Xn_sketch(p, :));
                end
            end
        end
        if breakup(n) > 1
            Z = ZT.';
        else
            if alpha > 0
                Z = (( G_sketch.'*G_sketch + alpha*eye(size(G_sketch,2)) ) \ ( G_sketch.'*Xn_sketch )).';
            else
                Z = (G_sketch \ Xn_sketch).';
            end
        end

        cores{n} = classical_mode_folding(Z, 2, size(cores{n}));
       
    end
    
    
    % Check convergence: Relative error
    if tol > 0 && strcmp(conv_crit, 'relative error')
        
        % Compute full tensor corresponding to cores
        Y = cores_2_tensor(cores, 'permute_for_speed', permute_for_speed);

        % Compute current relative error
        if X_mat_flag
            XX = X_mat.Y;
            er = norm(XX(:)-Y(:))/norm(XX(:));
        else
            er = norm(X(:)-Y(:))/norm(X(:));
        end        
        if verbose
            fprintf('\tRelative error after iteration %d: %.8f\n', it, er);
        end

        % Save current error to conv_vec if required
        if nargout > 1
            conv_vec(it) = er;
        end
        
        % Break if change in relative error below threshold
        if abs(er - er_old) < tol
            if verbose
                fprintf('\tRelative error change below tol; terminating...\n');
            end
            break
        end

        % Update old error
        er_old = er;
        
        
    % Check convergence: Norm change 
    elseif tol > 0 && strcmp(conv_crit, 'norm')
        
        % Compute norm of TR tensor and change if it > 1

        norm_new = normTR(cores);
        if it == 1
            norm_change = Inf;
        else
            norm_change = abs(norm_new - norm_old);
            if verbose
                fprintf('\tNorm change after iteration %d: %.8f\n', it, norm_change);
            end
        end
        
        % Save current norm_change to conv_vec
        if nargout > 1
            conv_vec(it) = norm_change;
        end
        
        % Break if change in relative error below threshold
        if norm_change < tol
            if verbose
                fprintf('\tNorm change below tol; terminating...\n');
            end
            break
        end
        
        % Update old norm
        norm_old = norm_new;
    
    % Just print iteration count
    else
        if verbose
            fprintf('\tIteration %d complete\n', it);
        end  
    end
    
end

if nargout > 1 && exist('conv_vec', 'var')
    varargout{1} = conv_vec(1:it);
else
    varargout{1} = nan;
end

if permute_for_speed
    cores = cores(inv_perm_vec);
end

end
