function [samples, sqrt_probs] = draw_samples_TNS_CP(A, AtA, n, no_samp)
%draw_samples_CP Draws samples for CP design matrix as discussed in our
%paper
%
%[samples, sqrt_probs] = draw_samples_CP(A, sketch, n, no_samp) returns
%no_samp samples organized into an no_samp by N matrix, where N is the
%number of modes of the tensor being decomposed, and a vector sqrt_probs
%which contains the square root of the probability of drawing each of the
%sampled rows. The cell A should contain all the CP factor matrices in
%standard order, and the n-th factor matrix is the one being solved for.
%The n-th column of samples will just be NaN, since that index is not being
%sampled.

N = length(A);

% if statement to avoid issues when n=1 and A{1} is empty
if n == 1
    R = size(A{2},2);
else 
    R = size(A{1},2);
end

% Compute matrix Phi
AmTAm = ones(R, R);
for j = 1:N
    if j ~= n
        AmTAm = AmTAm .* AtA{j};
    end
end
Phi = pinv(AmTAm);

% Precompute the two terms in Eq (28) in paper
M1 = cell(1,N);
M2 = cell(1,N);
for j = 1:N
    if j ~= n
        M1{j} = khatrirao(A{j}.', A{j}.');
        M2{j} = A{j}.' * A{j};
    end
end

% Also precompute the products in the second of those terms.
% Note that these products incorporate the Phi matrix as well.
term_2 = cell(1,N);
for m = 1:N
    term_2{m} = Phi(:);
    %term_2{m} = ones(numel(Phi),1);
    for j = m+1:N
        if j ~= n
            term_2{m} = term_2{m} .* M2{j}(:);
        end
    end
end

% Main loop for drawing all samples
samples = nan(no_samp, N); % Each row is a realization (i_j)_{j \neq n}
sqrt_probs = ones(no_samp, 1); % To store the square root of the probability of each drawn sample
if n == 1
    first_idx = 2;
else
    first_idx = 1;
end

%% Draw first subindex for each sample

m = first_idx;

% Compute conditional probability vector
%common_terms = Phi(:);
%for j = 1:N
%    if j > m && j ~= n
%        common_terms = common_terms .* M2{j}(:);
%    end
%end
%common_terms = common_terms.';

common_terms = term_2{m}.';

%const = common_terms * M2{m}(:);
%common_terms = common_terms / const;
prob_m = common_terms * M1{m};
prob_m = prob_m / sum(prob_m);

% Draw from the vector
samples(:, m) = randsample(length(prob_m), no_samp, true, prob_m);

% Update probability vector
sqrt_probs = sqrt(prob_m(samples(:, m)));


%% Compute rest of subindices

for samp = 1:no_samp
    % Compute P(i_m | (i_j)_{j < m, ~= n}) for each m (~=n) and draw i_m 
    current_sample = samples(samp, :);
    temp_common_terms = M1{first_idx}(:, current_sample(first_idx));
    for m = first_idx+1:N
        if m ~= n
            
            % Compute conditional probability vector
            %common_terms = Phi(:);
            %for j = 1:N
            %    if j < m && j ~= n
            %        %common_terms = common_terms .* M1{j}(:, samples(samp, j));
            %        common_terms = common_terms .* M1{j}(:, current_sample(j));
            %    elseif j > m && j ~= n
            %        common_terms = common_terms .* M2{j}(:);
            %    end
            %end

            %{
            common_terms = Phi(:);
            for j = 1:m-1
                if j ~= n
                    common_terms = common_terms .* M1{j}(:, current_sample(j)); 
                end
            end
            common_terms = common_terms .* term_2{m};

            common_terms_2 = temp_common_terms .* term_2{m};
            asdf = norm(common_terms - common_terms_2);
            if asdf ~= 0
                asdf
            end  
           %}

            common_terms = (temp_common_terms .* term_2{m}).';
            %common_terms = common_terms.';
            prob_m = common_terms * M1{m};
            prob_m = prob_m / sum(prob_m);

            % Draw from the vector
            sampled_idx = mt19937ar(prob_m);
            %samples(samp, m) = sampled_idx;
            current_sample(m) = sampled_idx;
            temp_common_terms = temp_common_terms .* M1{m}(:, current_sample(m));  

            % Update probability vector
            sqrt_probs(samp) = sqrt_probs(samp) * sqrt(prob_m(sampled_idx));

            
        end
    end
    samples(samp,:) = current_sample;
end


%{
first_idx_flag = true;
for samp = 1:no_samp
    % Compute P(i_m | (i_j)_{j < m, ~= n}) for each m (~=n) and draw i_m 
    for m = first_idx:N
        if m == first_idx && ~first_idx_flag
            continue
        end
        
        if m ~= n
            
            % Compute conditional probability vector
            common_terms = Phi(:);
            for j = 1:N
                if j < m && j ~= n
                    common_terms = common_terms .* M1{j}(:, samples(samp, j));
                elseif j > m && j ~= n
                    common_terms = common_terms .* M2{j}(:);
                end
            end
            common_terms = common_terms.';
            %const = common_terms * M2{m}(:);
            %common_terms = common_terms / const;
            prob_m = common_terms * M1{m};
            prob_m = prob_m / sum(prob_m);

            if first_idx_flag
                % Draw from the vector
                samples(:, m) = randsample(length(prob_m), no_samp, true, prob_m);
                
                % Update probability vector
                sqrt_probs = sqrt(prob_m(samples(:, m)));
                
                first_idx_flag = false;
            else
                % Draw from the vector
                samples(samp, m) = mt19937ar(prob_m);
                
                % Update probability vector
                sqrt_probs(samp) = sqrt_probs(samp) * sqrt(prob_m(samples(samp,m)));
            end
            
        end
    end
end
%}

sqrt_probs = sqrt_probs';

end
