clc;close all;clear all;
% Implementation for the Phase Transition of the Alternating Direction Method presented in the paper
% "Finding a Sparse Vector in a Subspace: Linear Sparsity using Alternating Directions",
% by Qing Qu, Ju Sun, and John Wright.
%
% Planted sparse vector model:
% Y = orth([x_0, G]), x_0: random k support sparse |x_0(i)| = 1/p,
% G~i.i.d. Gaussian N(0,1/sqrt(p)I). 
%
% gs.m: Gram-Schimdt orthogonalization function
% ADM.m: solve the following problem 
% min_{q,x} 1/2*||Y*q - x||_2^2 + lambda * ||x||_1, s.t. ||q||_2 = 1 
% by alternating minimization method (ADM), lambda is the penalty
% parameter.
% 
% Code written by Qing Qu, Ju Sun, and John Wright. 
% Last Updated: Sun 26 Apr 2015 12:24:57 PM EDT 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% parameter settings
n = 1:5:100; % subspace dimension
k = 1:20:600; % sparsity
C = 5; % the ratio between p and n: p = C * n * log n
T = 10; % simulation times for each pair (p,k)
MaxIter = 5000; % Max iteration before stop
tol_adm = 1e-5; % tolerance for convergence of the adm algorithm
tol_s = 1e-4; % error tolerance for judging the success of recovery
Prb = zeros(length(k),length(n));% record the recovery probability

%% main iterations
for t = 1:T
    for i = 1:length(k)
        for j = 1:length(n)
            
            % skip the impossible case when k>p
            p = round(C*n(j)*log(n(j)));
            
            if (k(i)>p)
                continue;
            end
            
            % generate the data
            
            % As described in the paper, for general data, it is enough to input
            % an arbitrary orthonormal basis for the subspace. Below the setting 
            % is for the purposes of verifying our theory. 
            
            % generate the planted sparse vector x_0 
            x0 = zeros(p,1);
            x0(randperm(p,k(i)),1) = randn(k(i),1);
            x0 = x0/norm(x0);
            % generate Gaussian basis
            G = randn(p,n(j)-1)/sqrt(p);
            Y = [x0,G];
            Y = gs(Y); % normalization by Gram-Schimdt 
            
            lambda = 1/sqrt(p);% penalty parameter
            q_mtx = zeros(n(j),p);
            q_true = [1;zeros(n(j)-1,1)];
            
            % exhausting all initializations
            flag = 0;
            for l = 1:p  
                q_0 = (Y(l,:)/norm(Y(l,:)))';
                q_mtx(:,l) = ADM(Y,q_0,lambda,MaxIter,tol_adm);
                cor_vec = abs(q_mtx(:,l)' * q_true);
                error = 1 - cor_vec;
                if(error<=tol_s)
                    flag = 1;
                    break;
                end
            end
            
            if (flag ==1)
                Prb(i,j) = Prb(i,j) + 1;
            end
            % print intermediate results
            fprintf('Simulation=%d, Ambient Dimension = %d, Sparsity = %d, Recovery Error = %f\n'...
                ,t,p,k(i),error);
            figure(1);
            imagesc(round(C*n.*log(n)), k, Prb/t);
            set(gca,'YDir','normal');
            xlabel('Ambient Dimension p');
            ylabel('Sparsity k');
            title('Phase Transition: p = 5nlog(n)');
            colormap('gray');
            colorbar;
            pause(.25);
            
        end
    end
    pause(5);
end

%% plot phase transition graph
figure(1);
imagesc(round(C*n.*log(n)), k, Prb/T);
set(gca,'YDir','normal');
xlabel('Ambient Dimension p');
ylabel('Sparsity k');
title('Phase Transition: p = 5nlog(n)');
colormap('gray');
colorbar;
