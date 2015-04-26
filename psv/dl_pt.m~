clc;close all;clear all;
% Implementation for the Phase Transition of the Alternating Direction Method presented in the paper
% "Finding a Sparse Vector in a Subspace: Linear Sparsity using Alternating Directions",
% by Qing Qu, Ju Sun, and John Wright.
%
% Dictionary Learning Model:
% Y = orth(Y_0)*Q, each column y_i of Y_0 has fixed k support, following
% i.i.d. Gaussian distribution, y_i is normalized to unitary.
% Q is a rotation matrix.
%
% gs.m: Gram-Schimdt orthogonalization function
% ADM.m: solve the following problem 
% min_{q,x} 1/2*||Y*q - x||_2^2 + lambda * ||x||_1, s.t. ||q||_2 = 1 
% by alternating minimization method (ADM), lambda is the penalty
% parameter.
% 
% Code written by Qing Qu, Ju Sun, and John Wright. 
% Last Updated: Sun 26 Apr 2015 12:23:53 PM EDT 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% parameter settings
n = 1:5:100; % subspace dimension
k = 5:20:600; % sparsity
C = 5; % the ratio between p and n: p = C * n * log n
T = 10; % simulation times for each pair (p,k)
MaxIter = 5000; % Max iteration before stop
tol_adm = 1e-5; % tolerance for convergence of the adm algorithm
tol_s = 1e-3; % error tolerance for judging the success of recovery
Prb = zeros(length(k),length(n));% record the recovery probability

for t = 1:T
    for i = 1:length(k)
        for j = 1:length(n)
            
            % skip the impossible case when k>p
            if (k(i)>C*n(j)*log(n(j)))
                continue;
            end
            p = round(C*n(j)*log(n(j)));
            
            % generate the data
            Y_0 = zeros(p,n(j));
            for l = 1:n(j)
                Y_0(randperm(p,k(i)),l) = randn(k(i),1);
                Y_0(:,l) = Y_0(:,l)/ norm(Y_0(:,l));
            end
            Y = orth(Y_0);
            Q = orth(randn(n(j),n(j)));
            Y = Y*Q; % rotate the matrix

            lambda = 1/sqrt(p);% penaltiy parameter
            
            % exhausting all initializations
            q_mtx = zeros(n(j),p);
            q_true = [1;zeros(n(j)-1,1)];
            count = 0;
            for l = 1:p
                q_0 = (Y(l,:)/norm(Y(l,:)))';
                q_mtx(:,l) = ADM(Y,q_0,lambda,MaxIter,tol_adm);
                x = Y*q_mtx(:,l);
                error = min([sum((x(:,ones(n(j),1))-Y_0).^2,1),sum((x(:,ones(n(j),1))+Y_0).^2,1)]);
                if(error<=tol_s)
                    count = count + 1;
                end
            end
            X = Y*q_mtx;            
            if (count~=0)
                Prb(i,j) = Prb(i,j) + 1;
            end
            
            % print intermediate results
            fprintf('Simulation=%d, Ambient Dimension = %d, Sparsity = %d\n'...
                ,t,p,k(i));            
            figure(1);
            imagesc(round(C*n.*log(n)), k, Prb/t);
            set(gca,'YDir','normal');
            xlabel('Sample Number p=5nlogn');
            ylabel('Sparsity k');
            colormap('gray');
            title('Phase Transition: p = 5nlog(n)');
            pause(.25);

        end
    end
    pause(5);
end

%% plot phase transition graph
figure(1);
imagesc(round(C*n.*log(n)), k, Prb/T);
set(gca,'YDir','normal');
xlabel('Sample Number p');
ylabel('Sparsity k');
colormap('gray');
title('Phase Transition: p = 5nlog(n)');
pause(.25);


