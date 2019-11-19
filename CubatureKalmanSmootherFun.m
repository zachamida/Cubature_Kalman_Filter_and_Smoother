%% Generic Code for Cubature Kalman Smoother
% Created by: Zachary Hamida
% November 19, 2019
%%
% This is a generic version of CKS, the original virsion is available
% at: https://www.haranarasaratnam.com/software.html
% Paper: Arasaratnam, Ienkaran, and Simon Haykin. "Cubature kalman smoothers."
% Automatica 47.10 (2011): 2245-2250.

%% INPUTS:
% 
% TransitionFunction:    non-linear transition function.
% Mu_KF: expected value of the state (Filter)
% Var_KF: variance of the state (Filter)
% Std_KF: standard deviation of the state (Filter)
% Q: process noise 

%% OUTPUTS:
%
% ExSmooth: expected value of the state (smoother)
% VarSmooth: variance of the state (smoother)

%% Note: This function is written to accommodate a single observation only
function [ExSmooth, VarSmooth]=CubatureKalmanSmootherFun(TransitionFunction, Mu_KF,...
    Var_KF,Std_KF,Q)
% size of state vector
nx=size(Mu_KF,1);
% Time series length
TotalTimeSteps=size(Mu_KF,2);
% Initilizing
Qsqrt=sqrt(Q);
[QPts,~,nPts]= FindCubaturePts(nx);
ExSmooth(:,TotalTimeSteps)=Mu_KF(:,TotalTimeSteps);
VarSmooth(:,:,TotalTimeSteps)=Var_KF(:,:,TotalTimeSteps);
StdSmooth(:,:,TotalTimeSteps)=Std_KF(:,:,TotalTimeSteps);

for t=TotalTimeSteps-1:-1:1
    X_sample = repmat(Mu_KF(:,t),1,nPts) + Std_KF(:,:,t)*QPts;
    Std_x_sample = (X_sample-repmat(Mu_KF(:,t),1,nPts))/sqrt(nPts);
    X_pred_sample = TransitionFunction(X_sample);
    Mu_Xpred_sample = sum(X_pred_sample,2)/nPts;
    Std_Xpred_sample = (X_pred_sample-repmat(Mu_Xpred_sample,1,nPts))/sqrt(nPts);
    [~,S] = qr([Std_Xpred_sample Qsqrt; Std_x_sample zeros(nx,nx)]',0);
    S = S';
    A = S(1:nx,1:nx);
    B = S(nx+1:end,1:nx);
    C = S(nx+1:end,nx+1:end);
    G = B/A;
    ExSmooth(:,t) = Mu_KF(:,t) + G*( ExSmooth(:,t+1) - Mu_Xpred_sample );
    [~,Std_x_update] = qr([ C  G*StdSmooth(:,:,t+1)]',0);
    StdSmooth(:,:,t) = Std_x_update';
    VarSmooth(:,:,t) = StdSmooth(:,:,t)*StdSmooth(:,:,t)';
end
end

function [QPts,wPts,nPts]= FindCubaturePts(n)

nPts = 2*n;
wPts = ones(1, nPts)/(2*n);
QPts = sqrt(n)*eye(n);
QPts = [QPts -QPts];
end