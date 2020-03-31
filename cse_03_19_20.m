function cse_03_19_20()
global phi0 tspan options dt

solver = struct('lsq',false);
% https://scicomp.stackexchange.com/questions/34672/automatically-generate-constraints-for-trajectory-optimization?noredirect=1&lq=1
ngrid = 10;  
dt = 0.01;
Dhat0  = 500*ones(ngrid-1,1); 
opts_fmin = optimoptions('fmincon','Display','iter', 'Algorithm', 'sqp'); %interior-point
opts_lsq = optimoptions('lsqnonlin','Display','iter', 'Algorithm', 'trust-region-reflective');
if solver.lsq == false
    % algorithm  = interior point (default), SQP, active set, and trust region reflective
   %[Dhat,~,~,output] = fmincon(@objfun,Dhat0,[],[],[],[],[],[],[], opts_fmin)
   [Dhat,~,~,output] = fmincon(@objfun,Dhat0,[],[],[],[],[],[],@defects, opts_fmin)

elseif solver.lsq == true
   % algorithm = trust region reflective(default) and Levenberg-Marquardt
   [Dhat,~,~,~,output] = lsqnonlin(@(Dhat) objfun(Dhat),Dhat0,[],[],opts_lsq)
end

function f = objfun(Dhat)

%% Integrator settings
phi0    = [5; 0; 0; 0; 0; 0; 0; 0; 0; 0];
tspan   = 0:dt:0.5;
options = odeset('abstol', 1e-10, 'reltol', 1e-9);

%% generate exact solution
    [t, phi]  = ode15s(@(t,phi) experimental(t,phi), tspan , phi0 ,options);


%% generate approximate solution

    [t, phi_tilde]  = ode15s(@(t,phi_tilde) predicted(t,phi_tilde, Dhat), tspan , phi0 ,options);


%% objective function for fminunc/fmincon
if solver.lsq == false
    f = sum((phi(:) - phi_tilde(:)).^2);
elseif solver.lsq == true     
%% objective function for lsqnonlin
    f  = phi - phi_tilde;
end
end

%--------------------------------------------------------------------------
% Experimental dphi/dt   = -M^TDMphi 
% this also forms the dynamic constraint for solving the optimizatin
% problem
%--------------------------------------------------------------------------
 function [dphi f_out] = experimental(t,phi)
     
 MT  = [-1  0  0  0   0  0  0  0  0;...
        1  -1  0  0   0  0  0  0  0;...
        0   1  -1 0   0  0  0  0  0;...
        0   0  1  -1  0  0  0  0  0;...
        0   0  0   1  -1 0  0  0  0;...
        0   0  0   0   1 -1 0  0  0;...
        0   0  0   0   0  1 -1 0  0;...
        0   0  0   0   0  0  1 -1 0;...
        0   0  0   0   0  0  0  1 -1;...
        0   0  0   0   0  0  0  0  1];

 
 M  = transpose(MT);
 D  = 5000*ones(ngrid-1,1);
 A  = MT*diag(D)*M;
 A  = A(2:ngrid-1,:);
 
 dphi(1,1) = 0;
 dphi(2:ngrid-1,1) = (-A*phi);
 dphi(ngrid,1) = (D(ngrid-1)*2*(phi(ngrid-1) - phi(ngrid)));
 
 
 end

%--------------------------------------------------------------------------
% Predicted
% -------------------------------------------------------------------------
 function dphi_hat = predicted(t,phi_hat, Dhat)

if size(phi_hat,2) ~= 1
    phi_hat = transpose(phi_hat);
end

 MT  = [-1  0  0  0   0  0  0  0  0;...
        1  -1  0  0   0  0  0  0  0;...
        0   1  -1 0   0  0  0  0  0;...
        0   0  1  -1  0  0  0  0  0;...
        0   0  0   1  -1 0  0  0  0;...
        0   0  0   0   1 -1 0  0  0;...
        0   0  0   0   0  1 -1 0  0;...
        0   0  0   0   0  0  1 -1 0;...
        0   0  0   0   0  0  0  1 -1;...
        0   0  0   0   0  0  0  0  1];
 
 M  = transpose(MT);
 A  = MT*diag(Dhat)*M;
 A  = A(2:ngrid-1,:);
 
 dphi_hat(1,1) = 0;
 dphi_hat(2:ngrid-1,1) = (-A*phi_hat);
 dphi_hat(ngrid,1) = (Dhat(ngrid-1)*2*(phi_hat(ngrid-1) - phi_hat(ngrid)));
 
 end
%--------------------------------------------------------------------------


function [c ceq] = defects(Dhat)
% ref: https://github.com/MatthewPeterKelly/OptimTraj
% This function computes the defects that are used to enforce the
% continuous dynamics of the system along the trajectory.
%
% INPUTS:
%   dt = time step (scalar)
%   x = [nState, nTime] = state at each grid-point along the trajectory
%   f = [nState, nTime] = dynamics of the state along the trajectory
%
% OUTPUTS:
%   defects = [nState, nTime-1] = error in dynamics along the trajectory
%   defectsGrad = [nState, nTime-1, nDecVars] = gradient of defects

% phi_tilde is replaced with x


afun = @(t,phi_tilde) predicted(t,phi_tilde,Dhat); % define the anonymous function once!
[t,phi_tilde] = ode15s(afun,tspan , phi0 ,options);
f = cellfun(afun,num2cell(t),num2cell(phi_tilde,2),'UniformOutput',false);
f = [f{:}].';

%   x = [nState, nTime] = state at each grid-point along the trajectory
%   f = [nState, nTime] = dynamics of the state along the trajectory
%   so taking transpose
phi_tilde = transpose(phi_tilde);
f = transpose(f);


nTime = size(phi_tilde,2);

idxLow = 1:(nTime-1);
idxUpp = 2:nTime;

xLow = phi_tilde(:,idxLow);
xUpp = phi_tilde(:,idxUpp);

fLow = f(:,idxLow);
fUpp = f(:,idxUpp);

% This is the key line:  (Trapazoid Rule)
defects = xUpp-xLow - 0.5*dt*(fLow+fUpp);
ceq = reshape(defects,numel(defects),1);
c = [];
end
end