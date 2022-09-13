function [opt,iter,errHist,qHist] = newtonsolve(q0,Td,lambda,tol)
global f Jvf Jpif
err = 1E3;
qk = q0;
dq = 1E2;
errHist = [];
iter = 0;
qHist = [];
while err > tol && dq > tol
    iter = iter + 1;
    errk = erfun(qk,Td);
    Jpi = Jpif(qk,lambda);
    qkp1 = qk + Jpi*errk;
    err = norm(errk);
    qHist = cat(2, qHist,qkp1);
    errHist = cat(1,errHist,err);
    dq = norm(qk-qkp1);
    qk = qkp1;
end
opt = qk;