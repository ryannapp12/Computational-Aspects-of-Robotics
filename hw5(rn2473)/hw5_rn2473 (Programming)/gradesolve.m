function [opt,iter,errHist, qHist] = gradesolve(q0,Td,alpha,tol)
global f Jvf Jpif
err = 1E3;
qk = q0;
dq = 1E2;
iter = 0;
errHist = [];
qHist = [];
while err > tol && dq > tol
    iter = iter + 1;
    errk = erfun(qk,Td);
    Jpi = Jvf(qk);
    qkp1 = qk + alpha*Jpi'*errk;
    err = norm(errk);
    errHist = cat(1,errHist,err);
    qHist = cat(2, qHist,qkp1);
    dq = norm(qk-qkp1);
    qk = qkp1;
end
opt = qk;