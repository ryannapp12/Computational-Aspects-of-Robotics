clear; clc; close all;
J = [-1.732, 2.598, 0;
    -1, 1.5, 0;
    0, 0, 1;
    0, -0.5, 0;
    0, 0.866, 0;
    1, 0, 0];
Jv = J(1:3,:);
Xee = ones(6,1);
lsqV = linspace(0.01,4,100);
err = [];
for i=1:length(lsqV)
    lsq = lsqV(i);
    Jpi = (Jv'*Jv+lsq*eye(size(Jv)))\(Jv');
    Jveecomp = Jpi*Xee(1:3);
    Xeecomp = Jv*Jveecomp;
    err = cat(1,err,norm(Xeecomp-Xee(1:3)));
end
plot(lsqV,err,'LineWidth',2,'Color',[1 0 0]);
title('Velocity error plot');
xlabel('\lambda^2'); ylabel('Velocity error');
grid on;
