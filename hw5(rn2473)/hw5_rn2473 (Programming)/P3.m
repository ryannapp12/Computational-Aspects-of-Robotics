clear; clc; close all;
global f Jvf Jpif
q0 = [0 0 0]';
d2 = 1;
Tod = [-0.014, 0.991, 0.135, 1;
    -0.1, -0.136, 0.986, 1;
    0.995, 0, 0.101, 1;
    0,0,0,1];
f = @(q) [cos(q(1))*cos(q(2)), -sin(q(1)), cos(q(1))*sin(q(2)),cos(q(1))*sin(q(2))*q(3)-d2*sin(q(1));
sin(q(1))*cos(q(2)), cos(q(1)), sin(q(1))*sin(q(2)),sin(q(1))*sin(q(2))*q(3)+d2*cos(q(1));
-sin(q(2)), 0, cos(q(2)), cos(q(2))*q(3);
0,0,0,1];
Jvf = @(q) [-q(3)*sin(q(1))*sin(q(2))-d2*cos(q(1)), q(3)*cos(q(1))*cos(q(2)),cos(q(1))*sin(q(2));
           q(3)*cos(q(1))*sin(q(2))-d2*sin(q(1)), q(3)*sin(q(1))*cos(q(2)), sin(q(1))*sin(q(2));
           0, -q(3)*sin(q(2)),cos(q(2));
           0, -sin(q(1)),0;
           0, cos(q(1)),0;
           1, 0, 0];
Jpif = @(q,lambda) (Jvf(q)'*Jvf(q)+ lambda*eye(3))\(Jvf(q)');
[qopt_newton, iter_newton, er_hist_newton, q_hist_newton] = newtonsolve(q0,Tod,0.02,1E-5);
[qopt_graddes, iter_graddes, er_hist_graddes, q_hist_graddes] = gradesolve(q0,Tod,0.58,1E-5);
figure(1);
plot(1:iter_newton,er_hist_newton,'LineWidth',2,'Color',[1,0,0],'DisplayName','Newton');
grid on;
xlabel('Iteration');
ylabel('Error norm');
title('Error norm plot for Newton algorithm');

figure(2);
plot(1:iter_graddes,er_hist_graddes,'LineWidth',2,'Color',[1,0,0],'DisplayName','Gradient Descent');
grid on;
xlabel('Iteration');
ylabel('Error norm');
title('Error norm plot for Gradient Descent');

figure(3);
plot(1:iter_graddes,q_hist_graddes(1,:),'LineWidth',2,'Color',[1,0,0],'DisplayName','\theta_1');
hold on;
plot(1:iter_graddes,q_hist_graddes(2,:),'LineWidth',2,'Color',[0,0,1],'DisplayName','\theta_2');
plot(1:iter_graddes,q_hist_graddes(3,:),'LineWidth',2,'Color',[1,0,1],'DisplayName','d_3');
grid on;
xlabel('Iteration');
ylabel('Joint variables');
title('Joint variables plot for Gradient Descent');
legend;

figure(4);
plot(1:iter_newton,q_hist_newton(1,:),'LineWidth',2,'Color',[1,0,0],'DisplayName','\theta_1');
hold on;
plot(1:iter_newton,q_hist_newton(2,:),'LineWidth',2,'Color',[0,0,1],'DisplayName','\theta_2');
plot(1:iter_newton,q_hist_newton(3,:),'LineWidth',2,'Color',[1,0,1],'DisplayName','d_3');
grid on;
xlabel('Iteration');
ylabel('Joint variables');
title('Joint variables plot for Newton');
legend;
