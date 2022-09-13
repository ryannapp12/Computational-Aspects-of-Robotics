function err = erfun(q1,Td)
global f Jvf Jpif
T1 = f(q1);
T2 = Td;
dth = 0.5*(cross(T1(1:3,1),T2(1:3,1))+cross(T1(1:3,2),T2(1:3,2))+cross(T1(1:3,3),T2(1:3,3)));
dr = (T1(1:3,end)-T2(1:3,end));
err = -[dr;dth];