t = tensor(rand(5,4,20));

timer1 = tic;
for i = 1:3
    CP_A = cp_apr(t, 20, 'printitn', 0);
end
cp_apr_time = toc(timer1);
disp(cp_apr_time);

timer2 = tic;
for i = 1:3
    CP_B = cp_als(t, 20, 'printitn', 0);
end
cp_als_time = toc(timer2);
disp(cp_als_time);