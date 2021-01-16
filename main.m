clear all
close all
clc

% Question 1.3
dim = 500;
b = 100 .* rand(dim, 1);
A = speye(dim);

f = @(x) (x-b)' * A * (x-b) + 3;
grad_f = @(x) 2 * A * (x-b);
hess_f = 2 * A;
f_star = f(b);

maxiter = 1e4;
epsilon = 1e-16;

% Starting point
x = zeros(dim,1);

% Gradient descent: fixed step size
[x_star, f_values] = gradient_descent(f, grad_f, hess_f, x, epsilon, maxiter);

residuals = f_values - f_star;
figure
len = length(residuals) ;
plot(linspace(0,len-1,len), residuals);
xlabel("Iterations k");
ylabel("Residuals: f(x^k) - f(x^*)");
hold on



% Graient descent; backtracking
alpha = 0.5;
beta = 0.5;

[x_star, f_values] = gradient_descent(f, grad_f, hess_f, x, epsilon, maxiter, alpha, beta);

residuals = f_values - f_star;

len = length(residuals);
plot(linspace(0,len-1,len), residuals);
xlabel("Iterations k");
ylabel("Residuals: f(x^k) - f(x^*)");

legend("Fixed step size", "Backtracking")
grid on


%% Question 1.4

clear all
close all
clc

dim = 500;
b = 99 .* rand(dim, 1) + 1;
A = sparse(diag((99 .* rand(dim, 1)) + 1));

f = @(x) (x-b)' * A * (x-b) + 3;
grad_f = @(x) 2 * A * (x-b);
hess_f = 2 * A;
f_star = f(b);

maxiter = 1e5;
epsilon = 1e-16;

% Starting point
x = zeros(dim,1);

% Gradient descent: fixed step size
[x_star, f_values] = gradient_descent(f, grad_f, hess_f, x, epsilon, maxiter);

residuals = f_values - f_star;
figure
len = length(residuals);
semilogy(linspace(0,len-1,len), residuals);
xlabel("Iterations k");
ylabel("Residuals: f(x^k) - f(x^*)");
hold on



% Graient descent; backtracking
alpha = 0.5;
beta = 0.5;

[x_star, f_values] = gradient_descent(f, grad_f, hess_f, x, epsilon, maxiter, alpha, beta);

residuals = f_values - f_star;

len = length(residuals);
semilogy(linspace(0,len-1,len), residuals);
xlabel("Iterations k");
ylabel("Residuals: f(x^k) - f(x^*)");

legend("Fixed step size", "Backtracking")
grid on

conditioning_number_A = condest(A)

%% Question 2.2: dual prohected gradient ascent
clear all
close all
clc

dim = 500;
b = 99 .* rand(dim, 1) + 1;
A = sparse(diag((99 .* rand(dim, 1)) + 1));

C = zeros(dim+1,dim);
C(1:dim,1:dim) = -1*eye(dim);
C(dim+1,1:dim) = ones(1,dim);
C = sparse(C);

d = zeros(dim+1,1);
d(dim+1) = 100;
d = sparse(d);

minus_g = @(u) 1/2*u'*C*A^-1*C'*u - u'*(C*b-d)-3;
%grad_minus_g = @(u) C*A^-1*C'*u-C*b+d;
hess_minus_g = C*inv(A)*C';

x_u = @(u) b - 0.5*inv(A)*C'*u;
grad_minus_g = @(x) -(C*x-d);


epsilon = 1e-8;
maxiter = int32(1/epsilon);
gamma = 1e-8;

% Starting point: feasible
u = ones(dim+1,1);

[u_star, residuals] = proj_gradient_descent(grad_minus_g, hess_minus_g, x_u, u, gamma, maxiter);

u_star;

figure
len = length(residuals);
semilogy(linspace(0,len-1,len), residuals);
xlabel("Iterations k");
ylabel("||u^k - u^{k-1}||_2");
grid on

x_star = b - 0.5*inv(A)*C'*u_star;

sum_x = sum(x_star)
negative_components = sum(x_star<-10*gamma)

f = @(x) (x-b)' * A * (x-b) + 3;
f_star = f(x_star)